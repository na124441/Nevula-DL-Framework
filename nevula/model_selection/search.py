"""
Hyperparameter optimization using Cross-Validated Grid and Randomized Search.
"""

from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Union
import itertools
import numpy as np

from nevula.model_selection.split import check_cv
from nevula.model_selection.utils import (
    clone_estimator,
    get_scorer,
    is_classifier,
    safe_indexing,
)
from nevula.model_selection.validation import cross_validate


class ParameterGrid:
    """
    Grid of parameters with a discrete number of values for each.

    Enables iterating over all Cartesian combinations of parameter lists.

    Args:
        param_grid: Dictionary with parameter names as keys and lists of
                    parameter settings to try as values, or a list of such dictionaries.
    """

    def __init__(self, param_grid: Union[Dict[str, Sequence[Any]], List[Dict[str, Sequence[Any]]]]):
        if isinstance(param_grid, dict):
            self.param_grid = [param_grid]
        else:
            self.param_grid = list(param_grid)

        for p in self.param_grid:
            for k, v in p.items():
                if isinstance(v, (str, bytes)) or not hasattr(v, "__iter__"):
                    raise TypeError(
                        f"Parameter values for key '{k}' must be an iterable list/sequence; got {type(v)}"
                    )

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        for p in self.param_grid:
            items = sorted(p.items())
            if not items:
                yield {}
                continue
            keys, values = zip(*items)
            for v in itertools.product(*values):
                yield dict(zip(keys, v))

    def __len__(self) -> int:
        total = 0
        for p in self.param_grid:
            product = 1
            for v in p.values():
                product *= len(v)
            total += product
        return total

    def __getitem__(self, index: int) -> Dict[str, Any]:
        for i, params in enumerate(self):
            if i == index:
                return params
        raise IndexError(f"Index {index} out of bounds for ParameterGrid of size {len(self)}")


class ParameterSampler:
    """
    Generator on parameters sampled from given distributions or candidate lists.

    Args:
        param_distributions: Dict with parameter names as keys and distributions or lists.
        n_iter: Number of parameter settings that are produced.
        random_state: Pseudo-random number generator seed.
    """

    def __init__(
        self,
        param_distributions: Dict[str, Any],
        n_iter: int = 10,
        random_state: Optional[Union[int, np.random.RandomState]] = None,
    ):
        self.param_distributions = param_distributions
        self.n_iter = int(n_iter)
        self.random_state = random_state

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        rng = np.random.RandomState(self.random_state) if not isinstance(self.random_state, np.random.RandomState) else self.random_state

        for _ in range(self.n_iter):
            params: Dict[str, Any] = {}
            for k, v in self.param_distributions.items():
                if hasattr(v, "rvs"):
                    # scipy distribution
                    params[k] = v.rvs(random_state=rng)
                elif isinstance(v, (list, tuple, np.ndarray)):
                    idx = rng.randint(0, len(v))
                    params[k] = v[idx]
                else:
                    params[k] = v
            yield params

    def __len__(self) -> int:
        return self.n_iter


def _set_params(estimator: Any, **params: Any) -> Any:
    """Safely updates hyperparameter attributes on an estimator instance."""
    if hasattr(estimator, "set_params"):
        return estimator.set_params(**params)

    if hasattr(estimator, "get_config"):
        try:
            config = estimator.get_config()
            config.update(params)
            return estimator.__class__(**config)
        except Exception:
            pass

    for k, v in params.items():
        setattr(estimator, k, v)
    return estimator


class BaseSearchCV:
    """Base class for hyperparameter search with cross-validation."""

    def __init__(
        self,
        estimator: Any,
        cv: Union[int, Any] = 5,
        scoring: Optional[Union[str, Callable]] = None,
        refit: bool = True,
        verbose: int = 0,
    ):
        self.estimator = estimator
        self.cv = cv
        self.scoring = scoring
        self.refit = refit
        self.verbose = verbose

        self.best_params_: Optional[Dict[str, Any]] = None
        self.best_score_: Optional[float] = None
        self.best_estimator_: Optional[Any] = None
        self.best_index_: Optional[int] = None
        self.cv_results_: Dict[str, Any] = {}
        self.scorer_: Optional[Callable] = None
        self.n_splits_: int = 0

    def _run_search(self, candidate_params: Sequence[Dict[str, Any]], X: Any, y: Optional[Any] = None, **fit_params: Any) -> "BaseSearchCV":
        candidate_list = list(candidate_params)
        n_candidates = len(candidate_list)
        if n_candidates == 0:
            raise ValueError("No parameter candidates to evaluate.")

        is_cls = is_classifier(self.estimator)
        cv_splitter = check_cv(self.cv, y=y, classifier=is_cls)
        scorer = get_scorer(self.scoring, estimator=self.estimator)
        self.scorer_ = scorer

        all_splits = list(cv_splitter.split(X, y))
        self.n_splits_ = len(all_splits)

        all_test_scores: List[np.ndarray] = []
        all_fit_times: List[np.ndarray] = []
        all_score_times: List[np.ndarray] = []

        best_score = -float("inf")
        best_index = -1
        best_params = None

        for cand_idx, params in enumerate(candidate_list):
            cand_est = clone_estimator(self.estimator)
            cand_est = _set_params(cand_est, **params)

            cv_res = cross_validate(
                estimator=cand_est,
                X=X,
                y=y,
                cv=cv_splitter,
                scoring=scorer,
                return_train_score=False,
                fit_params=fit_params,
                verbose=0,
            )

            scores = cv_res["test_score"]
            mean_score = float(np.mean(scores))
            all_test_scores.append(scores)
            all_fit_times.append(cv_res["fit_time"])
            all_score_times.append(cv_res["score_time"])

            if mean_score > best_score or best_params is None:
                best_score = mean_score
                best_params = params
                best_index = cand_idx

            if self.verbose > 0:
                print(
                    f"[{cand_idx + 1:3d}/{n_candidates:3d}] "
                    f"Score: {mean_score:.4f} (+/- {np.std(scores):.4f}) | Params: {params}"
                )

        self.best_params_ = best_params
        self.best_score_ = best_score
        self.best_index_ = best_index

        # Build cv_results_ dictionary
        all_test_scores_arr = np.array(all_test_scores)  # shape: (n_candidates, n_splits)
        mean_test_scores = np.mean(all_test_scores_arr, axis=1)
        std_test_scores = np.std(all_test_scores_arr, axis=1)

        # Rank test scores descending (rank 1 is best)
        order = np.argsort(-mean_test_scores)
        ranks = np.empty_like(order)
        ranks[order] = np.arange(1, len(order) + 1)

        results: Dict[str, Any] = {
            "params": candidate_list,
            "mean_test_score": mean_test_scores,
            "std_test_score": std_test_scores,
            "rank_test_score": ranks,
            "mean_fit_time": np.mean(all_fit_times, axis=1),
            "std_fit_time": np.std(all_fit_times, axis=1),
            "mean_score_time": np.mean(all_score_times, axis=1),
            "std_score_time": np.std(all_score_times, axis=1),
        }

        for split_i in range(self.n_splits_):
            results[f"split{split_i}_test_score"] = all_test_scores_arr[:, split_i]

        self.cv_results_ = results

        # Refit best estimator on complete dataset
        if self.refit:
            refitted = clone_estimator(self.estimator)
            refitted = _set_params(refitted, **best_params)
            if y is not None:
                refitted.fit(X, y, **fit_params)
            else:
                refitted.fit(X, **fit_params)
            self.best_estimator_ = refitted

        return self

    def predict(self, X: Any) -> Any:
        """Generates predictions using the refitted best estimator."""
        if not self.refit or self.best_estimator_ is None:
            raise RuntimeError("This GridSearchCV instance was not fitted with refit=True.")
        return self.best_estimator_.predict(X)

    def predict_proba(self, X: Any) -> Any:
        """Predicts class probabilities using the refitted best estimator."""
        if not self.refit or self.best_estimator_ is None:
            raise RuntimeError("This GridSearchCV instance was not fitted with refit=True.")
        if not hasattr(self.best_estimator_, "predict_proba"):
            raise AttributeError(f"{self.best_estimator_.__class__.__name__} does not support predict_proba.")
        return self.best_estimator_.predict_proba(X)

    def decision_function(self, X: Any) -> Any:
        """Computes decision function scores using the refitted best estimator."""
        if not self.refit or self.best_estimator_ is None:
            raise RuntimeError("This GridSearchCV instance was not fitted with refit=True.")
        if not hasattr(self.best_estimator_, "decision_function"):
            raise AttributeError(f"{self.best_estimator_.__class__.__name__} does not support decision_function.")
        return self.best_estimator_.decision_function(X)

    def evaluate(self, X: Any, y: Any, **kwargs: Any) -> float:
        """Evaluates test performance on the refitted best estimator."""
        if not self.refit or self.best_estimator_ is None:
            raise RuntimeError("This search instance was not fitted with refit=True.")
        return float(self.best_estimator_.evaluate(X, y, **kwargs))

    def score(self, X: Any, y: Optional[Any] = None) -> float:
        """Scores predictions on (X, y) using the configured scoring metric."""
        if not self.refit or self.best_estimator_ is None:
            raise RuntimeError("This search instance was not fitted with refit=True.")
        if self.scorer_ is None:
            scorer = get_scorer(self.scoring, estimator=self.best_estimator_)
        else:
            scorer = self.scorer_
        return float(scorer(self.best_estimator_, X, y))


class GridSearchCV(BaseSearchCV):
    """
    Exhaustive search over specified parameter values for an estimator.

    Optimizes hyperparameters using cross-validation over the Cartesian product
    of all candidate parameter settings.

    Args:
        estimator: Model instance to tune.
        param_grid: Dict or list of dicts with parameter names as keys and lists of settings.
        scoring: Metric string or callable to evaluate predictions.
        cv: Cross-validation generator or fold count (default: 5).
        refit: Whether to refit the best model on the complete dataset (default: True).
        verbose: Verbosity level (0: silent, 1: progress per combination).
    """

    def __init__(
        self,
        estimator: Any,
        param_grid: Union[Dict[str, Sequence[Any]], List[Dict[str, Sequence[Any]]]],
        scoring: Optional[Union[str, Callable]] = None,
        cv: Union[int, Any] = 5,
        refit: bool = True,
        verbose: int = 0,
    ):
        super().__init__(
            estimator=estimator,
            cv=cv,
            scoring=scoring,
            refit=refit,
            verbose=verbose,
        )
        self.param_grid = ParameterGrid(param_grid)

    def fit(self, X: Any, y: Optional[Any] = None, **fit_params: Any) -> "GridSearchCV":
        """Runs fit with all sets of parameters in param_grid."""
        self._run_search(list(self.param_grid), X, y, **fit_params)
        return self

    def __repr__(self) -> str:
        return (
            f"GridSearchCV(estimator={self.estimator.__class__.__name__}, "
            f"n_candidates={len(self.param_grid)}, cv={self.cv}, scoring='{self.scoring}')"
        )


class RandomizedSearchCV(BaseSearchCV):
    """
    Randomized search over hyperparameter distributions or lists.

    In contrast to GridSearchCV, not all parameter values are tried out,
    but rather a fixed number of parameter settings is sampled from the
    specified distributions.

    Args:
        estimator: Model instance to tune.
        param_distributions: Dict with parameters names as keys and distributions or lists.
        n_iter: Number of parameter settings sampled (default: 10).
        scoring: Metric string or callable.
        cv: Cross-validation generator or fold count (default: 5).
        refit: Whether to refit the best model on the complete dataset (default: True).
        random_state: Pseudo-random number generator seed.
        verbose: Verbosity level.
    """

    def __init__(
        self,
        estimator: Any,
        param_distributions: Dict[str, Any],
        n_iter: int = 10,
        scoring: Optional[Union[str, Callable]] = None,
        cv: Union[int, Any] = 5,
        refit: bool = True,
        random_state: Optional[Union[int, np.random.RandomState]] = None,
        verbose: int = 0,
    ):
        super().__init__(
            estimator=estimator,
            cv=cv,
            scoring=scoring,
            refit=refit,
            verbose=verbose,
        )
        self.param_distributions = param_distributions
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X: Any, y: Optional[Any] = None, **fit_params: Any) -> "RandomizedSearchCV":
        """Runs fit over randomly sampled parameter settings."""
        sampler = ParameterSampler(
            self.param_distributions,
            n_iter=self.n_iter,
            random_state=self.random_state,
        )
        self._run_search(list(sampler), X, y, **fit_params)
        return self

    def __repr__(self) -> str:
        return (
            f"RandomizedSearchCV(estimator={self.estimator.__class__.__name__}, "
            f"n_iter={self.n_iter}, cv={self.cv}, scoring='{self.scoring}')"
        )
