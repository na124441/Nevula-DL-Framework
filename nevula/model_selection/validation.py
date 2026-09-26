"""
Validation routines, cross-validation evaluation, out-of-fold predictions, and diagnostic curves.
"""

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
import math
import time
import numpy as np

from nevula.core.tensor import Tensor
from nevula.model_selection.split import check_cv, _num_samples
from nevula.model_selection.utils import (
    clone_estimator,
    get_scorer,
    is_classifier,
    safe_indexing,
    to_numpy,
)


def cross_val_score(
    estimator: Any,
    X: Any,
    y: Optional[Any] = None,
    cv: Union[int, Any] = 5,
    scoring: Optional[Union[str, Callable[[Any, Any, Any], float], Callable[[Any, Any], float]]] = None,
    fit_params: Optional[Dict[str, Any]] = None,
    verbose: int = 0,
) -> np.ndarray:
    """
    Evaluates an estimator's score by cross-validation.

    Args:
        estimator: The model to evaluate. Subclass of BaseModel or object implementing fit/predict.
        X: The input features (Tensor, ndarray, or sequence).
        y: The target values (Tensor, ndarray, or sequence).
        cv: Cross-validation generator or number of folds (default: 5).
        scoring: Metric string or callable(estimator, X_val, y_val).
        fit_params: Optional dictionary of keyword arguments passed to estimator.fit().
        verbose: Verbosity level (0: silent, 1: progress per fold).

    Returns:
        np.ndarray: Array of scores for each cross-validation fold.
    """
    results = cross_validate(
        estimator=estimator,
        X=X,
        y=y,
        cv=cv,
        scoring=scoring,
        return_train_score=False,
        return_estimator=False,
        fit_params=fit_params,
        verbose=verbose,
    )
    return results["test_score"]


def cross_validate(
    estimator: Any,
    X: Any,
    y: Optional[Any] = None,
    cv: Union[int, Any] = 5,
    scoring: Optional[Union[str, Callable, List[str], Dict[str, Union[str, Callable]]]] = None,
    return_train_score: bool = False,
    return_estimator: bool = False,
    fit_params: Optional[Dict[str, Any]] = None,
    verbose: int = 0,
) -> Dict[str, Any]:
    """
    Evaluates metric(s) by cross-validation and records fit and scoring runtimes.

    Args:
        estimator: The model to evaluate.
        X: Input features.
        y: Target values.
        cv: Cross-validation generator or number of folds.
        scoring: Single metric string/callable, list of metric names, or dict of named scorers.
        return_train_score: Whether to calculate and include training scores for each fold.
        return_estimator: Whether to return the fitted estimator for each fold.
        fit_params: Parameters passed into estimator.fit().
        verbose: Verbosity level.

    Returns:
        Dict[str, Any]: Dictionary containing 'fit_time', 'score_time', 'test_score',
                        and optionally 'train_score' and 'estimator'.
    """
    is_cls = is_classifier(estimator)
    cv_splitter = check_cv(cv, y=y, classifier=is_cls)
    fit_params = fit_params or {}

    # Standardize scorers into dict of {name: scorer_callable}
    scorers: Dict[str, Callable] = {}
    is_multimetric = False

    if isinstance(scoring, dict):
        is_multimetric = True
        for name, sc in scoring.items():
            scorers[name] = get_scorer(sc, estimator=estimator)
    elif isinstance(scoring, (list, tuple)):
        is_multimetric = True
        for sc in scoring:
            name = sc if isinstance(sc, str) else sc.__name__
            scorers[name] = get_scorer(sc, estimator=estimator)
    else:
        scorer = get_scorer(scoring, estimator=estimator)
        scorers["score"] = scorer

    fit_times: List[float] = []
    score_times: List[float] = []
    test_scores: Dict[str, List[float]] = {name: [] for name in scorers}
    train_scores: Dict[str, List[float]] = {name: [] for name in scorers}
    fitted_estimators: List[Any] = []

    splits = list(cv_splitter.split(X, y))
    n_splits = len(splits)

    for fold_idx, (train_idx, test_idx) in enumerate(splits):
        X_train = safe_indexing(X, train_idx)
        y_train = safe_indexing(y, train_idx) if y is not None else None
        X_test = safe_indexing(X, test_idx)
        y_test = safe_indexing(y, test_idx) if y is not None else None

        cloned_model = clone_estimator(estimator)

        # Fit
        t0_fit = time.perf_counter()
        if y_train is not None:
            cloned_model.fit(X_train, y_train, **fit_params)
        else:
            cloned_model.fit(X_train, **fit_params)
        t_fit = time.perf_counter() - t0_fit
        fit_times.append(t_fit)

        # Score test
        t0_score = time.perf_counter()
        for name, sc_fn in scorers.items():
            test_val = float(sc_fn(cloned_model, X_test, y_test))
            test_scores[name].append(test_val)

            if return_train_score:
                train_val = float(sc_fn(cloned_model, X_train, y_train))
                train_scores[name].append(train_val)
        t_score = time.perf_counter() - t0_score
        score_times.append(t_score)

        if return_estimator:
            fitted_estimators.append(cloned_model)

        if verbose > 0:
            first_metric = list(scorers.keys())[0]
            val = test_scores[first_metric][-1]
            print(f"[CV Fold {fold_idx + 1}/{n_splits}] {first_metric}={val:.4f} (fit: {t_fit:.3f}s)")

    results: Dict[str, Any] = {
        "fit_time": np.array(fit_times, dtype=np.float64),
        "score_time": np.array(score_times, dtype=np.float64),
    }

    if is_multimetric:
        for name, vals in test_scores.items():
            results[f"test_{name}"] = np.array(vals, dtype=np.float64)
            if return_train_score:
                results[f"train_{name}"] = np.array(train_scores[name], dtype=np.float64)
    else:
        results["test_score"] = np.array(test_scores["score"], dtype=np.float64)
        if return_train_score:
            results["train_score"] = np.array(train_scores["score"], dtype=np.float64)

    if return_estimator:
        results["estimator"] = fitted_estimators

    return results


def cross_val_predict(
    estimator: Any,
    X: Any,
    y: Optional[Any] = None,
    cv: Union[int, Any] = 5,
    method: str = "predict",
    fit_params: Optional[Dict[str, Any]] = None,
) -> Any:
    """
    Generates cross-validated out-of-fold estimates for each input sample.

    Args:
        estimator: The model to evaluate.
        X: Input features.
        y: Target values.
        cv: Cross-validation generator or fold count.
        method: Method to call on estimator ('predict', 'predict_proba', 'decision_function').
        fit_params: Keyword parameters passed to estimator.fit().

    Returns:
        Tensor or ndarray: Out-of-fold predictions with same sample count as X.
    """
    is_cls = is_classifier(estimator)
    cv_splitter = check_cv(cv, y=y, classifier=is_cls)
    fit_params = fit_params or {}

    n_samples = _num_samples(X)
    splits = list(cv_splitter.split(X, y))

    # Verify each sample is visited exactly once in test sets
    test_counts = np.zeros(n_samples, dtype=int)
    for _, test_idx in splits:
        test_counts[test_idx] += 1

    if not np.all(test_counts == 1):
        raise ValueError(
            "cross_val_predict requires cross-validation partitions where each sample "
            "is part of the test set exactly once (e.g., KFold, StratifiedKFold, LeaveOneOut)."
        )

    predictions: Optional[np.ndarray] = None

    for train_idx, test_idx in splits:
        X_train = safe_indexing(X, train_idx)
        y_train = safe_indexing(y, train_idx) if y is not None else None
        X_test = safe_indexing(X, test_idx)

        model = clone_estimator(estimator)
        if y_train is not None:
            model.fit(X_train, y_train, **fit_params)
        else:
            model.fit(X_train, **fit_params)

        pred_fn = getattr(model, method, None)
        if pred_fn is None:
            raise AttributeError(f"Estimator {model.__class__.__name__} has no method '{method}'.")

        fold_preds = pred_fn(X_test)
        fold_preds_np = to_numpy(fold_preds)

        if predictions is None:
            # Allocate container matching prediction shape
            out_shape = (n_samples,) + fold_preds_np.shape[1:]
            predictions = np.zeros(out_shape, dtype=fold_preds_np.dtype)

        predictions[test_idx] = fold_preds_np

    if isinstance(X, Tensor):
        return Tensor(predictions, device=X.device)
    return predictions


def learning_curve(
    estimator: Any,
    X: Any,
    y: Optional[Any] = None,
    train_sizes: Optional[Sequence[Union[float, int]]] = None,
    cv: Union[int, Any] = 5,
    scoring: Optional[Union[str, Callable]] = None,
    fit_params: Optional[Dict[str, Any]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes learning curve to diagnose model bias vs. variance.

    Evaluates training and test scores for varying training set sizes.

    Args:
        estimator: Model to evaluate.
        X: Input features.
        y: Target values.
        train_sizes: Relative fractions (e.g. [0.1, 0.33, 0.55, 0.78, 1.0]) or absolute counts.
        cv: Cross-validation generator or fold count.
        scoring: Metric string or callable.
        fit_params: Parameters for model.fit().

    Returns:
        Tuple of (train_sizes_abs, train_scores, test_scores):
            train_sizes_abs: 1D array of absolute training sample counts.
            train_scores: 2D array of shape (n_ticks, n_splits) with training scores.
            test_scores: 2D array of shape (n_ticks, n_splits) with validation scores.
    """
    if train_sizes is None:
        train_sizes = np.linspace(0.1, 1.0, 5)

    is_cls = is_classifier(estimator)
    cv_splitter = check_cv(cv, y=y, classifier=is_cls)
    scorer = get_scorer(scoring, estimator=estimator)
    fit_params = fit_params or {}

    splits = list(cv_splitter.split(X, y))
    n_splits = len(splits)

    # Compute maximum training samples across folds
    min_train_len = min(len(train_idx) for train_idx, _ in splits)

    train_sizes_abs: List[int] = []
    for s in train_sizes:
        if isinstance(s, float):
            count = int(math.floor(s * min_train_len))
        else:
            count = int(s)
        count = max(2, min(count, min_train_len))
        if count not in train_sizes_abs:
            train_sizes_abs.append(count)

    train_sizes_abs.sort()
    n_ticks = len(train_sizes_abs)

    train_scores = np.zeros((n_ticks, n_splits), dtype=np.float64)
    test_scores = np.zeros((n_ticks, n_splits), dtype=np.float64)

    for tick_idx, n_train_samples in enumerate(train_sizes_abs):
        for fold_idx, (train_idx, test_idx) in enumerate(splits):
            sub_train_idx = train_idx[:n_train_samples]
            X_train = safe_indexing(X, sub_train_idx)
            y_train = safe_indexing(y, sub_train_idx) if y is not None else None
            X_test = safe_indexing(X, test_idx)
            y_test = safe_indexing(y, test_idx) if y is not None else None

            model = clone_estimator(estimator)
            if y_train is not None:
                model.fit(X_train, y_train, **fit_params)
            else:
                model.fit(X_train, **fit_params)

            train_scores[tick_idx, fold_idx] = scorer(model, X_train, y_train)
            test_scores[tick_idx, fold_idx] = scorer(model, X_test, y_test)

    return np.array(train_sizes_abs, dtype=int), train_scores, test_scores


def validation_curve(
    estimator: Any,
    X: Any,
    y: Optional[Any] = None,
    param_name: str = "C",
    param_range: Sequence[Any] = (),
    cv: Union[int, Any] = 5,
    scoring: Optional[Union[str, Callable]] = None,
    fit_params: Optional[Dict[str, Any]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes validation curve to assess the impact of a single hyperparameter.

    Args:
        estimator: Model to evaluate.
        X: Input features.
        y: Target values.
        param_name: Name of hyperparameter to vary.
        param_range: Values of the parameter to evaluate.
        cv: Cross-validation generator or fold count.
        scoring: Metric string or callable.
        fit_params: Parameters for model.fit().

    Returns:
        Tuple of (train_scores, test_scores):
            train_scores: 2D array of shape (len(param_range), n_splits)
            test_scores: 2D array of shape (len(param_range), n_splits)
    """
    if len(param_range) == 0:
        raise ValueError("param_range must not be empty.")

    is_cls = is_classifier(estimator)
    cv_splitter = check_cv(cv, y=y, classifier=is_cls)
    scorer = get_scorer(scoring, estimator=estimator)
    fit_params = fit_params or {}

    splits = list(cv_splitter.split(X, y))
    n_splits = len(splits)
    n_params = len(param_range)

    train_scores = np.zeros((n_params, n_splits), dtype=np.float64)
    test_scores = np.zeros((n_params, n_splits), dtype=np.float64)

    for p_idx, param_val in enumerate(param_range):
        for fold_idx, (train_idx, test_idx) in enumerate(splits):
            X_train = safe_indexing(X, train_idx)
            y_train = safe_indexing(y, train_idx) if y is not None else None
            X_test = safe_indexing(X, test_idx)
            y_test = safe_indexing(y, test_idx) if y is not None else None

            model = clone_estimator(estimator)
            if hasattr(model, "set_params"):
                model.set_params(**{param_name: param_val})
            elif hasattr(model, param_name):
                setattr(model, param_name, param_val)
            elif hasattr(model, "get_config"):
                config = model.get_config()
                config[param_name] = param_val
                model = model.__class__(**config)
            else:
                setattr(model, param_name, param_val)

            if y_train is not None:
                model.fit(X_train, y_train, **fit_params)
            else:
                model.fit(X_train, **fit_params)

            train_scores[p_idx, fold_idx] = scorer(model, X_train, y_train)
            test_scores[p_idx, fold_idx] = scorer(model, X_test, y_test)

    return train_scores, test_scores
