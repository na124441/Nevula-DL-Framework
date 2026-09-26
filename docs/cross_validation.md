# Cross-Validation & Model Selection Guide

Nevula provides a unified, production-grade cross-validation and hyperparameter optimization suite. It supports native Nevula `Tensor`, NumPy `ndarray`, Python sequences, and Nevula `Dataset` objects with seamless type preservation.

---

## 1. Splitting Strategies & Cross-Validators

Cross-validation guards against data snooping and model overfitting by partitioning the dataset into complementary subsets, ensuring models generalize reliably to unseen data.

| Class | Splitting Strategy | Primary Use Case |
| :--- | :--- | :--- |
| `KFold(n_splits=5, shuffle=False, random_state=None)` | Standard $k$-fold partitioning | General regression and balanced classification |
| `StratifiedKFold(n_splits=5, shuffle=False, random_state=None)` | Stratified $k$-fold preserving class ratios | Class-imbalanced classification |
| `LeaveOneOut()` | Exhaustive leave-one-out ($N$ folds, test size 1) | Small sample regime ($N < 50$) |
| `LeavePOut(p=2)` | Exhaustive leave-$p$-out ($\binom{N}{p}$ combinations) | Combinatorial sensitivity analysis |
| `ShuffleSplit(n_splits=10, test_size=0.1, random_state=None)` | Monte Carlo random permutations | Arbitrary train/test fractions |
| `StratifiedShuffleSplit(n_splits=10, test_size=0.1, random_state=None)` | Stratified Monte Carlo permutations | Fast randomized classification splits |
| `TimeSeriesSplit(n_splits=5, max_train_size=None, test_size=None, gap=0)` | Rolling / forward-chaining walk | Sequential and temporal forecasting |
| `train_test_split(*arrays, test_size=0.25, shuffle=True, stratify=None)` | One-shot train/test split | Baseline benchmarking |

---

## 2. Evaluation Routines

### `cross_val_score`
Evaluates metric scores for each fold:

$$\text{Scores} = [s_1, s_2, \dots, s_k]$$

```python
import nevula as nv
from nevula.models.classification.logistic import LogisticRegression
from nevula.model_selection import cross_val_score

clf = LogisticRegression(in_features=4, n_classes=3)
scores = cross_val_score(clf, X, y, cv=5, scoring="accuracy")
print(f"Accuracy: {scores.mean():.4f} (+/- {scores.std():.4f})")
```

### `cross_validate`
Comprehensive cross-validation tracking fit times, score times, and multiple evaluation metrics:

```python
from nevula.model_selection import cross_validate

results = cross_validate(
    clf,
    X,
    y,
    cv=5,
    scoring={"acc": "accuracy", "f1": "f1_macro", "roc_auc": "roc_auc"},
    return_train_score=True,
    return_estimator=True,
)

print("Test Accuracy: ", results["test_acc"].mean())
print("Test F1:       ", results["test_f1"].mean())
print("Fit Time (ms): ", results["fit_time"].mean() * 1000)
```

### `cross_val_predict`
Generates out-of-fold clean predictions where every sample's prediction is produced by an estimator trained without that sample:

$$\hat{y}_i = f_{-k(i)}(x_i)$$

```python
from nevula.model_selection import cross_val_predict

oof_preds = cross_val_predict(clf, X, y, cv=5, method="predict")
oof_probs = cross_val_predict(clf, X, y, cv=5, method="predict_proba")
```

---

## 3. Diagnostic Curves

### `learning_curve`
Diagnoses bias (underfitting) vs. variance (overfitting) by tracking performance across increasing training sample sizes:

```python
from nevula.model_selection import learning_curve

train_sizes, train_scores, test_scores = learning_curve(
    estimator=model,
    X=X,
    y=y,
    train_sizes=[0.2, 0.4, 0.6, 0.8, 1.0],
    cv=5,
    scoring="r2",
)
```

### `validation_curve`
Assesses estimator sensitivity to a single hyperparameter:

```python
from nevula.model_selection import validation_curve

train_scores, test_scores = validation_curve(
    estimator=model,
    X=X,
    y=y,
    param_name="alpha",
    param_range=[0.001, 0.01, 0.1, 1.0, 10.0],
    cv=5,
    scoring="neg_mean_squared_error",
)
```

---

## 4. Hyperparameter Search & Tuning

### `GridSearchCV`
Exhaustive Cartesian product search over candidate parameter grids:

```python
from nevula.models.regression.ridge import RidgeRegression
from nevula.model_selection import GridSearchCV

param_grid = {
    "alpha": [0.001, 0.01, 0.1, 1.0, 10.0],
    "fit_intercept": [True, False],
}

grid = GridSearchCV(
    estimator=RidgeRegression(in_features=8, out_features=1),
    param_grid=param_grid,
    cv=5,
    scoring="r2",
    refit=True,  # Refits best model on entire dataset
    verbose=1,
)

grid.fit(X_train, y_train)

print("Best Parameters: ", grid.best_params_)
print("Best CV Score:   ", grid.best_score_)

# Direct inference using the refitted best model
y_pred = grid.predict(X_test)
```

### `RandomizedSearchCV`
Randomized parameter sampling for large or continuous search spaces:

```python
from nevula.model_selection import RandomizedSearchCV

param_dists = {
    "C": [0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0],
    "penalty": ["l1", "l2"],
}

rand_search = RandomizedSearchCV(
    estimator=LogisticRegression(in_features=10, n_classes=2),
    param_distributions=param_dists,
    n_iter=8,
    cv=4,
    scoring="accuracy",
    refit=True,
    random_state=42,
)

rand_search.fit(X_train, y_train, epochs=40, lr=0.1)
print("Best Params: ", rand_search.best_params_)
```

---

## 5. Scoring Metrics Reference

Nevula supports standard metric aliases:

- **Regression**: `'r2'`, `'mse'`, `'neg_mean_squared_error'`, `'mae'`, `'neg_mean_absolute_error'`, `'rmse'`, `'neg_root_mean_squared_error'`
- **Classification**: `'accuracy'`, `'precision'`, `'precision_macro'`, `'precision_weighted'`, `'recall'`, `'recall_macro'`, `'recall_weighted'`, `'f1'`, `'f1_macro'`, `'f1_weighted'`, `'roc_auc'`, `'log_loss'`, `'neg_log_loss'`, `'bce'`
- **Custom Callables**:
  - `scorer(estimator, X_val, y_val) -> float`
  - `metric(y_true, y_pred) -> float`
