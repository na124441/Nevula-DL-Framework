# 📚 Nevula Model Catalogue

> **Complete Registry and Architecture Directory for Machine Learning & Deep Learning Models in the Nevula Framework.**

The **Nevula Model Library** (`nevula.models`) is an extensible subsystem providing transparent, first-principles implementations of core machine learning and deep learning algorithms. Every model is built upon Nevula's foundational primitives—tensors, computational graphs, reverse-mode automatic differentiation, and optimizers—while adhering to a unified, clean API contract (`BaseModel`).

---

## 🗂️ Model Registry Quick Summary

You can inspect the live registry at runtime directly from Python:

```python
import nevula
print(nevula.models.summary())
```

```text
Available models in Nevula Model Library:

Classification:
  - LogisticRegression
  - SVC

Ensemble:
  - GradientBoostingRegressor
  - RandomForestRegressor
  - VotingRegressor

Regression:
  - LassoRegression
  - LinearRegression
  - RidgeRegression
  - SVR

Trees:
  - DecisionTreeRegressor
```

---

## 📊 Comprehensive Comparison Matrix

| Model | Category | Canonical Key | Aliases | Paradigm | Key Mechanism | Best For | Docs |
|---|---|---|---|---|---|---|---|
| **Logistic Regression** | `classification` | `logistic_regression` | `logistic` | Parametric Linear Probabilistic | Sigmoid Link, Softmax, BCEWithLogits & Autograd GD | Binary & multiclass classification, calibrated class probabilities | [Docs](docs/models/logistic_regression.md) |
| **Support Vector Machine (SVC)** | `classification` | `svc` | `svm`, `support_vector_classifier` | Margin-Based Linear | Primal Hinge / Squared Hinge Loss & $L_2$ Regularization | Maximum margin classification, outlier resilience, support vector localization | [Docs](docs/models/svm.md) |
| **Linear Regression** | `regression` | `linear_regression` | `linear`, `ols` | Parametric Linear | Closed-form OLS & Autograd GD | Baseline linear problems, continuous targets | [Docs](docs/models/linear_regression.md) |
| **Ridge Regression** | `regression` | `ridge_regression` | `ridge`, `tikhonov` | Parametric Regularized Linear | $L_2$ Regularization ($\frac{\alpha}{2}\|w\|^2$) | Multicollinear features, ill-conditioned matrices | [Docs](docs/models/ridge_regression.md) |
| **Lasso Regression** | `regression` | `lasso_regression` | `lasso`, `l1_regression` | Parametric Sparse Linear | $L_1$ Regularization ($\alpha\|w\|_1$) & Subgradient / Coordinate Descent | High-dimensional data, automatic feature selection | [Docs](docs/models/lasso_regression.md) |
| **Support Vector Regression (SVR)** | `regression` | `svr` | `support_vector_regression` | Margin-Based Convex | $\epsilon$-Insensitive Tube Loss + $L_2$ Regularization | Outlier-heavy datasets, robust error margins | [Docs](docs/models/svr.md) |
| **Decision Tree Regressor** | `trees` | `decision_tree_regressor` | `decision_tree`, `decision_tree_regression` | Non-Parametric Recursive Partitioning | CART Variance Reduction Splitting | Non-linear relationships, step functions, interactions | [Docs](docs/models/decision_tree_regression.md) |
| **Voting Regressor** | `ensemble` | `voting_regressor` | `voting` | Heterogeneous Ensembling | Weighted/Uniform Model Prediction Averaging | Combining diverse inductive biases (e.g. Ridge + SVR + Trees) | [Docs](docs/models/ensemble_learning.md) |
| **Random Forest Regressor** | `ensemble` | `random_forest_regressor` | `random_forest` | Bagging Ensembling | Bootstrap Aggregating + Random Feature Subspaces | Overcoming tree variance/overfitting, robust non-linear modeling | [Docs](docs/models/ensemble_learning.md) |
| **Gradient Boosting Regressor** | `ensemble` | `gradient_boosting_regressor` | `gradient_boosting` | Boosting Ensembling | Forward Stage-wise Pseudo-Residual Fitting with Shrinkage | High-accuracy non-linear prediction, bias minimization | [Docs](docs/models/ensemble_learning.md) |

---

## 🔬 Model Deep Dive

### 1. Classification Models (`nevula.models.classification`)

#### 1.1 Logistic Regression (`LogisticRegression`)
- **Module**: `nevula.models.classification.logistic`
- **Link Functions**:
  - Binary: Sigmoid link function $\sigma(z) = \frac{1}{1 + e^{-z}}$, trained with numerically stable `BCEWithLogitsLoss`.
  - Multiclass: Softmax normalized exponential distribution $P(y = c \mid x) = \frac{e^{z_c}}{\sum_k e^{z_k}}$, trained with `CrossEntropyLoss`.
- **Objective Function**:
  $$\mathcal{L}_{\text{BCE}}(W, b) = -\frac{1}{N} \sum_{i=1}^N \left[ y_i \log(\sigma(z_i)) + (1 - y_i) \log(1 - \sigma(z_i)) \right] + \mathcal{R}(W)$$
- **Key Features**:
  - Probabilistic classification with calibrated class likelihoods via `.predict_proba()`.
  - Linear decision boundary hyperplane extraction via `.decision_function()`.
  - Regularization options: $L_2$ (Ridge penalty) and $L_1$ (Lasso penalty with subgradients).
  - Evaluates accuracy, precision, recall, F1 score, confusion matrix, and binary cross-entropy loss.
- **Example**: `examples/logistic_regression.py`
- **Interactive Lab**: `model_lab/classification/logistic_regression/`

#### 1.2 Support Vector Machine (`SVC` / `SVM`)
- **Module**: `nevula.models.classification.svm`
- **Aliases**: `svm`, `support_vector_classifier`
- **Objective Function**:
  $$\min_{W, b} \mathcal{L}(W, b) = \frac{1}{N} \sum_{i=1}^N \max(0, 1 - y_i (x_i^T W + b))^p + \frac{1}{2C} \|W\|_2^2$$
  where $p = 1$ for standard Hinge Loss and $p = 2$ for Squared Hinge Loss.
- **Key Features**:
  - Maximum geometric margin classification with subgradients computed via autograd.
  - Native Multiclass One-vs-Rest (OvR) hyperplane training.
  - Diagnostic Support Vector extraction (`support_vectors`, `support_vectors_mask`, `support_vectors_ratio`, `n_support_`).
  - Evaluates accuracy, precision, recall, F1 score, confusion matrix, and hinge loss score.
- **Example**: `examples/svm.py`
- **Interactive Lab**: `model_lab/classification/svm/`

---

### 2. Regression Models (`nevula.models.regression`)

#### 2.1 Linear Regression (`LinearRegression`)
- **Module**: `nevula.models.regression.linear`
- **Objective Function**:
  $$\mathcal{L}_{\text{MSE}}(W, b) = \frac{1}{2N} \sum_{i=1}^N \left( \hat{y}_i - y_i \right)^2$$
- **Dual Solving Modes**:
  - `mode='framework'`: Iterative gradient descent using Nevula's autograd computation graph and optimizers (`SGD`, `Adam`).
  - `mode='analytical'`: Direct analytical normal equation: $W^* = (X^T X)^{-1} X^T y$.
- **Example**: `examples/linear_regression.py`
- **Interactive Lab**: `model_lab/regression/linear_regression/`

#### 1.2 Ridge Regression (`RidgeRegression`)
- **Module**: `nevula.models.regression.ridge`
- **Objective Function**:
  $$\mathcal{L}_{\text{Ridge}}(W, b) = \frac{1}{2N} \|XW + b - y\|_2^2 + \frac{\alpha}{2} \|W\|_2^2$$
- **Key Features**:
  - Tikhonov regularization dampens large oscillating weights.
  - Closed-form analytical solution: $W^* = (X^T X + \alpha I)^{-1} X^T y$.
  - Autograd-driven gradient descent with weight decay.
- **Example**: `examples/ridge_regression.py`
- **Interactive Lab**: `model_lab/regression/ridge_regression/`

#### 1.3 Lasso Regression (`LassoRegression`)
- **Module**: `nevula.models.regression.lasso`
- **Objective Function**:
  $$\mathcal{L}_{\text{Lasso}}(W, b) = \frac{1}{2N} \|XW + b - y\|_2^2 + \alpha \|W\|_1$$
- **Key Features**:
  - Promotes structural parameter sparsity (automatic feature selection).
  - Subgradient descent using Nevula's differentiable `Tensor.abs()`.
  - Fast analytical Coordinate Descent with soft-thresholding operator $\mathcal{S}_{\lambda}(z) = \text{sign}(z) \max(|z| - \lambda, 0)$.
  - `.sparsity()` diagnostic reporting exact proportion of zeroed weights.
- **Example**: `examples/lasso_regression.py`
- **Interactive Lab**: `model_lab/regression/lasso_regression/`

#### 1.4 Support Vector Regression (`SVR`)
- **Module**: `nevula.models.regression.svr`
- **Objective Function**:
  $$\mathcal{L}_{\text{SVR}}(W, b) = \frac{1}{N} \sum_{i=1}^N \max\left(0, |y_i - \hat{y}_i| - \epsilon\right) + \frac{1}{2C} \|W\|_2^2$$
- **Key Features**:
  - $\epsilon$-insensitive error tube ignores small residuals $\le \epsilon$.
  - Linear penalty outside the tube ensures robustness against extreme anomalous outliers.
  - `.support_vectors(X, y)` extraction identifying boundary data points.
- **Example**: `examples/svr.py`
- **Interactive Lab**: `model_lab/regression/svr/`

---

### 2. Tree-Based Models (`nevula.models.trees`)

#### 2.1 Decision Tree Regressor (`DecisionTreeRegressor`)
- **Module**: `nevula.models.trees.decision_tree`
- **Partitioning Model**:
  $$\hat{y}(x) = \sum_{m=1}^M c_m \cdot \mathbb{I}(x \in R_m), \quad c_m = \frac{1}{N_m}\sum_{x_i \in R_m} y_i$$
- **Key Features**:
  - CART greedy recursive binary partitioning maximizing variance reduction (MSE decrease).
  - Completely non-parametric: handles complex step functions, non-linear waves, and high-order interactions without basis expansion.
  - Invariant to monotonic feature scaling.
  - Configurable stopping criteria: `max_depth`, `min_samples_split`, `min_samples_leaf`, `min_impurity_decrease`.
  - Normalized `.feature_importances()` vector and structural diagnostics (`tree_depth()`, `n_leaves()`).
- **Example**: `examples/decision_tree_regression.py`
- **Interactive Lab**: `model_lab/trees/decision_tree_regressor/`

---

### 3. Ensemble Learning Models (`nevula.models.ensemble`)

#### 3.1 Voting Regressor (`VotingRegressor`)
- **Module**: `nevula.models.ensemble.voting`
- **Aggregation Formula**:
  $$\hat{y}(x) = \sum_{k=1}^K w_k \cdot \hat{f}_k(x), \quad \text{where } \sum_{k=1}^K w_k = 1$$
- **Key Features**:
  - Combines diverse, heterogeneous Nevula estimators (e.g. `Ridge` + `SVR` + `DecisionTree`).
  - Supports custom positive weighting or default uniform weighting.
  - Fits each constituent model independently.

#### 3.2 Random Forest Regressor (`RandomForestRegressor`)
- **Module**: `nevula.models.ensemble.random_forest`
- **Aggregation & Variance Reduction**:
  $$\hat{y}(x) = \frac{1}{B} \sum_{b=1}^B h_b(x), \quad \text{Var}(\hat{y}) = \rho \sigma^2 + \frac{1 - \rho}{B} \sigma^2$$
- **Key Features**:
  - Bootstrap Aggregation (Bagging) with replacement over $B$ regression trees.
  - Random feature subspace selection (`max_features="sqrt"`, `"log2"`, fraction, or count) at each split de-correlates trees ($\rho \downarrow$).
  - Built-in Out-of-Bag (OOB) generalization evaluation (`oob_score=True`).
  - Forest-wide ensemble `.feature_importances()`.

#### 3.3 Gradient Boosting Regressor (`GradientBoostingRegressor`)
- **Module**: `nevula.models.ensemble.gradient_boosting`
- **Stage-wise Additive Model**:
  $$F_m(x) = F_{m-1}(x) + \eta \cdot h_m(x), \quad r_{im} = y_i - F_{m-1}(x_i)$$
- **Key Features**:
  - Sequential forward stage-wise gradient descent fitting shallow trees to pseudo-residuals.
  - Learning rate shrinkage $\eta \in (0, 1]$ controls convergence and prevents overfitting.
  - Stochastic Gradient Boosting (`subsample < 1.0`) for faster iterations and regularized splits.
  - Stage-by-stage loss tracking in `.train_score_`.
- **Example**: `examples/ensemble_learning.py`
- **Interactive Lab**: `model_lab/ensemble/`

---

## 🛠️ Unified API Contract

All models in Nevula implement the common `BaseModel` interface:

```python
import numpy as np
from nevula.models import RandomForestRegressor

# 1. Instantiate
model = RandomForestRegressor(n_estimators=50, max_depth=6, random_state=42)

# 2. Fit
model.fit(X_train, y_train)

# 3. Predict (automatically under no_grad)
y_pred = model.predict(X_test)

# 4. Evaluate
r2 = model.evaluate(X_test, y_test, metric="r2")
mse = model.evaluate(X_test, y_test, metric="mse")

# 5. Inspect
importances = model.feature_importances()
config = model.get_config()
state = model.state_dict()
```

---

## 🧩 Dynamic Registry Access

The Nevula Model Library incorporates a centralized registry for programmatically discovering, inspecting, and loading models:

```python
from nevula.models import get_model, list_models, summary

# List all models in a specific category
regression_models = list_models(category="regression")
ensemble_models = list_models(category="ensemble")
tree_models = list_models(category="trees")

# Instantiate a model dynamically by name or alias
model_cls = get_model("random_forest")
model = model_cls(n_estimators=100)

# Print tabular overview
print(summary())
```

---

## 🚀 Adding a New Model

To register a new model into the Nevula Model Library:

1. **Subclass `BaseModel`**:
   ```python
   from nevula.models.base import BaseModel
   from nevula.models.registry import register_model

   @register_model(name="my_model", category="regression")
   class MyModel(BaseModel):
       def __init__(self, ...):
           super().__init__()
           ...

       def fit(self, X, y, **kwargs):
           ...
           self._is_fitted = True
           return self

       def forward(self, X):
           ...

       def predict(self, X):
           self.eval()
           with no_grad():
               return self.forward(X)
   ```
2. **Expose Publicly**: Export the class in `nevula/models/<category>/__init__.py` and `nevula/models/__init__.py`.
3. **Register Aliases**: Call `register_model(MyModel, name="alias_name", category="category")`.
4. **Create Verification Suite**: Add tests under `tests/models/<category>/`, docs in `docs/models/`, and runnable script in `examples/`.
