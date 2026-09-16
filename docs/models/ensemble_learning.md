# Ensemble Learning: Voting, Random Forest, & Gradient Boosting

Ensemble learning combines multiple base models (estimators) to produce a predictive model that achieves superior generalization performance compared to any single constituent model.

Nevula provides three complementary ensemble paradigms under `nevula.models.ensemble`:
1. **Voting Regressor (`VotingRegressor`)**: Combines heterogeneous models via weighted or uniform prediction averaging.
2. **Random Forest Regressor (`RandomForestRegressor`)**: Reduces variance via Bootstrap Aggregation (Bagging) and random feature subspaces over decision trees.
3. **Gradient Boosting Regressor (`GradientBoostingRegressor`)**: Reduces bias via sequential forward stage-wise fitting of shallow regression trees to pseudo-residuals.

---

## 1. The Bias-Variance Dilemma & Ensemble Motivation

Any supervised estimator incurs generalization error composed of three orthogonal components:

$$\text{Expected Error} = \text{Bias}^2 + \text{Variance} + \text{Irreducible Noise} \ \sigma_{\epsilon}^2$$

- **High-Variance Models (e.g., Deep Decision Trees)**: Fit the training data with near-zero training error, but fluctuate drastically across training sets. They overfit noise and specific sample quirks.
- **High-Bias Models (e.g., Shallow Linear Models on Non-Linear Data)**: Underfit because their structural assumptions are too rigid to capture underlying data relationships.

Ensemble methods target these two failure modes:
- **Bagging / Random Forests**: Drive **variance down** without increasing bias by averaging de-correlated estimators.
- **Boosting**: Drives **bias down** sequentially by building an additive ensemble where each new stage compensates for the errors of the preceding stages.
- **Voting**: Dampens model-specific inductive bias by blending diverse mathematical architectures (e.g. Ridge + SVR + Trees).

---

## 2. Mathematical Formulations

### 2.1 Voting Regressor
Given $K$ trained base estimators $\hat{f}_1(x), \dots, \hat{f}_K(x)$ and non-negative weights $w_1, \dots, w_K$ such that $\sum_{k=1}^K w_k = 1$:

$$\hat{y}(x) = \sum_{k=1}^K w_k \cdot \hat{f}_k(x)$$

For uniform voting: $w_k = \frac{1}{K}$.

### 2.2 Random Forest Regressor (Bagging + Feature Subspaces)
Let $B$ be the number of trees (`n_estimators`). Each tree $h_b(x)$ is trained on a bootstrap sample $\mathcal{D}_b^*$ drawn with replacement from training dataset $\mathcal{D}$ of size $N$.

At each candidate split within tree $b$, only a random subset of $m$ features (`max_features`) is considered:
$$m = \lfloor \sqrt{D} \rfloor \quad \text{or} \quad m = \lfloor \log_2 D \rfloor$$

The ensemble prediction is the arithmetic mean across all $B$ trees:

$$\hat{y}(x) = \frac{1}{B} \sum_{b=1}^B h_b(x)$$

#### Theoretical Variance Reduction
If each individual tree has variance $\sigma^2$ and the average pairwise correlation between trees is $\rho$:

$$\text{Var}(\hat{y}(x)) = \rho \sigma^2 + \frac{1 - \rho}{B} \sigma^2$$

As $B \to \infty$, the second term approaches zero, leaving $\rho \sigma^2$. Random feature subspace sampling minimizes $\rho$ (de-correlating the trees), allowing the forest to reach far lower variance than any single tree.

#### Out-of-Bag (OOB) Estimation
Since each bootstrap draw samples with replacement, the probability that an observation is *not* chosen in a bootstrap sample of size $N$ is:
$$\lim_{N \to \infty} \left(1 - \frac{1}{N}\right)^N = \frac{1}{e} \approx 36.8\%$$
For each observation $i$, the OOB prediction averages only those trees where sample $i$ was omitted during training. The OOB $R^2$ score offers an unbiased generalization estimate without requiring a dedicated validation set.

### 2.3 Gradient Boosting Regressor (GBDT)
Gradient boosting constructs an additive model in a forward stage-wise fashion:

$$F_M(x) = F_0(x) + \sum_{m=1}^M \eta \cdot h_m(x)$$

where:
1. **Initial Model**: $F_0(x) = \arg\min_c \sum_{i=1}^N L(y_i, c) = \frac{1}{N} \sum_{i=1}^N y_i$ (empirical mean of $y$).
2. **Pseudo-Residuals**: For squared error loss $L(y, F) = \frac{1}{2}(y - F)^2$, the negative gradient at stage $m$ is:
   $$r_{im} = -\left[ \frac{\partial L(y_i, F(x_i))}{\partial F(x_i)} \right]_{F=F_{m-1}} = y_i - F_{m-1}(x_i)$$
3. **Weak Estimator**: A shallow regression tree $h_m(x)$ (typically `max_depth` $\in [2, 4]$) is fit to the pseudo-residuals $\{(x_i, r_{im})\}_{i=1}^N$.
4. **Shrinkage (Learning Rate)**: $\eta \in (0, 1]$ scales the step size of each stage, providing regularization against overfitting.
5. **Stochastic Subsampling**: If `subsample` $< 1.0$, each stage fits $h_m$ on a random subset without replacement, reducing variance and computational time.

---

## 3. Ensemble Comparison

| Metric / Property | VotingRegressor | RandomForestRegressor | GradientBoostingRegressor |
|---|---|---|---|
| **Base Estimator Type** | Heterogeneous (any `BaseModel`) | Homogeneous (`DecisionTreeRegressor`) | Homogeneous shallow trees (`max_depth` 2–4) |
| **Training Scheme** | Independent / Parallel | Independent / Parallel bootstrap | Sequential stage-wise residuals |
| **Primary Error Targeted** | Inductive architecture bias | **Variance (Overfitting)** | **Bias (Underfitting)** |
| **Tree Correlation** | Dependent on model diversity | De-correlated via random subspaces | De-correlated by residual shifts |
| **Hyperparameter Sensitivity** | Low | Low (robust to tuning) | Moderate to high (`learning_rate`, `max_depth`) |
| **Feature Importances** | N/A | Mean normalized impurity decrease | Mean normalized impurity decrease |
| **Out-of-Bag (OOB) Evaluation**| No | **Yes (`oob_score=True`)** | No (uses loss trajectory `train_score_`) |

---

## 4. Usage & Examples

### 4.1 Random Forest Regressor
```python
from nevula.models import RandomForestRegressor
import numpy as np

# Training data
X = np.random.uniform(-3, 3, size=(200, 5))
y = np.sin(X[:, 0]) + 0.5 * (X[:, 1] ** 2) + np.random.normal(0, 0.1, 200)

# Instantiate with 50 trees, sqrt feature sampling, and OOB scoring
rf = RandomForestRegressor(
    n_estimators=50,
    max_depth=6,
    max_features="sqrt",
    bootstrap=True,
    oob_score=True,
    random_state=42
)
rf.fit(X, y)

print(f"OOB R^2 Score: {rf.oob_score_:.4f}")
print(f"Feature Importances: {rf.feature_importances()}")
```

### 4.2 Gradient Boosting Regressor
```python
from nevula.models import GradientBoostingRegressor

gbdt = GradientBoostingRegressor(
    n_estimators=60,
    learning_rate=0.1,
    max_depth=3,
    subsample=0.8,
    random_state=42
)
gbdt.fit(X, y)

print(f"Train R^2: {gbdt.evaluate(X, y, metric='r2'):.4f}")
print(f"Final Stage Loss: {gbdt.train_score_[-1]:.4f}")
```

### 4.3 Voting Regressor
```python
from nevula.models import VotingRegressor, RidgeRegression, SVR, DecisionTreeRegressor

ridge = RidgeRegression(in_features=5, alpha=1.0)
svr = SVR(in_features=5, epsilon=0.1, C=1.0)
tree = DecisionTreeRegressor(max_depth=4)

voting = VotingRegressor(
    estimators=[("ridge", ridge), ("svr", svr), ("tree", tree)],
    weights=[0.2, 0.3, 0.5]
)
voting.fit(X, y.reshape(-1, 1), epochs=100, lr=0.01, verbose=False)
y_pred = voting.predict(X)
```

---

## 5. Computational Complexity

Let $N$ be the number of training samples, $D$ the number of features, $B$ the number of estimators, and $d$ the maximum tree depth:

| Algorithm | Training Time Complexity | Inference Time Complexity | Space Complexity |
|---|---|---|---|
| **VotingRegressor** | $\sum_{k=1}^K \mathcal{O}(\text{Fit}_k)$ | $\sum_{k=1}^K \mathcal{O}(\text{Pred}_k)$ | $\sum_{k=1}^K \mathcal{O}(\text{Space}_k)$ |
| **RandomForestRegressor** | $\mathcal{O}(B \cdot \sqrt{D} \cdot N \log N \cdot d)$ | $\mathcal{O}(B \cdot M \cdot d)$ | $\mathcal{O}(B \cdot 2^d)$ |
| **GradientBoostingRegressor** | $\mathcal{O}(M \cdot D \cdot N \log N \cdot d)$ | $\mathcal{O}(M \cdot d)$ | $\mathcal{O}(M \cdot 2^d)$ |
