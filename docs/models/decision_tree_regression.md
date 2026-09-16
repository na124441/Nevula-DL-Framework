# Decision Tree Regression

Decision Tree Regression is a non-parametric supervised learning algorithm that models complex, non-linear relationships by recursively partitioning the feature space into disjoint axis-aligned hyper-rectangles.

---

## 1. Intuition

Parametric models (such as Linear Regression, Ridge, Lasso, and SVR) enforce a global algebraic functional form ($y = XW + b$). In contrast, a Decision Tree makes **no parametric assumptions**:
- It divides feature space into smaller rectangular regions through simple binary tests (e.g. $x_1 \le 3.5$).
- For any test observation, it traverses the tree from root to leaf and predicts the **average target value** of all training observations that landed in that leaf.
- It naturally models non-linear step functions, thresholds, and high-order feature interactions without requiring non-linear feature engineering or polynomial basis expansions.

---

## 2. Mathematical Formulation

### Regional Partition Model
Given a partition of the feature space $\mathbb{R}^D$ into $M$ disjoint regions $R_1, R_2, \dots, R_M$:

$$\hat{y}(x) = \sum_{m=1}^M c_m \cdot \mathbb{I}(x \in R_m)$$

where $\mathbb{I}(\cdot)$ is the indicator function, and the optimal leaf constant $c_m$ minimizing squared error is the empirical mean:

$$c_m = \text{mean}(y_i \mid x_i \in R_m) = \frac{1}{N_m} \sum_{x_i \in R_m} y_i$$

### Splitting Criterion: Variance Reduction (CART)
At each node $Q$ containing $N_Q$ observations, the node impurity is the variance of target values:

$$\text{Impurity}(Q) = \text{MSE}(Q) = \frac{1}{N_Q} \sum_{i \in Q} (y_i - \bar{y}_Q)^2 = \text{Var}(y_Q)$$

For candidate feature $j$ and split threshold $s$, the data is partitioned into:
$$Q_L = \{i \in Q \mid x_{ij} \le s\}, \quad Q_R = \{i \in Q \mid x_{ij} > s\}$$

The algorithm selects split $(j^*, s^*)$ maximizing the impurity decrease (variance reduction):

$$\Delta \mathcal{I}(j, s) = \text{Impurity}(Q) - \left( \frac{N_L}{N_Q} \text{Impurity}(Q_L) + \frac{N_R}{N_Q} \text{Impurity}(Q_R) \right)$$

### Feature Importance
The importance of feature $j$ is the sum over all internal nodes splitting on $j$ of the sample-weighted impurity reduction:

$$\text{Importance}(j) = \frac{\sum_{t \in \text{splits}(j)} N_t \cdot \Delta \mathcal{I}_t}{\sum_{t \in \text{all splits}} N_t \cdot \Delta \mathcal{I}_t}$$

The vector is normalized to sum to $1.0$.

---

## 3. Hyperparameters & Stopping Criteria

- `max_depth` (`int`, default `5`): Maximum depth of the tree. Controls model complexity and prevents overfitting.
- `min_samples_split` (`int`, default `2`): Minimum number of samples required to split an internal node.
- `min_samples_leaf` (`int`, default `1`): Minimum number of samples required to form a leaf node.
- `min_impurity_decrease` (`float`, default `0.0`): Minimum variance reduction required to accept a split.
- `max_features` (`int`, `float`, or `None`, default `None`): Number of features to consider when looking for best split (useful for random subsampling).

---

## 4. Comparison: Parametric vs Non-Parametric Models

| Property | Parametric (Linear, Ridge, Lasso, SVR) | Non-Parametric (DecisionTreeRegressor) |
|---|---|---|
| **Model Structure** | Global hyperplane $y = XW + b$ | Local piecewise-constant partitioning |
| **Non-Linear Relationships** | Requires basis expansion / kernels | **Learns non-linear functions directly** |
| **Feature Scaling** | Sensitive to feature scales | **Invariant to monotonic feature scaling** |
| **Feature Interactions** | Must be explicitly modeled ($x_1 \cdot x_2$) | **Naturally captured hierarchically** |
| **Overfitting Tendency** | Low/moderate (regularized) | High if unpruned (`max_depth` controls this) |
| **Optimization Method** | Gradient / subgradient descent | Greedy recursive partitioning (CART) |

---

## 5. Usage & Example

```python
from nevula.models import DecisionTreeRegressor
import numpy as np

# 1. Non-linear step dataset
X = np.linspace(-3, 3, 100).reshape(-1, 1)
y = np.sin(X[:, 0]) + 0.5 * (X[:, 0] > 0)

# 2. Instantiate and fit DecisionTreeRegressor
tree = DecisionTreeRegressor(max_depth=4, min_samples_leaf=2)
tree.fit(X, y)

# 3. Predict & Evaluate
preds = tree.predict(X)
r2 = tree.evaluate(X, y, metric="r2")
print(f"R^2 Score: {r2:.4f}")
print(f"Tree Depth: {tree.tree_depth()}, Leaves: {tree.n_leaves()}")
print(f"Feature Importances: {tree.feature_importances()}")
```

---

## 6. Computational Complexity

| Operation | Time Complexity | Space Complexity |
|---|---|---|
| **Training (`fit`)** | $\mathcal{O}(D \cdot N \log N \cdot d_{\text{max}})$ | $\mathcal{O}(N \cdot D + 2^{d_{\text{max}}})$ |
| **Inference (`predict`)** | $\mathcal{O}(M \cdot d_{\text{max}})$ | $\mathcal{O}(M)$ |

Where $N$ is sample size, $D$ is feature count, $d_{\text{max}}$ is tree depth, and $M$ is prediction batch size.

---

## 7. Limitations

1. **High Variance**: Small changes in training data can result in completely different tree structures (mitigated in subsequent phases by Random Forests).
2. **Step Function Artifacts**: Decision trees approximate smooth curves using stair-step piecewise-constant functions.
3. **Extrapolation Inability**: Cannot predict values outside the range $[\min(y), \max(y)]$ observed during training.

---

## 8. References

- Breiman, L., Friedman, J., Stone, C. J., & Olshen, R. A. (1984). *Classification and Regression Trees*. CRC Press.
- Quinlan, J. R. (1986). *Induction of decision trees*. Machine Learning, 1(1), 81-106.
- Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning*. Springer.
