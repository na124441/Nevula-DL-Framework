# Experimental Support Vector Regression (`model_lab`)

Design specification and experimental lab for Support Vector Regression (SVR).

---

## 1. Problem Formulation

Ordinary Least Squares and Ridge Regression minimize squared residuals $\sum (y_i - \hat{y}_i)^2$. Because errors are squared, extreme anomalies and outliers exert immense leverage on the learned hyperplane, tilting predictions away from genuine data trends.

**Support Vector Regression (SVR)** combats this by introducing:
1. An **$\epsilon$-insensitive tube**: Deviations smaller than $\epsilon$ incur zero penalty ($L_\epsilon = 0$).
2. **Linear penalty outside the margin**: Errors greater than $\epsilon$ are penalized linearly ($|e| - \epsilon$), bounding the gradient leverage of outliers.

---

## 2. Objective Function

$$\mathcal{L}(W, b) = \frac{1}{N} \sum_{i=1}^N \max(0, |\hat{y}_i - y_i| - \epsilon) + \frac{1}{2C} \|W\|_2^2$$

where:
- $\epsilon \ge 0$ defines the radius of the tube.
- $C > 0$ balances empirical tube violation penalty against weight flatness $\|W\|_2^2$.
- Observations lying on or outside the tube boundary ($|\hat{y}_i - y_i| \ge \epsilon$) are the **Support Vectors**.

---

## 3. Subgradient Formulation in Nevula

In Nevula, the $\epsilon$-insensitive loss is composed natively as:
$$\text{tube\_loss} = \text{ReLU}(|\hat{y} - y| - \epsilon)$$

Since both $\text{ReLU}$ and $|\cdot|$ possess exact subgradients in Nevula's Autograd engine:
- If $|y_i - \hat{y}_i| < \epsilon$: $\nabla_{W} = 0$ (inside tube, zero gradient).
- If $y_i - \hat{y}_i > \epsilon$: $\nabla_{W} = -X_i$ (under-prediction).
- If $\hat{y}_i - y_i > \epsilon$: $\nabla_{W} = +X_i$ (over-prediction).

Unlike OLS where gradients scale with $2(y_i - \hat{y}_i)$, SVR gradients are constant ($\pm 1$), capping outlier influence.

---

## 4. Graduation Status

- [x] Mathematical specification formalized.
- [x] Implemented using Nevula primitives (`Tensor`, `ReLU`, `Abs`, `Linear`).
- [x] Outlier robustness verified experimentally against contaminated datasets.
- [x] Graduated to official library: `nevula.models.SVR`.
