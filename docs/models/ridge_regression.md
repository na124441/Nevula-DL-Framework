# Ridge Regression

Ridge Regression (also known as **Tikhonov Regularization**) is a regularized linear model designed to mitigate overfitting and combat the numerical instability caused by **multicollinearity** among input features.

---

## 1. Intuition

In Ordinary Least Squares (OLS) regression, when two or more predictor variables are highly correlated, the covariance matrix $X^T X$ becomes ill-conditioned (nearly singular). This causes standard linear regression weights to explode in magnitude with opposite signs, leading to massive variance in predictions.

Ridge regression solves this problem by imposing an **$L_2$ penalty** (shrinkage constraint) on the size of the coefficients:
- Weights are shrunk toward zero, drastically lowering model variance.
- Coefficients are never set strictly to zero (unlike Lasso $L_1$ regularization), preserving all input features with scaled influence.
- Geometrically, the solution corresponds to the point where the elliptical loss contours of MSE intersect the spherical $L_2$ constraint ball $\|\mathbf{w}\|_2^2 \le t$.

---

## 2. Mathematical Formulation

### Model Equation
For a feature vector $x \in \mathbb{R}^{1 \times D}$ and target $y \in \mathbb{R}^{1 \times K}$:

$$\hat{y} = x W + b$$

For a dataset batch of $N$ observations $X \in \mathbb{R}^{N \times D}$:

$$\hat{Y} = X W + \mathbf{1}_N b^T$$

where:
- $W \in \mathbb{R}^{D \times K}$ is the weight matrix.
- $b \in \mathbb{R}^K$ is the unpenalized intercept vector.

### Objective Function (Loss with $L_2$ Regularization)

$$\mathcal{L}(W, b) = \frac{1}{N} \sum_{i=1}^N \|\hat{y}_i - y_i\|_2^2 + \alpha \|W\|_F^2$$

where:
- $\alpha \ge 0$ is the regularization strength hyperparameter.
- $\|W\|_F^2 = \sum_{j=1}^D \sum_{k=1}^K w_{jk}^2$ is the squared Frobenius norm of the weights.
- When $\alpha = 0$, the objective reduces strictly to standard MSE (OLS).
- As $\alpha \to \infty$, weights approach zero: $\|W\| \to 0$.

### Analytical Gradients

$$\nabla_W \mathcal{L} = \frac{2}{N} X^T (\hat{Y} - Y) + 2\alpha W$$

$$\nabla_b \mathcal{L} = \frac{2}{N} \sum_{i=1}^N (\hat{y}_i - y_i)$$

In Nevula, **Autograd** evaluates these gradients automatically through reverse-mode automatic differentiation.

### Closed-Form Solution (Normal Equation)
Assuming zero-centered data (or absorbing the bias into $X$ without penalizing it):

$$W^* = (X^T X + \alpha I)^{-1} X^T Y$$

Adding $\alpha I$ to $X^T X$ adds positive eigenvalues to the diagonal, guaranteeing the matrix is strictly invertible and well-conditioned even when $N < D$ or features are collinear.

---

## 3. Parameters & Hyperparameters

### Trainable Parameters
- `weight` (`Parameter`): Weight matrix of shape `(in_features, out_features)`.
- `bias` (`Parameter`): Intercept vector of shape `(out_features,)` (when `fit_intercept=True`).

### Hyperparameters
- `in_features` (`int`): Dimensionality of input space $D$.
- `out_features` (`int`, default `1`): Dimensionality of target space $K$.
- `alpha` (`float`, default `1.0`): $L_2$ penalty coefficient. Must be non-negative.
- `fit_intercept` (`bool`, default `True`): Whether to fit an additive bias vector.
- `mode` (`str`, default `"framework"`): Implementation style (`"framework"` via `nn.Linear` or `"tensor"` via explicit tensor multiplications).
- `lr` (`float`, default `0.01`): Learning rate for gradient descent.
- `epochs` (`int`, default `100`): Maximum training iterations.
- `batch_size` (`int`, optional): Mini-batch size for `DataLoader`.
- `optimizer` (`str` or `Optimizer`, default `"sgd"`): Optimization algorithm (`"sgd"`, `"adam"`, `"adamw"`).

---

## 4. Comparison: OLS vs Ridge vs Lasso

| Property | Linear Regression (OLS) | Ridge Regression ($L_2$) | Lasso Regression ($L_1$) |
|---|---|---|---|
| **Penalty Term** | None | $\alpha \sum w_j^2$ | $\alpha \sum \|w_j\|$ |
| **Collinearity Handling** | Unstable, large variance | **Very Stable, shrinks weights** | Selects one feature arbitrarily |
| **Sparsity** | No | No (dense weights) | Yes (zeros out features) |
| **Solution Type** | Closed-form / Gradient | Closed-form / Gradient | Subgradient / Coordinate Descent |

---

## 5. Usage & Example

```python
from nevula.models import RidgeRegression
import numpy as np

# 1. Prepare data with correlated features
X = np.random.randn(100, 4)
y = X @ np.array([[2.0], [2.0], [-1.0], [0.5]]) + 1.5

# 2. Instantiate Ridge model with regularization strength alpha=1.0
model = RidgeRegression(in_features=4, out_features=1, alpha=1.0)

# 3. Fit model
model.fit(X, y, epochs=150, lr=0.03, optimizer="sgd")

# 4. Predict
preds = model.predict(X)

# 5. Evaluate
r2 = model.evaluate(X, y, metric="r2")
mse = model.evaluate(X, y, metric="mse")
print(f"R^2 Score: {r2:.4f}, MSE: {mse:.4f}")
```

---

## 6. Computational Complexity

| Operation | Time Complexity | Space Complexity |
|---|---|---|
| **Forward Pass** | $\mathcal{O}(N \cdot D \cdot K)$ | $\mathcal{O}(N \cdot K)$ |
| **Loss & $L_2$ Backward** | $\mathcal{O}(N \cdot D \cdot K)$ | $\mathcal{O}(N \cdot K)$ |
| **Training (per epoch)** | $\mathcal{O}(N \cdot D \cdot K)$ | $\mathcal{O}(D \cdot K)$ |
| **Inference (`predict`)** | $\mathcal{O}(M \cdot D \cdot K)$ | $\mathcal{O}(M \cdot K)$ |

---

## 7. Limitations

1. **Feature Retention**: Ridge does not perform feature selection (coefficients shrink asymptotically towards zero but rarely reach absolute zero).
2. **Feature Scaling Sensitivity**: Because the $L_2$ penalty treats all weights equally, features should ideally be on comparable scales.
3. **Linearity**: Assumes a linear relationship between features and target variables.

---

## 8. References

- Hoerl, A. E., & Kennard, R. W. (1970). *Ridge regression: Biased estimation for nonorthogonal problems*. Technometrics, 12(1), 55-67.
- Tikhonov, A. N. (1963). *Solution of incorrectly formulated problems and the regularization method*. Soviet Mathematics Doklady, 4, 1035-1038.
- Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning*. Springer.
