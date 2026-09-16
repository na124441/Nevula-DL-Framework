# Lasso Regression

Lasso (**Least Absolute Shrinkage and Selection Operator**) is an $L_1$-regularized linear model that combines parameter estimation with automatic **feature selection**, yielding sparse and highly interpretable models.

---

## 1. Intuition

While Ridge Regression ($L_2$) shrinks coefficients toward zero, it rarely drives any coefficient to absolute zero. Every feature remains active, which is problematic when dealing with high-dimensional data containing dozens or hundreds of irrelevant variables.

Lasso solves this by constraining the model with an **$L_1$ norm penalty** $\sum |w_j|$:
- **Automatic Sparsity**: Redundant and noisy feature weights are forced strictly to zero ($w_j = 0$).
- **Geometric Origin**: The $L_1$ constraint region is a rotated hypercube (cross-polytope / diamond) with sharp vertices along coordinate axes. The elliptical MSE loss contours are geometrically much more likely to make first contact at one of these vertices, setting one or more coefficients strictly to zero.
- **Interpretability**: By pruning non-essential features, Lasso produces a concise subset of predictors.

---

## 2. Mathematical Formulation

### Model Equation
For a feature vector $x \in \mathbb{R}^{1 \times D}$ and target $y \in \mathbb{R}^{1 \times K}$:

$$\hat{y} = x W + b$$

For a batch of $N$ observations $X \in \mathbb{R}^{N \times D}$:

$$\hat{Y} = X W + \mathbf{1}_N b^T$$

where:
- $W \in \mathbb{R}^{D \times K}$ is the weight matrix.
- $b \in \mathbb{R}^K$ is the unpenalized intercept vector.

### Objective Function (Loss with $L_1$ Penalty)

$$\mathcal{L}(W, b) = \frac{1}{N} \sum_{i=1}^N \|\hat{y}_i - y_i\|_2^2 + \alpha \|W\|_1$$

where:
- $\alpha \ge 0$ is the regularization strength hyperparameter.
- $\|W\|_1 = \sum_{j=1}^D \sum_{k=1}^K |w_{jk}|$ is the $L_1$ norm of the weight parameters.
- When $\alpha = 0$, the objective reduces strictly to standard Ordinary Least Squares (OLS).
- Larger values of $\alpha$ produce sparser models with more coefficients set to zero.

### Subgradient Calculus

The absolute value function $|w|$ is not differentiable at $w = 0$. Its subgradient set is:

$$\partial |w| = \begin{cases} \{+1\} & \text{if } w > 0 \\ \{-1\} & \text{if } w < 0 \\ [-1, 1] & \text{if } w = 0 \end{cases}$$

The subgradient of the loss with respect to $W$:

$$\nabla_W \mathcal{L} = \frac{2}{N} X^T (\hat{Y} - Y) + \alpha \cdot \text{sign}(W)$$

$$\nabla_b \mathcal{L} = \frac{2}{N} \sum_{i=1}^N (\hat{y}_i - y_i)$$

Nevula's `Tensor.abs()` autograd function computes this subgradient during reverse-mode automatic differentiation.

---

## 3. Parameters & Hyperparameters

### Trainable Parameters
- `weight` (`Parameter`): Weight matrix of shape `(in_features, out_features)`.
- `bias` (`Parameter`): Intercept vector of shape `(out_features,)` (when `fit_intercept=True`).

### Hyperparameters
- `in_features` (`int`): Dimensionality of input space $D$.
- `out_features` (`int`, default `1`): Dimensionality of target space $K$.
- `alpha` (`float`, default `1.0`): $L_1$ penalty coefficient. Must be non-negative. Higher values enforce greater sparsity.
- `fit_intercept` (`bool`, default `True`): Whether to learn an additive bias vector.
- `tol` (`float`, default `1e-3`): Tolerance threshold below which weights are trimmed to 0.0.
- `mode` (`str`, default `"framework"`): Implementation style (`"framework"` via `nn.Linear` or `"tensor"` via explicit tensor multiplications).
- `lr` (`float`, default `0.01`): Learning rate for gradient descent.
- `epochs` (`int`, default `150`): Maximum training iterations.
- `optimizer` (`str` or `Optimizer`, default `"sgd"`): Optimization algorithm.

---

## 4. Comparison: OLS vs Ridge vs Lasso

| Property | Linear Regression (OLS) | Ridge Regression ($L_2$) | Lasso Regression ($L_1$) |
|---|---|---|---|
| **Penalty Function** | None | $\alpha \sum w_j^2$ | $\alpha \sum \|w_j\|$ |
| **Constraint Shape** | None | Hypersphere (smooth) | Cross-polytope / Diamond (corners) |
| **Feature Selection** | None (keeps all features) | None (keeps all features) | **Yes (forces weights to 0)** |
| **Sparsity** | Dense | Dense | **Sparse** |
| **Handling Collinearity** | Unstable | Groups collinear weights | Arbitrarily picks one feature |

---

## 5. Usage & Example

```python
from nevula.models import LassoRegression
import numpy as np

# 1. 10 features, but only features 0 and 1 are true predictors (8 are noise)
X = np.random.randn(150, 10)
y = 3.0 * X[:, 0:1] - 2.0 * X[:, 1:2] + 1.0

# 2. Instantiate Lasso with alpha=0.5
model = LassoRegression(in_features=10, out_features=1, alpha=0.5)

# 3. Fit model
model.fit(X, y, epochs=200, lr=0.03, tol=1e-2)

# 4. Check sparsity
print(f"Sparsity: {model.sparsity() * 100:.1f}% zeroed out!")

# 5. Predict & Evaluate
preds = model.predict(X)
r2 = model.evaluate(X, y, metric="r2")
print(f"R^2 Score: {r2:.4f}")
```

---

## 6. Computational Complexity

| Operation | Time Complexity | Space Complexity |
|---|---|---|
| **Forward Pass** | $\mathcal{O}(N \cdot D \cdot K)$ | $\mathcal{O}(N \cdot K)$ |
| **Loss & $L_1$ Backward** | $\mathcal{O}(N \cdot D \cdot K)$ | $\mathcal{O}(N \cdot K)$ |
| **Training (per epoch)** | $\mathcal{O}(N \cdot D \cdot K)$ | $\mathcal{O}(D \cdot K)$ |
| **Inference (`predict`)** | $\mathcal{O}(M \cdot D \cdot K)$ | $\mathcal{O}(M \cdot K)$ |

---

## 7. Limitations

1. **Collinear Grouping**: If two features are highly correlated, Lasso tends to pick one arbitrarily and zero out the other (ElasticNet resolves this by combining $L_1$ and $L_2$).
2. **Dimension Constraints**: When $D > N$, standard Lasso can select at most $N$ variables before saturation.
3. **Subgradient Convergence Rate**: First-order subgradient descent converges at rate $\mathcal{O}(1/\sqrt{k})$ near points of non-differentiability.

---

## 8. References

- Tibshirani, R. (1996). *Regression shrinkage and selection via the lasso*. Journal of the Royal Statistical Society: Series B (Methodological), 58(1), 267-288.
- Hastie, T., Tibshirani, R., & Wainwright, M. (2015). *Statistical Learning with Sparsity: The Lasso and Generalizations*. CRC Press.
