# Experimental Lasso Regression (`model_lab`)

Design specification and experimental lab for $L_1$-regularized linear regression (Lasso).

---

## 1. Problem Formulation

While Ridge Regression ($L_2$) shrinks coefficients toward zero, it retains all features in the model. When dealing with high-dimensional datasets where many features are irrelevant or purely noise, **Lasso (Least Absolute Shrinkage and Selection Operator)** enforces sparsity by setting non-informative coefficients strictly to zero.

$$\hat{y} = x W + b$$

For a batch of $N$ observations:

$$\hat{Y} = X W + \mathbf{1}_N b^T$$

---

## 2. Objective Function

$$\mathcal{L}(W, b) = \frac{1}{N} \sum_{i=1}^N \|\hat{y}_i - y_i\|_2^2 + \alpha \|W\|_1 = \text{MSE}(\hat{Y}, Y) + \alpha \sum_{j=1}^D \sum_{k=1}^K |w_{jk}|$$

where:
- $\alpha \ge 0$ is the regularization parameter.
- The diamond-shaped $L_1$ ball constraint $\|\mathbf{w}\|_1 \le t$ has sharp corners along coordinate axes, making optimal solutions naturally lie on axes where one or more parameters equal zero.
- The intercept $b$ is unpenalized.

---

## 3. Subgradient Optimization

Since the absolute value function is non-differentiable at zero, we use subgradient calculus:

$$\partial |w| = \begin{cases} \{+1\} & \text{if } w > 0 \\ \{-1\} & \text{if } w < 0 \\ [-1, 1] & \text{if } w = 0 \end{cases}$$

The subgradient with respect to $W$:

$$\nabla_W \mathcal{L} = \frac{2}{N} X^T (\hat{Y} - Y) + \alpha \cdot \text{sign}(W)$$

$$\nabla_b \mathcal{L} = \frac{2}{N} \sum_{i=1}^N (\hat{y}_i - y_i)$$

Nevula's `Tensor.abs()` handles subgradient propagation automatically during `loss.backward()`.

---

## 4. Graduation Status

- [x] Mathematical specification formalized.
- [x] Differentiable `Tensor.abs()` added to framework core.
- [x] Sparsity and feature selection verified against high-dimensional noisy data.
- [x] Graduated to official library: `nevula.models.LassoRegression`.
