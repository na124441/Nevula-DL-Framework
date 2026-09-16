# Experimental Ridge Regression (`model_lab`)

Design specification and experimental lab for L2-regularized linear regression (Ridge Regression).

---

## 1. Problem Formulation

Ordinary Linear Regression often overfits or yields unstable parameters when features are highly correlated (multicollinear) or when feature noise dominates. Ridge Regression introduces an $L_2$ penalty (Tikhonov regularization) to constrain weight magnitude:

$$\hat{y} = x W + b$$

For a batch of $N$ samples:

$$\hat{Y} = X W + \mathbf{1}_N b^T$$

---

## 2. Objective Function

$$\mathcal{L}(W, b) = \frac{1}{N} \sum_{i=1}^N \|\hat{y}_i - y_i\|_2^2 + \alpha \|W\|_F^2$$

where:
- $\alpha \ge 0$ is the regularization penalty strength.
- As $\alpha \to 0$, Ridge converges to Ordinary Least Squares (OLS).
- As $\alpha \to \infty$, weights shrink towards zero ($\|W\|_2 \to 0$).
- Intercept $b$ is unpenalized so the overall level/mean of target values is preserved without distortion.

---

## 3. Analytical Gradients

$$\nabla_W \mathcal{L} = \frac{2}{N} X^T (\hat{Y} - Y) + 2\alpha W$$
$$\nabla_b \mathcal{L} = \frac{2}{N} \sum_{i=1}^N (\hat{y}_i - y_i)$$

Autograd handles gradient calculation through reverse-mode AD on $(W^2).\text{sum}() * \alpha$.

---

## 4. Normal Equations (Closed-Form)

When centered:
$$W^* = (X^T X + \alpha I)^{-1} X^T Y$$
Adding $\alpha I$ ensures $(X^T X + \alpha I)$ is strictly positive-definite and well-conditioned even when $X^T X$ is singular.

---

## 5. Graduation Status

- [x] Mathematics formalized.
- [x] Implemented using Nevula primitives (`Tensor`, `Parameter`, `Linear`, `MSELoss`).
- [x] Weight shrinkage validated against collinear benchmarks.
- [x] Graduated to official library: `nevula.models.RidgeRegression`.
