# Support Vector Regression (SVR)

Support Vector Regression (SVR) is a margin-based regression algorithm that constructs an **$\epsilon$-insensitive tube** around predictions. Errors smaller than $\epsilon$ incur zero penalty, while deviations exceeding $\epsilon$ are penalized linearly, providing exceptional **robustness against outliers**.

---

## 1. Intuition

In standard Least Squares regression (OLS and Ridge), loss scales quadratically with error: $\mathcal{L} \propto (y - \hat{y})^2$. If an anomalous outlier has an error of 100, its loss contribution is $10,000$, which drastically tilts the regression plane and ruins predictions on normal data.

SVR solves this with two geometric principles:
1. **The $\epsilon$-Insensitive Tube**: Any prediction that falls within distance $\epsilon$ of the true target is considered "good enough" and carries zero loss ($L_\epsilon = 0$). This acts as a dead-band filter against small noise.
2. **Linear Penalty Outside the Margin**: For deviations larger than $\epsilon$, loss grows only linearly ($|y - \hat{y}| - \epsilon$). Consequently, its gradient is bounded to $\pm 1$, preventing any single outlier from dominating the optimization.
3. **Support Vectors**: Points that lie strictly on or outside the tube boundary ($|y_i - \hat{y}_i| \ge \epsilon$) are the **Support Vectors** that actively determine the position and orientation of the regression hyperplane.

---

## 2. Mathematical Formulation

### Model Equation
For a feature vector $x \in \mathbb{R}^{1 \times D}$ and target $y \in \mathbb{R}^{1 \times K}$:

$$\hat{y} = x W + b$$

For a batch of $N$ observations $X \in \mathbb{R}^{N \times D}$:

$$\hat{Y} = X W + \mathbf{1}_N b^T$$

where:
- $W \in \mathbb{R}^{D \times K}$ is the weight matrix.
- $b \in \mathbb{R}^K$ is the intercept vector.

### $\epsilon$-Insensitive Loss Function
$$L_\epsilon(y, \hat{y}) = \begin{cases} 0 & \text{if } |y - \hat{y}| \le \epsilon \\ |y - \hat{y}| - \epsilon & \text{otherwise} \end{cases}$$

In Nevula, this is formulated differentiably via `ReLU` and `Abs`:
$$L_\epsilon(y, \hat{y}) = \text{ReLU}(|y - \hat{y}| - \epsilon)$$

### Primal Objective Function

$$\mathcal{L}(W, b) = \frac{1}{N} \sum_{i=1}^N \text{ReLU}(|\hat{y}_i - y_i| - \epsilon) + \frac{1}{2C} \|W\|_2^2$$

where:
- $\epsilon \ge 0$ specifies the margin of tolerance.
- $C > 0$ controls the trade-off between margin violation penalties and model smoothness (flatness of $W$).
- Lower $C$ emphasizes weight regularization (flatter model); higher $C$ emphasizes fitting training data closely.

### Subgradient Calculus

$$\nabla_W L_\epsilon(y_i, \hat{y}_i) = \begin{cases} 0 & \text{if } |y_i - \hat{y}_i| < \epsilon \\ \text{sign}(\hat{y}_i - y_i) \cdot X_i & \text{if } |y_i - \hat{y}_i| \ge \epsilon \end{cases}$$

$$\nabla_W \mathcal{L} = \frac{1}{N} \sum_{i \in \text{SV}} \text{sign}(\hat{y}_i - y_i) \cdot X_i + \frac{1}{C} W$$

---

## 3. Parameters & Hyperparameters

### Trainable Parameters
- `weight` (`Parameter`): Weight matrix of shape `(in_features, out_features)`.
- `bias` (`Parameter`): Intercept vector of shape `(out_features,)` (when `fit_intercept=True`).

### Hyperparameters
- `in_features` (`int`): Dimensionality of input space $D$.
- `out_features` (`int`, default `1`): Dimensionality of target space $K$.
- `epsilon` (`float`, default `0.1`): Width of the insensitive tube. Must be non-negative.
- `C` (`float`, default `1.0`): Regularization parameter. Must be strictly positive.
- `fit_intercept` (`bool`, default `True`): Whether to learn an additive bias vector.
- `mode` (`str`, default `"framework"`): Implementation style (`"framework"` via `nn.Linear` or `"tensor"`).
- `lr` (`float`, default `0.01`): Learning rate for gradient descent.
- `epochs` (`int`, default `150`): Maximum training iterations.
- `optimizer` (`str` or `Optimizer`, default `"sgd"`): Optimization algorithm.

---

## 4. Comparison: The 4 Nevula Regression Models

| Property | Linear Regression (OLS) | Ridge ($L_2$) | Lasso ($L_1$) | SVR ($\epsilon$-tube) |
|---|---|---|---|---|
| **Loss Type** | Squared error $(e^2)$ | Squared error $(e^2)$ | Squared error $(e^2)$ | **$\epsilon$-Insensitive $(\text{ReLU}(\|e\| - \epsilon))$** |
| **Penalty** | None | $\|W\|_2^2$ | $\|W\|_1$ | $\|W\|_2^2$ |
| **Outlier Resilience** | Very Poor (quadratic) | Poor (quadratic) | Poor (quadratic) | **Excellent (linear bounded)** |
| **Feature Sparsity** | No | No | **Yes** | No |
| **Primary Strength** | Unbiased baseline | Multicollinearity | Feature selection | Outlier robustness |

---

## 5. Usage & Example

```python
from nevula.models import SVR
import numpy as np

# 1. Prepare training data with occasional anomalous spikes
X = np.random.uniform(-3.0, 3.0, size=(100, 2))
y = 2.0 * X[:, 0:1] - 1.5 * X[:, 1:2] + 0.5
y[0:5] += 50.0  # extreme outliers

# 2. Instantiate SVR with epsilon margin = 0.2
model = SVR(in_features=2, out_features=1, epsilon=0.2, C=1.0)

# 3. Fit model
model.fit(X, y, epochs=200, lr=0.03, optimizer="sgd")

# 4. Predict & Support Vectors Inspection
preds = model.predict(X)
sv_ratio = model.support_vectors_ratio(X, y)
print(f"Support Vectors: {sv_ratio * 100:.1f}% of data")
```

---

## 6. Computational Complexity

| Operation | Time Complexity | Space Complexity |
|---|---|---|
| **Forward Pass** | $\mathcal{O}(N \cdot D \cdot K)$ | $\mathcal{O}(N \cdot K)$ |
| **Loss & Subgradient** | $\mathcal{O}(N \cdot D \cdot K)$ | $\mathcal{O}(N \cdot K)$ |
| **Training (per epoch)** | $\mathcal{O}(N \cdot D \cdot K)$ | $\mathcal{O}(D \cdot K)$ |
| **Inference (`predict`)** | $\mathcal{O}(M \cdot D \cdot K)$ | $\mathcal{O}(M \cdot K)$ |

---

## 7. Limitations

1. **Epsilon Sensitivity**: Setting $\epsilon$ too high causes underfitting by treating real trends as noise; setting $\epsilon$ too low increases the number of support vectors and vulnerability to small fluctuations.
2. **Linear Kernel**: Linear SVR models flat hyperplanes. Non-linear manifolds require feature mappings or kernel extensions.

---

## 8. References

- Vapnik, V. (1995). *The Nature of Statistical Learning Theory*. Springer-Verlag.
- Smola, A. J., & Schölkopf, B. (2004). *A tutorial on support vector regression*. Statistics and Computing, 14(3), 199-222.
- Drucker, H., Burges, C. J., Kaufman, L., Smola, A., & Vapnik, V. (1997). *Support vector regression machines*. Advances in Neural Information Processing Systems, 9, 155-161.
