# Linear Regression

Ordinary Linear Regression is a fundamental supervised learning algorithm for predicting continuous real-valued targets by modeling a linear relationship between input features and target variables.

---

## 1. Intuition

Given a set of input observations with multiple features, linear regression fits a hyperplane that minimizes the vertical distance (squared error) between the data points and the fitted hyperplane.

In 1D space, it fits a straight line:
$$y = wx + b$$

In $D$-dimensional space, it fits a flat hyperplane:
$$y = w_1 x_1 + w_2 x_2 + \dots + w_D x_D + b$$

---

## 2. Mathematical Formulation

### Model Equation

For a single observation $x \in \mathbb{R}^{1 \times D}$, target $y \in \mathbb{R}^{1 \times K}$ (typically $K=1$):

$$\hat{y} = x W + b$$

For a batch of $N$ observations arranged into matrix $X \in \mathbb{R}^{N \times D}$:

$$\hat{Y} = X W + \mathbf{1}_N b^T$$

where:
- $X \in \mathbb{R}^{N \times D}$ is the input feature matrix.
- $W \in \mathbb{R}^{D \times K}$ is the weight parameter matrix.
- $b \in \mathbb{R}^K$ is the bias (intercept) parameter vector.
- $\hat{Y} \in \mathbb{R}^{N \times K}$ is the model prediction matrix.

### Loss Function (Mean Squared Error)

Linear regression optimizes the empirical Mean Squared Error (MSE):

$$\mathcal{L}(W, b) = \frac{1}{N} \sum_{i=1}^N \|\hat{y}_i - y_i\|_2^2 = \frac{1}{N} \|\hat{Y} - Y\|_F^2$$

where $\|\cdot\|_F$ is the Frobenius norm.

### Analytical Gradients

Differentiating the loss function with respect to parameters:

$$\nabla_W \mathcal{L} = \frac{2}{N} X^T (\hat{Y} - Y)$$

$$\nabla_b \mathcal{L} = \frac{2}{N} \sum_{i=1}^N (\hat{y}_i - y_i)$$

In Nevula, these derivatives are automatically computed by **Autograd** through reverse-mode automatic differentiation.

---

## 3. Training Algorithm

Nevula trains Linear Regression iteratively via **Gradient Descent** (or mini-batch Stochastic Gradient Descent):

1. **Forward Pass**: Compute predictions $\hat{Y} = XW + b$.
2. **Loss Evaluation**: Compute scalar loss $\mathcal{L} = \text{MSELoss}(\hat{Y}, Y)$.
3. **Backward Pass**: Execute `loss.backward()`, accumulating gradients into $W.\text{grad}$ and $b.\text{grad}$.
4. **Parameter Update**: Update parameters via the optimizer:
   $$W \leftarrow W - \eta \nabla_W \mathcal{L}$$
   $$b \leftarrow b - \eta \nabla_b \mathcal{L}$$
5. **Zero Gradients**: Clear gradient buffers with `optimizer.zero_grad()`.
6. Repeat for $E$ epochs until convergence.

---

## 4. Parameters & Hyperparameters

### Trainable Parameters
- `weight` (`Parameter`): Weight matrix of shape `(in_features, out_features)`.
- `bias` (`Parameter`): Bias vector of shape `(out_features,)` (present when `fit_intercept=True`).

### Hyperparameters
- `in_features` (`int`): Dimensionality of input space $D$.
- `out_features` (`int`, default `1`): Dimensionality of target space $K$.
- `fit_intercept` (`bool`, default `True`): Whether to include bias term $b$.
- `mode` (`str`, default `"framework"`): Implementation style (`"framework"` using `nn.Linear` or `"tensor"` using direct tensor operations).
- `lr` (`float`, default `0.01`): Learning rate for gradient descent.
- `epochs` (`int`, default `100`): Maximum training iterations.
- `batch_size` (`int`, optional): Size of mini-batches. If `None`, executes full-batch gradient descent.
- `optimizer` (`str` or `Optimizer`, default `"sgd"`): Optimization algorithm (`"sgd"`, `"adam"`, `"adamw"`).

---

## 5. Two Implementation Modes in Nevula

Nevula's `LinearRegression` supports two educational modes:

### Mode A: Direct Tensor Parameterization (`mode="tensor"`)
Directly instantiates `Parameter(Tensor(...))` for weights and bias, and expresses the computation purely via tensor matrix multiplication:
$$\hat{Y} = X @ W + b$$

### Mode B: Framework Composition (`mode="framework"`)
Leverages Nevula's high-level neural network primitives (`nevula.nn.Linear`, `nevula.nn.MSELoss`, `nevula.optim.SGD`), demonstrating how classical ML algorithms naturally compose from standard deep-learning blocks.

---

## 6. Usage & Public API

```python
from nevula.models import LinearRegression
import numpy as np

# 1. Prepare data
X = np.random.randn(100, 3)
y = X @ np.array([[2.0], [-1.0], [0.5]]) + 1.2

# 2. Instantiate model
model = LinearRegression(in_features=3, out_features=1)

# 3. Fit model
model.fit(X, y, epochs=150, lr=0.05, optimizer="sgd")

# 4. Predict
predictions = model.predict(X)

# 5. Evaluate
r2 = model.evaluate(X, y, metric="r2")
mse = model.evaluate(X, y, metric="mse")
print(f"R^2 Score: {r2:.4f}, MSE: {mse:.4f}")
```

---

## 7. Computational Complexity

| Operation | Time Complexity | Space Complexity |
|---|---|---|
| **Forward Pass** | $\mathcal{O}(N \cdot D \cdot K)$ | $\mathcal{O}(N \cdot K)$ |
| **Loss & Backward** | $\mathcal{O}(N \cdot D \cdot K)$ | $\mathcal{O}(N \cdot K)$ |
| **Training (per epoch)** | $\mathcal{O}(N \cdot D \cdot K)$ | $\mathcal{O}(D \cdot K)$ |
| **Inference (`predict`)** | $\mathcal{O}(M \cdot D \cdot K)$ | $\mathcal{O}(M \cdot K)$ |

Where $N$ is sample count, $D$ is feature dimensionality, $K$ is target count, and $M$ is query batch size.

---

## 8. Limitations

1. **Linearity Assumption**: Cannot capture non-linear relationships without explicit basis expansion or polynomial features.
2. **Sensitivity to Outliers**: The quadratic penalty of MSE makes linear regression sensitive to anomalous points.
3. **Multicollinearity**: High correlation among input features can lead to numerical instability and large parameter variance (mitigated by Ridge/Lasso regularization in subsequent phases).

---

## 9. References

- Legendre, A. M. (1805). *Nouvelles méthodes pour la détermination des orbites des comètes*.
- Gauss, C. F. (1809). *Theoria Motus Corporum Coelestium in Sectionibus Conicis Solem Ambientium*.
- Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning* (2nd ed.). Springer.
