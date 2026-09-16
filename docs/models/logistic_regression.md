# Logistic Regression

Logistic Regression is the foundational parametric model for binary and multiclass classification, modeling posterior probabilities via linear log-odds transformed by the sigmoid or softmax function.

---

## 1. Intuition

While Linear Regression fits an unbounded real hyperplane $\hat{y} = Xw + b \in (-\infty, \infty)$, classification requires predicting bounded class membership probabilities $P(y = 1 \mid x) \in [0, 1]$.

Logistic Regression resolves this by applying a non-linear **squashing function** (the **Sigmoid** activation $\sigma(z)$) to the linear combination:
- Large positive linear values $z \to +\infty$ produce probabilities $\sigma(z) \to 1.0$.
- Large negative linear values $z \to -\infty$ produce probabilities $\sigma(z) \to 0.0$.
- A linear score $z = 0$ corresponds to maximum uncertainty: $\sigma(0) = 0.5$, defining the **hyperplane decision boundary** $\{x \mid x^T w + b = 0\}$.

---

## 2. Mathematical Formulation

### 2.1 The Sigmoid Link Function
The standard logistic (sigmoid) function is defined as:

$$\sigma(z) = \frac{1}{1 + e^{-z}} = \frac{e^z}{1 + e^z}$$

Key mathematical properties:
1. **Symmetry**: $\sigma(-z) = 1 - \sigma(z)$
2. **Derivative**:
   $$\frac{d\sigma(z)}{dz} = \sigma(z)(1 - \sigma(z))$$
3. **Log-Odds (Logit)**:
   $$\log\left(\frac{P(y=1 \mid x)}{1 - P(y=1 \mid x)}\right) = x^T w + b$$

### 2.2 Binary Cross-Entropy Loss (BCE)
Given training observations $\{(x_i, y_i)\}_{i=1}^N$ with binary targets $y_i \in \{0, 1\}$:

$$\mathcal{L}(w, b) = -\frac{1}{N} \sum_{i=1}^N \left[ y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i) \right] + \mathcal{R}(w)$$

where $\hat{y}_i = \sigma(x_i^T w + b)$.

#### Numerically Stable BCE with Logits
To avoid numerical underflow/overflow when computing $\log(\sigma(z))$, Nevula employs the log-sum-exp trick:

$$\ell(z, y) = \max(z, 0) - z \cdot y + \log\left(1 + e^{-|z|}\right)$$

Analytical gradient:
$$\frac{\partial \ell}{\partial z} = \sigma(z) - y$$

### 2.3 Multiclass Classification (Softmax Regression)
For $K > 2$ classes, the model outputs $K$ raw logits $z_c = x^T w_c + b_c$. Probabilities are normalized via Softmax:

$$P(y = c \mid x) = \frac{e^{z_c}}{\sum_{j=1}^K e^{z_j}}$$

The loss is Categorical Cross-Entropy:
$$\mathcal{L}(W, b) = -\frac{1}{N} \sum_{i=1}^N \log P(y = y_i \mid x_i)$$

### 2.4 Regularization
Nevula's `LogisticRegression` supports:
- **$L_2$ Regularization**: $\frac{1}{2C} \|w\|_2^2$ (prevents exploding weights on separable data)
- **$L_1$ Regularization**: $\frac{1}{C} \|w\|_1$ (promotes weight sparsity and feature selection)

Where parameter $C > 0$ represents the inverse regularization strength (smaller $C \implies$ stronger regularization).

---

## 3. Hyperparameters & Configuration

| Parameter | Type | Default | Description |
|---|---|---|---|
| `in_features` | `int` | *required* | Dimensionality of input feature space $D$. |
| `n_classes` | `int` | `2` | Number of distinct classes (2 for binary, $> 2$ for multinomial). |
| `penalty` | `str` or `None` | `'l2'` | Regularization norm (`'l2'`, `'l1'`, or `None`). |
| `C` | `float` | `1.0` | Inverse regularization strength (must be positive). |
| `fit_intercept` | `bool` | `True` | Whether to calculate the bias term $b$. |

---

## 4. Usage Example

```python
import nevula as nv
from nevula.models import LogisticRegression
import numpy as np

# 1. Generate linearly separable binary data
X = np.random.normal(0, 1, size=(200, 4))
y = (X[:, 0] + 2 * X[:, 1] > 0).astype(int)

# 2. Instantiate and train LogisticRegression
clf = LogisticRegression(in_features=4, n_classes=2, penalty="l2", C=1.0)
clf.fit(X, y, epochs=100, lr=0.1, verbose=True)

# 3. Predict probabilities and discrete classes
probs = clf.predict_proba(X)   # Shape (N, 2)
preds = clf.predict(X)         # Shape (N, 1)

# 4. Evaluate performance
acc = clf.evaluate(X, y, metric="accuracy")
f1 = clf.evaluate(X, y, metric="f1")
print(f"Accuracy: {acc * 100:.2f}% | F1 Score: {f1:.4f}")
print(f"Learned Weights: {clf.coef_}")
print(f"Learned Intercept: {clf.intercept_}")
```

---

## 5. Computational Complexity

| Operation | Time Complexity | Space Complexity |
|---|---|---|
| **Training (`fit`)** | $\mathcal{O}(\text{epochs} \cdot N \cdot D \cdot K)$ | $\mathcal{O}(D \cdot K + N)$ |
| **Inference (`predict`)** | $\mathcal{O}(N_{\text{test}} \cdot D \cdot K)$ | $\mathcal{O}(N_{\text{test}} \cdot K)$ |
