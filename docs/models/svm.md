# Support Vector Machine (SVC / SVM)

The Support Vector Classifier (`SVC` / `SVM`) is a maximum-margin classification algorithm that constructs optimal linear hyperplanes with maximum geometric separation between classes. Equipped with soft-margin **Hinge Loss** and $L_2$ regularization, it provides outstanding generalization on high-dimensional data and robust boundary decisions governed strictly by **Support Vectors**.

---

## 1. Intuition

While probabilistic classifiers like Logistic Regression optimize likelihood across all data points (even those far away from the decision boundary), the Support Vector Machine focuses exclusively on the boundary itself:

1. **Maximum Margin Principle**: Among all hyperplanes that separate two classes, the SVM finds the unique hyperplane that maximizes the geometric distance to the closest training examples from either class.
2. **Support Vectors**: The only observations that dictate the position and angle of the decision boundary are those that fall strictly on the margin or violate it ($y_i (w^T x_i + b) \le 1$). Points located comfortably deep inside their correct class territory have zero subgradient and exert no force on the separating hyperplane.
3. **Soft Margin & Regularization ($C$)**: Real data is rarely linearly separable. The soft-margin formulation introduces a trade-off parameter $C$:
   - **Large $C$**: Penalizes margin violations harshly, prioritizing accurate training classification with a narrower margin.
   - **Small $C$**: Emphasizes maximizing the margin width ($2 / \|w\|_2$), allowing a softer margin with more tolerance for boundary overlap.

---

## 2. Mathematical Formulation

### 2.1 Hyperplane & Decision Function
For an input vector $x \in \mathbb{R}^D$, the linear decision score is:

$$z = x^T w + b$$

The predicted class is:
$$\hat{y} = \text{sign}(z) = \begin{cases} +1 & \text{if } z \ge 0 \\ -1 & \text{if } z < 0 \end{cases}$$

### 2.2 Primal Soft-Margin Objective
Given training pairs $\{(x_i, y_i)\}_{i=1}^N$ with labels $y_i \in \{-1, +1\}$:

$$\min_{w, b} \mathcal{L}(w, b) = \frac{1}{N} \sum_{i=1}^N \max\left(0, 1 - y_i (x_i^T w + b)\right)^p + \frac{1}{2C} \|w\|_2^2$$

where:
- $p = 1$ for standard **Hinge Loss** (`loss='hinge'`)
- $p = 2$ for **Squared Hinge Loss** (`loss='squared_hinge'`)
- $C > 0$ is the inverse regularization parameter

### 2.3 Subgradient Calculus
For the Hinge loss $\ell(z_i, y_i) = \max(0, 1 - y_i z_i)$, the subgradient with respect to $w$ and $b$ is:

$$\frac{\partial \ell}{\partial w} = \begin{cases} -y_i x_i & \text{if } y_i (w^T x_i + b) < 1 \\ 0 & \text{if } y_i (w^T x_i + b) > 1 \end{cases}$$

$$\nabla_w \mathcal{L} = -\frac{1}{N} \sum_{i \in \text{SV}} y_i x_i + \frac{1}{C} w$$

$$\nabla_b \mathcal{L} = -\frac{1}{N} \sum_{i \in \text{SV}} y_i$$

In Nevula, subgradients are computed automatically via reverse-mode automatic differentiation through the `relu()` primitive:
$$\max(0, 1 - y_i z_i) = \text{ReLU}(1 - y_i z_i)$$

### 2.4 Multiclass One-vs-Rest (OvR)
For $K > 2$ classes, Nevula trains $K$ hyperplanes $W \in \mathbb{R}^{D \times K}, b \in \mathbb{R}^K$. For class $k$, samples belonging to class $k$ have target $+1$, and all other samples have target $-1$:
$$\hat{y} = \arg\max_{k \in \{0, \dots, K-1\}} \left( x^T w_k + b_k \right)$$

---

## 3. Hyperparameters & Configuration

| Parameter | Type | Default | Description |
|---|---|---|---|
| `in_features` | `int` | *required* | Dimensionality of input feature space $D$. |
| `n_classes` | `int` | `2` | Number of classes ($2$ for binary, $> 2$ for multiclass OvR). |
| `C` | `float` | `1.0` | Regularization parameter (must be positive). |
| `loss` | `str` | `'hinge'` | Loss function (`'hinge'` or `'squared_hinge'`). |
| `fit_intercept` | `bool` | `True` | Whether to calculate the bias term $b$. |
| `mode` | `str` | `'framework'` | Architecture mode (`'framework'` via `nn.Linear` or `'tensor'`). |

---

## 4. Support Vector Extraction Methods

| Method | Return Type | Description |
|---|---|---|
| `support_vectors_mask(X, y, tol=1e-3)` | `np.ndarray` (bool) | Boolean mask of samples on or violating the margin boundary. |
| `support_vectors(X, y, tol=1e-3)` | `np.ndarray` (float) | 2D coordinates of the identified support vector samples. |
| `support_vectors_ratio(X, y, tol=1e-3)` | `float` | Fraction of dataset comprising support vectors $\in [0, 1]$. |
| `n_support_(X, y, tol=1e-3)` | `int` | Total count of support vectors. |

---

## 5. Usage Example

```python
import numpy as np
import nevula as nv
from nevula.models import SVC

# 1. Prepare binary classification dataset
X = np.random.normal(0, 1, size=(200, 2))
y = (X[:, 0] + X[:, 1] > 0).astype(int)

# 2. Instantiate and train SVC
clf = SVC(in_features=2, n_classes=2, C=5.0, loss="hinge")
clf.fit(X, y, epochs=100, lr=0.05, optimizer="adam")

# 3. Predict and evaluate
preds = clf.predict(X)
acc = clf.evaluate(X, y, metric="accuracy")
f1 = clf.evaluate(X, y, metric="f1")

# 4. Inspect Support Vectors & Margin
n_sv = clf.n_support_(X, y)
sv_ratio = clf.support_vectors_ratio(X, y)
w = clf.coef_.ravel()
margin_width = 2.0 / np.linalg.norm(w)

print(f"Accuracy: {acc * 100:.2f}% | F1 Score: {f1:.4f}")
print(f"Margin Width: {margin_width:.4f}")
print(f"Support Vectors: {n_sv} ({sv_ratio * 100:.1f}%)")
```

---

## 6. Comparison: Logistic Regression vs Support Vector Classifier

| Characteristic | Logistic Regression | Support Vector Classifier (SVC) |
|---|---|---|
| **Loss Function** | Binary Cross-Entropy / Log-Loss | Hinge Loss / Squared Hinge |
| **Output Type** | Calibrated posterior probabilities $P(y \mid x)$ | Geometric signed distances (margins) $z$ |
| **Active Data Points** | All samples influence gradient | Only Support Vectors influence boundary |
| **Sensitivity to Outliers** | Mildly sensitive to far-off points | Highly robust (bounded subgradient) |
| **Inductive Bias** | Maximum Likelihood Estimation | Maximum Geometric Margin |
| **Multi-class Strategy** | Multinomial Softmax | One-vs-Rest (OvR) Hyperplanes |
| **Nevula Implementation** | `LogisticRegression` | `SVC` / `SVM` |
