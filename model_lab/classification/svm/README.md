# 🔬 Model Lab: Support Vector Machine (SVC / SVM)

> **In-depth mathematical exploration, subgradient derivation, and experimental laboratory for Support Vector Classification in the Nevula Framework.**

---

## 1. Mathematical Foundations

### 1.1 Maximum Margin Hyperplane

Consider a binary classification training dataset:
$$\mathcal{D} = \{(x_i, y_i)\}_{i=1}^N, \quad x_i \in \mathbb{R}^D, \quad y_i \in \{-1, +1\}$$

A linear decision surface is defined by the affine hyperplane:
$$\mathcal{H} = \{x \in \mathbb{R}^D : w^T x + b = 0\}$$

The orthogonal Euclidean distance from an arbitrary point $x_i$ to the hyperplane is:
$$\gamma_i = \frac{y_i (w^T x_i + b)}{\|w\|_2}$$

In a linearly separable dataset, the **hard margin** SVM seeks the hyperplane that maximizes the geometric margin $\gamma = \min_i \gamma_i$:
$$\max_{w, b} \frac{1}{\|w\|_2} \min_{i} \left[ y_i (w^T x_i + b) \right]$$

Setting the functional margin of the canonical boundary points to 1 ($y_i (w^T x_i + b) \ge 1$), maximizing $\frac{1}{\|w\|}$ is strictly equivalent to minimizing the convex quadratic objective:
$$\min_{w, b} \frac{1}{2} \|w\|_2^2 \quad \text{subject to} \quad y_i(w^T x_i + b) \ge 1, \quad \forall i=1,\dots,N$$

---

### 1.2 Soft-Margin Formulation & Slack Variables

Real-world datasets are rarely linearly separable. To tolerate noise, outliers, and overlapping class distributions, Cortes & Vapnik (1995) introduced non-negative slack variables $\xi_i \ge 0$:
$$\min_{w, b, \xi} \frac{1}{2} \|w\|_2^2 + C \sum_{i=1}^N \xi_i$$
$$\text{subject to} \quad y_i (w^T x_i + b) \ge 1 - \xi_i, \quad \xi_i \ge 0 \quad \forall i$$

Here:
- $\xi_i = 0$: The point lies on or outside the correct margin ($y_i z_i \ge 1$).
- $0 < \xi_i \le 1$: The point is correctly classified, but lies within the margin tube.
- $\xi_i > 1$: The point violates the decision boundary and is misclassified.
- $C > 0$: Regularization parameter balancing margin maximization against penalty violations.

---

### 1.3 Primal Formulation as Unconstrained Hinge Loss

Since $\xi_i = \max(0, 1 - y_i(w^T x_i + b))$, the constrained soft-margin problem maps directly to unconstrained empirical risk minimization with the **Hinge Loss**:
$$\min_{w, b} \mathcal{L}(w, b) = \frac{1}{N} \sum_{i=1}^N \max(0, 1 - y_i (w^T x_i + b)) + \frac{1}{2C} \|w\|_2^2$$

For the **Squared Hinge Loss** (L2-SVM):
$$\min_{w, b} \mathcal{L}_{\text{sq}}(w, b) = \frac{1}{N} \sum_{i=1}^N \max(0, 1 - y_i (w^T x_i + b))^2 + \frac{1}{2C} \|w\|_2^2$$

---

### 1.4 Subgradient Calculus & Autograd Execution

The Hinge loss is convex but non-differentiable at $1 - y_i z_i = 0$. Its subgradient with respect to the pre-activation logit $z_i = w^T x_i + b$ is:
$$\partial_{z_i} \max(0, 1 - y_i z_i) = \begin{cases} -y_i & \text{if } 1 - y_i z_i > 0 \\ 0 & \text{if } 1 - y_i z_i < 0 \\ [-y_i, 0] & \text{if } 1 - y_i z_i = 0 \end{cases}$$

Applying the chain rule:
$$\nabla_w \mathcal{L} = -\frac{1}{N} \sum_{i \in \mathcal{V}} y_i x_i + \frac{1}{C} w$$
$$\nabla_b \mathcal{L} = -\frac{1}{N} \sum_{i \in \mathcal{V}} y_i$$
where $\mathcal{V} = \{i : 1 - y_i (w^T x_i + b) > 0\}$ is the active set of margin violators and margin points.

In Nevula, this subgradient is evaluated automatically and stably through the computational graph:
```python
diff = 1.0 - y * scores
margin_loss = diff.relu().mean()
reg_loss = (self.weight ** 2).sum() * (0.5 / self.C)
total_loss = margin_loss + reg_loss
total_loss.backward()
```

---

### 1.5 Support Vector Identification

Points $(x_i, y_i)$ satisfying:
$$1 - y_i (w^T x_i + b) \ge -\epsilon_{\text{tol}}$$
are **Support Vectors**. These are the critical observations that determine the orientation and position of the separating hyperplane. Points with $y_i (w^T x_i + b) > 1$ could be removed from the training set without altering the optimal decision boundary!

---

### 1.6 Multiclass One-vs-Rest (OvR)

For $K > 2$ classes, Nevula optimizes $K$ hyperplanes simultaneously:
$$Z = X W + b \in \mathbb{R}^{N \times K}$$
with target matrix $Y \in \{-1, +1\}^{N \times K}$ where $Y_{ik} = +1$ if sample $i$ belongs to class $k$, and $-1$ otherwise.
$$\hat{y}_i = \arg\max_{k \in \{0, \dots, K-1\}} Z_{ik}$$

---

## 2. Laboratory Experiments

Run the benchmark script:
```powershell
python model_lab/classification/svm/experiment.py
```
