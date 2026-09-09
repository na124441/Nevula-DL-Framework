# Experimental Linear Regression (`model_lab`)

Design specification and experimental log for ordinary linear regression.

---

## 1. Problem Formulation

Ordinary Linear Regression predicts a continuous scalar or multi-output target vector $y \in \mathbb{R}^K$ from a $D$-dimensional feature vector $x \in \mathbb{R}^D$:

$$\hat{y} = x W + b$$

For a batch of $N$ samples:

$$\hat{Y} = X W + \mathbf{1}_N b^T$$

where:
- $X \in \mathbb{R}^{N \times D}$
- $W \in \mathbb{R}^{D \times K}$
- $b \in \mathbb{R}^K$
- $\hat{Y} \in \mathbb{R}^{N \times K}$

---

## 2. Objective Function

Trained to minimize Mean Squared Error (MSE):

$$\mathcal{L}(W, b) = \frac{1}{N} \sum_{i=1}^N \|\hat{y}_i - y_i\|_2^2 = \frac{1}{N} \|\hat{Y} - Y\|_F^2$$

Analytical gradients:

$$\nabla_W \mathcal{L} = \frac{2}{N} X^T (\hat{Y} - Y)$$

$$\nabla_b \mathcal{L} = \frac{2}{N} \sum_{i=1}^N (\hat{y}_i - y_i)$$

Autograd computes these gradients automatically via backward accumulation.

---

## 3. Parameter Update

Using gradient descent:

$$W \leftarrow W - \eta \nabla_W \mathcal{L}$$
$$b \leftarrow b - \eta \nabla_b \mathcal{L}$$

where $\eta > 0$ is the learning rate.

---

## 4. Graduation Status

- [x] Mathematical specification defined.
- [x] Implemented using Nevula primitives (`Tensor`, `Autograd`, `Module`).
- [x] Forward and predict validated.
- [x] Gradient descent convergence verified on synthetic lines ($y = 3x + 2$).
- [x] Graduated to `nevula.models.LinearRegression`.
