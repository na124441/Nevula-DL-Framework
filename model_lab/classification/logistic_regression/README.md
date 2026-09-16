# Logistic Regression (`model_lab`)

Design specification and experimental lab for Logistic Regression in Nevula.

---

## 1. Mathematical Formulation

Logistic regression models the posterior probability of class membership using the **sigmoid (logistic) link function**:

$$P(y = 1 \mid x) = \sigma(z) = \frac{1}{1 + e^{-z}}, \quad \text{where } z = x^T w + b$$

### Log-Odds (Logit)
$$\log\left(\frac{P(y=1 \mid x)}{1 - P(y=1 \mid x)}\right) = x^T w + b$$

The log-odds of the positive class are a strictly linear function of the input features $x$.

### Negative Log-Likelihood (Binary Cross-Entropy Loss)
Given training observations $\{(x_i, y_i)\}_{i=1}^N$ with $y_i \in \{0, 1\}$:

$$\mathcal{L}(w, b) = -\frac{1}{N} \sum_{i=1}^N \left[ y_i \log(\sigma(z_i)) + (1 - y_i) \log(1 - \sigma(z_i)) \right] + \mathcal{R}(w)$$

Where $\mathcal{R}(w)$ is the optional regularization term:
- **$L_2$ Regularization**: $\frac{1}{2C} \|w\|_2^2$
- **$L_1$ Regularization**: $\frac{1}{C} \|w\|_1$

### Gradient Derivation
$$\frac{\partial \mathcal{L}}{\partial z_i} = \sigma(z_i) - y_i = \hat{y}_i - y_i$$
$$\nabla_w \mathcal{L} = \frac{1}{N} X^T (\sigma(Xw + b) - y) + \nabla_w \mathcal{R}(w)$$
$$\frac{\partial \mathcal{L}}{\partial b} = \frac{1}{N} \sum_{i=1}^N (\sigma(z_i) - y_i)$$

---

## 2. Experimental Objectives
- Verify linear decision boundary recovery on separable Gaussian point clouds.
- Measure convergence speed under different learning rates and regularizations ($L_2$ vs $L_1$).
- Benchmark binary vs multiclass (Softmax) accuracy.
