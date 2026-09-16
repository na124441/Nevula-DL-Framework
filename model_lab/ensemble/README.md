# Experimental Ensemble Learning (`model_lab`)

Design specification and experimental lab for ensemble methods in Nevula:
- **Voting Regressor**: Model averaging across heterogeneous estimators.
- **Random Forest Regressor**: Bootstrap aggregating (bagging) + random feature subspaces.
- **Gradient Boosting Regressor**: Forward stage-wise additive modeling fitting pseudo-residuals.

---

## 1. Bias-Variance Tradeoff in Ensembles

Single estimators typically suffer from one of two extremes:
- High bias (underfitting): e.g. shallow models or simple linear assumptions.
- High variance (overfitting): e.g. deep unpruned decision trees sensitive to training sample fluctuations.

Ensemble methods address these dual challenges:

### Bagging (Variance Reduction)
Given $B$ independent estimators each with variance $\sigma^2$ and pairwise correlation $\rho$:
$$\text{Var}\left(\frac{1}{B} \sum_{b=1}^B h_b(X)\right) = \rho \sigma^2 + \frac{1 - \rho}{B} \sigma^2$$
Random feature subspace sampling forces trees to de-correlate ($\rho \downarrow$), allowing the $\frac{1-\rho}{B}$ term to drive variance down dramatically.

### Boosting (Bias Reduction)
Boosting fits sequential weak learners $h_m(x)$ to the residual errors of the current ensemble:
$$F_m(x) = F_{m-1}(x) + \eta \cdot h_m(x)$$
Iteratively reducing training error and driving bias toward zero.

---

## 2. Graduation Status

- [x] Mathematics and algorithms formalized.
- [x] Implemented within `BaseModel` lifecycle.
- [x] Verified on complex noisy non-linear surfaces.
- [x] Graduated to official library: `nevula.models.ensemble`.
