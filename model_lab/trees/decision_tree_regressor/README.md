# Experimental Decision Tree Regression (`model_lab`)

Design specification and experimental incubation lab for Decision Tree Regression (CART).

---

## 1. Problem Formulation

Parametric linear models (OLS, Ridge, Lasso, SVR) assume an underlying global planar relationship $\hat{y} = XW + b$. When datasets exhibit non-linear step responses, sharp discontinuities, or high-order interactions, linear models fail.

**Decision Tree Regression** solves this non-parametrically by recursively partitioning feature space into orthogonal axis-aligned hyper-rectangles:

$$\hat{y}(x) = \sum_{m=1}^M c_m \cdot \mathbb{I}(x \in R_m)$$

where $c_m = \frac{1}{N_m} \sum_{i \in R_m} y_i$ is the average of training targets in leaf region $R_m$.

---

## 2. Greedy Recursive Partitioning (CART)

At each node $Q$ containing $N_Q$ samples with target variance $\text{Var}(y_Q)$, the algorithm greedily searches across all candidate features $j$ and split thresholds $s$ to maximize variance reduction:

$$\Delta \mathcal{I} = \text{Var}(y_Q) - \left( \frac{N_L}{N_Q} \text{Var}(y_L) + \frac{N_R}{N_Q} \text{Var}(y_R) \right)$$

Splitting continues recursively until reaching stopping criteria:
- `max_depth`
- `min_samples_split`
- `min_samples_leaf`
- `min_impurity_decrease`

---

## 3. Feature Importance Metric

The total importance of feature $j$ is the sum of impurity reductions across all internal nodes splitting on $j$, weighted by the number of samples visiting that node:

$$\text{Importance}(j) = \frac{\sum_{t \in \text{splits}(j)} N_t \cdot \Delta \mathcal{I}_t}{\sum_{t \in \text{all splits}} N_t \cdot \Delta \mathcal{I}_t}$$

---

## 4. Graduation Status

- [x] Mathematics and CART partitioning formalized.
- [x] Implemented within `BaseModel` lifecycle.
- [x] Verified non-linear step and wave fitting against linear models.
- [x] Graduated to official library: `nevula.models.DecisionTreeRegressor`.
