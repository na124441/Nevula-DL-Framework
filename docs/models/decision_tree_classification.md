# Decision Tree Classification

`DecisionTreeClassifier` is a non-parametric supervised learning classifier implemented in Nevula using the standard **CART (Classification and Regression Trees)** framework. It partitions the feature space into orthogonal hyper-rectangles by recursively maximizing class impurity reduction.

---

## 1. Mathematical Formulation

### Impurity Measures

At any node $Q$ with $N_Q$ observations across $K$ classes ($k \in \{0, 1, \dots, K-1\}$), the empirical class distribution is:

$$p_k = \frac{1}{N_Q} \sum_{i \in Q} \mathbb{I}(y_i = k)$$

Nevula supports two classical impurity criteria:

1. **Gini Impurity (`criterion="gini"`, default)**:
   Measures the probability of a randomly chosen element being incorrectly labeled if randomly labeled according to the distribution:
   $$I_{\text{Gini}}(Q) = 1 - \sum_{k=0}^{K-1} p_k^2 = \sum_{k=0}^{K-1} p_k (1 - p_k)$$
   - Minimum value: $0.0$ (node is completely pure, all samples belong to one class).
   - Maximum value: $1 - 1/K$ (uniform distribution across classes).

2. **Entropy / Information Gain (`criterion="entropy"`)**:
   Measures Shannon entropy of the class distribution in bits:
   $$I_{\text{Entropy}}(Q) = - \sum_{k=0, p_k > 0}^{K-1} p_k \log_2(p_k)$$
   - Minimum value: $0.0$ (pure node).
   - Maximum value: $\log_2(K)$.

### Best Split Selection
For candidate feature $j$ and split threshold $s$, the node $Q$ is partitioned into:
$$Q_L = \{i \in Q \mid x_{ij} \le s\}, \quad Q_R = \{i \in Q \mid x_{ij} > s\}$$

The impurity reduction (Gain) achieved by split $(j, s)$ is:
$$\Delta I(j, s) = I(Q) - \left( \frac{N_L}{N_Q} I(Q_L) + \frac{N_R}{N_Q} I(Q_R) \right)$$

The algorithm searches across candidate features and thresholds to find:
$$(j^*, s^*) = \arg\max_{j, s} \Delta I(j, s)$$

### Leaf Node Predictions
For an observation $x$ routed to leaf node $L$:
- **Class Probabilities**:
  $$\hat{P}(Y = k \mid x \in L) = p_{k, L} = \frac{N_{k, L}}{N_L}$$
- **Predicted Class**:
  $$\hat{y} = \arg\max_k p_{k, L}$$

### Feature Importances
The total importance of feature $j$ is the sum of sample-weighted impurity decreases over all internal splits on feature $j$:
$$\text{Importance}(j) = \frac{\sum_{t \in \text{splits}(j)} N_t \cdot \Delta I_t}{\sum_{t \in \text{all splits}} N_t \cdot \Delta I_t}$$

Normalized such that $\sum_j \text{Importance}(j) = 1.0$.

---

## 2. Model API & Hyperparameters

```python
from nevula.models.trees import DecisionTreeClassifier

clf = DecisionTreeClassifier(
    criterion="gini",             # 'gini' or 'entropy'
    max_depth=5,                  # Maximum tree depth (None for unlimited)
    min_samples_split=2,          # Minimum samples to consider splitting
    min_samples_leaf=1,           # Minimum samples required in leaves
    min_impurity_decrease=0.0,    # Minimum gain required for split
    max_features=None,            # Subsample features per split
    n_classes=None,               # Optional class count (inferred if None)
)
```

### Primary Methods
- `fit(X, y)`: Builds tree recursively from training data.
- `predict(X)`: Returns predicted class labels tensor of shape `(N, 1)`.
- `predict_proba(X)`: Returns posterior class probabilities tensor of shape `(N, n_classes)`.
- `evaluate(X, y, metric="accuracy")`: Evaluates accuracy, precision, recall, f1, or log_loss.
- `feature_importances()`: Returns normalized 1D feature importance array.
- `tree_depth()`: Returns actual depth of the fitted tree.
- `n_leaves()`: Returns count of terminal leaf nodes.
- `state_dict()` / `load_state_dict()`: Full checkpoint serialization.

---

## 3. Quickstart Example

```python
import numpy as np
from nevula.models.trees import DecisionTreeClassifier

# 1. Prepare synthetic multiclass data
X = np.array([
    [1.0, 2.0], [1.2, 1.8],
    [5.0, 6.0], [5.2, 5.8],
    [9.0, 1.0], [9.2, 1.2],
])
y = np.array([0, 0, 1, 1, 2, 2])

# 2. Fit DecisionTreeClassifier
clf = DecisionTreeClassifier(criterion="gini", max_depth=3)
clf.fit(X, y)

# 3. Predict classes and probabilities
preds = clf.predict(X)
probs = clf.predict_proba(X)

print("Predictions:", preds.to_list())
print("Probabilities:", probs.to_list())
print("Accuracy:", clf.evaluate(X, y, metric="accuracy"))
```
