# Evaluation Metrics Guide

Nevula provides a unified evaluation metrics library covering regression and classification tasks. All metric functions seamlessly accept Nevula `Tensor`, NumPy `ndarray`, or standard Python sequences.

---

## 1. Regression Metrics

### Mean Absolute Error (MAE)
$$\text{MAE} = \frac{1}{N} \sum_{i=1}^N |y_i - \hat{y}_i|$$
- **When to use**: When errors should be penalized linearly and the distribution contains occasional extreme outliers.
- **Function**: `mean_absolute_error(y_true, y_pred)` (alias: `mae_score`)

### Mean Squared Error (MSE)
$$\text{MSE} = \frac{1}{N} \sum_{i=1}^N (y_i - \hat{y}_i)^2$$
- **When to use**: Strongly penalizes large errors. Serves as default loss and objective function in linear models.
- **Function**: `mean_squared_error(y_true, y_pred)` (alias: `mse_score`)

### Root Mean Squared Error (RMSE)
$$\text{RMSE} = \sqrt{\text{MSE}} = \sqrt{\frac{1}{N} \sum_{i=1}^N (y_i - \hat{y}_i)^2}$$
- **When to use**: Measures error in the **same units as the target variable**, making it easily interpretable while retaining quadratic sensitivity to large errors.
- **Function**: `root_mean_squared_error(y_true, y_pred)` (alias: `rmse_score`)

### Coefficient of Determination ($R^2$ Score)
$$R^2 = 1 - \frac{\sum_{i=1}^N (y_i - \hat{y}_i)^2}{\sum_{i=1}^N (y_i - \bar{y})^2}$$
- **When to use**: Quantifies the fraction of total target variance explained by the model ($1.0$ indicates perfect predictions, $0.0$ matches the baseline mean predictor, and negative values indicate performance worse than the sample mean).
- **Function**: `r2_score(y_true, y_pred)` (alias: `r_squared_score`)

---

## 2. Classification Metrics

### Confusion Matrix
Computes the matrix $C$ where $C_{i, j}$ represents the count of observations with true class $i$ and predicted class $j$.
- **Function**: `confusion_matrix(y_true, y_pred, labels=None)`

### Accuracy
$$\text{Accuracy} = \frac{\sum_{i=1}^N \mathbb{I}(y_i = \hat{y}_i)}{N}$$
- **When to use**: Balanced classes with equal costs for type I and type II errors.
- **Function**: `accuracy_score(y_true, y_pred)`

### Precision
$$\text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}}$$
- **When to use**: When false positives are costly (e.g. spam filters, fraud alerts).
- **Function**: `precision_score(y_true, y_pred, average='binary' | 'macro' | 'micro' | 'weighted', pos_label=1, zero_division=0.0)`

### Recall (Sensitivity)
$$\text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}}$$
- **When to use**: When false negatives are catastrophic (e.g. disease diagnostics, anomaly detection).
- **Function**: `recall_score(y_true, y_pred, average='binary' | 'macro' | 'micro' | 'weighted', pos_label=1, zero_division=0.0)`

### F1 Score
$$F_1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$$
- **When to use**: Harmonic balance between precision and recall, especially on imbalanced datasets.
- **Function**: `f1_score(y_true, y_pred, average='binary' | 'macro' | 'micro' | 'weighted', pos_label=1, zero_division=0.0)`

### ROC Curve & ROC-AUC
- **ROC Curve**: Plots True Positive Rate ($TPR = TP / P$) against False Positive Rate ($FPR = FP / N$) across varying classification thresholds.
  - **Function**: `fpr, tpr, thresholds = roc_curve(y_true, y_score, pos_label=1)`
- **Area Under Curve (AUC)**: Integrates the ROC curve via trapezoidal rule.
  - **Function**: `auc(fpr, tpr)`
- **ROC-AUC Score**: Computes ROC-AUC directly for binary or multiclass classification via One-vs-Rest (OvR).
  - **Function**: `roc_auc_score(y_true, y_score, average='macro' | 'weighted')`

---

## 3. Quick Usage Example

```python
import numpy as np
import nevula as nv
from nevula.metrics import (
    mean_absolute_error,
    mean_squared_error,
    root_mean_squared_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    roc_auc_score,
)

# Regression evaluation
y_true_reg = nv.tensor([10.0, 20.0, 30.0, 40.0])
y_pred_reg = nv.tensor([11.0, 19.5, 29.0, 42.0])

print("MAE: ", mean_absolute_error(y_true_reg, y_pred_reg))
print("MSE: ", mean_squared_error(y_true_reg, y_pred_reg))
print("RMSE:", root_mean_squared_error(y_true_reg, y_pred_reg))
print("R2:  ", r2_score(y_true_reg, y_pred_reg))

# Classification evaluation
y_true_cls = [1, 0, 1, 1, 0, 0]
y_pred_cls = [1, 0, 1, 0, 0, 1]
y_scores   = [0.9, 0.1, 0.85, 0.45, 0.2, 0.6]

print("Confusion Matrix:\n", confusion_matrix(y_true_cls, y_pred_cls))
print("Accuracy: ", accuracy_score(y_true_cls, y_pred_cls))
print("Precision:", precision_score(y_true_cls, y_pred_cls))
print("Recall:   ", recall_score(y_true_cls, y_pred_cls))
print("F1 Score: ", f1_score(y_true_cls, y_pred_cls))
print("ROC AUC:  ", roc_auc_score(y_true_cls, y_scores))
```
