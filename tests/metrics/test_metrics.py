import os
import sys

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import unittest
import numpy as np

import nevula as nv
from nevula.core.tensor import Tensor
from nevula.metrics import (
    mean_squared_error,
    mean_absolute_error,
    root_mean_squared_error,
    r2_score,
    mse_score,
    mae_score,
    rmse_score,
    r_squared_score,
    mae,
    mse,
    rmse,
    r2,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    auc,
    roc_curve,
    roc_auc_score,
    log_loss_score,
    accuracy,
    precision,
    recall,
    f1,
    roc_auc,
    log_loss,
)



class TestRegressionMetrics(unittest.TestCase):
    """
    Test suite for regression metrics: MSE, MAE, RMSE, and R^2.
    """

    def test_perfect_predictions(self):
        y_true = [1.0, 2.0, 3.0, 4.0]
        y_pred = [1.0, 2.0, 3.0, 4.0]

        self.assertAlmostEqual(mean_squared_error(y_true, y_pred), 0.0)
        self.assertAlmostEqual(mean_absolute_error(y_true, y_pred), 0.0)
        self.assertAlmostEqual(root_mean_squared_error(y_true, y_pred), 0.0)
        self.assertAlmostEqual(r2_score(y_true, y_pred), 1.0)

    def test_known_values(self):
        y_true = np.array([3.0, -0.5, 2.0, 7.0])
        y_pred = np.array([2.5, 0.0, 2.0, 8.0])

        # Errors: [0.5, -0.5, 0.0, -1.0]
        # Squared: [0.25, 0.25, 0.0, 1.0] -> sum = 1.5 -> mean = 0.375
        # Absolute: [0.5, 0.5, 0.0, 1.0] -> sum = 2.0 -> mean = 0.5
        self.assertAlmostEqual(mean_squared_error(y_true, y_pred), 0.375)
        self.assertAlmostEqual(mse_score(y_true, y_pred), 0.375)

        self.assertAlmostEqual(mean_absolute_error(y_true, y_pred), 0.5)
        self.assertAlmostEqual(mae_score(y_true, y_pred), 0.5)

        self.assertAlmostEqual(root_mean_squared_error(y_true, y_pred), np.sqrt(0.375))
        self.assertAlmostEqual(rmse_score(y_true, y_pred), np.sqrt(0.375))

        # Variance of y_true: mean = 2.875, ss_tot = sum((y - 2.875)^2) = 29.1875
        # ss_res = 1.5
        # r2 = 1 - 1.5 / 29.1875 = 0.9486098...
        expected_r2 = 1.0 - (1.5 / np.sum((y_true - np.mean(y_true)) ** 2))
        self.assertAlmostEqual(r2_score(y_true, y_pred), expected_r2, places=5)
        self.assertAlmostEqual(r_squared_score(y_true, y_pred), expected_r2, places=5)

    def test_tensor_inputs(self):
        t_true = Tensor([10.0, 20.0, 30.0])
        t_pred = Tensor([12.0, 19.0, 28.0])

        self.assertAlmostEqual(mean_squared_error(t_true, t_pred), (4.0 + 1.0 + 4.0) / 3.0)
        self.assertAlmostEqual(mean_absolute_error(t_true, t_pred), (2.0 + 1.0 + 2.0) / 3.0)
        self.assertAlmostEqual(root_mean_squared_error(t_true, t_pred), np.sqrt(9.0 / 3.0))

    def test_empty_and_mismatched_inputs(self):
        with self.assertRaises(ValueError):
            mean_squared_error([], [])
        with self.assertRaises(ValueError):
            mean_absolute_error([1.0], [1.0, 2.0])
        with self.assertRaises(ValueError):
            r2_score([1.0], [1.0, 2.0])


class TestClassificationMetrics(unittest.TestCase):
    """
    Test suite for classification metrics: Accuracy, Precision, Recall, F1, Confusion Matrix, and ROC-AUC.
    """

    def test_confusion_matrix_binary(self):
        y_true = [0, 1, 0, 1, 1, 0]
        y_pred = [0, 1, 1, 1, 0, 0]

        # TN: y_true=0, y_pred=0 -> 2
        # FP: y_true=0, y_pred=1 -> 1
        # FN: y_true=1, y_pred=0 -> 1
        # TP: y_true=1, y_pred=1 -> 2
        cm = confusion_matrix(y_true, y_pred)
        expected_cm = np.array([[2, 1], [1, 2]])
        np.testing.assert_array_equal(cm, expected_cm)

    def test_confusion_matrix_multiclass(self):
        y_true = [0, 1, 2, 0, 1, 2]
        y_pred = [0, 2, 2, 0, 1, 1]

        cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
        self.assertEqual(cm.shape, (3, 3))
        # Class 0: 2 correctly predicted as 0
        self.assertEqual(cm[0, 0], 2)
        # Class 1: 1 predicted as 1, 1 predicted as 2
        self.assertEqual(cm[1, 1], 1)
        self.assertEqual(cm[1, 2], 1)
        # Class 2: 1 predicted as 1, 1 predicted as 2
        self.assertEqual(cm[2, 1], 1)
        self.assertEqual(cm[2, 2], 1)

    def test_accuracy_score(self):
        y_true = [0, 1, 2, 3]
        y_pred = [0, 2, 2, 3]
        self.assertAlmostEqual(accuracy_score(y_true, y_pred), 0.75)

    def test_precision_recall_f1_binary(self):
        # 3 positive samples, 3 negative samples
        # TP: 2, FP: 1, FN: 1, TN: 2
        y_true = [1, 1, 1, 0, 0, 0]
        y_pred = [1, 1, 0, 1, 0, 0]

        p = precision_score(y_true, y_pred, average="binary")
        r = recall_score(y_true, y_pred, average="binary")
        f1 = f1_score(y_true, y_pred, average="binary")

        # Precision = 2 / (2 + 1) = 2/3
        self.assertAlmostEqual(p, 2.0 / 3.0)
        # Recall = 2 / (2 + 1) = 2/3
        self.assertAlmostEqual(r, 2.0 / 3.0)
        # F1 = 2 * (2/3 * 2/3) / (4/3) = 2/3
        self.assertAlmostEqual(f1, 2.0 / 3.0)

    def test_multiclass_averages(self):
        y_true = [0, 0, 1, 1, 2, 2]
        y_pred = [0, 1, 1, 1, 2, 0]

        # Class 0: TP=1, FP=1, FN=1 -> p=1/2, r=1/2, f1=1/2
        # Class 1: TP=2, FP=1, FN=0 -> p=2/3, r=2/2=1.0, f1=4/5=0.8
        # Class 2: TP=1, FP=0, FN=1 -> p=1/1=1.0, r=1/2=0.5, f1=2/3
        p_macro = precision_score(y_true, y_pred, average="macro")
        self.assertAlmostEqual(p_macro, (0.5 + 2.0 / 3.0 + 1.0) / 3.0)

        r_macro = recall_score(y_true, y_pred, average="macro")
        self.assertAlmostEqual(r_macro, (0.5 + 1.0 + 0.5) / 3.0)

        # Micro precision = Micro recall = Accuracy
        p_micro = precision_score(y_true, y_pred, average="micro")
        acc = accuracy_score(y_true, y_pred)
        self.assertAlmostEqual(p_micro, acc)

        # Weighted average
        f1_weighted = f1_score(y_true, y_pred, average="weighted")
        self.assertGreater(f1_weighted, 0.5)

    def test_roc_curve_and_binary_auc(self):
        y_true = np.array([0, 0, 1, 1])
        y_scores = np.array([0.1, 0.4, 0.35, 0.8])

        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        score_auc = auc(fpr, tpr)
        roc_score = roc_auc_score(y_true, y_scores)

        self.assertAlmostEqual(score_auc, roc_score)
        self.assertGreaterEqual(roc_score, 0.7)
        self.assertLessEqual(roc_score, 1.0)

    def test_perfect_roc_auc(self):
        y_true = [0, 0, 1, 1]
        y_scores = [0.1, 0.2, 0.8, 0.9]

        roc_score = roc_auc_score(y_true, y_scores)
        self.assertAlmostEqual(roc_score, 1.0)

    def test_multiclass_roc_auc_ovr(self):
        y_true = np.array([0, 1, 2, 0, 1, 2])
        # High confidence probabilities matching ground truth
        y_probs = np.array([
            [0.9, 0.05, 0.05],
            [0.1, 0.8, 0.1],
            [0.05, 0.15, 0.8],
            [0.85, 0.1, 0.05],
            [0.2, 0.75, 0.05],
            [0.1, 0.1, 0.8],
        ])

        macro_auc = roc_auc_score(y_true, y_probs, average="macro")
        weighted_auc = roc_auc_score(y_true, y_probs, average="weighted")

        self.assertAlmostEqual(macro_auc, 1.0)
        self.assertAlmostEqual(weighted_auc, 1.0)

    def test_tensor_support_in_classification(self):
        t_true = Tensor([1, 0, 1, 1])
        t_pred = Tensor([1, 0, 0, 1])

        self.assertAlmostEqual(accuracy_score(t_true, t_pred), 0.75)
        self.assertAlmostEqual(precision_score(t_true, t_pred), 1.0)
        self.assertAlmostEqual(recall_score(t_true, t_pred), 2.0 / 3.0)

    def test_log_loss_binary(self):
        y_true = [1, 0, 1, 0]
        y_prob = [0.9, 0.1, 0.8, 0.2]

        score = log_loss_score(y_true, y_prob)
        # -[ln(0.9) + ln(0.9) + ln(0.8) + ln(0.8)] / 4
        expected = -0.25 * (np.log(0.9) + np.log(0.9) + np.log(0.8) + np.log(0.8))
        self.assertAlmostEqual(score, expected, places=5)
        self.assertAlmostEqual(log_loss(y_true, y_prob), score)

        # Tensor input
        t_true = Tensor(y_true)
        t_prob = Tensor(y_prob)
        self.assertAlmostEqual(log_loss_score(t_true, t_prob), score)

    def test_log_loss_multiclass(self):
        y_true = [0, 1, 2]
        y_prob = np.array([
            [0.8, 0.1, 0.1],
            [0.1, 0.8, 0.1],
            [0.2, 0.2, 0.6],
        ])
        score = log_loss_score(y_true, y_prob)
        expected = -(np.log(0.8) + np.log(0.8) + np.log(0.6)) / 3.0
        self.assertAlmostEqual(score, expected, places=5)

        # One-hot true targets
        y_true_onehot = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ])
        score_oh = log_loss_score(y_true_onehot, y_prob)
        self.assertAlmostEqual(score_oh, expected, places=5)

    def test_sklearn_parity(self):
        from sklearn import metrics as skm

        # Classification
        y_true = np.array([0, 1, 1, 0, 1, 0, 1, 1])
        y_pred = np.array([0, 1, 0, 0, 1, 1, 1, 0])
        y_score = np.array([0.1, 0.85, 0.4, 0.2, 0.9, 0.6, 0.75, 0.35])

        self.assertAlmostEqual(accuracy_score(y_true, y_pred), skm.accuracy_score(y_true, y_pred))
        self.assertAlmostEqual(precision_score(y_true, y_pred), skm.precision_score(y_true, y_pred))
        self.assertAlmostEqual(recall_score(y_true, y_pred), skm.recall_score(y_true, y_pred))
        self.assertAlmostEqual(f1_score(y_true, y_pred), skm.f1_score(y_true, y_pred))
        self.assertAlmostEqual(roc_auc_score(y_true, y_score), skm.roc_auc_score(y_true, y_score))
        self.assertAlmostEqual(log_loss_score(y_true, y_score), skm.log_loss(y_true, y_score), places=5)
        np.testing.assert_array_equal(confusion_matrix(y_true, y_pred), skm.confusion_matrix(y_true, y_pred))

        # Regression
        y_reg_true = np.array([3.0, -0.5, 2.0, 7.0])
        y_reg_pred = np.array([2.5, 0.0, 2.0, 8.0])

        self.assertAlmostEqual(mean_squared_error(y_reg_true, y_reg_pred), skm.mean_squared_error(y_reg_true, y_reg_pred))
        self.assertAlmostEqual(mean_absolute_error(y_reg_true, y_reg_pred), skm.mean_absolute_error(y_reg_true, y_reg_pred))
        self.assertAlmostEqual(root_mean_squared_error(y_reg_true, y_reg_pred), skm.root_mean_squared_error(y_reg_true, y_reg_pred))
        self.assertAlmostEqual(r2_score(y_reg_true, y_reg_pred), skm.r2_score(y_reg_true, y_reg_pred))

    def test_aliases(self):
        y_true = [1, 0, 1]
        y_pred = [1, 0, 0]
        y_prob = [0.9, 0.1, 0.4]

        self.assertEqual(accuracy(y_true, y_pred), accuracy_score(y_true, y_pred))
        self.assertEqual(precision(y_true, y_pred), precision_score(y_true, y_pred))
        self.assertEqual(recall(y_true, y_pred), recall_score(y_true, y_pred))
        self.assertEqual(f1(y_true, y_pred), f1_score(y_true, y_pred))
        self.assertEqual(roc_auc(y_true, y_prob), roc_auc_score(y_true, y_prob))
        self.assertEqual(log_loss(y_true, y_prob), log_loss_score(y_true, y_prob))

        y_r_true = [1.0, 2.0]
        y_r_pred = [1.5, 2.5]
        self.assertEqual(mae(y_r_true, y_r_pred), mean_absolute_error(y_r_true, y_r_pred))
        self.assertEqual(mse(y_r_true, y_r_pred), mean_squared_error(y_r_true, y_r_pred))
        self.assertEqual(rmse(y_r_true, y_r_pred), root_mean_squared_error(y_r_true, y_r_pred))
        self.assertEqual(r2(y_r_true, y_r_pred), r2_score(y_r_true, y_r_pred))


if __name__ == "__main__":
    unittest.main()

