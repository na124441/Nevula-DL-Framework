"""
Classification models package for Nevula.
"""

from nevula.models.classification.logistic import LogisticRegression
from nevula.models.classification.svm import SVC, SVM
from nevula.models.trees.decision_tree import DecisionTreeClassifier

__all__ = ["LogisticRegression", "SVC", "SVM", "DecisionTreeClassifier"]
