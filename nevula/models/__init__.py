from nevula.models.base import BaseModel
from nevula.models.registry import register_model, get_model, list_models, summary
from nevula.models.regression.linear import LinearRegression
from nevula.models.regression.ridge import RidgeRegression
from nevula.models.regression.lasso import LassoRegression
from nevula.models.regression.svr import SVR
from nevula.models.trees.decision_tree import DecisionTreeRegressor, DecisionTreeClassifier
from nevula.models.ensemble.voting import VotingRegressor
from nevula.models.ensemble.random_forest import RandomForestRegressor
from nevula.models.ensemble.gradient_boosting import GradientBoostingRegressor
from nevula.models.classification.logistic import LogisticRegression
from nevula.models.classification.svm import SVC, SVM
from nevula.models import regression
from nevula.models import classification
from nevula.models import trees
from nevula.models import ensemble
from nevula.models import deep

__all__ = [
    "BaseModel",
    "LinearRegression",
    "RidgeRegression",
    "LassoRegression",
    "SVR",
    "DecisionTreeRegressor",
    "DecisionTreeClassifier",
    "VotingRegressor",
    "RandomForestRegressor",
    "GradientBoostingRegressor",
    "LogisticRegression",
    "SVC",
    "SVM",
    "register_model",
    "get_model",
    "list_models",
    "summary",
    "regression",
    "classification",
    "trees",
    "ensemble",
    "deep",
]
