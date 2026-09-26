"""
Nevula Model Selection and Cross-Validation Module.

Provides comprehensive data splitting, cross-validation evaluation routines,
out-of-fold predictions, diagnostic learning curves, and hyperparameter
tuning (GridSearchCV, RandomizedSearchCV).
"""

from nevula.model_selection.split import (
    BaseCrossValidator,
    KFold,
    StratifiedKFold,
    LeaveOneOut,
    LeavePOut,
    ShuffleSplit,
    StratifiedShuffleSplit,
    TimeSeriesSplit,
    train_test_split,
    check_cv,
)
from nevula.model_selection.validation import (
    cross_val_score,
    cross_validate,
    cross_val_predict,
    learning_curve,
    validation_curve,
)
from nevula.model_selection.search import (
    ParameterGrid,
    ParameterSampler,
    BaseSearchCV,
    GridSearchCV,
    RandomizedSearchCV,
)
from nevula.model_selection.utils import (
    clone_estimator,
    safe_indexing,
    to_numpy,
    is_classifier,
    get_scorer,
)

__all__ = [
    # Splitters & Cross-Validators
    "BaseCrossValidator",
    "KFold",
    "StratifiedKFold",
    "LeaveOneOut",
    "LeavePOut",
    "ShuffleSplit",
    "StratifiedShuffleSplit",
    "TimeSeriesSplit",
    "train_test_split",
    "check_cv",
    # Validation Routines
    "cross_val_score",
    "cross_validate",
    "cross_val_predict",
    "learning_curve",
    "validation_curve",
    # Hyperparameter Optimization
    "ParameterGrid",
    "ParameterSampler",
    "BaseSearchCV",
    "GridSearchCV",
    "RandomizedSearchCV",
    # Utilities
    "clone_estimator",
    "safe_indexing",
    "to_numpy",
    "is_classifier",
    "get_scorer",
]
