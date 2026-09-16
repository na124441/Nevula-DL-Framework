from abc import ABC
from typing import Any, Dict, List, Optional, Union
import numpy as np

from nevula.core.tensor import Tensor
from nevula.models.base import BaseModel


class BaseEnsemble(BaseModel, ABC):
    """
    Abstract base class for all ensemble models in Nevula.

    Provides common infrastructure for managing collections of base estimators,
    parallel or sequential predictions, and state serialization.
    """

    def __init__(self):
        super().__init__()
        self.estimators_: List[Any] = []
        self.n_features_in_: Optional[int] = None

    def __len__(self) -> int:
        """Returns the number of estimators in the ensemble."""
        return len(self.estimators_)

    def __iter__(self):
        """Iterates over the underlying estimators."""
        return iter(self.estimators_)

    def __getitem__(self, index: int) -> Any:
        """Accesses an estimator by index."""
        return self.estimators_[index]
