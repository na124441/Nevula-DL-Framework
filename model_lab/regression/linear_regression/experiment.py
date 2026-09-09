"""
Reproducible Experiment Runner for Linear Regression.
Records hyperparameters, dataset specifications, loss trajectory, and evaluation metrics.
"""

import os
import sys

# Ensure repository root is on sys.path when script is executed directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional
import time
import numpy as np

from nevula.core.tensor import Tensor
from nevula.models.regression.linear import LinearRegression
from nevula.metrics.regression import mean_squared_error, r2_score


@dataclass
class ExperimentConfig:
    experiment_name: str = "linear_regression_benchmark"
    random_seed: int = 42
    n_train_samples: int = 250
    n_test_samples: int = 50
    in_features: int = 1
    true_weights: List[float] = field(default_factory=lambda: [3.0])
    true_bias: float = 2.0
    noise_std: float = 0.05
    epochs: int = 120
    learning_rate: float = 0.03
    batch_size: Optional[int] = None
    optimizer: str = "sgd"
    mode: str = "framework"


@dataclass
class ExperimentResult:
    config: Dict[str, Any]
    final_train_loss: float
    test_mse: float
    test_r2: float
    learned_weights: List[float]
    learned_bias: float
    duration_seconds: float
    loss_history: List[float]


def run_experiment(config: Optional[ExperimentConfig] = None) -> ExperimentResult:
    if config is None:
        config = ExperimentConfig()

    start_time = time.time()
    np.random.seed(config.random_seed)

    # 1. Dataset Generation
    X_train = np.random.uniform(-4.0, 4.0, size=(config.n_train_samples, config.in_features))
    true_w = np.array(config.true_weights, dtype=np.float64).reshape((config.in_features, 1))
    noise_train = np.random.normal(0.0, config.noise_std, size=(config.n_train_samples, 1))
    y_train = X_train @ true_w + config.true_bias + noise_train

    X_test = np.random.uniform(-4.0, 4.0, size=(config.n_test_samples, config.in_features))
    noise_test = np.random.normal(0.0, config.noise_std, size=(config.n_test_samples, 1))
    y_test = X_test @ true_w + config.true_bias + noise_test

    # 2. Model Instantiation
    model = LinearRegression(
        in_features=config.in_features,
        out_features=1,
        fit_intercept=True,
        mode=config.mode,
    )

    # 3. Training
    model.fit(
        X_train,
        y_train,
        epochs=config.epochs,
        lr=config.learning_rate,
        batch_size=config.batch_size,
        optimizer=config.optimizer,
        verbose=False,
    )

    # 4. Evaluation
    test_mse = model.evaluate(X_test, y_test, metric="mse")
    test_r2 = model.evaluate(X_test, y_test, metric="r2")

    w_learned = [float(x) for x in model.weight.data]
    b_learned = float(model.bias_param.data[0]) if model.bias_param is not None else 0.0

    duration = time.time() - start_time

    result = ExperimentResult(
        config=asdict(config),
        final_train_loss=model.loss_history[-1] if model.loss_history else 0.0,
        test_mse=test_mse,
        test_r2=test_r2,
        learned_weights=w_learned,
        learned_bias=b_learned,
        duration_seconds=duration,
        loss_history=model.loss_history,
    )

    print("=" * 60)
    print(f" Experiment: {config.experiment_name}")
    print("=" * 60)
    print(f" True Parameters:    W={config.true_weights}, b={config.true_bias}")
    print(f" Learned Parameters: W={result.learned_weights}, b={result.learned_bias:.4f}")
    print(f" Final Train MSE:    {result.final_train_loss:.6f}")
    print(f" Test Set MSE:       {result.test_mse:.6f}")
    print(f" Test Set R^2:       {result.test_r2:.6f}")
    print(f" Completed in:       {result.duration_seconds * 1000:.2f} ms")
    print("=" * 60)

    return result


if __name__ == "__main__":
    run_experiment()
