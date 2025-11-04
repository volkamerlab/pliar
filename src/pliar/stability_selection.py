import numpy as np
from sklearn.linear_model import Lasso
from sklearn.utils import resample
from joblib import Parallel, delayed


def _select_variables(
    X: np.ndarray,
    y: np.ndarray,
    num_samples: int,
    regularisation: float,
    positive_only: bool,
):
    X_sub, y_sub = resample(X, y, n_samples=num_samples, random_state=None)
    lasso = Lasso(alpha=regularisation, positive=positive_only)
    lasso.fit(X_sub, y_sub)
    return (lasso.coef_ != 0).astype(np.float32)


def compute_probability_paths(
    X: np.ndarray,
    y: np.ndarray,
    regularisation_grid: np.ndarray,
    num_bootstraps: int,
    bootstrap_ratio: float,
    positive_only: bool,
):
    probability_paths = np.zeros(
        (regularisation_grid.shape[0], X.shape[1]), dtype=np.float32
    )
    bootstrap_size = int(X.shape[0] * bootstrap_ratio)
    for j_reg, regularisation in enumerate(regularisation_grid):
        selected = Parallel(n_jobs=-1)(
            delayed(_select_variables)(
                X, y, bootstrap_size, regularisation, positive_only
            )
            for _ in range(num_bootstraps)
        )
        selection_probabilities = np.mean(selected, axis=0)
        probability_paths[j_reg] = selection_probabilities
    return probability_paths


def stability_selection(
    X: np.ndarray,
    y: np.ndarray,
    regularisation_grid: np.ndarray | None = None,
    num_bootstraps: int | None = None,
    positive_only: bool = False,
    bootstrap_ratio: float = 0.8,
    threshold: float = 0.6,
):
    if regularisation_grid is None:
        regularisation_grid = np.array([0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0])
    if num_bootstraps is None:
        num_bootstraps = X.shape[0] // 2
    probability_paths = compute_probability_paths(
        X, y, regularisation_grid, num_bootstraps, bootstrap_ratio, positive_only
    )
    max_probabilities = np.max(probability_paths, axis=0)
    selected_features = np.where(max_probabilities > threshold)[0]
    return selected_features


if __name__ == "__main__":
    from sklearn.datasets import make_regression

    X, y = make_regression(n_samples=1000, n_features=100, noise=0.1, n_informative=3)
    selected = stability_selection(X, y)
    print(selected)
