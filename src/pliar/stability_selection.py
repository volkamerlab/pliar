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
    """Compute selection probabilities for each feature across a regularisation grid.

    For every value of regularisation in the grid, runs `num_bootstraps` parallel
    bootstrap iterations of Lasso fitting and averages the resulting binary selection
    masks to produce an empirical selection probability per feature.

    Parameters
    ----------
    X : np.ndarray of shape (n_samples, n_features)
        Feature matrix.
    y : np.ndarray of shape (n_samples,)
        Target vector.
    regularisation_grid : np.ndarray of shape (n_lambdas,)
        Sequence of Lasso alpha values to evaluate. Typically a decreasing grid
        from strong to weak regularisation.
    num_bootstraps : int
        Number of bootstrap resamples to run at each regularisation value.
    bootstrap_ratio : float
        Fraction of the training data to include in each bootstrap subsample.
        Must be in (0, 1].
    positive_only : bool
        If True, constrains Lasso coefficients to be non-negative at every grid point.

    Returns
    -------
    np.ndarray of shape (n_lambdas, n_features), dtype float32
        Matrix of selection probabilities. Entry [j, k] is the fraction of bootstrap
        runs at regularisation index j in which feature k was selected.
    """
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
    """Identify stable features via stability selection with Lasso.

    Implements the stability selection procedure of Meinshausen & Bühlmann (2010).
    Repeatedly fits Lasso models on random subsamples across a grid of regularisation
    strengths, then retains only the features whose maximum selection probability
    across the grid exceeds `threshold`. This approach controls the expected number
    of falsely selected variables while remaining largely insensitive to the exact
    choice of regularisation.

    Parameters
    ----------
    X : np.ndarray of shape (n_samples, n_features)
        Feature matrix.
    y : np.ndarray of shape (n_samples,)
        Target vector.
    regularisation_grid : np.ndarray of shape (n_lambdas,) or None, optional
        Lasso alpha values to sweep over. Defaults to
        ``[0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]`` if None.
    num_bootstraps : int or None, optional
        Number of bootstrap iterations per regularisation value. Defaults to
        ``n_samples // 2`` if None.
    positive_only : bool, optional
        If True, restricts Lasso to non-negative coefficients. Default is False.
    bootstrap_ratio : float, optional
        Proportion of samples drawn for each bootstrap subsample. Default is 0.8.
    threshold : float, optional
        Minimum selection probability required for a feature to be retained.
        Must be in (0.5, 1.0] for theoretical guarantees. Default is 0.6.

    Returns
    -------
    np.ndarray of shape (n_selected,), dtype int
        Indices of the selected features, sorted in ascending order.

    References
    ----------
    Meinshausen, N. and Bühlmann, P. (2010). Stability selection.
    Journal of the Royal Statistical Society: Series B, 72(4), 417–473.

    Examples
    --------
    >>> from sklearn.datasets import make_regression
    >>> X, y = make_regression(n_samples=1000, n_features=100,
    ...                        noise=0.1, n_informative=3)
    >>> selected = stability_selection(X, y)
    >>> print(selected)
    """
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
