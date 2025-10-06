import numpy as np
from sklearn.linear_model import Ridge
import warnings

def STLSQ(X: np.ndarray,
          Y: np.ndarray,
          alpha: float = 1e-6,
          threshold: float = 0.1,
          max_iter: int = 20,
          tol: float = 1e-8,
          handle_nans: bool = True) -> np.ndarray:
    
    N, P = X.shape
    d = Y.shape[1]
    coef = np.zeros((P, d))

    # Initial fit (multi-output Ridge if no NaNs)
    if not handle_nans or not np.isnan(Y).any():
        reg = Ridge(alpha=alpha, fit_intercept=False)
        coef = reg.fit(X, Y).coef_.T
    else:
        for j in range(d):
            valid_idx = ~np.isnan(Y[:, j])
            if np.any(valid_idx):
                reg = Ridge(alpha=alpha, fit_intercept=False)
                coef[:, j] = reg.fit(X[valid_idx], Y[valid_idx, j]).coef_

    # Iterative thresholding
    for _ in range(max_iter):
        coef_old = coef.copy()
        small = np.abs(coef) < threshold
        coef[small] = 0.0

        for j in range(d):
            valid_idx = ~np.isnan(Y[:, j]) if handle_nans else np.ones(N, dtype=bool)
            big_idx = ~small[:, j]
            if np.any(valid_idx) and np.any(big_idx):
                reg = Ridge(alpha=alpha, fit_intercept=False)
                coef[big_idx, j] = reg.fit(X[valid_idx][:, big_idx], Y[valid_idx, j]).coef_

        # Convergence check
        num = np.linalg.norm(coef - coef_old, ord="fro")
        den = np.linalg.norm(coef_old, ord="fro") + 1e-12
        if num / den < tol:
            break

    # Warn if everything zero
    if np.all(coef == 0):
        warnings.warn("All coefficients were pruned to zero.")

    return coef

def STLSQ_weak(X: np.ndarray,
          Y: np.ndarray,
          alpha: float = 1e-6,
          threshold: float = 0.1,
          max_iter: int = 20,
          tol: float = 1e-8,
          handle_nans: bool = True) -> np.ndarray:
    
    N, P = X.shape
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)
    d = Y.shape[1]
    coef = np.zeros((P, d))

    # Initial fit
    if not handle_nans or not np.isnan(Y).any():
        reg = Ridge(alpha=alpha, fit_intercept=False)
        coef_fit = reg.fit(X, Y).coef_   # (d, P) or (P,)
        if coef_fit.ndim == 1:
            coef = coef_fit.reshape(P, 1)
        else:
            coef = coef_fit.T            # (P, d)
    else:
        for j in range(d):
            valid_idx = ~np.isnan(Y[:, j])
            if np.any(valid_idx):
                reg = Ridge(alpha=alpha, fit_intercept=False)
                coef[:, j] = reg.fit(X[valid_idx], Y[valid_idx, j]).coef_

    # Iterative thresholding
    for _ in range(max_iter):
        coef_old = coef.copy()
        small = np.abs(coef) < threshold
        coef[small] = 0.0

        for j in range(d):
            valid_idx = ~np.isnan(Y[:, j]) if handle_nans else np.ones(N, dtype=bool)
            big_idx = ~small[:, j]
            if np.any(valid_idx) and np.any(big_idx):
                reg = Ridge(alpha=alpha, fit_intercept=False)
                coef[big_idx, j] = reg.fit(X[valid_idx][:, big_idx], Y[valid_idx, j]).coef_

        # Convergence check
        num = np.linalg.norm(coef - coef_old, ord="fro")
        den = np.linalg.norm(coef_old, ord="fro") + 1e-12
        if num / den < tol:
            break

    if np.all(coef == 0):
        warnings.warn("All coefficients were pruned to zero.")

    return coef