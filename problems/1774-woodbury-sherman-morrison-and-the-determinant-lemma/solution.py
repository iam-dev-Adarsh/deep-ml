import numpy as np


def woodbury_solve(sigma_diag, X, B):
    sigma_diag = np.asarray(sigma_diag, dtype=float)
    X = np.asarray(X, dtype=float)
    B = np.asarray(B, dtype=float)

    # Sigma^-1 * X and Sigma^-1 * B via broadcasting (avoids N x N matrix)
    inv_sigma_X = X / sigma_diag[:, None]
    inv_sigma_B = B / sigma_diag[:, None] if B.ndim == 2 else B / sigma_diag

    # Inner D x D system: (I_D + X^T Sigma^-1 X)
    D = X.shape[1]
    inner = np.eye(D) + X.T @ inv_sigma_X

    # Solve inner * Z = X^T Sigma^-1 B
    rhs = X.T @ inv_sigma_B
    Z = np.linalg.solve(inner, rhs)

    # Sigma^-1 B - Sigma^-1 X Z
    return inv_sigma_B - inv_sigma_X @ Z


def sherman_morrison(A_inv, u, v):
    A_inv = np.asarray(A_inv, dtype=float)
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)

    A_inv_u = A_inv @ u
    denom = 1.0 + v @ A_inv_u

    if np.isclose(denom, 0.0):
        raise ValueError("Rank-1 update produces a singular matrix.")

    v_A_inv = v @ A_inv
    return A_inv - np.outer(A_inv_u, v_A_inv) / denom


def logdet_rank_one_update(logdet_A, A_inv, u, v):
    A_inv = np.asarray(A_inv, dtype=float)
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)

    factor = 1.0 + v @ (A_inv @ u)
    if factor <= 0.0:
        raise ValueError("Updated determinant is not positive.")

    return float(logdet_A + np.log(factor))