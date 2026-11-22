import numpy as np


def min_norm_solution(A, b):
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)
    # Solve (A @ A.T) @ w = b, then x_hat = A.T @ w
    w = np.linalg.solve(A @ A.T, b)
    return A.T @ w


def min_norm_via_kkt(A, b):
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)
    m, n = A.shape

    # From L(x, lam) = x.T @ x + lam.T @ (A @ x - b):
    # dL/dx = 2x + A.T @ lam = 0  ->  2x + A.T @ lam = 0
    # dL/dlam = A @ x - b = 0     ->  A @ x = b
    top = np.hstack([2.0 * np.eye(n), A.T])
    bottom = np.hstack([A, np.zeros((m, m))])
    KKT = np.vstack([top, bottom])
    rhs = np.concatenate([np.zeros(n), b])

    sol = np.linalg.solve(KKT, rhs)
    x = sol[:n]
    lam = sol[n:]
    return x, lam


def min_norm_from_particular(A, x_p):
    A = np.asarray(A, dtype=float)
    x_p = np.asarray(x_p, dtype=float)
    # Project x_p onto row(A), removing its null(A) part: A.T @ (A @ A.T)^-1 @ A @ x_p
    w = np.linalg.solve(A @ A.T, A @ x_p)
    return A.T @ w