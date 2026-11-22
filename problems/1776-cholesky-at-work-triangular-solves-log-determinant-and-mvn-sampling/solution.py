import numpy as np


def forward_sub(L, b):
    L = np.asarray(L, dtype=float)
    b = np.asarray(b, dtype=float)
    n = b.shape[0]
    y = np.zeros(n, dtype=float)

    for i in range(n):
        known = np.dot(L[i, :i], y[:i]) if i > 0 else 0.0
        y[i] = (b[i] - known) / L[i, i]
    return y


def back_sub(U, y):
    U = np.asarray(U, dtype=float)
    y = np.asarray(y, dtype=float)
    n = y.shape[0]
    x = np.zeros(n, dtype=float)

    for i in range(n - 1, -1, -1):
        known = np.dot(U[i, i + 1 :], x[i + 1 :]) if i + 1 < n else 0.0
        x[i] = (y[i] - known) / U[i, i]
    return x


def chol_solve(A, b):
    L = np.linalg.cholesky(np.asarray(A, dtype=float))
    y = forward_sub(L, b)
    return back_sub(L.T, y)


def logdet_chol(A):
    L = np.linalg.cholesky(np.asarray(A, dtype=float))
    return float(2.0 * np.sum(np.log(np.diag(L))))


def mvn_sample(mu, Sigma, eps):
    mu = np.asarray(mu, dtype=float)
    eps = np.asarray(eps, dtype=float)
    L = np.linalg.cholesky(np.asarray(Sigma, dtype=float))
    # Each row i is L @ eps[i] + mu -> eps @ L.T + mu
    return eps @ L.T + mu