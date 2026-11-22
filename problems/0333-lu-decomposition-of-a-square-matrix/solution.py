import numpy as np


def lu_decomposition(A: list) -> tuple:
    M = np.asarray(A, dtype=float)
    n = M.shape[0]

    L = np.eye(n, dtype=float)
    U = np.zeros((n, n), dtype=float)

    for i in range(n):
        # Upper triangular entries in row i
        for j in range(i, n):
            U[i, j] = M[i, j] - np.dot(L[i, :i], U[:i, j])

        # Lower triangular entries below diagonal in column i
        for k in range(i + 1, n):
            L[k, i] = (M[k, i] - np.dot(L[k, :i], U[:i, i])) / U[i, i]

    return L, U