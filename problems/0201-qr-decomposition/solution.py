import numpy as np


def qr_decomposition(
    A: list[list[float]],
) -> tuple[list[list[float]], list[list[float]]]:
    """Perform QR decomposition using the Gram-Schmidt process."""
    M = np.asarray(A, dtype=float)
    m, n = M.shape

    Q = np.zeros((m, n), dtype=float)
    R = np.zeros((n, n), dtype=float)

    for j in range(n):
        v = M[:, j].copy()

        # Remove components along previous orthonormal columns
        for i in range(j):
            R[i, j] = np.dot(Q[:, i], M[:, j])
            v = v - R[i, j] * Q[:, i]

        norm_v = np.linalg.norm(v)
        R[j, j] = norm_v
        if norm_v > 1e-12:
            Q[:, j] = v / norm_v

    return Q.tolist(), R.tolist()