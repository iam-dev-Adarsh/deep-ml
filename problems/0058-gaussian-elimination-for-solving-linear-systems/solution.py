import numpy as np


def gaussian_elimination(A, b):
    A = np.asarray(A, dtype=float).copy()
    b = np.asarray(b, dtype=float).copy()
    n = len(b)

    # Forward elimination with partial pivoting
    for col in range(n):
        pivot = col + np.argmax(np.abs(A[col:, col]))
        if pivot != col:
            A[[col, pivot]] = A[[pivot, col]]
            b[[col, pivot]] = b[[pivot, col]]

        for row in range(col + 1, n):
            factor = A[row, col] / A[col, col]
            A[row, col:] -= factor * A[col, col:]
            b[row] -= factor * b[col]

    # Backward substitution
    x = np.zeros(n, dtype=float)
    for row in range(n - 1, -1, -1):
        x[row] = (b[row] - np.dot(A[row, row + 1 :], x[row + 1 :])) / A[row, row]

    return x