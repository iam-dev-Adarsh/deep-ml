import numpy as np

def cholesky_decomposition(A):
    """
    Perform Cholesky decomposition on a symmetric positive-definite matrix.
    
    Args:
        A: A symmetric positive-definite matrix (2D list or numpy array)
    
    Returns:
        L: Lower triangular matrix such that A = L @ L.T as a 2D list,
           or -1 if decomposition is not possible
    """
    # Your code here
    try:
        M = np.asarray(A, dtype=float)
    except (TypeError, ValueError):
        return -1

    if M.ndim != 2 or M.shape[0] == 0 or M.shape[0] != M.shape[1]:
        return -1

    if not np.allclose(M, M.T, atol=1e-8):
        return -1

    n = M.shape[0]
    L = np.zeros((n, n), dtype=float)

    for i in range(n):
        for j in range(i + 1):
            dot_sum = np.dot(L[i, :j], L[j, :j])
            if i == j:
                val = M[i, i] - dot_sum
                if val <= 0.0:
                    return -1
                L[i, j] = np.sqrt(val)
            else:
                if np.isclose(L[j, j], 0.0):
                    return -1
                L[i, j] = (M[i, j] - dot_sum) / L[j, j]

    return L.tolist()
    