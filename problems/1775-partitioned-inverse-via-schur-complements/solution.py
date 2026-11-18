import numpy as np

def partitioned_inverse(M, k):
    """
    Return M^-1 assembled from its four blocks via the partitioned inverse formula 
    built on the Schur complement with respect to H.
    """
    n = M.shape[0]
    
    # Split M into 4 blocks
    # E is k x k, F is k x (n-k), G is (n-k) x k, H is (n-k) x (n-k)
    E = M[:k, :k]
    F = M[:k, k:]
    G = M[k:, :k]
    H = M[k:, k:]
    
    # H_inv and Schur complement M/H = E - F @ H^-1 @ G
    H_inv = np.linalg.inv(H)
    schur_H = E - F @ H_inv @ G
    schur_H_inv = np.linalg.inv(schur_H)
    
    # Components of the block inverse
    top_left = schur_H_inv
    top_right = -schur_H_inv @ F @ H_inv
    bottom_left = -H_inv @ G @ schur_H_inv
    bottom_right = H_inv + H_inv @ G @ schur_H_inv @ F @ H_inv
    
    # Construct the full inverse matrix block by block
    top_row = np.hstack((top_left, top_right))
    bottom_row = np.hstack((bottom_left, bottom_right))
    return np.vstack((top_row, bottom_row))


def block_slogdet(M, k):
    """
    Return (sign, logabsdet) of det(M) using det(M) = det(M/H) * det(H).
    """
    n = M.shape[0]
    E = M[:k, :k]
    F = M[:k, k:]
    G = M[k:, :k]
    H = M[k:, k:]
    
    H_inv = np.linalg.inv(H)
    schur_H = E - F @ H_inv @ G
    
    # Compute sign and log-absolute-value of determinants for both H and M/H
    sign_H, log_H = np.linalg.slogdet(H)
    sign_schur, log_schur = np.linalg.slogdet(schur_H)
    
    # Combine via product of signs and sum of logs
    total_sign = sign_H * sign_schur
    total_logabsdet = log_H + log_schur
    
    return (float(total_sign), float(total_logabsdet))


def bordered_inverse(A_inv, b, c):
    """
    Given A^-1, vector b, and scalar c, return the inverse of the (n+1)x(n+1) 
    bordered matrix in O(n^2) time using the Schur complement s = c - b^T A^-1 b.
    """
    # s = c - b^T @ A^-1 @ b (scalar Schur complement)
    # Note: b is shape (n,) or (n, 1), handle safely with matrix multiplication
    b_col = np.atleast_2d(b).T
    b_row = np.atleast_2d(b)
    
    s = c - (b_row @ A_inv @ b_col).item()
    
    # Raise ValueError if s is too close to zero
    if abs(s) < 1e-12:
        raise ValueError("Schur complement s is singular or nearly singular.")
    
    s_inv = 1.0 / s
    
    # Block inverse components for bordered matrix [A, b; b^T, c]
    # Here E = A, F = b, G = b^T, H = [c]
    top_left = A_inv + s_inv * (A_inv @ b_col) @ (b_row @ A_inv)
    top_right = -s_inv * (A_inv @ b_col)
    bottom_left = -s_inv * (b_row @ A_inv)
    bottom_right = np.array([[s_inv]])
    
    top_row = np.hstack((top_left, top_right))
    bottom_row = np.hstack((bottom_left, bottom_right))
    return np.vstack((top_row, bottom_row))