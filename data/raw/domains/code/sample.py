import numpy as np

def conjugate_gradient(A, b, tol=1e-8, max_steps=1000):
    x = np.zeros_like(b)
    r = b - A @ x
    p = r.copy()
    rs_old = r @ r
    for _ in range(max_steps):
        Ap = A @ p
        alpha = rs_old / (p @ Ap)
        x += alpha * p
        r -= alpha * Ap
        rs_new = r @ r
        if np.sqrt(rs_new) < tol:
            break
        p = r + (rs_new / rs_old) * p
        rs_old = rs_new
    return x
