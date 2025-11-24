import pytest
import numpy as np
from pangolin.ir import *
from include import Sample_prob  

rng = np.random.default_rng()

def test_linear_alg():
    A_np = rng.normal(size=(4, 3))         # 4×3
    B_np = rng.normal(size=(3, 4))         # 3×4
    M_np = rng.normal(size=(4, 4))
    M_np = M_np @ M_np.T + np.eye(4) * 1e-6   # PD 4×4 for invertibility

    # Expected results using NumPy
    C_np = A_np @ B_np                      # 4×4
    M_inv_np = np.linalg.inv(M_np)          # 4×4
    Z_np = C_np @ M_inv_np                  # 4×4
    S_np = np.sum(Z_np, axis=1)             # 4-vector

    # Wrap in RVs
    A = RV(Constant(A_np))
    B = RV(Constant(B_np))
    M = RV(Constant(M_np))

    # Define graph:
    # C = A @ B
    # M_inv = inv(M)
    # Z = (A @ B) @ inv(M)
    # S = sum(Z, axis=1)
    C = RV(Matmul(), A, B)
    M_inv = RV(Inv(), M)
    Z = RV(Matmul(), C, M_inv)
    S = RV(Sum(axis=1), Z)

    # Evaluate
    sp = Sample_prob()
    sp.sample([S], {})
    coda = sp.read_coda()

    S_val = coda['v' + str(S._n)]

    # Tests
    assert S_val.shape == (4,)
    assert np.allclose(S_val, S_np, atol=1e-6)

    # For additional safety, check intermediate nodes too
    sp = Sample_prob()
    sp.sample([C, M_inv, Z, S], {})
    coda = sp.read_coda()

    C_val = coda['v' + str(C._n)]
    M_inv_val = coda['v' + str(M_inv._n)]
    Z_val = coda['v' + str(Z._n)]
    S_val = coda['v' + str(S._n)]

    assert np.allclose(C_val, C_np, atol=1e-6)
    assert np.allclose(M_inv_val, M_inv_np, atol=1e-6)
    assert np.allclose(Z_val, Z_np, atol=1e-6)
    assert np.allclose(S_val, S_np, atol=1e-6)