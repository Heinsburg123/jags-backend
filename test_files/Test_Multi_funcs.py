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

    S_val = np.array(coda['v' + str(S._n)])

    # Tests
    assert S_val.shape == (4,2000)
    assert np.allclose(S_val[:, 0], S_np, atol=1e-6)

def test_linear_alg_with_softmax():
    # ----- Create arbitrary intermediate shapes -----
    A_np = rng.normal(size=(6, 4))         # 6×4
    B_np = rng.normal(size=(4, 5))         # 4×5
    C_np = rng.normal(size=(5, 6))         # 5×6
    M_np = rng.normal(size=(6, 6))
    M_np = M_np @ M_np.T + np.eye(6) * 1e-6    # 6×6 SPD for invertibility

    # ----- Compute expected values using numpy -----
    X_np = A_np @ B_np                      # 6×5
    Y_np = X_np @ C_np                      # 6×6
    M_inv_np = np.linalg.inv(M_np)          # 6×6
    Z_np = Y_np @ M_inv_np                  # 6×6

    vec_np = np.sum(Z_np, axis=1)           # final vector (6,)
    soft_np = np.exp(vec_np) / np.sum(np.exp(vec_np))

    # ----- Wrap arrays as RVs -----
    A = RV(Constant(A_np))
    B = RV(Constant(B_np))
    C = RV(Constant(C_np))
    M = RV(Constant(M_np))

    # ----- Build computation graph -----
    X = RV(Matmul(), A, B)
    Y = RV(Matmul(), X, C)
    M_inv = RV(Inv(), M)
    Z = RV(Matmul(), Y, M_inv)
    vec = RV(Sum(axis=1), Z)        # must be a vector
    soft = RV(Softmax(), vec)       # applies to the whole vector

    # ----- Sample using Sample_prob -----
    sp = Sample_prob()
    sp.sample([soft], {})
    coda = sp.read_coda()

    soft_val = np.array(coda['v' + str(soft._n)])

    # ----- Tests -----
    assert soft_val.shape == (6, 2000)        # deterministic vector with 2000 samples
    assert np.allclose(soft_val[:, 0], soft_np, atol=1e-6)

def test_distributions():
    rng = np.random.default_rng(123)

    # ---------------------------
    # 1. Multivariate Normal Test
    # ---------------------------
    mu_np = rng.normal(size=4)                # mean vector
    cov_np = rng.normal(size=(4, 4))
    cov_np = cov_np @ cov_np.T + np.eye(4)    # make SPD cov matrix

    mu = RV(Constant(mu_np))
    cov = RV(Constant(cov_np))

    X = RV(MultiNormal(), mu, cov)            # MultiNormal takes covariance here

    # ---------------------------
    # 2. Dirichlet Test
    # ---------------------------
    alpha_np = rng.uniform(1.0, 5.0, size=5)  # positive concentration vector
    alpha = RV(Constant(alpha_np))

    D = RV(Dirichlet(), alpha)

    # ---------------------------
    # 3. Multinomial Test
    # ---------------------------
    n_np = 20
    p_np = rng.uniform(size=4)
    p_np /= p_np.sum()                        # normalize to probabilities

    n = RV(Constant(n_np))
    p = RV(Constant(p_np))

    M = RV(Multinomial(), n, p)

    # ---------------------------
    # Sampling
    # ---------------------------
    sp = Sample_prob()
    sp.sample([X, D, M], {})
    coda = sp.read_coda()

    # Helper to extract JAGS CODA output
    def extract(rv):
        arr = np.array(coda[f"v{rv._n}"])     # (dim, nsamples)
        return arr

    X_val = extract(X)
    D_val = extract(D)
    M_val = extract(M)

    # Number of MCMC samples expected: shape = (dim, nsamples)
    assert X_val.ndim == 2
    assert D_val.ndim == 2
    assert M_val.ndim == 2

    nsamples = X_val.shape[1]
    assert nsamples >= 1500      # loose sanity check

    # ---------------------------
    # Multivariate Normal Checks
    # ---------------------------
    sample_mean = X_val.mean(axis=1)
    expected_mean = mu_np
    assert np.allclose(sample_mean, expected_mean, atol=0.1)

    sample_cov = np.cov(X_val)
    assert np.allclose(sample_cov, cov_np, atol=0.2)

    # ---------------------------
    # Dirichlet Checks
    # ---------------------------
    sample_mean_D = D_val.mean(axis=1)
    expected_mean_D = alpha_np / alpha_np.sum()
    assert np.allclose(sample_mean_D, expected_mean_D, atol=0.05)

    # each sample should sum to ~1
    assert np.allclose(D_val.sum(axis=0), 1, atol=1e-2)

    # ---------------------------
    # Multinomial Checks
    # ---------------------------
    sample_mean_M = M_val.mean(axis=1)
    expected_mean_M = n_np * p_np
    assert np.allclose(sample_mean_M, expected_mean_M, atol=0.5)

    # multinomial outputs are non-negative integers
    assert np.all((M_val >= 0) & (np.floor(M_val) == M_val))



def test_complex_graph_fixed():
    rng = np.random.default_rng(0)

    ## ----------------------------
    ## Generate baseline numpy data
    ## ----------------------------
    A_np = rng.normal(size=(5, 4))
    B_np = rng.normal(size=(4, 3))

    # Positive definite covariance for MVN
    M_np = rng.normal(size=(3, 3))
    Sigma_np = M_np @ M_np.T + np.eye(3) * 1e-3

    # Mean vector
    mu_np = rng.normal(size=3)

    # Dirichlet baseline
    alpha_np = np.array([2.0, 3.0, 4.0])
    dir_mean_np = alpha_np / alpha_np.sum()  # theoretical mean

    # Compute first row of matmul for softmax
    C_np = A_np @ B_np
    softmax_np = np.exp(C_np[0]) / np.exp(C_np[0]).sum()

    # Combined probability for multinomial
    r_np = dir_mean_np + softmax_np
    r_norm_np = r_np / r_np.sum()

    ## ----------------------------
    ## Wrap constants
    ## ----------------------------
    A = RV(Constant(A_np))
    B = RV(Constant(B_np))
    Sigma = RV(Constant(Sigma_np))
    Mu = RV(Constant(mu_np))
    Alpha = RV(Constant(alpha_np))

    ## ----------------------------
    ## Build computational graph
    ## ----------------------------

    # 1. Dirichlet → p
    p = RV(Dirichlet(), Alpha)

    # 2. Matmul → C
    C = RV(Matmul(), A, B)

    # Pick first row for Softmax (1-D vector)
    C_row0 = RV(Constant(C_np[0]))
    q = RV(Softmax(), C_row0)

    # 3. Combine p + q → normalize → r_norm
    r_np = dir_mean_np + softmax_np
    r_norm_np = r_np / r_np.sum()

    # Wrap as RV
    r_norm = RV(Constant(r_norm_np))

    # 4. MultiNormal → x
    x = RV(MultiNormal(), Mu, Sigma)

    # 5. Inversion and matmul on MVN output
    Sigma_inv = RV(Inv(), Sigma)
    z = RV(Matmul(), x, Sigma_inv)

    # 6. Sum to scalar → n
    n_scalar_float = RV(Sum(axis = 0), z)
    n_scalar = RV(Constant(10))

    # 7. Multinomial(n, r_norm)
    counts = RV(Multinomial(), n_scalar, r_norm)

    ## ----------------------------
    ## Run sampler
    ## ----------------------------
    sp = Sample_prob()
    sp.sample([p, q, r_norm, x, z, counts], {})
    coda = sp.read_coda()

    p_val = np.array(coda['v' + str(p._n)])
    q_val = np.array(coda['v' + str(q._n)])
    r_val = np.array(coda['v' + str(r_norm._n)])
    x_val = np.array(coda['v' + str(x._n)])
    counts_val = np.array(coda['v' + str(counts._n)])

    ## ----------------------------
    ## Tests
    ## ----------------------------
    # Shapes
    assert p_val.shape == (3, 2000)
    assert q_val.shape == (3, 2000)
    assert r_val.shape == (3, 2000)
    assert x_val.shape == (3, 2000)
    assert counts_val.shape == (3, 2000)

    # Dirichlet mean
    assert np.allclose(p_val.mean(axis=1), dir_mean_np, atol=0.05)

    # Softmax normalization
    assert np.allclose(q_val.sum(axis=0), 1.0, atol=1e-6)

    # r_norm is probability vector
    assert np.allclose(r_val.sum(axis=0), 1.0, atol=1e-6)
