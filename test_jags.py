import pytest
import numpy as np
from pangolin.ir import RV, Add, Constant, Normal
from include import Sample_prob  

@pytest.fixture
def pangolin_model():
    # Build the graph
    a = RV(Constant(3.0))
    b = RV(Constant(4.0))
    mu = RV(Add(), a, b)      
    c = RV(Constant(1.0))         
    y = RV(Normal(), mu, c)  
    z = RV(Add(), y, mu)               
    sp = Sample_prob()
    return sp, a, b, mu, y, z

def test_mu_deterministic(pangolin_model):
    sp, a, b, mu, y, z = pangolin_model
    sp.sample([mu, z], {})  
    coda = sp.read_coda()

    mu_name = 'v'+str(mu._n)
    mu_samples = np.array(coda[mu_name])

    assert np.allclose(mu_samples, 7.0, atol=1e-10), f"mu samples differ from expected 7: {mu_samples[:5]}"

def test_z_mean(pangolin_model):
    sp, a, b, mu, y, z = pangolin_model
    sp.sample([mu, z], {})
    coda = sp.read_coda()

    z_name = 'v'+str(z._n)
    z_samples = np.array(coda[z_name])

    z_mean = np.mean(z_samples)
    assert np.isclose(z_mean, 14.0, atol=0.5), f"z mean {z_mean:.3f} deviates from 14.0"

def test_z_std_from_precision(pangolin_model):
    sp, a, b, mu, y, z = pangolin_model

    sp.sample([mu, z], {})

    coda = sp.read_coda()
    z_name = 'v'+str(z._n)
    z_samples = np.array(coda[z_name])


    tau = 1.0
    expected_std = 1 / np.sqrt(tau)

    z_std = np.std(z_samples)
    assert np.isclose(z_std, expected_std, atol=0.1), \
        f"z std {z_std:.3f} deviates from expected {expected_std:.3f}"