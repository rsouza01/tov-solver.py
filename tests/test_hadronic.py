import pytest
import os
import numpy as np
from tov_solver.domains.eos.hadronic.swrdp_loader import SWRDPLoader

data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'swrdp', 'eos_swrdp_l0040_L97a32.dat')


def test_swrdp_loader_consistency():
    loader = SWRDPLoader(data_path) # Adjust path as needed
    
    # Check that we have data
    assert len(loader.mu_B) > 0
    
    # Test for monotonicity (Physics sanity check: Pressure must increase with mu_B)
    # The derivative of Pressure with respect to chemical potential is the density (n_B)
    # So dP/dmu must be positive.
    dPdmu = np.gradient(loader.pressure, loader.mu_B)
    assert np.all(dPdmu > 0), "Pressure is not monotonically increasing with chemical potential!"

def test_interpolation_logic():
    loader = SWRDPLoader(data_path)
    # Test a point within the range
    test_mu = np.mean(loader.mu_B)
    p, eps = loader.get_eos(test_mu)
    assert p > 0, "Pressure should be positive."
    assert eps > 0, "Energy density should be positive."