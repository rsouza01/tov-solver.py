import pytest
import os

from tov_solver.domains.eos.mftqcd.mftqcd import MFTQCD
from tov_solver.domains.eos.hadronic.swrdp_loader import SWRDPLoader

@pytest.fixture
def loader():
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'swrdp', 'eos_swrdp_l0040_L97a32.dat')
    # Return your Hadronic loader pointing to your data
    return SWRDPLoader(data_path)

@pytest.fixture
def model():
    # Try lowering B_bag. If you were at 75.7, try 50.0 or 40.0.
    # This will shift the Quark pressure curve 'upward'.
    return MFTQCD(m_g=1.0, g_coupling=1.8199247, B_bag=5.0)    
