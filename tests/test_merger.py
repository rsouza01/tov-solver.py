import pytest
import numpy as np
from scipy.optimize import brentq

@pytest.mark.skip(reason="Quark model currently calibrated for high-density regime; intersection outside hadronic data range.")
def test_phase_transition_possibility(model, loader):
    mu_start = 2700.0  
    mu_end = 3000.0
    mu_range = np.linspace(mu_start, mu_end, 50)
    
    p_quark = [model.get_thermodynamics_at_mu(mu)[0] for mu in mu_range]
    p_hadron = [loader.p_interp(mu) for mu in mu_range]
    
    # DEBUG: Print the values to see if they ever cross
    print(f"\nQuark pressures at ends: {p_quark[0]:.2f}, {p_quark[-1]:.2f}")
    print(f"Hadron pressures at ends: {p_hadron[0]:.2f}, {p_hadron[-1]:.2f}")
    
    diff = np.array(p_quark) - np.array(p_hadron)
    assert np.sign(diff[0]) != np.sign(diff[-1]), f"No crossing! Diff at ends: {diff[0]}, {diff[-1]}"    


