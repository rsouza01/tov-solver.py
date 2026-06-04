import numpy as np

from tov_solver.domains.eos.mftqcd.mftqcd import MFTQCD


def test_legacy_quark_pressure():
    # 1. Initialize the class with your 2016 parameters
    # (Insert the exact g, m_g, and B_bag used to generate the table)
    model = MFTQCD(m_g=..., g_coupling=..., B_bag=...)
    
    # 2. Pick one row from your tabulated data
    # (e.g., density = X, k_u = ..., k_d = ..., k_s = ..., k_e = ...)
    k_u_test = ... 
    k_d_test = ...
    k_s_test = ...
    k_e_test = ...
    rho_B_test = ...
    
    expected_pressure = ... # The pressure value from your table
    
    # 3. Calculate using the new Python class
    calculated_pressure = model.total_pressure(k_u_test, k_d_test, k_s_test, k_e_test, rho_B_test)
    
    # 4. Assert they match within an acceptable tolerance
    assert np.isclose(calculated_pressure, expected_pressure, rtol=1e-5), \
        f"Mismatch! Expected: {expected_pressure}, Got: {calculated_pressure}"