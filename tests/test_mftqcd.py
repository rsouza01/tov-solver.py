import numpy as np
from tov_solver.domains.eos.mftqcd.mftqcd import MFTQCD

def test_legacy_quark_pressure_row_1():
    # 1. Initialize with the correct Bag Constant (75.7)
    model = MFTQCD(m_g=1.0, g_coupling=1.8199247472859184, B_bag=75.7) 
    
    # 2. Extract values from the first row of MFTQCD_B_75_7MeV.csv
    k_u_test = 2.007700661482983   # 'x' column
    k_d_test = 2.077054135957307   # 'y' column
    k_s_test = 1.9331657117152636  # 'z' column
    k_e_test = 0.06952521078724577 # 'w' column
    rho_test = 4.5                 # 'rho' column 
    
    # Updated expectations mapped to the correct columns
    expected_fermi_pressure = 235.86684374042994 # 'fermip' (Pure Kinetic)
    expected_mit_pressure = 160.16684374042993   # 'pressaomit' (Kinetic - B)
    expected_pressure = 273.3484023281014        # 'pressao' (Total)
    
    # 3. Calculate using the new Python class
    pq, pe = model.kinetic_pressure(k_u_test, k_d_test, k_s_test, k_e_test)
    calculated_kinetic = pq + pe
    
    # --- ASSERTIONS ---
    
    # A. Test Pure Kinetic Pressure (fermip)
    assert np.isclose(calculated_kinetic, expected_fermi_pressure, rtol=1e-5), \
        f"Fermi mismatch! Expected: {expected_fermi_pressure}, Got: {calculated_kinetic}"
        
    # B. Test MIT Pressure (fermip - B)
    calculated_mit = calculated_kinetic - model.B_bag
    assert np.isclose(calculated_mit, expected_mit_pressure, rtol=1e-5), \
        f"MIT mismatch! Expected: {expected_mit_pressure}, Got: {calculated_mit}"
        
    # C. Test Final Total Pressure (with Hard Gluons)
    calculated_total = model.total_pressure(k_u_test, k_d_test, k_s_test, k_e_test, rho_test)
    assert np.isclose(calculated_total, expected_pressure, rtol=1e-5), \
        f"Total mismatch! Expected: {expected_pressure}, Got: {calculated_total}"

def test_beta_equilibrium():
    model = MFTQCD(m_g=1.0, g_coupling=1.8199247, B_bag=75.7) 
    
    # The true baryon density from the 'nB' column
    # The CSV density divided by the legacy scale factor of 9.0 to get true fm^-3
    nB_test = 7.379622924835158 / 9.0    
    
    # Ask the new solver to find the momenta
    k_u, k_d, k_s, k_e = model.solve_beta_equilibrium(nB_test)
    
    # Assert they match your old Mathematica output
    assert np.isclose(k_u, 2.00770066, rtol=1e-4), f"k_u failed: {k_u}"
    assert np.isclose(k_d, 2.07705413, rtol=1e-4), f"k_d failed: {k_d}"
    assert np.isclose(k_s, 1.93316571, rtol=1e-4), f"k_s failed: {k_s}"
    
    # Electrons are small, let's allow a slightly larger relative tolerance
    assert np.isclose(k_e, 0.06952521, rtol=1e-3), f"k_e failed: {k_e}"