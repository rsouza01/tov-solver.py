import matplotlib.pyplot as plt
from tov_solver.domains.eos.hadronic.swrdp_loader import SWRDPLoader

def test_raw_data_plot():
    # Load your real thesis data
    hadronic = SWRDPLoader("data/swrdp/eos_swrdp_l0040_L97a32.dat")
    
    # Plot the raw points (not the splines)
    plt.figure()
    plt.scatter(hadronic.mu_B, hadronic.pressure, label="Raw Thesis Data")
    plt.savefig("outputs/plots/diagnostic_raw_data.png")