import numpy as np
import pandas as pd
from tov_solver.domains.eos.hadronic.swrdp_loader import SWRDPLoader

def swrdp(args):
	hadronic = SWRDPLoader(args.path)
	mu_grid = hadronic.mu_B
	eos_model = hadronic
	save_spline_to_csv(eos_model, mu_grid, args.output, args.precision)


def save_spline_to_csv(eos_model, mu_grid, filename=None, precision="%.3f"):
    p_values = eos_model.p_interp(mu_grid)
    eps_values = eos_model.eps_interp(mu_grid)
    
    df = pd.DataFrame({
        'mu_B': mu_grid,
        'pressure': p_values,
        'energy_density': eps_values
    })
    
    if filename:
        df.to_csv(filename, index=False, float_format=precision)
        print(f"EoS exported to {filename} with precision {precision}")
    else:
        print(df.to_csv(index=False, sep=",", float_format=precision))