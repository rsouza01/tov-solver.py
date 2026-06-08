import pandas as pd
import numpy as np
from scipy.interpolate import CubicSpline

class SWRDPLoader:
    def __init__(self, file_path):
        # Skip the first 3 columns as instructed
        # Using space-delimited parsing
        df = pd.read_csv(file_path, sep='\s+', header=None)        
        # Mapping according to your new descriptor:
        # Col 3: Energy Density (MeV/fm^3)
        # Col 4: Pressure (MeV/fm^3)
        # Col 7: Baryon Chemical Potential (MeV)
        
        self.energy_density = df.iloc[:, 3].values
        self.pressure = df.iloc[:, 4].values
        self.mu_B = df.iloc[:, 7].values
        
        # Sort by mu_B (required for CubicSpline monotonicity)
        idx = np.argsort(self.mu_B)
        self.mu_B = self.mu_B[idx]
        self.pressure = self.pressure[idx]
        self.energy_density = self.energy_density[idx]
        
        # Interpolators
        self.p_interp = CubicSpline(self.mu_B, self.pressure)
        self.eps_interp = CubicSpline(self.mu_B, self.energy_density)

    def get_eos(self, mu_B):
        return self.p_interp(mu_B), self.eps_interp(mu_B)        