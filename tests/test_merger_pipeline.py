import os
import pytest
import numpy as np
from scipy.interpolate import CubicSpline
from tov_solver.domains.eos.merger import run_merger_pipeline

# You'll need to mock your loaders for the test to be fast
class MockEOS:
    def __init__(self):
        # Create dummy data for testing
        mu = np.linspace(100, 500, 100)
        # Create actual CubicSplines instead of lambdas
        self.p_interp = CubicSpline(mu, mu**2 * 0.01)
        self.eps_interp = CubicSpline(mu, mu * 0.5)
        self.mu_B = mu


def test_merger_pipeline_generates_outputs():
    # Setup
    hadronic = MockEOS()
    quark = MockEOS()
    
    # Run the pipeline
    p_merged, eps_merged = run_merger_pipeline(hadronic, quark)
    
    # Verify outputs exist
    assert os.path.exists("outputs/data/merged_eos.csv")
    assert os.path.exists("outputs/plots/final_eos_comparison.png")
    assert os.path.exists("outputs/plots/cs2_comparison.png")    

    # Verify data properties
    # p_merged contains hadronic part (up to idx) + quark part (from idx to end)
    # Since both are 500, the result should be 1000
    assert len(p_merged) == 1000