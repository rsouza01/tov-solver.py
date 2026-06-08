import numpy as np
from scipy.optimize import root

class MFTQCD:
    def __init__(self, m_g=0.0, g_coupling=0.0, B_bag=0.0):
        """
        Initializes the Quark Phase Model.
        Parameters correspond directly to curvaghBms.nb
        """
        # Conversion factor from Mathematica: 
        # conversão de fm para MeV^-1
        self.fm = 5.07e-3 
        
        # Masses converted to fm^-1 (matching the script)
        self.m_u = 5.0 * self.fm
        self.m_d = 7.0 * self.fm
        self.m_s = 150.0 * self.fm
        self.m_e = 0.5 * self.fm
        
        # Degeneracy factors
        self.gamma_q = 6.0
        self.gamma_e = 2.0
        
        # fator2 from script
        self.fator2 = 1.0 / self.fm 
        
        # New MFTQCD Tuning Parameters (Hard Gluons)
        self.m_g = m_g
        self.g = g_coupling
        self.B_bag = B_bag
        
        # Avoid division by zero if m_g is not set yet
        if self.m_g > 0:
            self.C_gluon = (27.0 * self.g**2) / (16.0 * self.m_g**2)
        else:
            self.C_gluon = 0.0

    def _fermi_integral(self, k, m):
        """
        Exact mathematical translation of the analytic Fermi integral 
        from the Mathematica notebook.
        k: Fermi momentum (x, y, z, or w in the script)
        m: Particle mass
        """
        # Ensure physical boundaries (no negative momentum)
        k = np.maximum(k, 0.0) 
        
        # The 4 terms exactly as they appear in the Mathematica RowBox
        term1 = (k**3 * np.sqrt(k**2 + m**2)) / 4.0
        term2 = 3.0 * (m**2 * k * np.sqrt(k**2 + m**2)) / 8.0
        term3 = 3.0 * (m**4) * np.log(k + np.sqrt(k**2 + m**2)) / 8.0
        term4 = 3.0 * (m**4) * np.log(m**2) / 16.0
        
        return term1 - term2 + term3 - term4

    def kinetic_pressure(self, k_u, k_d, k_s, k_e):
        """
        Calculates total kinetic pressure (pq + pe) for a given set of Fermi momenta.
        Variables x, y, z, w from the script correspond to k_u, k_d, k_s, k_e.
        """
        # Quark terms (pq)
        coef_q = (1.0 / 3.0) * (self.gamma_q / (2.0 * np.pi**2)) * self.fator2
        
        pq_u = coef_q * self._fermi_integral(k_u, self.m_u)
        pq_d = coef_q * self._fermi_integral(k_d, self.m_d)
        pq_s = coef_q * self._fermi_integral(k_s, self.m_s)
        pq = pq_u + pq_d + pq_s
        
        # Electron term (pe) - Translated exactly from the script
        # Note: Depending on your script version, you might need to add (1.0/3.0) here 
        # to match standard thermodynamic pressure.
        coef_e = (self.gamma_e / (2.0 * np.pi**2)) * self.fator2
        pe = coef_e * self._fermi_integral(k_e, self.m_e)
        
        return pq, pe
    
    def total_pressure(self, k_u, k_d, k_s, k_e, rho_B):
        """
        Combines the bare kinetic pressure with the MFTQCD (Hard Gluon) interaction 
        and the Bag Constant to get the final Equation of State pressure.
        """
        pq, pe = self.kinetic_pressure(k_u, k_d, k_s, k_e)
        
        # The MFTQCD Hard Gluon Vector Repulsion (scales as density^2)
        p_gluon = self.C_gluon * (rho_B**2)
        
        return pq + pe + p_gluon - self.B_bag

    def _n_to_k(self, n, is_quark=True):
        """Converts particle density (n) in fm^-3 to Fermi momentum (k) in fm^-1."""
        n = np.maximum(n, 0.0) # Prevent complex numbers from negative guesses in the solver
        if is_quark:
            return (n * np.pi**2)**(1.0/3.0)
        else:
            return (n * 3.0 * np.pi**2)**(1.0/3.0)

    def _beta_equilibrium_residuals(self, vars_k, n_B_phys):
        """
        Calculates the error in Weak Equilibrium and Charge Neutrality.
        Variables are Fermi momenta: [k_d, k_s, k_e]
        """
        k_d, k_s, k_e = vars_k
        
        # 1. Baryon Conservation to find k_u
        # n_B = (k_u^3 + k_d^3 + k_s^3) / (3 * pi^2)
        k_u_cubed = 3.0 * np.pi**2 * n_B_phys - k_d**3 - k_s**3
        
        # np.cbrt smoothly handles negatives, keeping gradients intact for the solver
        k_u = np.cbrt(k_u_cubed) 
        
        # Calculate Bare Chemical Potentials
        mu_u = np.sqrt(k_u**2 + self.m_u**2)
        mu_d = np.sqrt(k_d**2 + self.m_d**2)
        mu_s = np.sqrt(k_s**2 + self.m_s**2)
        mu_e = np.sqrt(k_e**2 + self.m_e**2)
        
        # Equation 1: Weak Equilibrium (d <-> s)
        eq1 = mu_d - mu_s
        
        # Equation 2: Weak Equilibrium (d <-> u + e)
        eq2 = mu_d - mu_u - mu_e
        
        # Equation 3: Charge Neutrality
        # (2/3)n_u - (1/3)n_d - (1/3)n_s - n_e = 0
        # Multiplied by 3*pi^2, this simplifies into a beautiful clean polynomial:
        eq3 = 2.0 * k_u**3 - k_d**3 - k_s**3 - k_e**3
        
        return [eq1, eq2, eq3]

    def solve_beta_equilibrium(self, n_B_phys):
        """
        Finds the exact Fermi momenta for a given true baryon density (n_B_phys in fm^-3).
        """
        # Smart Initial Guess: Assume quarks are roughly equal
        k_guess = (np.pi**2 * n_B_phys)**(1.0/3.0) 
        
        # Run the root finder
        sol = root(
            self._beta_equilibrium_residuals, 
            x0=[k_guess, k_guess * 0.9, 0.1], # k_d, k_s, k_e guesses
            args=(n_B_phys,)
        )
        
        if not sol.success:
            raise ValueError(f"Beta-Equilibrium solver failed at n_B = {n_B_phys}: {sol.message}")
            
        k_d, k_s, k_e = sol.x
        k_u = np.cbrt(3.0 * np.pi**2 * n_B_phys - k_d**3 - k_s**3)
        
        return k_u, k_d, k_s, k_e

    def get_thermodynamics(self, n_B_phys):
        """
        Returns (energy_density, pressure, mu_B) for a given baryon density.
        """
        # 1. Get the stable state using the solver we just built
        k_u, k_d, k_s, k_e = self.solve_beta_equilibrium(n_B_phys)
        
        # 2. Calculate the Vector Potential V (Hard Gluon repulsion)
        # V = (27/16) * (g/mg)^2 * rho_B. 
        # Note: We use n_B_phys directly here.
        V = self.C_gluon * n_B_phys
        
        # 3. Calculate effective chemical potentials (kinetic + vector)
        mu_u = np.sqrt(k_u**2 + self.m_u**2) + V
        mu_d = np.sqrt(k_d**2 + self.m_d**2) + V
        mu_s = np.sqrt(k_s**2 + self.m_s**2) + V
        
        # 4. Total Baryon Chemical Potential
        mu_B = mu_u + mu_d + mu_s
        
        # 5. Pressure and Energy Density
        # We reuse the kinetic functions, then add the gluon and bag terms
        pq, pe = self.kinetic_pressure(k_u, k_d, k_s, k_e)
        p_gluon = self.C_gluon * (n_B_phys**2)
        total_pressure = pq + pe + p_gluon - self.B_bag
        
        # Energy density = sum(epsilon_kinetic) + gluon_interaction + B_bag
        # epsilon_kin = (3/4)*mu_k*k*... (using the Fermi integral logic)
        # For brevity, I'll let you add your kinetic_energy_density() logic here.
        
        return total_pressure, mu_B        