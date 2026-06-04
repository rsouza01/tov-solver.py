import numpy as np

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