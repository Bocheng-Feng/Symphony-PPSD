import numpy as np
from scipy.integrate import cumulative_trapezoid as cumtrapz
import symlib

def einasto_model():
    """
    Returns two functions:
        1. einasto_rho(r, alpha, r2, rho2): Einasto density profile.
        2. jeans_sigma_r(r_grid, beta_grid, alpha, c): Radial velocity dispersion from Jeans equation.
    """

    def einasto_rho(r, alpha, r2, rho2):
        """
        Einasto density profile in linear scale.
        Parameters:
            r (array_like): Radius array.
            alpha (float): Einasto shape parameter.
            r2 (float): Scale radius (r_-2).
            rho2 (float): Density at scale radius r2.
        Returns:
            rho (array_like): Density at radius r.
        """
        x = (r / r2)**alpha
        return rho2 * np.exp(-(2.0/alpha)*(x - 1.0))

    def jeans_sigma_r(r_grid, beta_grid, alpha, c):
        """
        Compute radial velocity dispersion sigma_r(r)/Vvir on a given radius grid by solving the Jeans equation.
        Parameters:
            r_grid (array_like): Radius grid.
            beta_grid (array_like): Velocity anisotropy profile beta(r) on the same grid.
            alpha (float): Einasto shape parameter.
            c (float): Concentration parameter (c = 1 / r_-2).
        Returns:
            sigma (array_like): Radial velocity dispersion normalized by virial velocity at r_grid.
        """
        def einasto_mass(r, alpha, r2, rho2):
            s = r
            rho = einasto_rho(s, alpha, r2, rho2)
            integrand = 4*np.pi * rho * s**2
            M = np.concatenate(([0.0], cumtrapz(integrand, s)))
            return M

        def rho2_for_Mvir_equals_one(r_grid, alpha, r2):
            rho2_trial = 1.0
            M_trial = einasto_mass(r_grid, alpha, r2, rho2_trial)
            M1 = np.interp(1.0, r_grid, M_trial)
            if M1 <= 0 or not np.isfinite(M1):
                return np.nan
            return 1.0 / M1

        r = r_grid
        r2 = 1.0 / c
        rho2 = rho2_for_Mvir_equals_one(r, alpha, r2)
        if not np.isfinite(rho2) or rho2 <= 0:
            return np.full_like(r, np.nan)
        rho = einasto_rho(r, alpha, r2, rho2)
        M   = einasto_mass(r, alpha, r2, rho2)
        integrand = np.where(r > 0, beta_grid / r, 0.0)
        I_beta = np.concatenate(([0.0], cumtrapz(integrand, r)))
        base = np.where(r > 0, rho * M / (r**2), 0.0)
        B = base * np.exp(2.0 * I_beta)
        P = np.concatenate(([0.0], cumtrapz(B, r)))
        J = (P[-1] - P) * np.exp(-2.0 * I_beta)
        sigma2 = np.where(rho > 0, J / rho, np.nan)
        sigma  = np.sqrt(np.maximum(sigma2, 0.0))
        return sigma

    return einasto_rho, jeans_sigma_r

def get_snap_from_z(sim_dir, redshift):
    """
    Infer the closet snapshot and redshift of the symphony suite at the target redshift.
    """
    scale = symlib.scale_factors(sim_dir)                      # Array of scale factors for snapshots
    z = 1/scale - 1                                            # Convert scale factors to redshifts
    idx = np.argmin(np.abs(z - redshift))    # Find index of snapshot closest to target redshift
    closest_snapshot = idx                   # Snapshot ID corresponding to closest redshift
    closest_redshift = z[idx]                # Actual redshift of the chosen snapshot

    return closest_snapshot, closest_redshift