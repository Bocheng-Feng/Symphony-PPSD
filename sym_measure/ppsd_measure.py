import numpy as np
import pandas as pd
import os
import pynumdiff
import pynumdiff.optimize

def ppsd_profiles(base_dir, suite_name):
    """
    Compute and save pseudo phase-space density (PPSD) profiles for host halos in a simulation suite.

    The PPSD is a measure combining density and velocity dispersion, often used to characterize 
    the dynamical state of dark matter halos. It is computed as:
        - Q_r   = rho / sigma_rad^3, where sigma_rad is the radial velocity dispersion.
        - Q_tot = rho / sigma_total^3, where sigma_total is the total (3D) velocity dispersion.

    These profiles provide insight into the structure and dynamics of halos and are saved per halo 
    as CSV files for further analysis.

    Parameters
    ----------
    base_dir : str
        Root directory containing pre-computed profiles for the simulation suite.

    suite_name : str
        Identifier for the simulation suite (e.g., 'SymphonyLMC').

    Returns
    -------
    None
        Saves PPSD profiles to disk in the 'ppsd_profiles' subdirectory under the suite directory.

    Notes
    -----
    This function expects the following pre-processed profiles to exist for each halo:
        - Density profiles ('rho_scaled') in 'density_profiles'
        - Enclosed mass profiles ('m_scaled') in 'mass_profiles'
        - Velocity dispersion profiles ('sigma_rad_scaled', 'sigma_total_scaled') in 'velocity_profiles'
    """
    # Define directories for input profiles and output PPSD profiles
    density_dir = os.path.join(base_dir, suite_name, "density_profiles")
    mass_dir = os.path.join(base_dir, suite_name, "mass_profiles")
    velocity_dir = os.path.join(base_dir, suite_name, "velocity_profiles")
    output_dir = os.path.join(base_dir, suite_name, "ppsd_profiles")
    os.makedirs(output_dir, exist_ok=True)

    # Gather and sort all profile files to ensure consistent halo ordering
    density_files = sorted([f for f in os.listdir(density_dir) if f.endswith(".csv")])
    mass_files = sorted([f for f in os.listdir(mass_dir) if f.endswith(".csv")])
    velocity_files = sorted([f for f in os.listdir(velocity_dir) if f.endswith(".csv")])

    # Iterate over halos to compute and save PPSD profiles
    for halo_idx, (f_rho, f_mass, f_vel) in enumerate(zip(density_files, mass_files, velocity_files)):
        # Load density, mass, and velocity dispersion profiles for the current halo
        df_rho = pd.read_csv(os.path.join(density_dir, f_rho))      # Density profile data
        df_mass = pd.read_csv(os.path.join(mass_dir, f_mass))       # Enclosed mass profile data
        df_vel = pd.read_csv(os.path.join(velocity_dir, f_vel))     # Velocity dispersion profile data

        # Extract scaled radius, density, enclosed mass, and velocity dispersions
        r = df_rho["r_scaled"].values           # Scaled radial coordinate (dimensionless)
        rho = df_rho["rho_scaled"].values       # Scaled density (mass per unit volume)
        m = df_mass["m_scaled"].values          # Scaled enclosed mass within radius r
        sigma_rad = df_vel["sigma_rad_scaled"].values   # Radial velocity dispersion (1D)
        sigma_tot = df_vel["sigma_total_scaled"].values # Total (3D) velocity dispersion

        # Calculate pseudo phase-space densities:
        # Q_r uses radial velocity dispersion, representing anisotropic dynamics.
        # Q_tot uses total velocity dispersion, representing isotropic approximation.
        Q_r = np.where(sigma_rad > 0, rho / sigma_rad**3, np.nan)
        Q_tot = np.where(sigma_tot > 0, rho / sigma_tot**3, np.nan)

        # Compile results into a DataFrame and save to CSV
        df_out = pd.DataFrame({
            "r_scaled": r,
            "m_scaled": m,
            "Q_r": Q_r,
            "Q_tot": Q_tot,
        })
        df_out.to_csv(f"{output_dir}/halo_{halo_idx:03d}_profile.csv", index=False)

    print(f"[Saved] PPSD profiles for {suite_name}.")

def smooth_ppsd_slopes(base_dir, suite_name, method='constant_jerk', tvgamma=None):
    """
    Compute smoothed logarithmic slopes of pseudo phase-space density (PPSD) profiles for halos.

    The logarithmic slopes characterize how PPSD varies with radius and enclosed mass, providing 
    insight into halo structure and dynamical state. Slopes are computed as derivatives of:
        - log(Q_r) with respect to log(r) and log(m)
        - log(Q_tot) with respect to log(r) and log(m)

    Derivatives are estimated using advanced numerical differentiation methods from the pynumdiff 
    package, optionally with total variation regularization.

    Parameters
    ----------
    base_dir : str
        Root directory containing the simulation suite's profiles.

    suite_name : str
        Name of the simulation suite (e.g., 'SymphonyLMC').

    method : str, optional
        Name of the pynumdiff derivative method to use (default is 'constant_jerk').

    tvgamma : float or None, optional
        Regularization strength for total variation-based differentiation methods. Ignored if 
        the chosen method does not support it.

    Returns
    -------
    None
        Smoothed slope profiles are saved as CSV files in 'ppsd_slope_profiles_r' and 
        'ppsd_slope_profiles_m' directories under the suite directory.

    Raises
    ------
    ValueError
        If the specified differentiation method is not found in pynumdiff.

    Notes
    -----
    The function attempts to fit derivatives robustly and reports halos for which saving slopes fails.
    """
    def get_diff_and_optimize_funcs(method):
        """
        Locate the differentiation and optimization functions corresponding to the specified method.

        Searches pynumdiff submodules for the requested derivative method and its optimizer.

        Parameters
        ----------
        method : str
            Name of the derivative method to find.

        Returns
        -------
        diff_func : callable
            Differentiation function that computes derivative given data, step size, and parameters.

        optimize_func : callable
            Optimization function that estimates best-fit parameters for differentiation.

        Raises
        ------
        ValueError
            If the method is not found in any pynumdiff submodule.
        """
        submodules = [
            'kalman_smooth',
            'smooth_finite_difference',
            'finite_difference',
            'total_variation_regularization',
            'linear_model'
        ]
        for submod in submodules:
            try:
                mod_optimize = getattr(pynumdiff.optimize, submod)
                mod_diff = getattr(pynumdiff, submod)
                if hasattr(mod_optimize, method) and hasattr(mod_diff, method):
                    return getattr(mod_diff, method), getattr(mod_optimize, method)
            except AttributeError:
                continue
        raise ValueError(f"Method '{method}' not found in any submodule.")

    # Define directories for input PPSD profiles and output slope profiles
    ppsd_dir = os.path.join(base_dir, suite_name, "ppsd_profiles")
    slope_r_dir = os.path.join(base_dir, suite_name, "ppsd_slope_profiles_r")
    slope_m_dir = os.path.join(base_dir, suite_name, "ppsd_slope_profiles_m")
    os.makedirs(slope_r_dir, exist_ok=True)
    os.makedirs(slope_m_dir, exist_ok=True)

    # List of PPSD profile files for all halos
    ppsd_files = sorted([f for f in os.listdir(ppsd_dir) if f.endswith(".csv")])
    n_halos = len(ppsd_files)

    # ----- Helper function: fit derivative for a profile -----
    def fit_derivative(y, dt):
        """
        Fit the derivative of a 1D signal y with respect to independent variable spacing dt.

        Uses pynumdiff to obtain a smooth and robust derivative estimate, which reduces noise 
        amplification common in numerical differentiation.

        Parameters
        ----------
        y : array_like
            1D array of dependent variable values (e.g., log PPSD).

        dt : float
            Step size in the independent variable (e.g., mean spacing of log radius).

        Returns
        -------
        dydx : array_like or None
            Smoothed derivative of y with respect to x, or None if fitting fails.
        """
        try:
            diff_func, optimize_func = get_diff_and_optimize_funcs(method)
            kwargs = {'tvgamma': tvgamma} if 'tvgamma' in optimize_func.__code__.co_varnames else {}
            params, _ = optimize_func(y, dt, **kwargs)   # Optimize parameters for differentiation
            _, dydx = diff_func(y, dt, params)           # Compute derivative using optimized params
            return dydx
        except Exception as e:
            print(f"{method} derivative fit failed: {e}")
            return None

    failed_halos = []
    saved_count = 0

    # Process each halo to compute and save smoothed PPSD slopes
    for halo_idx in range(n_halos):
        try:
            df_Q = pd.read_csv(os.path.join(ppsd_dir, ppsd_files[halo_idx]))
        except Exception as e:
            print(f"[Halo {halo_idx}] loading profiles failed: {e}")
            failed_halos.append(halo_idx)
            continue

        # Extract scaled radius, enclosed mass, and PPSD values
        r = df_Q["r_scaled"].values
        m = df_Q["m_scaled"].values
        Q_tot = df_Q["Q_tot"].values
        Q_r = df_Q["Q_r"].values

        # Mask valid points
        mask = (Q_r>0) & (Q_tot>0)
        Q_r = Q_r[mask]
        Q_tot = Q_tot[mask]
        r = r[mask]
        m = m[mask]

        # Compute average step sizes in log-space for radius and mass grids
        dt_r = np.diff(np.log10(r)).mean()
        dt_m = np.diff(np.log10(m)).mean()

        # Convert PPSD profiles to logarithmic scale for slope calculation
        log_Q_r = np.log10(Q_r)
        log_Q_tot = np.log10(Q_tot)

        # Calculate smoothed logarithmic derivatives:
        # slope_Q_r and slope_Q_tot represent local power-law indices of PPSD versus radius or mass.
        slope_Q_tot_r = fit_derivative(log_Q_tot, dt_r)
        slope_Q_rad_r = fit_derivative(log_Q_r, dt_r)
        slope_Q_tot_m = fit_derivative(log_Q_tot, dt_m)
        slope_Q_rad_m = fit_derivative(log_Q_r, dt_m)

        # Save slope profiles as CSV files, handling possible exceptions
        try:
            df_r = pd.DataFrame({
                "r_scaled": r,
                "slope_Q_r": slope_Q_rad_r,
                "slope_Q_tot": slope_Q_tot_r
            })
            df_r.to_csv(os.path.join(slope_r_dir, f"halo_{halo_idx:03d}_profile.csv"), index=False)
            df_m = pd.DataFrame({
                "m_scaled": m,
                "slope_Q_r": slope_Q_rad_m,
                "slope_Q_tot": slope_Q_tot_m
            })
            df_m.to_csv(os.path.join(slope_m_dir, f"halo_{halo_idx:03d}_profile.csv"), index=False)

            saved_count += 1
            print(f"[Saved] {saved_count}/{n_halos} PPSD slope for suite {suite_name}", end='\r', flush=True)
            
        except Exception as e:
            print(f"[Halo {halo_idx}] saving PPSD slope of m failed: {e}")
            failed_halos.append(halo_idx)

    print()  # Print newline to separate progress output from subsequent logs

    if failed_halos:
        failed_unique = sorted(set(failed_halos))
        print(f"[Warning] Failed to save PPSD slope profiles for halos: {failed_unique}")
