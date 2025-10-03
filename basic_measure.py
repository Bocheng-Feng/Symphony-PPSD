import numpy as np
import pandas as pd
from astropy.constants import G
from astropy import units as u
import symlib
import os

def density_velocity_mass(input_dir, output_dir, suite_name, snap, n_bins=40, r_min=1e-3, r_max=1.5):
    """
    Compute and save radial profiles of density, velocity dispersion (including anisotropy), 
    and enclosed mass for host halos in a given simulation suite at a specified snapshot.

    The profiles are normalized to facilitate comparison across halos and simulation setups:
        - Density is normalized by the background matter density at redshift zero, 
          defined as ρ_m = Ω_m * ρ_crit, where ρ_crit is the critical density of the Universe.
        - Velocity dispersions are normalized by the virial velocity of the halo at the snapshot.
        - Enclosed mass profiles are normalized by the virial mass of the halo.

    Parameters
    ----------
    input_dir : str
        Root directory containing the simulation suite's particle and halo datasets.

    output_dir : str
        Directory path where computed profiles will be saved as CSV files.

    suite_name : str
        Identifier for the simulation suite (e.g., 'SymphonyLMC').

    snap : int
        Snapshot index specifying the simulation output time at which to analyze halos.

    n_bins : int, optional
        Number of logarithmically spaced radial bins used for profile calculations.
        Default is 40.

    r_min : float, optional
        Minimum radius for radial binning, expressed in units of the host halo's virial radius (R_vir).
        Default is 1e-3.

    r_max : float, optional
        Maximum radius for radial binning, expressed in units of the host halo's virial radius (R_vir).
        Default is 1.5.

    Returns
    -------
    None
        The function saves computed profiles to disk and prints warnings if saving fails.
    """

    # ----------------------------------------------------------------
    # Shared helper functions
    # ----------------------------------------------------------------
    def load_halo_and_particles(halo_idx):
        """
        Load the host halo properties and particle data for a given halo index.

        Parameters
        ----------
        halo_idx : int
            Index of the halo within the simulation suite.

        Returns
        -------
        host : dict or None
            Dictionary containing host halo properties at the specified snapshot.
            Returns None if Rockstar halo catalog is unavailable.

        p : object or None
            Particle data object containing particle positions and velocities at the snapshot.
            Returns None if particle snapshot file is missing.
        """
        sim_dir = symlib.get_host_directory(input_dir, suite_name, halo_idx)

        try:
            r, hist = symlib.read_rockstar(sim_dir)
            host = r[0, snap]   # Extract host halo properties at snapshot "snap"
        except FileNotFoundError:
            print(f"[Warning] Rockstar file not found for Halo {halo_idx}")
            return None, None

        try:
            part = symlib.Particles(sim_dir)
            p = part.read(snap) # Load particle data at snapshot "snap"
        except FileNotFoundError:
            print(f"[Warning] Particle snapshot not found for Halo {halo_idx}")
            return host, None

        return host, p

    def create_bins():
        """
        Generate logarithmically spaced radial bins and their centers in units of R_vir.

        Returns
        -------
        bins : ndarray
            Array of bin edges, length n_bins + 1.

        bin_centers : ndarray
            Array of bin center radii, length n_bins.
        """
        bins = np.logspace(np.log10(r_min), np.log10(r_max), n_bins + 1)
        bin_centers = 0.5 * (bins[1:] + bins[:-1])
        return bins, bin_centers

    def save_profile(df, profile_type, halo_idx, count, total, failed_list):
        """
        Save the computed profile DataFrame to a CSV file in the appropriate subdirectory.

        Parameters
        ----------
        df : pandas.DataFrame
            DataFrame containing the profile data to save.

        profile_type : str
            Subdirectory name indicating the profile type (e.g., 'density_profiles').

        halo_idx : int
            Index of the halo corresponding to this profile.

        count : int
            Current count of successfully processed halos.

        total : int
            Total number of halos to process.

        failed_list : list
            List to which the halo_idx will be appended if saving fails.

        Returns
        -------
        None
        """
        out_path = f"{output_dir}/{suite_name}/{profile_type}"
        os.makedirs(out_path, exist_ok=True)
        try:
            df.to_csv(f"{out_path}/halo_{halo_idx:03d}_profile.csv", index=False)
            print(f"[Saved] {count}/{total} {profile_type.replace('_',' ')} for suite {suite_name}", end='\r', flush=True)
        except Exception:
            failed_list.append(halo_idx)

    # ----------------------------------------------------------------
    # Subfunction 1: Compute density profiles
    # ----------------------------------------------------------------
    def density_profile():
        """
        Compute radial density profiles normalized by the background matter density.

        The density is computed by counting particles in spherical shells around the halo center,
        converting counts to mass density, and normalizing by ρ_m = Ω_m * ρ_crit(z=0).

        Returns
        -------
        failed_density : list
            List of halo indices for which saving the density profile failed.
        """
        n_halos = symlib.n_hosts(suite_name)
        params = symlib.simulation_parameters(suite_name)
        h = params['h100'] # Dimensionless Hubble parameter scaled by 100 km/s/Mpc
        mp = params['mp'] / h # Particle mass in solar masses (Msun), corrected for h scaling
        Om0 = params["Om0"] # Matter density parameter at redshift zero

        # Calculate the critical density of the Universe at z=0 in units of Msun/kpc^3
        H0_si = params["H0"] * u.km / u.s / u.Mpc
        G_si = G.to(u.Mpc**3 / u.Msun / u.s**2)
        rho_crit = (3 * H0_si**2 / (8 * np.pi * G_si)).to(u.Msun / u.kpc**3).value
        rho_m = Om0 * rho_crit  # Background matter density

        cvir, mvir = [], []
        bins, bin_centers = create_bins()
        failed_density = []

        count = 0
        for halo_idx in range(n_halos):
            host, p = load_halo_and_particles(halo_idx)
            if host is None or p is None:
                continue

            center, r_vir = host['x'], host['rvir']
            cvir.append(host['cvir'])
            mvir.append(host['m'])

            # Compute normalized radial distances of particles from halo center
            radi = np.linalg.norm(p[0]['x'] - center, axis=1) / r_vir

            # Count particles in each radial bin
            counts, _ = np.histogram(radi, bins=bins)

            # Compute volume of spherical shells in physical units (kpc^3)
            shell_volumes = (4/3) * np.pi * ((bins[1:]*r_vir)**3 - (bins[:-1]*r_vir)**3)

            # Calculate density in each shell, normalized by background matter density
            rho_scaled = (counts * mp / shell_volumes) / rho_m

            # Prepare DataFrame for saving
            df = pd.DataFrame({"r_scaled": bin_centers, "rho_scaled": rho_scaled})
            count += 1
            save_profile(df, "density_profiles", halo_idx, count, n_halos, failed_density)
        print()  # newline after progress output

        return failed_density

    # ----------------------------------------------------------------
    # Subfunction 2: Compute velocity dispersion profiles
    # ----------------------------------------------------------------
    def velocity_profile():
        """
        Compute radial profiles of velocity dispersions and velocity anisotropy parameter β.

        Velocity dispersions are normalized by the halo's virial velocity. The anisotropy parameter β
        quantifies the relative importance of radial versus tangential velocity dispersion components.

        Returns
        -------
        failed_velocity : list
            List of halo indices for which saving the velocity profile failed.
        """
        n_halos = symlib.n_hosts(suite_name)
        bins, bin_centers = create_bins()
        failed_velocity = []

        count = 0
        for halo_idx in range(n_halos):
            host, p = load_halo_and_particles(halo_idx)
            if host is None or p is None:
                continue

            center, v_host, r_vir = host['x'], host['v'], host['rvir']

            # Compute virial velocity v_vir = sqrt(G * M_vir / R_vir) in km/s
            m_vir = host['m'] * u.Msun
            r_vir_u = r_vir * u.kpc
            G_kpc = G.to(u.kpc * (u.km/u.s)**2 / u.Msun)
            v_vir = np.sqrt(G_kpc * m_vir / r_vir_u).to(u.km / u.s).value

            # Compute particle positions and velocities relative to halo center and bulk velocity
            dx = p[0]['x'] - center
            dv = p[0]['v'] - v_host

            # Radial distance normalized by R_vir
            radi = np.linalg.norm(dx, axis=1) / r_vir

            # Unit vectors pointing from halo center to particles
            r_hat = dx / np.linalg.norm(dx, axis=1)[:, None]

            # Radial velocity component of each particle
            v_rad = np.sum(dv * r_hat, axis=1)

            # Initialize lists to store scaled velocity dispersions and anisotropy parameter
            sigma_rad_scaled, sigma_tan_scaled, sigma_total_scaled, beta_profile = [], [], [], []

            # Loop over radial bins to compute velocity dispersion components
            for i in range(n_bins):
                in_bin = (radi >= bins[i]) & (radi < bins[i+1])
                sigma_rad = np.std(v_rad[in_bin])  # Radial velocity dispersion (standard deviation)
                # Total velocity dispersion from standard deviations of all 3 velocity components
                sigma_total = np.linalg.norm([np.std(dv[in_bin][:, j]) for j in range(3)])
                # Tangential velocity dispersion derived by subtracting radial component in quadrature
                sigma_tan = np.sqrt(max(0, sigma_total**2 - sigma_rad**2))
                # Velocity anisotropy parameter β = 1 - (σ_tan^2) / (2 σ_rad^2)
                beta = np.nan if sigma_rad == 0 else 1 - sigma_tan**2 / (2 * sigma_rad**2)

                # Normalize dispersions by virial velocity
                sigma_rad_scaled.append(sigma_rad / v_vir)
                sigma_tan_scaled.append(sigma_tan / v_vir)
                sigma_total_scaled.append(sigma_total / v_vir)
                beta_profile.append(beta)

            # Prepare DataFrame for saving
            df = pd.DataFrame({
                "r_scaled": bin_centers,
                "sigma_rad_scaled": sigma_rad_scaled,
                "sigma_tan_scaled": sigma_tan_scaled,
                "sigma_total_scaled": sigma_total_scaled,
                "beta": beta_profile
            })
            count += 1
            save_profile(df, "velocity_profiles", halo_idx, count, n_halos, failed_velocity)
        print()  # newline after progress output

        return failed_velocity

    # ----------------------------------------------------------------
    # Subfunction 3: Compute enclosed mass profiles
    # ----------------------------------------------------------------
    def mass_profile():
        """
        Compute enclosed mass profiles normalized by the host halo virial mass.

        The enclosed mass at each radius is computed by cumulatively summing particle masses within that radius.

        Returns
        -------
        failed_mass : list
            List of halo indices for which saving the mass profile failed.
        """
        n_halos = symlib.n_hosts(suite_name)
        bins, bin_centers = create_bins()
        params = symlib.simulation_parameters(suite_name)
        mp = params['mp'] / params['h100']  # Particle mass in Msun, corrected for h scaling
        failed_mass = []

        count = 0
        for halo_idx in range(n_halos):
            host, p = load_halo_and_particles(halo_idx)
            if host is None or p is None:
                continue

            center, r_vir, m_vir = host['x'], host['rvir'], host['m']

            # Compute normalized radial distances of particles from halo center
            radi = np.linalg.norm(p[0]['x'] - center, axis=1) / r_vir

            # Count particles in each radial bin
            counts, _ = np.histogram(radi, bins=bins)

            # Compute cumulative enclosed mass in units of Msun
            enclosed_mass = np.cumsum(counts * mp)

            # Normalize enclosed mass by virial mass
            m_scaled = enclosed_mass / m_vir

            # Prepare DataFrame for saving
            df = pd.DataFrame({"r_scaled": bin_centers, "m_scaled": m_scaled})
            count += 1
            save_profile(df, "mass_profiles", halo_idx, count, n_halos, failed_mass)
        print()  # newline after progress output

        return failed_mass

    # ----------------------------------------------------------------
    # Execute all profile computations sequentially
    # ----------------------------------------------------------------
    failed_density = []
    failed_velocity = []
    failed_mass = []
    n_halos = symlib.n_hosts(suite_name)

    failed_density = density_profile()
    failed_velocity = velocity_profile()
    failed_mass = mass_profile()

    # Report any halos for which profile saving failed
    if failed_density:
        print(f"[Warning] Failed to save density profiles for halos: {sorted(set(failed_density))}")
    if failed_velocity:
        print(f"[Warning] Failed to save velocity profiles for halos: {sorted(set(failed_velocity))}")
    if failed_mass:
        print(f"[Warning] Failed to save mass profiles for halos: {sorted(set(failed_mass))}")
