import os
import csv
import warnings
import traceback
import numpy as np
import pandas as pd
from astropy.constants import G
from astropy import units as u
import symlib
from colossus.cosmology import cosmology

def process_halo_properties(input_dir, output_dir, suite_name, snap):
    """
    Process halo properties for a given simulation suite and snapshot.

    This function performs several key analyses on halo data, including:
    - Saving fundamental halo properties such as virial mass, radius, velocity, and concentration.
    - Computing the radius of maximum circular velocity (r_vmax) from mass profiles.
    - Calculating the Jeans deviation profile and its integrated total as a measure of dynamical equilibrium.
    - Estimating the dynamical mass accretion rates over the last dynamical time.

    Parameters
    ----------
    input_dir : str
        Root directory containing the simulation suite's particle and halo datasets.

    output_dir : str
        Directory where processed outputs will be saved. A subdirectory named after the suite will be created.

    suite_name : str
        Name of the simulation suite (e.g., 'SymphonyLMC').

    snap : int
        Snapshot index specifying which snapshot to use for halo property extraction.

    Returns
    -------
    None
        Outputs are saved as CSV files in the designated output directory.
    """

    output_dir = os.path.join(output_dir, suite_name)
    os.makedirs(output_dir, exist_ok=True)

    # ------------------- Save halo basic properties -------------------
    def save_basic_properties():
        """
        Extract and save fundamental halo properties for all host halos in the suite.

        Properties saved include:
        - Virial mass (mvir)
        - Virial radius (rvir)
        - Virial velocity (vvir), computed via v_vir = sqrt(G * M_vir / R_vir)
        - Concentration parameter (cvir)

        The virial velocity is calculated using consistent physical units (km/s).

        Halo IDs are formatted as zero-padded three-digit strings.

        Outputs are saved as separate CSV files for each property.

        Returns
        -------
        None
        """
        n_halos = symlib.n_hosts(suite_name)

        halo_ids, mvir_list, rvir_list, vvir_list, cvir_list = [], [], [], [], []

        for halo_idx in range(n_halos):
            sim_dir = symlib.get_host_directory(input_dir, suite_name, halo_idx)
            try:
                r, _ = symlib.read_rockstar(sim_dir)
                host = r[0, snap]

                halo_ids.append(f"{halo_idx:03d}")

                # Virial mass in solar masses
                mvir_list.append(host["m"])

                # Virial radius in kpc
                rvir_list.append(host["rvir"])

                # Compute virial velocity: v_vir = sqrt(G * M_vir / R_vir)
                m = host["m"] * u.Msun
                r_kpc = host["rvir"] * u.kpc
                G_kpc = G.to(u.kpc * (u.km/u.s)**2 / u.Msun)
                vvir = np.sqrt(G_kpc * m / r_kpc).to(u.km/u.s).value
                vvir_list.append(vvir)

                # Halo concentration parameter
                cvir_list.append(host["cvir"])

            except FileNotFoundError:
                print(f"[Warning] Rockstar file not found for Halo {halo_idx}")
                continue

        # Save each property to its respective CSV file
        pd.DataFrame({"halo_id": halo_ids, "mvir": mvir_list}).to_csv(
            os.path.join(output_dir, "halo_mass.csv"), index=False)
        pd.DataFrame({"halo_id": halo_ids, "rvir": rvir_list}).to_csv(
            os.path.join(output_dir, "virial_radius.csv"), index=False)
        pd.DataFrame({"halo_id": halo_ids, "vvir": vvir_list}).to_csv(
            os.path.join(output_dir, "virial_velocity.csv"), index=False)
        pd.DataFrame({"halo_id": halo_ids, "cvir": cvir_list}).to_csv(
            os.path.join(output_dir, "halo_concentrations.csv"), index=False)

        print(f"[Saved] Halo basic properties for {suite_name}")

    # ------------------- Compute rvmax from mass profiles -------------------
    def save_rvmax():
        """
        Compute and save the radius of maximum circular velocity (r_vmax) for each halo.

        The circular velocity profile is computed as v_c(r) = sqrt(M(<r) / r) in scaled units.
        The radius at which v_c(r) attains its maximum defines r_vmax.

        The scaled radius r_vmax is converted to physical units by multiplying with the virial radius.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        mass_dir = os.path.join(output_dir, "mass_profiles")
        df_rvir = pd.read_csv(os.path.join(output_dir, "virial_radius.csv"), dtype={"halo_id": str})
        os.makedirs(output_dir, exist_ok=True)

        def compute_rvmax_from_mass_profile(r, m):
            # Compute circular velocity profile v_c(r) = sqrt(M(<r)/r)
            vc = np.sqrt(m / r)
            # Return radius corresponding to maximum circular velocity
            return r[np.argmax(vc)]

        files = sorted([f for f in os.listdir(mass_dir) if f.endswith(".csv")])
        halo_ids, rvmax_list = [], []

        for f in files:
            try:
                df = pd.read_csv(os.path.join(mass_dir, f))
                r = df["r_scaled"].values
                m = df["m_scaled"].values
                rvmax = compute_rvmax_from_mass_profile(r, m)

                halo_id = f.split("_")[1]
                if halo_id not in df_rvir["halo_id"].values:
                    print(f"[Warning] halo_id {halo_id} not found in virial_radius.csv")
                    continue
                rvir = df_rvir.loc[df_rvir.halo_id == halo_id, "rvir"].values[0]

                # Convert scaled r_vmax to physical units by multiplying with rvir
                halo_ids.append(halo_id)
                rvmax_list.append(rvmax * rvir)
            except Exception as e:
                print(f"[Warning] Failed to process {f}: {e}")
                continue

        # Save r_vmax values to CSV
        pd.DataFrame({"halo_id": halo_ids, "rvmax": rvmax_list}).to_csv(
            os.path.join(output_dir, "max_radius.csv"), index=False)
        print(f"[Saved] rvmax for {suite_name}")

    # ------------------- Compute Jeans deviation -------------------
    def compute_jeans_deviation():
        """
        Calculate the Jeans deviation profile δ_J(r) and its integrated total δ_J_tot for each halo.

        The Jeans deviation δ_J(r) quantifies the deviation from Jeans equilibrium in spherical systems,
        defined as the absolute value of the ratio between the residual of the Jeans equation numerator
        and its denominator:

            δ_J(r) = |(dP/dr + 2βP/r + ρ dΦ/dr) / (ρ dΦ/dr)|

        where
        - P = ρ σ_r^2 is the radial pressure,
        - β is the velocity anisotropy parameter,
        - ρ is the density,
        - Φ is the gravitational potential approximated via enclosed mass M(<r).

        The total deviation δ_J_tot is computed by integrating δ_J(r) over radius r ≤ r_vir (scaled radius ≤ 1).

        Outputs:
        - Radial profiles saved as CSV files per halo.
        - Total integrated δ_J saved as a summary CSV.

        Returns
        -------
        None
        """
        density_dir = os.path.join(output_dir, "density_profiles")
        velocity_dir = os.path.join(output_dir, "velocity_profiles")
        mass_dir = os.path.join(output_dir, "mass_profiles")
        jeans_dir = os.path.join(output_dir, "jeans_deviation")
        os.makedirs(jeans_dir, exist_ok=True)

        halo_files = sorted([f for f in os.listdir(density_dir) if f.endswith(".csv")])
        n_halos = len(halo_files)
        saved_count = 0
        failed_halos = []

        for f in halo_files:
            halo_id = f.split("_")[1]
            try:
                df_rho = pd.read_csv(os.path.join(density_dir, f))
                df_vel = pd.read_csv(os.path.join(velocity_dir, f))
                df_mass = pd.read_csv(os.path.join(mass_dir, f))

                r = df_rho["r_scaled"].values           # Scaled radius (r/r_vir)
                rho = df_rho["rho_scaled"].values       # Scaled density
                sigma_r2 = df_vel["sigma_rad_scaled"].values ** 2  # Radial velocity dispersion squared
                beta = df_vel["beta"].values             # Velocity anisotropy parameter
                m = df_mass["m_scaled"].values           # Enclosed mass profile

                # Radial pressure P(r) = ρ(r) * σ_r^2(r)
                P_r = rho * sigma_r2
                # Radial derivative of pressure dP/dr
                dPdr = np.gradient(P_r, r)
                # Gravitational acceleration dΦ/dr = M(<r) / r^2 (in scaled units)
                dPhidr = m / (r ** 2)
                # Numerator of Jeans equation residual
                numer = dPdr + 2 * beta * P_r / r + rho * dPhidr
                # Denominator of Jeans equation residual
                denom = rho * dPhidr
                # Compute Jeans deviation δ_J(r)
                delta_J = np.abs(numer / denom)

                try:
                    # Save radial Jeans deviation profile per halo
                    pd.DataFrame({"halo_id": int(halo_id), "r_scaled": r, "delta_J": delta_J}).to_csv(
                        os.path.join(jeans_dir, f"halo_{int(halo_id):03d}_profile.csv"), index=False)
                    saved_count += 1
                    print(f"[Saved] {saved_count}/{n_halos} Jeans deviation for suite {suite_name}", end='\r', flush=True)
                except Exception as e:
                    print(f"[Warning] Failed to save Jeans deviation for halo {halo_id}: {e}")
                    failed_halos.append(int(halo_id))

            except Exception as e:
                print(f"[Warning] Failed to compute Jeans deviation for halo {halo_id}: {e}")
                failed_halos.append(int(halo_id))
        print()  # newline after progress

        # Compute total integrated Jeans deviation δ_J_tot by trapezoidal integration up to scaled radius 1
        results = []
        files = sorted([f for f in os.listdir(jeans_dir) if f.endswith(".csv")])
        for f in files:
            halo_id = int(f.split("_")[1])
            df = pd.read_csv(os.path.join(jeans_dir, f))
            r = df["r_scaled"].values
            delta_J = df["delta_J"].values
            mask = r <= 1.0
            delta_J_tot = np.trapz(delta_J[mask], r[mask])
            results.append({"halo_id": halo_id, "delta_J_tot": delta_J_tot})

        # Save total Jeans deviation per halo
        pd.DataFrame(results).to_csv(os.path.join(output_dir, "jeans_deviation_total.csv"), index=False)
        print(f"[Saved] Total Jeans deviation for {suite_name}")

        if failed_halos:
            print(f"[Warning] Jeans deviation failed for halos: {sorted(failed_halos)}")

    # ------------------- Compute dynamical accretion rates -------------------
    def compute_accretion_rates():
        """
        Compute the mass accretion rates γ for each host halo over the last dynamical time.

        The accretion rate is defined as:

            γ = (M_now - M_past) / t_dyn

        where
        - M_now is the halo mass at the latest snapshot,
        - M_past is the halo mass one dynamical time ago,
        - t_dyn is the dynamical time computed from the virial overdensity and cosmology.

        The dynamical time is estimated as:

            t_dyn = 1 / sqrt((4/3) * π * G * ρ_vir)

        where ρ_vir is the virial density threshold times the mean matter density at z=0.

        Cosmological parameters are read from the simulation metadata to ensure consistency.

        Outputs a CSV file listing halo IDs and their corresponding accretion rates.

        Returns
        -------
        None
        """
        import traceback
        output_csv = os.path.join(output_dir, "accretion_rates.csv")
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)

        n_halos = symlib.n_hosts(suite_name)
        saved_count = 0
        failed_halos = []

        with open(output_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["halo_id", "gamma"])

            n_halos = symlib.n_hosts(suite_name)
            for i in range(n_halos):
                try:
                    sim_dir = symlib.get_host_directory(input_dir, suite_name, i)
                    scale = symlib.scale_factors(sim_dir)
                    r, _ = symlib.read_rockstar(sim_dir)
                    snap_last = len(scale) - 1
                    m_now = r[0, snap_last]["m"]

                    # Load cosmological parameters for this simulation
                    sim_params = symlib.simulation_parameters(sim_dir)
                    cosmo = cosmology.setCosmology("custom", {
                        "flat": sim_params["flat"],
                        "H0": sim_params["H0"],
                        "Om0": sim_params["Om0"],
                        "Ob0": sim_params["Ob0"],
                        "sigma8": sim_params["sigma8"],
                        "ns": sim_params["ns"]
                    })

                    # Mean matter density at z=0 in Msun/Mpc^3
                    rho_m0 = cosmo.rho_m(0)
                    Delta = 99  # Virial overdensity threshold
                    # Virial density threshold in physical units, accounting for h100 scaling
                    rho_vir = Delta * rho_m0 * 1e9 / sim_params["h100"]**2
                    # Gravitational constant in Mpc^3 / (Msun Gyr^2)
                    G_val = G.to(u.Mpc**3 / (u.Msun * u.Gyr**2)).value
                    # Dynamical time scale (Gyr)
                    t_dyn = 1.0 / np.sqrt((4/3) * np.pi * G_val * rho_vir)

                    # Compute cosmic times at each snapshot scale factor
                    times = cosmo.age(1 / scale - 1)
                    t0 = times[snap_last]
                    t_past = t0 - t_dyn
                    # Find snapshot closest to one dynamical time ago
                    snap_past = np.argmin(np.abs(times - t_past))
                    m_past = r[0, snap_past]["m"]
                    # Calculate accretion rate γ
                    gamma = (m_now - m_past) / t_dyn
                    writer.writerow([f"{i:03d}", gamma])
                    saved_count += 1
                    print(f"[Saved] {saved_count}/{n_halos} accretion rates for suite {suite_name}", end='\r', flush=True)

                except Exception as e:
                    traceback.print_exc()
                    warnings.warn(f"[{suite_name} - halo {i}] Error calculating accretion rate: {e}")
                    writer.writerow([i, "nan"])
                    failed_halos.append(i)
        print()  # newline after progress

        if failed_halos:
            print(f"[Warning] Accretion rate calculation failed for halos: {sorted(failed_halos)}")
            
    def best_fit_ppsd_slope():
        """
        Fit the PPSD slope with singel power law model using non-linear least squares curve fitting.
        """

        def const_model(r, a):
            """A one-parameter constant model: y(r) = a."""
            return a * np.ones_like(r)
        
        ppsd_slope_dir = os.path.join(output_dir, "ppsd_slope_profiles_r")
        halo_files = sorted([f for f in os.listdir(ppsd_slope_dir) if f.endswith(".csv")])

        halo_ids, ppsd_slope = [], [] 
        for f in halo_files:
            halo_id = f.split("_")[1]
            halo_ids.append(halo_id)

            df_ppsd_slope = pd.read_csv(os.path.join(ppsd_slope_dir, f))
            r = df_ppsd_slope["r_scaled"].values
            slope = df_ppsd_slope["slope_Q_r"].values

            try:
                popt, _ = curve_fit(const_model, r, slope, p0=[-1.875])
                best_slope = popt[0]
            except Exception:
                best_slope = -1.875

            ppsd_slope.append(best_slope)

        pd.DataFrame({"halo_id": halo_ids, "slope": ppsd_slope}).to_csv(
        os.path.join(output_dir, "best_fit_slope.csv"), index=False)
        print(f"[Saved] best-fit PPSD slope for {suite_name}")

    # ------------------- Run sub-functions you need-------------------
    save_basic_properties()
    # save_rvmax()
    compute_jeans_deviation()
    compute_accretion_rates()
    best_fit_ppsd_slope()