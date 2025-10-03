import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import seaborn as sns
from glob import glob
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d

from utils import einasto_model

sim_colors = {
        "SymphonyLMC": sns.color_palette("colorblind")[4],
        "SymphonyMilkyWay": sns.color_palette("colorblind")[0],
        "SymphonyGroup": sns.color_palette("colorblind")[2],
        "SymphonyLCluster": sns.color_palette("colorblind")[1],
        "SymphonyCluster": sns.color_palette("colorblind")[3],
    }

sim_names = {
        "SymphonyLMC": "LMC",
        "SymphonyMilkyWay": "Milky~Way",
        "SymphonyGroup": "Group",
        "SymphonyLCluster": "L-Cluster",
        "SymphonyCluster": "Cluster",
    }

def plot_density_velocity(base_dir, suite_names, einasto_fit=False, mask_range=[1e-3,1.5], plot_range=None):
    """
    Plot density and radial velocity dispersion profiles for specified simulation suites.

    Parameters:
        base_dir (str): Base directory containing 'data' and 'figure' subdirectories.
        suite_names (list of str): List of simulation suite names to process.
        einasto_fit (bool): Whether to perform and overlay Einasto profile fits. If True, measured anisotropy profiles are used to compute Einasto velocity dispersion profiles.
        mask_range (list or tuple): Radius range [r_min, r_max] to mask data for plotting and fitting.
        plot_range (tuple or list, optional): (xmin, xmax) range for the x-axis; if None, use default autoscaling.

    Returns:
        None

    Notes:
        - Loads density and velocity profile CSV files for each suite.
        - Computes mean and standard deviation of profiles across halos.
        - Optionally fits Einasto parameters to density and velocity dispersion profiles.
        - Saves figure as 'density_velocity_profiles.pdf' in base_dir/figure.
        - Uses log-log scale for density and semi-log for velocity dispersion.
    """
    plt.figure(figsize=(12, 5), dpi=500)
    plt.rcParams["text.usetex"] = True  # Use LaTeX for text rendering

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=500)
    ax_density, ax_sigma = axes

    # Add legend entries for Einasto fits if enabled
    if einasto_fit:
        ax_density.plot([], [], ls="--", color="black", lw=0.9, label=r"$\mathrm{Einasto\ fit}$")
        ax_sigma.plot([], [], ls="--", color="black", lw=0.9, label=r"$\mathrm{Einasto\ fit}$")

    # Get Einasto model functions
    einasto_rho, jeans_sigma_r = einasto_model()

    for suite in suite_names:
        # Load density profile CSV files for the suite
        data_dir = os.path.join(base_dir, 'data')
        prof_dir = os.path.join(data_dir, suite, "density_profiles")
        files = sorted(glob(os.path.join(prof_dir, "halo_*_profile.csv")))
        if not files:
            continue
        profs = [pd.read_csv(f) for f in files]

        r_all = profs[0]["r_scaled"].to_numpy()
        # Apply radius mask if specified
        mask = (r_all >= mask_range[0]) & (r_all <= mask_range[1]) if mask_range else slice(None)
        r = r_all[mask]

        # Stack density profiles and compute mean and std deviation across halos
        rho_stack = np.array([
            (df["rho_scaled"].to_numpy()[mask] if mask_range else df["rho_scaled"].to_numpy())
            for df in profs
        ])
        rho_mean = np.nanmean(rho_stack, axis=0)
        rho_std  = np.nanstd(rho_stack, axis=0)

        # Plot mean density times r^2 (to highlight profile shape) with shaded std deviation
        color = sim_colors[suite]
        label = rf"$\mathrm{{{sim_names[suite]}}}$"
        ax_density.loglog(r, rho_mean * r**2, lw=1.0, color=color, label=label)
        ax_density.fill_between(r, (rho_mean - rho_std) * r**2, (rho_mean + rho_std) * r**2,
                         color=color, alpha=0.25, lw=0)

        if einasto_fit:
            # Fit Einasto profile to density in log space over valid positive data
            valid = np.isfinite(r) & np.isfinite(rho_mean) & (r > 0) & (rho_mean > 0)
            r_fit  = r[valid]
            yln_fit = np.log(rho_mean[valid])

            # Initial guesses and bounds for fitting parameters
            p0     = [np.log(np.median(np.exp(yln_fit))), 0.18, 0.1]
            bounds = ([-50.0, 0.05, 1e-3], [50.0, 0.6, 0.5])

            def _log_einasto_rho_model(rr, log_rho2, alpha_fit, r_minus2):
                # Convert log rho2 back to linear, evaluate Einasto density, and return log-density
                rho2_lin = np.exp(log_rho2)
                rho_lin = einasto_rho(rr, alpha_fit, r_minus2, rho2_lin)
                # Guard against log of zero
                return np.log(np.maximum(rho_lin, 1e-300))

            popt, _ = curve_fit(_log_einasto_rho_model, r_fit, yln_fit,
                                p0=p0, bounds=bounds, maxfev=50000)
            log_rho2, alpha, r_minus2 = popt

            # Overlay fitted Einasto profile on density plot
            label = rf"$\mathrm{{{sim_names[suite]}}}\;(\alpha={alpha:.3f})$"
            ax_density.loglog(r, np.exp(_log_einasto_rho_model(r, log_rho2, alpha, r_minus2)) * r**2,
                              ls="--", lw=0.9, color=color)

        # Load velocity dispersion profiles for the suite
        vdir = os.path.join(base_dir, "data", suite, "velocity_profiles")
        files_v = sorted(glob(os.path.join(vdir, "halo_*_profile.csv")))
        if not files_v:
            continue

        profs_v   = [pd.read_csv(f) for f in files_v]
        r_v       = profs_v[0]["r_scaled"].to_numpy()
        # Apply radius mask for velocity data
        mask_v = (r_v >= mask_range[0]) & (r_v <= mask_range[1]) if mask_range else slice(None)
        r_v = r_v[mask_v]

        # Stack radial velocity dispersion and anisotropy beta profiles
        sig_r   = np.vstack([p["sigma_rad_scaled"].to_numpy()[mask_v] for p in profs_v])
        beta    = np.vstack([p["beta"].to_numpy()[mask_v]              for p in profs_v])

        # Compute mean and std deviation across halos
        sig_mean, sig_std   = np.nanmean(sig_r, axis=0), np.nanstd(sig_r, axis=0)
        beta_mean, beta_std = np.nanmean(beta, axis=0), np.nanstd(beta, axis=0)

        # Plot mean radial velocity dispersion with shaded std deviation
        ax_sigma.plot(r_v, sig_mean, lw=1, color=color, label=label)
        ax_sigma.fill_between(r_v, sig_mean - sig_std, sig_mean + sig_std,
                            color=color, alpha=0.25, lw=0)

        if einasto_fit:
            # Fit (alpha, c) by matching Jeans-model sigma_r to the mean profile over a fitting range
            alpha_min, alpha_max = 0.1, 0.3
            c_min, c_max = 5.0, 15.0
            fit_mask = (r_v >= 1e-3) & (r_v <= 1.0) & np.isfinite(sig_mean) & np.isfinite(beta_mean)
            r_fit_v = r_v[fit_mask]
            sig_fit = sig_mean[fit_mask]
            beta_fit = beta_mean[fit_mask]

            def _sigma_model_for_fit(rr, alpha_fit, c_fit):
                # Use the same masked beta grid as rr for the fit
                # (curve_fit will pass rr == r_fit_v here)
                return jeans_sigma_r(rr, beta_fit, alpha_fit, c_fit)

            p0_v = [0.18, 10.0]
            bounds_v = ([alpha_min, c_min], [alpha_max, c_max])
            try:
                popt_v, _ = curve_fit(_sigma_model_for_fit, r_fit_v, sig_fit,
                                      p0=p0_v, bounds=bounds_v, maxfev=50000)
                alpha_hat, c_hat = popt_v
                sig_model_full = jeans_sigma_r(r_v, beta_mean, alpha_hat, c_hat)
                ax_sigma.plot(r_v, sig_model_full, ls="--", lw=1.1, color=color)
            except Exception:
                # If the Jeans fit fails, skip overlay for this suite
                pass

    # Add vertical reference line at radius corresponding to 3.38e-3 Rvir
    ax_density.axvline(3.38e-3, ls=":" ,lw=0.9, color='black')
    ax_density.set_xlabel(r"$r / R_{\mathrm{vir}}$", fontsize=18)
    ax_density.set_ylabel(r"$(\rho / \rho_m)\,(r/R_{\mathrm{vir}})^2$", fontsize=18)
    ax_density.set_xscale("log")
    ax_density.set_yscale("log")
    #ax_density.set_xlim(1e-3, 1.5)
    #ax_density.set_ylim(1, 4e2)
    if plot_range is not None:
        ax_density.set_xlim(plot_range)
        ax_sigma.set_xlim(plot_range)
    ax_density.legend(loc="best", fontsize=12, frameon=False)

    ax_sigma.set_xlabel(r"$r / R_{\mathrm{vir}}$", fontsize=18)
    ax_sigma.set_ylabel(r"$\sigma_{\rm rad}/V_{\rm vir}$", fontsize=18)
    ax_sigma.set_xscale("log")
    #ax_sigma.set_xlim(1e-3, 1.5)
    #ax_sigma.set_ylim(0, None)
    ax_sigma.axvline(3.38e-3, ls=":" ,lw=0.9, color='black')
    ax_sigma.legend(loc="best", fontsize=12, frameon=False)

    plt.tight_layout()
    plt.savefig(os.path.join(base_dir, 'figure', "density_velocity_profiles.pdf"))
    plt.close()

def plot_anisotropy(base_dir, suite_names, mask_range=[1e-3,1.5], plot_range=None):
    """
    Plot velocity anisotropy profiles beta(r) for specified simulation suites.

    Parameters:
        base_dir (str): Base directory containing 'data' and 'figure' subdirectories.
        suite_names (list of str): List of simulation suite names to process.
        plot_range (tuple or list, optional): (xmin, xmax) range for the x-axis; if None, use default autoscaling.

    Returns:
        None

    Notes:
        - Loads velocity profile CSV files for each suite.
        - Computes mean and standard deviation of beta across halos.
        - Plots beta(r) = 1 - sigma_tan^2 / (2 sigma_rad^2) on semilog-x scale.
        - Saves figure as 'velocity_anisotropy.pdf' in base_dir/figure.
    """
        
    fig, ax_beta = plt.subplots(1, 1, figsize=(6, 5), dpi=500)
    plt.rcParams['text.usetex'] = True   

    for suite in suite_names:
        # Load velocity profile CSV files for the suite
        vdir = os.path.join(base_dir, "data", suite, "velocity_profiles")
        files = sorted(glob(os.path.join(vdir, "halo_*_profile.csv")))
        if not files:
            continue

        profs = [pd.read_csv(f) for f in files]
        r_all = profs[0]["r_scaled"].to_numpy()
        beta_all = np.vstack([p["beta"].to_numpy() for p in profs])

        mask = (r_all >= mask_range[0]) & (r_all <= mask_range[1]) if mask_range else slice(None)
        r = r_all[mask]
        beta = beta_all[:, mask]

        # Compute mean and standard deviation of beta across halos
        beta_mean, beta_std = np.nanmean(beta, axis=0), np.nanstd(beta, axis=0)

        # Plot mean beta with shaded std deviation
        ax_beta.semilogx(r, beta_mean, lw=1, color=sim_colors[suite], label=rf"$\mathrm{{{sim_names[suite]}}}$")
        ax_beta.fill_between(r, beta_mean - beta_std, beta_mean + beta_std,
                            color=sim_colors[suite], alpha=0.25, lw=0)

    ax_beta.set_xlabel(r"$r/R_{\mathrm{vir}}$", fontsize=18)
    ax_beta.set_ylabel(r"$\beta = 1 - \sigma_{\rm tan}^2 / (2 \sigma_{\rm rad}^2)$", fontsize=18)
    #ax_beta.set_ylim(-0.5, 0.8)
    ax_beta.axhline(0, color="k", ls="--", lw=0.8)  # Reference line at isotropy beta=0
    ax_beta.axvline(3.38e-3, ls=":" ,lw=0.8, color='black')  # Reference radius
    if plot_range is not None:
        ax_beta.set_xlim(plot_range)

    ax_beta.legend(loc="best", fontsize=12, frameon=False)

    plt.tick_params(axis='both', labelsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(base_dir, 'figure', "velocity_anisotropy.pdf"))
    plt.close()
