import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import seaborn as sns

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

def plot_normalized_ppsd(base_dir, suite_names, plot_range=None):
    """
    Plot normalized pseudo phase-space density profiles.

    Parameters:
    base_dir (str): Base directory containing data and where to save figures.
    suite_names (list): List of suite names to process.
    plot_range (tuple, optional): (xmin, xmax) range for the x-axis; only data within this range will be plotted.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=500)
    plt.rcParams['text.usetex'] = True

    all_data = {suite: [] for suite in suite_names}
    r = None
    m = None
    convergence_radius = 3.38e-3
    convergence_mass = 1e-3

    def interpolate_curves(x_target, x_list, y_list):
        return np.array([
            np.interp(x_target, x, y, left=np.nan, right=np.nan)
            for x, y in zip(x_list, y_list)
        ])

    for suite in suite_names:
        input_dir = os.path.join(base_dir, 'data', suite, "ppsd_profiles")

        files = sorted([f for f in os.listdir(input_dir) if f.endswith(".csv")])
        if not files:
            continue

        # Load profile axis from the first file
        r = pd.read_csv(os.path.join(input_dir, files[0]))["r_scaled"].values
        m = pd.read_csv(os.path.join(input_dir, files[0]))["m_scaled"].values

        Qr = np.array([pd.read_csv(os.path.join(input_dir, f))["Q_r"].values for f in files])
        Qtot = np.array([pd.read_csv(os.path.join(input_dir, f))["Q_tot"].values for f in files])
        all_data[suite] = [Qr, Qtot, r, m]

        m_list = [pd.read_csv(os.path.join(input_dir, f))["m_scaled"].values for f in files]
        Qr_interp_m = interpolate_curves(m, m_list, Qr)

        # Mask data by plot_range for r
        if plot_range is not None:
            mask_r = (r >= plot_range[0]) & (r <= plot_range[1])
            r_plot = r[mask_r]
            Qr_plot = Qr[:, mask_r]
            mean_Qr = np.nanmean(Qr_plot, axis=0)
            std_Qr = np.nanstd(Qr_plot, axis=0)
        else:
            r_plot = r
            mean_Qr = np.nanmean(Qr, axis=0)
            std_Qr = np.nanstd(Qr, axis=0)

        axes[0].plot(r_plot, mean_Qr, color=sim_colors[suite], lw=1.5, label = rf"$\mathrm{{{sim_names[suite]}}}$")
        axes[0].fill_between(r_plot, mean_Qr - std_Qr, mean_Qr + std_Qr, color=sim_colors[suite], alpha=0.2)

        # Mask data by plot_range for m
        if plot_range is not None:
            mask_m = (m >= plot_range[0]) & (m <= plot_range[1])
            m_plot = m[mask_m]
            Qr_interp_m_plot = Qr_interp_m[:, mask_m]
            mean_Qr_m = np.nanmean(Qr_interp_m_plot, axis=0)
            std_Qr_m = np.nanstd(Qr_interp_m_plot, axis=0)
        else:
            m_plot = m
            mean_Qr_m = np.nanmean(Qr_interp_m, axis=0)
            std_Qr_m = np.nanstd(Qr_interp_m, axis=0)

        axes[1].plot(m_plot, mean_Qr_m, color=sim_colors[suite], lw=1.5, label=rf"$\mathrm{{{sim_names[suite]}}}$")
        axes[1].fill_between(m_plot, mean_Qr_m - std_Qr_m, mean_Qr_m + std_Qr_m, color=sim_colors[suite], alpha=0.3)

    #axes[0].set_xlim(2e-3, 1.5)
    #axes[0].set_ylim(1e4, 1e10)
    axes[0].set_xlabel(r"$r / R_{\mathrm{vir}}$",fontsize=18)
    axes[0].set_ylabel(r"$Q_r=\rho/\sigma_{\mathrm{r}}^3$",fontsize=18)
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].grid(True, which="major", linestyle=":")
    axes[0].axvline(convergence_radius, ls=":" ,lw=1.3, color='black')
    axes[0].legend(fontsize=15, loc="best", frameon=True)

    #axes[1].set_xlim(3e-4, 1.5)
    #axes[1].set_ylim(1e4, 1e10)
    axes[1].set_xlabel(r"$M(<r) / M_{\mathrm{vir}}$",fontsize=18)
    axes[1].set_ylabel(r"$Q_r=\rho/\sigma_{\mathrm{r}}^3$",fontsize=18)
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].grid(True, which="major", linestyle=":")
    axes[1].axvline(convergence_mass, ls=":", lw=1.3, color='black')

    if plot_range is not None:
        axes[0].set_xlim(plot_range)
        axes[1].set_xlim(plot_range)

    fig.tight_layout(rect=[0, 0, 1, 1])
    save_path = os.path.join(base_dir, 'figure', "normalized_ppsd_profiles.pdf")
    axes[0].tick_params(axis='both', labelsize=13)
    axes[1].tick_params(axis='both', labelsize=13)
    axes[0].xaxis.label.set_size(18)
    axes[0].yaxis.label.set_size(18)
    axes[1].xaxis.label.set_size(18)
    axes[1].yaxis.label.set_size(18)
    plt.savefig(save_path, format="pdf")
    plt.close(fig)

def plot_ppsd_slope(base_dir, suite_names, plot_range=None):
    """
    Plot slope of pseudo phase-space density profiles.

    Parameters:
    base_dir (str): Base directory containing data and where to save figures.
    suite_names (list): List of suite names to process.
    plot_range (tuple, optional): (xmin, xmax) range for the x-axis; only data within this range will be plotted.
    """
    fig, ax = plt.subplots(1, 1, figsize=(7, 5), dpi=500)
    plt.rcParams['text.usetex'] = True

    r = None
    convergence_radius = 3.38e-3

    for suite in suite_names:
        radius_dir = os.path.join(base_dir, "data", suite, "ppsd_slope_profiles_r")

        files = sorted([f for f in os.listdir(radius_dir) if f.endswith(".csv")])
        if not files:
            continue

        # Load profile axis from the first file
        r = pd.read_csv(os.path.join(radius_dir, files[0]))["r_scaled"].values

        slope_Qr = np.array([pd.read_csv(os.path.join(radius_dir, f))["slope_Q_r"].values for f in files])
        Qtot = np.array([pd.read_csv(os.path.join(radius_dir, f))["slope_Q_tot"].values for f in files])

        # Mask data by plot_range for r
        if plot_range is not None:
            mask_r = (r >= plot_range[0]) & (r <= plot_range[1])
            r_plot = r[mask_r]
            slope_Qr_plot = slope_Qr[:, mask_r]
            slope_mean_Qr = np.nanmean(slope_Qr_plot, axis=0)
        else:
            r_plot = r
            slope_mean_Qr = np.nanmean(slope_Qr, axis=0)

        std_Qr = np.nanstd(slope_mean_Qr, axis=0)
        ax.plot(r_plot, slope_mean_Qr, color=sim_colors[suite], lw=1.5, label = rf"$\mathrm{{{sim_names[suite]}}}$")
        #ax.fill_between(r_plot, slope_mean_Qr - std_Qr, slope_mean_Qr + std_Qr, color=sim_colors[suite], alpha=0.3)

    #ax.set_xlim(2e-3, 1.5)
    #ax.set_ylim(-2.3, -1.2)
    ax.set_xlabel(r"$r / R_{\mathrm{vir}}$",fontsize=18)
    ax.set_ylabel(r"$d\log Q_r / d\log r$",fontsize=18)
    ax.set_xscale("log")
    ax.set_yscale("linear")
    ax.grid(True, which="major", linestyle=":")
    ax.axvline(convergence_radius, ls=":" ,lw=1.3, color='black')
    ax.legend(fontsize=11, loc="best", frameon=True)
    ax.axhline(-1.875, ls="--", color="black", lw=1, label=r"$\frac{d\log Q_r}{d\log r}=-1.875$")

    if plot_range is not None:
        ax.set_xlim(plot_range)

    fig.tight_layout()
    save_path = os.path.join(base_dir, 'figure', "ppsd_slope_profiles.pdf")
    plt.tick_params(axis='both', labelsize=13)
    plt.savefig(save_path, format="pdf")
    plt.close(fig)
