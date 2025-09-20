import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
import os
import symlib
from scipy.stats import binned_statistic_2d
from scipy.ndimage import uniform_filter
from astropy.stats import sigma_clip

def field_map(suite_name, halo_id, input_dir, snapshot, grid_res, n_neighbors, output_dir):

    # --- Create output directory ---
    index = symlib.get_host_directory(input_dir, suite_name, halo_id)[-3:]
    os.makedirs(os.path.join(output_dir, f"{suite_name}", f"{index}"), exist_ok=True)
    save_path = os.path.join(output_dir, f"{suite_name}", f"{index}")

    # --- Load particle & simulation parameters ---
    sim_dir = symlib.get_host_directory(input_dir, suite_name, halo_id)
    params = symlib.simulation_parameters(suite_name)
    h = params['h100']            
    mp = params['mp'] / h           
    part = symlib.Particles(sim_dir)
    p = part.read(snapshot)        

    # --- Load host halo properties from Rockstar catalog ---
    r_data, hist = symlib.read_rockstar(sim_dir)
    host = r_data[0,snapshot]     
    center = host["x"]            
    v_host = host["v"]              
    r_vir = host["rvir"]           

    # --- Convert slice thickness and grid box to physical units ---
    slice_thickness = 0.5 * r_vir       
    box_size = r_vir                

    # --- Positions & velocities in halo-centric frame ---
    x = p[0]["x"] - center         
    v = p[0]["v"] - v_host        

    # --- Apply Z-slice selection ---
    z_mask = np.abs(x[:, 2]) < slice_thickness / 2
    x = x[z_mask]                   
    v = v[z_mask]                   

    # --- Build KDTree for fast nearest-neighbor queries ---
    tree = cKDTree(x)
    dist, idx = tree.query(x, k=n_neighbors)

    # --- Compute local quantities per particle
    rho = np.zeros(len(x))    # local density
    sigma_r = np.zeros(len(x))  # local radial velocity dispersion
    sigma_tot = np.zeros(len(x)) # local radial velocity dispersion
    for i in range(len(x)):
        neighbors = idx[i]
        v_neighbors = v[neighbors]

        # local density estimate 
        rho[i] = n_neighbors * mp / ((4/3) * np.pi * dist[i, -1]**3)

        # velocity dispersion
        sigma_tot[i] = np.sqrt(np.var(v_neighbors[:,0]) + np.var(v_neighbors[:,1]) + np.var(v_neighbors[:,2]))
        if sigma_tot[i] < 0:
            sigma_tot[i] = np.nan

        r_hat = x[i] / np.linalg.norm(x[i])
        v_radial = np.dot(v_neighbors, r_hat)
        sigma_r[i] = np.std(v_radial)
        if sigma_r[i] < 0:
            sigma_r[i] = np.nan

    # --- PPSD definition ---
    Q_r = rho / sigma_r**3
    Q_tot = rho / sigma_tot**3  
    x_plot, y_plot = x[:, 0], x[:, 1]
    mask = (np.abs(x_plot) < box_size) & (np.abs(y_plot) < box_size)

    # --- Bin into 2D image grid ---
    Q_r_field, xedges, yedges, _ = binned_statistic_2d(
        x_plot[mask], y_plot[mask], Q_r[mask],
        statistic="mean", bins=grid_res
    )
    Q_tot_field, xedges, yedges, _ = binned_statistic_2d(
        x_plot[mask], y_plot[mask], Q_tot[mask],
        statistic="mean", bins=grid_res
    )
    rho_field, xedges, yedges, _ = binned_statistic_2d(
        x_plot[mask], y_plot[mask], rho[mask],
        statistic="mean", bins=grid_res
    )
    temp_field, xedges, yedges, _ = binned_statistic_2d(
        x_plot[mask], y_plot[mask], (sigma_tot**2)[mask],
        statistic="mean", bins=grid_res
    )
    sigma_r_field, xedges, yedges, _ = binned_statistic_2d(
        x_plot[mask], y_plot[mask], sigma_r[mask],
        statistic="mean", bins=grid_res
    )

    # --- Smooth grids ---
    # Q_r_field = uniform_filter(np.nan_to_num(Q_r_field, nan=0), size=5)
    # Q_tot_field = uniform_filter(np.nan_to_num(Q_tot_field, nan=0), size=5)
    # rho_field = uniform_filter(np.nan_to_num(rho_field, nan=0), size=5)
    # temp_field = uniform_filter(np.nan_to_num(temp_field, nan=0), size=5)
    # sigma_r_field = uniform_filter(np.nan_to_num(sigma_r_field, nan=0), size=5)

    # --- Sigma clip ---
    # Q_r_field = sigma_clip(Q_r_field, sigma=5, maxiters=10)
    # Q_tot_field = sigma_clip(Q_tot_field, sigma=5, maxiters=10)
    # rho_field = sigma_clip(rho_field, sigma=5, maxiters=10)
    # temp_field = sigma_clip(temp_field, sigma=5, maxiters=10)
    # sigma_r_field = sigma_clip(sigma_r_field, sigma=5, maxiters=10)

    # --- Function to plot a 2D field ---
    def plot_field(field, cmap, label, scale):
        plt.rcParams["text.usetex"] = False
        plt.figure(figsize=(8,7), dpi=500)

        # Apply log scale if requested
        if scale == 'log':
            field = np.where(field > 0, field, np.nan)
            field = np.log(field)

        plt.imshow(
            field.T, origin='lower',
            extent=[-box_size, box_size, -box_size, box_size],
            cmap=cmap,
            vmin=np.nanpercentile(field, 2),
            vmax=np.nanpercentile(field, 98)
        )
        # --- Construct halo name string ---
        index = symlib.get_host_directory(input_dir, suite_name, halo_id)[-3:]
        halo_name = f"{suite_name} {index}"

        plt.xlabel(r"$X [\mathrm{kpc}]$")
        plt.ylabel(r"$Y [\mathrm{kpc}]$")
        plt.title(rf"{halo_name}", fontsize=16)

        # --- Colorbar with unit ---
        cbar = plt.colorbar()
        print(f'Visualizing {halo_name}')
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, f"{label}.pdf"))
        plt.close()

    # --- Plot density and velocity fields ---
    plot_field(rho_field, cmap='magma', label='Density', scale='log')
    plot_field(temp_field, cmap='coolwarm', label='Temperature', scale='linear')
    plot_field(sigma_r_field, cmap='coolwarm', label='Radial Velocity Dispersion',scale='linear')
    plot_field(Q_r_field, cmap='magma', label='Radial PPSD', scale='log')
    plot_field(Q_tot_field, cmap='magma', label='Total PPSD', scale='log')
