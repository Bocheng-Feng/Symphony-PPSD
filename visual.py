import numpy as np
import os
import symlib
from astropy import units as u
from astropy.constants import G
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize

def field_map(suite_name, halo_id, input_dir, snapshot, n_neighbors, output_dir, gridsize=1000):

    # --- Create output directory ---
    index = symlib.get_host_directory(input_dir, suite_name, halo_id)[-3:]
    os.makedirs(os.path.join(output_dir, f"{suite_name}", f"{index}"), exist_ok=True)
    save_path = os.path.join(output_dir, f"{suite_name}", f"{index}")
    print(f"Visualizing {suite_name}_{index}")
    
    # --- Load particle & simulation parameters ---
    sim_dir = symlib.get_host_directory(input_dir, suite_name, halo_id)
    params = symlib.simulation_parameters(suite_name)
    h = params['h100']            
    mp = params['mp'] / h # particle mass [Msun]          
    part = symlib.Particles(sim_dir)
    p = part.read(snapshot)        

    # --- Load host halo properties from Rockstar catalog ---
    r_data, hist = symlib.read_rockstar(sim_dir)
    host = r_data[0,snapshot]     
    center = host["x"]            
    v_host = host["v"]  # [km/s]         
    r_vir = host["rvir"]  # [kpc]         
    Om0 = params["Om0"] # Matter density parameter at redshift zero

    # --- Slice thickness & box size ---
    slice_thickness = 0.2 * r_vir       
    box_size = 1 * r_vir                

    # --- Positions & velocities in halo-centric frame ---
    x = p[0]["x"] - center         
    v = p[0]["v"] - v_host        

    # --- Apply Z-slice selection ---
    z_mask = np.abs(x[:, 2]) < slice_thickness / 2
    x = x[z_mask]                   
    v = v[z_mask]                   

    # --- Build KDTree & compute local quantities ---
    tree = cKDTree(x)
    dist, idx = tree.query(x, k=n_neighbors)

    rho = np.zeros(len(x))
    sigma_r = np.zeros(len(x))
    sigma_tot = np.zeros(len(x))

    for i in range(len(x)):
        neighbors = idx[i]
        v_neighbors = v[neighbors]

        # local density
        rho[i] = (n_neighbors-1) * mp / ((4/3) * np.pi * dist[i, -1]**3)
        # velocity dispersion
        sigma_tot[i] = np.sqrt(np.var(v_neighbors[:,0]) + np.var(v_neighbors[:,1]) + np.var(v_neighbors[:,2]))
        r_hat = x[i] / np.linalg.norm(x[i])
        v_radial = np.dot(v_neighbors, r_hat)
        sigma_r[i] = np.std(v_radial)

    # --- Normalize ---
    m = host["m"] * u.Msun
    r_kpc = r_vir * u.kpc
    v_vir = np.sqrt(G * m / r_kpc).to(u.km/u.s).value

    H0_si = params["H0"] * u.km / u.s / u.Mpc
    rho_crit = (3 * H0_si**2 / (8 * np.pi * G)).to(u.Msun / u.kpc**3).value
    rho_m = Om0 * rho_crit  
    
    rho /= rho_m
    sigma_r /= v_vir
    sigma_tot /= v_vir

    # --- PPSD ---
    Q_r = rho / sigma_r**3
    Q_tot = rho / sigma_tot**3

    # --- Function to plot using hexbin ---
    def plot_hexbin(field, label, cmap='magma', logscale=True):
        plt.figure(figsize=(8,7), dpi=500)
        if logscale:
            norm = LogNorm(vmin=np.nanpercentile(field, 2), vmax=np.nanpercentile(field, 98))
        else:
            norm = Normalize(vmin=np.nanpercentile(field, 2), vmax=np.nanpercentile(field, 98))

        x_plot, y_plot = x[:, 0], x[:, 1]
        mask = (np.abs(x_plot) < box_size) & (np.abs(y_plot) < box_size)
        x_plot = x_plot[mask]
        y_plot= y_plot[mask]
        field = field[mask]

        hb = plt.hexbin(x_plot, y_plot, C=field, gridsize=gridsize, cmap=cmap, reduce_C_function=np.mean, norm=norm, alpha=0.8)
        plt.colorbar(hb, label=label)
        plt.xlabel(r"$X [\mathrm{kpc}]$")
        plt.ylabel(r"$Y [\mathrm{kpc}]$")
        halo_name = f"{suite_name} {index}"
        plt.title(halo_name, fontsize=16)
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, f"{label}.png"))
        plt.close()

    # --- Plot all fields ---
    plot_hexbin(rho, 'Density', cmap='viridis', logscale=True)
    plot_hexbin(sigma_tot**2, 'Temperature', cmap='coolwarm', logscale=False)
    plot_hexbin(sigma_r, 'Radial Velocity Dispersion', cmap='coolwarm', logscale=False)
    plot_hexbin(Q_r, 'Radial PPSD', cmap='magma', logscale=True)
    plot_hexbin(Q_tot, 'Total PPSD', cmap='magma', logscale=True)
