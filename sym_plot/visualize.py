import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
import os
import symlib

def visual(suite_name, halo_id, input_dir, snapshot,
           grid_res, n_neighbors, slice_thickness,
           output_dir):
    """
    Visualize projected density, radial and total velocity dispersion, and PPSD
    for a given halo in a simulation suite.

    Parameters:
    -----------
    suite_name : str
        Name of the simulation suite.
    halo_id : int
        ID of the target halo.
    input_dir : str
        Path to the directory containing simulation data.
    snapshot : int
        Snapshot number to read particle data from.
    grid_res : int
        Resolution of the 2D grid for visualization.
    n_neighbors : int
        Number of nearest neighbors to use for local estimates.
    slice_thickness : float
        Thickness of the Z-slice (in units of halo virial radius).
    output_dir : str
        Directory where plots will be saved.
    """

    # --- Create output directory ---
    os.makedirs(os.path.join(output_dir, 'visualization', f"{suite_name}", f"{halo_id}"), exist_ok=True)
    save_path = os.path.join(output_dir, 'visualization', f"{suite_name}", f"{halo_id}")

    # --- Load particle & simulation parameters ---
    sim_dir = symlib.get_host_directory(input_dir, suite_name, halo_id)
    params = symlib.simulation_parameters(suite_name)
    h = params['h100']              # Hubble parameter scaling factor
    mp = params['mp'] / h           # Particle mass in proper units
    part = symlib.Particles(sim_dir)
    p = part.read(snapshot)         # Read particle data at specified snapshot

    # --- Load host halo properties from Rockstar catalog ---
    r_data, hist = symlib.read_rockstar(sim_dir)
    host = r_data[0, snapshot]      # Select the first halo (host) at this snapshot
    center = host["x"]              # Halo center position
    v_host = host["v"]              # Bulk halo velocity
    r_vir = host["rvir"]            # Virial radius of the halo

    # --- Convert slice thickness and grid box to physical units ---
    slice_thickness *= r_vir        # Slice thickness in kpc
    box_size = r_vir                # Visualization box extends to virial radius

    # --- Positions & velocities in halo-centric frame ---
    x = p[0]["x"] - center          # Particle positions relative to halo center
    v = p[0]["v"] - v_host          # Particle velocities relative to halo bulk motion

    # --- Apply Z-slice selection ---
    z_mask = np.abs(x[:, 2]) < slice_thickness / 2
    x = x[z_mask]                   # Keep only particles within slice
    v = v[z_mask]                   # Corresponding velocities

    # --- Build KDTree for fast nearest-neighbor queries ---
    tree = cKDTree(x)

    # --- Define 2D grid for visualization ---
    x_edges = np.linspace(-box_size, box_size, grid_res+1)
    y_edges = np.linspace(-box_size, box_size, grid_res+1)
    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])   # Grid cell centers
    y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])

    # --- Allocate arrays to store local quantities ---
    rho_field = np.zeros((grid_res, grid_res))          # Local mass density
    vel_radial_field = np.zeros((grid_res, grid_res))   # Radial velocity dispersion
    vel_total_field = np.zeros((grid_res, grid_res))    # Total velocity dispersion
    ppsd_field = np.zeros((grid_res, grid_res))         # Pseudo-phase-space density (radial)
    ppsd_total_field = np.zeros((grid_res, grid_res))   # PPSD using total velocity

    # --- Compute per-grid quantities ---
    for i, xc in enumerate(x_centers):
        for j, yc in enumerate(y_centers):
            pixel_center = np.array([xc, yc, 0.0])   # 2D pixel center in XY-plane

            # --- Find n_neighbors nearest particles ---
            dists, idxs = tree.query(pixel_center, k=n_neighbors)
            neighbors_pos = x[idxs]
            neighbors_vel = v[idxs]

            # --- Compute local density using median distance to neighbors ---
            R_med = np.median(dists)                      # Median distance
            rho_field[i,j] = n_neighbors * mp / ((4/3) * np.pi * R_med**3)  # Volume-averaged density

            # --- Compute radial velocity dispersion ---
            r_vec = neighbors_pos - pixel_center          # Vector from pixel center to particles
            r_hat = r_vec / np.linalg.norm(r_vec, axis=1)[:, None]  # Unit vector along radius
            v_radial = np.sum(neighbors_vel * r_hat, axis=1)        # Radial component
            vel_radial_field[i,j] = np.std(v_radial)                 # Standard deviation = dispersion

            # --- Compute total velocity dispersion (sigma_x^2 + sigma_y^2 + sigma_z^2)^(1/2) ---
            sigma_x = np.var(neighbors_vel[:,0])
            sigma_y = np.var(neighbors_vel[:,1])
            sigma_z = np.var(neighbors_vel[:,2])
            vel_total_field[i,j] = np.sqrt(sigma_x + sigma_y + sigma_z)

            # --- Avoid zero values ---
            vel_radial_field[i,j] = max(vel_radial_field[i,j], 0.1)
            vel_total_field[i,j] = max(vel_total_field[i,j], 0.1)

    # --- Compute pseudo-phase-space density ---
    ppsd_field = rho_field / vel_radial_field**3
    ppsd_total_field = rho_field / vel_total_field**3

    # --- Function to plot a 2D field ---
    def plot_field(field, cmap, fname, label):
        plt.figure(figsize=(8,7), dpi=500)
        plt.imshow(field.T, origin='lower',
                   extent=[-box_size, box_size, -box_size, box_size],
                   cmap=cmap,
                   vmin=np.nanpercentile(field, 2),
                   vmax=np.nanpercentile(field, 98))
        plt.xlabel("X [kpc]")
        plt.ylabel("Y [kpc]")
        plt.title(label)
        plt.colorbar(label=label)
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, f"{fname}.pdf"))
        plt.close()

    # --- Plot density and velocity fields ---
    plot_field(np.log(rho_field), cmap='inferno', fname='density', label='Density')
    plot_field(vel_radial_field, cmap='coolwarm', fname='radial_velocity_dispersion', label='Radial Velocity Dispersion')
    plot_field(vel_total_field, cmap='coolwarm', fname='total_velocity_dispersion', label='Total Velocity Dispersion')
    plot_field(np.log(ppsd_field), cmap='cividis', fname='ppsd_radial', label='PPSD Radial')
    plot_field(np.log(ppsd_total_field), cmap='cividis', fname='ppsd_total', label='PPSD Total')
