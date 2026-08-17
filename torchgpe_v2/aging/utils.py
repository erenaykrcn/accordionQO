from pathlib import Path

import h5py
import numpy as np
import torch
import matplotlib.pyplot as plt

def to_numpy(x):
    """Convert Torch tensors or array-like objects to NumPy arrays."""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


    
def save_quench_run(
    output_dir,
    temperature,
    result1,
    result2,
    cavity_monitor1,
    cavity_monitor2,
    *,
    N_particles1,
    N_particles2,
    gamma,
    dt,
    thermalization_time,
    grid_size,
    final_time1,
    final_time2,
    J,
    detuning,
    VP,
    a_s,
    prefix,
):
    """
    Save the SGPE states, cavity fields, and run parameters in HDF5 format.
    """
    output_path, run_id = get_next_quench_path(
        output_dir=output_dir,
        temperature=temperature,
        gamma=gamma,
        N_particles1=N_particles1,
        N_particles2=N_particles2,
        thermalization_time=thermalization_time,
        grid_size=grid_size,
        J=J,
        detuning=detuning,
        VP=VP,
        a_s=a_s,
        prefix=prefix,
    )

    states1 = to_numpy(result1["states"])
    states2 = to_numpy(result2["states"])

    if "alpha" not in result1:
        raise KeyError(
            "result1 does not contain 'alpha'. "
            "Check monitor_cavity and monitor_every in propagate_bilayer_sgpe."
        )

    if "alpha" not in result2:
        raise KeyError(
            "result2 does not contain 'alpha'. "
            "Check monitor_cavity and monitor_every in propagate_bilayer_sgpe."
        )

    alpha1 = to_numpy(result1["alpha"])
    alpha2 = to_numpy(result2["alpha"])

    with h5py.File(output_path, "x") as h5file:
        # "x" creates a new file and raises an error if it already exists,
        # providing an additional safeguard against overwriting.

        states_group = h5file.create_group("states")

        states_group.create_dataset(
            "pump_ramp",
            data=states1,
            compression="gzip",
            compression_opts=4,
            shuffle=True,
        )

        states_group.create_dataset(
            "post_quench",
            data=states2,
            compression="gzip",
            compression_opts=4,
            shuffle=True,
        )

        cavity_group = h5file.create_group("cavity")

        cavity_group.create_dataset(
            "alpha_pump_ramp",
            data=alpha1,
            compression="gzip",
            compression_opts=4,
            shuffle=True,
        )

        cavity_group.create_dataset(
            "alpha_post_quench",
            data=alpha2,
            compression="gzip",
            compression_opts=4,
            shuffle=True,
        )

        # Store scalar run parameters as file attributes.
        h5file.attrs["run_id"] = run_id
        h5file.attrs["temperature"] = temperature
        h5file.attrs["N_particles1"] = N_particles1
        h5file.attrs["N_particles2"] = N_particles2
        h5file.attrs["gamma"] = gamma
        h5file.attrs["dt"] = dt
        h5file.attrs["thermalization_time"] = thermalization_time
        h5file.attrs["grid_size"] = grid_size
        h5file.attrs["final_time1"] = final_time1
        h5file.attrs["final_time2"] = final_time2
        h5file.attrs["J"] = J
        h5file.attrs["detuning"] = detuning

    print(f"Saved run {run_id} to: {output_path}")

    return output_path



def get_next_state_path(
    output_dir,
    temperature,
    thermalization_time,
    omegar,
    grid_size,
    N_particles,
    prefix="thermal_state",
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    run_id = 0

    while True:
        filename = (
            f"{prefix}"
            f"_T{temperature:g}"
            f"_tth{thermalization_time:g}"
            f"_wr{omegar:g}"
            f"_L{grid_size:g}"
            f"_N{N_particles:g}"
            f"_id{run_id:03d}.hdf5"
        )

        path = output_dir / filename

        if not path.exists():
            return path, run_id

        run_id += 1

def get_next_quench_path(
    output_dir,
    temperature,
    gamma,
    N_particles1,
    N_particles2,
    thermalization_time,
    grid_size,
    J,
    detuning,
    VP,
    a_s,
    prefix="SO_quench",
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    run_id = 0

    while True:
        filename = (
            f"{prefix}"
            f"_T{temperature:g}"
            f"_g{gamma:g}"
            f"_N1{N_particles1:g}"
            f"_N2{N_particles2:g}"
            f"_tth{thermalization_time:g}"
            f"_L{grid_size:g}"
            f"_J{J:g}"
            f"_D{detuning:g}"
            f"_VP{VP:g}"
            f"_a_s{a_s:g}"
            f"_id{run_id:03d}.hdf5"
        )

        path = output_dir / filename

        if not path.exists():
            return path, run_id

        run_id += 1


def save_state(
    output_dir,
    temperature,
    state,
    *,
    thermalization_time,
    omegar,
    grid_size,
    N_particles,
    prefix="thermal_state",
):
    output_path, run_id = get_next_state_path(
        output_dir=output_dir,
        temperature=temperature,
        thermalization_time=thermalization_time,
        omegar=omegar,
        grid_size=grid_size,
        N_particles=N_particles,
        prefix=prefix,
    )

    state = to_numpy(state)

    with h5py.File(output_path, "x") as h5file:
        h5file.create_dataset(
            "state",
            data=state,
            compression="gzip",
            compression_opts=4,
            shuffle=True,
        )

        h5file.attrs["run_id"] = run_id
        h5file.attrs["temperature"] = temperature
        h5file.attrs["thermalization_time"] = thermalization_time
        h5file.attrs["omegar"] = omegar
        h5file.attrs["grid_size"] = grid_size
        h5file.attrs["N_particles"] = N_particles

    print(f"Saved run {run_id} to: {output_path}")

    return output_path



def plot_psi(psi, x_um, y_um,
    density_mask=None,
    density_threshold=0.05, ):
    X_um, Y_um = np.meshgrid(x_um, y_um, indexing="ij")
    
    dens_f = (torch.abs(psi) ** 2).detach().cpu().numpy()
    phase_f = torch.angle(psi).detach().cpu().numpy()
    
    extent = [x_um.min(), x_um.max(), y_um.min(), y_um.max()]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    im2 = axes[0].imshow(dens_f, origin="lower", extent=extent, aspect="equal")
    axes[0].set_title("Relaxed density")
    axes[0].set_xlabel("x (µm)")
    axes[0].set_ylabel("y (µm)")
    axes[0].grid(False)
    plt.colorbar(im2, ax=axes[0])
    
    im3 = axes[1].imshow(phase_f, origin="lower", extent=extent, aspect="equal")
    axes[1].set_title("Relaxed phase")
    axes[1].set_xlabel("x (µm)")
    axes[1].set_ylabel("y (µm)")
    plt.colorbar(im3, ax=axes[1])
    plt.grid(False)
    plt.tight_layout()
    plt.show()
    
    
    psi_np = psi.detach().cpu().numpy()
    phase = np.angle(psi_np)
    density = np.abs(psi_np)**2
    
    vort, antiv = detect_vortices_masked(
        psi,
        density_mask=density_mask,
        density_threshold=density_threshold
    )
    
    # Convert plaquette-center indices [row, col] to physical coordinates
    x_v = np.interp(vort[:, 1], np.arange(len(x_um)), x_um)
    y_v = np.interp(vort[:, 0], np.arange(len(y_um)), y_um)
    
    x_av = np.interp(antiv[:, 1], np.arange(len(x_um)), x_um)
    y_av = np.interp(antiv[:, 0], np.arange(len(y_um)), y_um)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    im0 = axes[0].imshow(
        density,
        origin="lower",
        extent=extent,
        aspect="equal",
    )
    axes[0].scatter(
        x_v, y_v,
        s=12,
        facecolors="none",
        edgecolors="red",
        label=f"vortex ({len(vort)})",
    )
    axes[0].scatter(
        x_av, y_av,
        s=12,
        marker="x",
        color="yellow",
        label=f"antivortex ({len(antiv)})",
    )
    axes[0].set_title("Density with vortices")
    axes[0].set_xlabel("x (µm)")
    axes[0].set_ylabel("y (µm)")
    axes[0].legend()
    
    im1 = axes[1].imshow(phase_f, origin="lower", extent=extent, aspect="equal")
    
    axes[1].scatter(
        x_v, y_v,
        s=12,
        facecolors="none",
        edgecolors="red",
    )
    axes[1].scatter(
        x_av, y_av,
        s=12,
        marker="x",
        color="yellow",
    )
    axes[1].set_title("Phase with vortices")
    axes[1].set_xlabel("x (µm)")
    axes[1].set_ylabel("y (µm)")
    axes[0].grid(False)
    axes[1].grid(False)
    plt.colorbar(im0, ax=axes[0])
    plt.colorbar(im1, ax=axes[1], label="Phase")
    plt.tight_layout()
    plt.show()
    

def phase_wrap(x):
    return (x + np.pi) % (2 * np.pi) - np.pi


def detect_vortices_from_phase(phase):
    # Plaquettes only; no periodic wrap across boundaries
    th00 = phase[:-1, :-1]
    th10 = phase[1:, :-1]
    th11 = phase[1:, 1:]
    th01 = phase[:-1, 1:]

    winding = (
        phase_wrap(th10 - th00)
        + phase_wrap(th11 - th10)
        + phase_wrap(th01 - th11)
        + phase_wrap(th00 - th01)
    ) / (2 * np.pi)

    winding_int = np.rint(winding).astype(int)

    vort = np.argwhere(winding_int == 1)
    antiv = np.argwhere(winding_int == -1)

    # Plaquette-center coordinates in array-index units
    return vort + 0.5, antiv + 0.5


def detect_vortices_masked(
    psi,
    density_mask=None,
    density_threshold=0.05,
):

    if torch.is_tensor(psi):
        psi = psi.detach().cpu().numpy()
    else:
        psi = np.asarray(psi)

    density = np.abs(psi)**2
    phase = np.angle(psi)

    # ------------------------------------------------------------
    # Construct density mask if none was supplied
    # ------------------------------------------------------------
    if density_mask is None:
        density_mask = density > density_threshold * density.max()
    else:
        density_mask = np.asarray(density_mask, dtype=bool)

        if density_mask.shape != density.shape:
            raise ValueError(
                f"density_mask shape {density_mask.shape} "
                f"does not match psi shape {density.shape}"
            )

    # ------------------------------------------------------------
    # Convert site mask -> plaquette mask
    #
    # Require all four corners of the plaquette to lie inside
    # the accepted density region.
    # ------------------------------------------------------------
    plaquette_mask = (
        density_mask[:-1, :-1]
        & density_mask[1:, :-1]
        & density_mask[1:, 1:]
        & density_mask[:-1, 1:]
    )

    # ------------------------------------------------------------
    # Detect phase winding
    # ------------------------------------------------------------
    vort, antiv = detect_vortices_from_phase(phase)

    # argwhere() coordinates correspond directly to the
    # lower-left index of each plaquette
    vort_i = np.floor(vort).astype(int)
    antiv_i = np.floor(antiv).astype(int)

    if len(vort):
        vort = vort[plaquette_mask[vort_i[:, 0], vort_i[:, 1]]]

    if len(antiv):
        antiv = antiv[plaquette_mask[antiv_i[:, 0], antiv_i[:, 1]]]

    return vort, antiv