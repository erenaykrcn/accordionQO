from pathlib import Path

import h5py
import numpy as np
import torch

def to_numpy(x):
    """Convert Torch tensors or array-like objects to NumPy arrays."""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def get_next_run_path(
    output_dir,
    temperature,
    prefix="SO_quench",
):
    """
    Return the first unused filename:

        SO_quench_T40_id000.hdf5
        SO_quench_T40_id001.hdf5
        ...
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    run_id = 0

    while True:
        filename = (
            f"{prefix}_T{temperature:g}_id{run_id:03d}.hdf5"
        )
        path = output_dir / filename

        if not path.exists():
            return path, run_id

        run_id += 1


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
    omegar,
    grid_size,
    final_time1,
    final_time2,
    J,
    detuning,
    prefix="SO_quench",
):
    """
    Save the SGPE states, cavity fields, and run parameters in HDF5 format.
    """
    output_path, run_id = get_next_quench_path(
        output_dir=output_dir,
        temperature=temperature,
        N_particles1=N_particles1,
        N_particles2=N_particles2,
        thermalization_time=thermalization_time,
        omegar=omegar,
        grid_size=grid_size,
        J=J,
        detuning=detuning,
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
        h5file.attrs["omegar"] = omegar
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
    N_particles1,
    N_particles2,
    thermalization_time,
    omegar,
    grid_size,
    J,
    detuning,
    prefix="SO_quench",
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    run_id = 0

    while True:
        filename = (
            f"{prefix}"
            f"_T{temperature:g}"
            f"_N1{N_particles1:g}"
            f"_N2{N_particles2:g}"
            f"_tth{thermalization_time:g}"
            f"_wr{omegar:g}"
            f"_L{grid_size:g}"
            f"_J{J:g}"
            f"_D{detuning:g}"
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