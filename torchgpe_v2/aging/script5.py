import argparse
from pathlib import Path
import re
import h5py
import torch
import numpy as np


# ============================================================
# Argument parsing
# ============================================================

def str2bool(v):
    if isinstance(v, bool):
        return v

    if v.lower() in ("yes", "true", "t", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "0"):
        return False

    raise argparse.ArgumentTypeError(
        "Expected a boolean value."
    )


parser = argparse.ArgumentParser()

parser.add_argument(
    "--temperature",
    type=float,
    required=True
)

parser.add_argument(
    "--N_particles1",
    type=int,
    required=True
)

parser.add_argument(
    "--N_particles2",
    type=int,
    required=True
)

parser.add_argument(
    "--grid_size",
    type=float,
    default=20e-6
)

parser.add_argument(
    "--thermalization_time",
    type=float,
    required=True
)

parser.add_argument(
    "--gamma",
    type=float,
    required=True
)

parser.add_argument(
    "--VP",
    type=float,
    required=True
)

parser.add_argument(
    "--enable_temperature",
    type=str2bool,
    default=True
)

parser.add_argument(
    "--final_time",
    type=float,
    default=60e-3
)

parser.add_argument(
    "--T_ramp_trap",
    type=float,
    default=4e-3
)

parser.add_argument(
    "--T_ramp_TP",
    type=float,
    default=8e-3
)

parser.add_argument(
    "--t_delay_trap_ramp",
    type=float,
    default=0
)

parser.add_argument(
    "--box_length",
    type=float,
    default=15e-6
)

parser.add_argument(
    "--box_trap",
    type=str2bool,
    required=True
)

parser.add_argument(
    "--final_length",
    type=float,
    default=12.5e-6
)

# ------------------------------------------------------------
# Optional pre-loaded result
# ------------------------------------------------------------

parser.add_argument(
    "--result_path",
    type=str,
    default=None,
    help=(
        "Path relative to ./results/ containing a previous result "
        "file. If provided, the last state in states/pump_ramp "
        "is used as the initial state instead of generating a "
        "thermal/BEC state."
    )
)

args = parser.parse_args()


# ============================================================
# Imports
# ============================================================

from thermal import (
    get_thermal_state,
    make_bilayer,
    estimate_mu,
    get_BEC,
    BoxTrap,
)

from SGPE_SO import get_SO_SGPE_state

from torchgpe.utils import parse_config

from utils import (
    save_quench_run,
    save_state,
)

from torchgpe.bec2D.potentials import (
    Contact,
    DispersiveCavity,
    Trap,
)

from torchgpe.bec2D.callbacks import CavityMonitor

import os
import psutil

_process = psutil.Process(os.getpid())

import sys
sys.path.append("../../")

from torchgpe_v2.bec2D.bilayer_v5 import (
    BilayerGas,
    propagate_bilayer,
    propagate_bilayer_sgpe,
    make_momentum_projector,
)


# ============================================================
# Thermal-state cache
# ============================================================

def get_or_make_thermal_state(
    temperature,
    N_particles,
    gamma,
    thermalization_time,
    grid_size,
    trap,
    cavity_monitor,
    seed,
    imaginary_steps=500,
    cache_dir="thermal_states",
):

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # Make a filename encoding the important preparation parameters.
    cache_path = cache_dir / (
        f"thermal_"
        f"T{temperature:g}_"
        f"N{N_particles}_"
        f"gamma{gamma:g}_"
        f"t{thermalization_time:g}_"
        f"grid{grid_size:g}_"
        f"L{box_length:g}.pt"
    )

    if cache_path.exists():

        print(
            f"[THERMAL] Loading cached state: {cache_path}",
            flush=True
        )

        # Load onto CPU first; make_bilayer can then put/use
        # it as appropriate.
        state = torch.load(
            cache_path,
            map_location="cpu"
        )

    else:

        print(
            f"[THERMAL] No cached state found.",
            flush=True
        )

        print(
            f"[THERMAL] Generating: {cache_path}",
            flush=True
        )

        result = get_thermal_state(
            temperature,
            thermalization_time=thermalization_time,
            grid_size=grid_size,
            N_particles=N_particles,
            monitor_cavity=cavity_monitor,
            monitor_every=1000,
            gamma=gamma,
            contact_as=100,
            trap=trap,
            seed=seed,
            imaginary_steps=imaginary_steps,
            J=0,
        )

        state = result["states"][-1]

        # Save CPU copy so the cache isn't tied to a
        # particular GPU/device.
        torch.save(
            state.detach().cpu(),
            cache_path
        )

        print(
            f"[THERMAL] Saved state: {cache_path}",
            flush=True
        )

    return state


# ============================================================
# Memory
# ============================================================

def print_mem(label):

    rss = (
        _process.memory_info().rss
        / 1024**3
    )

    print(
        f"[MEM] {label}: {rss:.3f} GB",
        flush=True
    )


# ============================================================
# Trap / lattice ramps
# ============================================================

def omega_of_t(
    t,
    omega_initial,
    omega_final,
    T_ramp,
    t_delay=0
):

    if t is None:
        t = 0.0

    if t <= t_delay:
        return omega_initial

    if t >= t_delay + T_ramp:
        return omega_final

    x = (
        t - t_delay
    ) / T_ramp

    s = (
        10*x**3
        - 15*x**4
        + 6*x**5
    )

    diff = (
        omega_final
        - omega_initial
    )

    return (
        diff * s
        + omega_initial
    )


def L_of_t(
    t,
    L_initial,
    L_final,
    T_ramp,
    t_delay=0
):

    if t is None:
        t = 0.0

    if t < t_delay:
        return L_initial

    if t < T_ramp + t_delay:

        return (
            L_initial
            + (L_final - L_initial)
            * (t - t_delay)
            / T_ramp
        )

    return L_final


def lattice_ramp(
    t,
    T_ramp=8e-3,
    t_delay=0,
    VP=4
):

    if t <= t_delay:
        return 0.0

    if t >= t_delay + T_ramp:
        return VP

    x = (
        t - t_delay
    ) / T_ramp

    s = (
        10*x**3
        - 15*x**4
        + 6*x**5
    )

    return VP * s


# ============================================================
# Configuration
# ============================================================

config = parse_config("config.yaml")

temperature = args.temperature

N_particles1 = args.N_particles1
N_particles2 = args.N_particles2

grid_size = args.grid_size

VP = args.VP

gamma = args.gamma

thermalization_time = (
    args.thermalization_time
)

enable_temperature = (
    args.enable_temperature
)

box_length = args.box_length

final_time = args.final_time

T_ramp_trap = (
    args.T_ramp_trap
)

T_ramp_TP = (
    args.T_ramp_TP
)

final_length = (
    args.final_length
)

box_trap = args.box_trap

t_delay_trap_ramp = (
    args.t_delay_trap_ramp
)

result_path = args.result_path

seed = np.random.randint(1e6)


# ============================================================
# Simulation parameters
# ============================================================

dt = 1e-6

t_ramp = 2e-3

J = 0
detuning = -10e6
imaginary_steps = int(1000)

monitor_every = 500

VP = float(VP)


def lattice_static(t):
    return VP


lattice_ramp_ = lambda t: lattice_ramp(
    t,
    T_ramp=T_ramp_TP,
    t_delay=T_ramp_TP,
    VP=VP
)


# ============================================================
# Initial trap
# ============================================================

trap_initial_ = Trap(
    omegax=box_length,
    omegay=box_length
)

if box_trap:

    print(
        "Using BoxTrap",
        flush=True
    )

    trap_initial_ = BoxTrap(
        box_length=box_length
    )


# ============================================================
# Potentials
# ============================================================

config = parse_config("config.yaml")

contact = Contact(
    a_s=100
)

detuning = -10e6


# ============================================================
# Main SO propagation
# ============================================================

def call_SO(
    trap_ramp_time,
    enable_temp,
    temperature,
    gamma,
    final_length=12.5e-6,
    t_delay=22e-3,
    VP=15,
    T_ramp_TP=20e-3,
    final_time=50e-3,
    thermalization_time=50e-3,
    N_particles=int(20e3),
):

    # --------------------------------------------------------
    # Cavity
    # --------------------------------------------------------

    lattice_ramp_ = lambda t: lattice_ramp(
        t,
        T_ramp=T_ramp_TP,
        t_delay=0,
        VP=VP
    )

    cavity = DispersiveCavity(
        lattice_depth=lattice_ramp_,
        cavity_detuning=detuning,
        **config["potentials"]["cavity"]
    )

    cavity_monitor = CavityMonitor(
        cavity
    )


    # --------------------------------------------------------
    # Initial state
    # --------------------------------------------------------

    print_mem("before initial state")


    if result_path is not None:

        # ====================================================
        # LOAD PREVIOUS RESULT
        # ====================================================

        output_path = (
            f"./results/{result_path}"
        )

        print(
            f"[PRELOAD] Loading previous result:",
            flush=True
        )

        print(
            f"[PRELOAD] {output_path}",
            flush=True
        )

        if not os.path.isfile(output_path):

            raise FileNotFoundError(
                f"Could not find result file:\n"
                f"{output_path}"
            )

        with h5py.File(
            output_path,
            "r"
        ) as f:

            result1 = {
                "states": torch.from_numpy(
                    f["states/pump_ramp"][:]
                ),
                "alpha": torch.from_numpy(
                    f["cavity/alpha_pump_ramp"][:]
                ),
            }

        states_preLoaded = result1["states"]

        psi_thermal = states_preLoaded[-1]

        # This is the state that will be passed to make_bilayer.
        state = psi_thermal

        print(
            "[PRELOAD] Successfully loaded state.",
            flush=True
        )

        print(
            f"[PRELOAD] states shape: "
            f"{tuple(states_preLoaded.shape)}",
            flush=True
        )

        print(
            f"[PRELOAD] Initial state shape: "
            f"{tuple(state.shape)}",
            flush=True
        )


    else:

        # ====================================================
        # GENERATE NEW INITIAL STATE
        # ====================================================

        print(
            "[PRELOAD] No result_path supplied.",
            flush=True
        )

        print(
            "[THERMAL] Generating/loading initial state.",
            flush=True
        )

        if enable_temp:

            state = get_or_make_thermal_state(
                temperature=temperature,
                N_particles=N_particles,
                gamma=gamma,
                thermalization_time=thermalization_time,
                grid_size=grid_size,
                trap=trap_initial_,
                cavity_monitor=cavity_monitor,
                seed=seed,
                imaginary_steps=imaginary_steps,
            )

        else:

            state = get_BEC(
                0,
                int(500),
                trap=trap_initial_,
                N_particles=N_particles,
                grid_size=grid_size,
            )[1]


    print_mem("after initial state")


    # --------------------------------------------------------
    # Re-create cavity
    # --------------------------------------------------------

    lattice_ramp_ = lambda t: lattice_ramp(
        t,
        T_ramp=T_ramp_TP,
        t_delay=0,
        VP=VP
    )

    cavity = DispersiveCavity(
        lattice_depth=lattice_ramp_,
        cavity_detuning=detuning,
        **config["potentials"]["cavity"]
    )

    cavity_monitor = CavityMonitor(
        cavity
    )


    # --------------------------------------------------------
    # Dynamical trap
    # --------------------------------------------------------

    trap_dyn = Trap(
        omegax=lambda t: omega_of_t(
            t,
            box_length,
            final_length,
            trap_ramp_time,
            t_delay=t_delay
        ),

        omegay=lambda t: omega_of_t(
            t,
            box_length,
            final_length,
            trap_ramp_time,
            t_delay=t_delay
        ),
    )


    if box_trap:

        print(
            "Using BoxTrap",
            flush=True
        )

        trap_dyn = BoxTrap(
            box_length=lambda t: L_of_t(
                t,
                box_length,
                final_length,
                trap_ramp_time,
                t_delay=t_delay
            )
        )


    # --------------------------------------------------------
    # Bilayer
    # --------------------------------------------------------

    bilayer, pots1, pots2, P1, P2 = make_bilayer(
        state,
        state,
        1,
        trap=trap_dyn,
        N_particles=N_particles,
        grid_size=grid_size
    )


    # --------------------------------------------------------
    # Chemical potential
    # --------------------------------------------------------

    mu = estimate_mu(
        bilayer.layer1,
        [
            trap_dyn,
            contact
        ]
    )


    # --------------------------------------------------------
    # SGPE propagation
    # --------------------------------------------------------

    res = propagate_bilayer_sgpe(
        bilayer,

        final_time=final_time,

        time_step=1e-6,

        J=0,

        temperature=(
            temperature
            if enable_temp
            else 0
        ),

        gamma=(
            gamma
            if enable_temp
            else 0
        ),

        chemical_potential=mu,

        potentials1=[
            trap_dyn,
            contact,
            cavity,
        ],

        potentials2=[
            trap_dyn,
            contact,
            cavity,
        ],

        projector1=P1,
        projector2=P2,

        leave_progress_bar=False,

        monitor_cavity=cavity,
        monitor_every=50,
    )


    print_mem("after SO")


    return res, cavity_monitor


# ============================================================
# Run
# ============================================================

res, cm = call_SO(
    T_ramp_trap,
    enable_temperature,
    temperature,
    gamma,

    final_length=final_length,

    t_delay=t_delay_trap_ramp,

    VP=VP,

    T_ramp_TP=T_ramp_TP,

    final_time=final_time,

    thermalization_time=thermalization_time,

    N_particles=N_particles1,
)


# ============================================================
# Save
# ============================================================

save_path = save_quench_run(
    output_dir="results",

    temperature=temperature,

    result1=res,
    result2=res,

    cavity_monitor1=cm,
    cavity_monitor2=cm,

    N_particles1=N_particles1,
    N_particles2=N_particles2,

    gamma=gamma,
    N_grid=256,
    dt=dt,

    thermalization_time=thermalization_time,

    grid_size=grid_size,

    final_time1=final_time,
    final_time2=final_time,

    J=J,

    detuning=detuning,

    VP=VP,

    a_s=100,

    prefix="Box_Quench_",
)