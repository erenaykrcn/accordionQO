import argparse
from pathlib import Path
import os
import sys
import psutil
import torch
import numpy as np


# ============================================================
# Arguments
# ============================================================

parser = argparse.ArgumentParser()

parser.add_argument("--temperature1", type=float, default=75)
parser.add_argument("--temperature2", type=float, required=True)

parser.add_argument("--N_particles1", type=int, required=True)
parser.add_argument("--N_particles2", type=int, required=True)

parser.add_argument("--grid_size", type=float, default=20e-6)
parser.add_argument("--thermalization_time", type=float, required=True)

parser.add_argument("--gamma", type=float, required=True)
parser.add_argument("--VP", type=float, default=15)

parser.add_argument("--enable_temperature", type=bool, default=True)

parser.add_argument("--final_time", type=float, default=60e-3)

parser.add_argument("--T_ramp_trap", type=float, default=4e-3)
parser.add_argument("--T_ramp_TP", type=float, default=8e-3)

parser.add_argument("--t_delay_temp1", type=float, default=0)

parser.add_argument("--trap_initial", type=float, default=None)
parser.add_argument("--trap_final", type=float, default=None)

parser.add_argument("--box_length", type=float, default=15e-6)
parser.add_argument("--final_length", type=float, default=15e-6)

# If supplied, try to skip the first SO propagation.
# If the requested cache does not exist, SO is performed and saved.
parser.add_argument(
    "--preload_SO",
    action="store_true",
    help="Load cached self-organized state instead of running first SO propagation.",
)

parser.add_argument(
    "--SO_cache_dir",
    type=str,
    default="SO_states",
)

parser.add_argument(
    "--thermal_cache_dir",
    type=str,
    default="thermal_states",
)

args = parser.parse_args()


# ============================================================
# Imports from project
# ============================================================

from thermal import (
    get_thermal_state,
    make_bilayer,
    estimate_mu,
    get_BEC,
    BoxTrap,
)

from torchgpe.utils import parse_config

from utils import save_quench_run

from torchgpe.bec2D.potentials import (
    Contact,
    DispersiveCavity,
)
from torchgpe.bec2D.potentials import Contact, DispersiveCavity, Trap

from torchgpe.bec2D.callbacks import CavityMonitor

sys.path.append("../../")

from torchgpe_v2.bec2D.bilayer_v5 import (
    propagate_bilayer_sgpe,
)


# ============================================================
# Parameters
# ============================================================

#monitor_every, monitor_every_th = (50, 1000)
monitor_every, monitor_every_th = (100000, 100000)

temperature1 = args.temperature1
temperature2 = args.temperature2

N_particles1 = args.N_particles1
N_particles2 = args.N_particles2

grid_size = args.grid_size
thermalization_time = args.thermalization_time

gamma = args.gamma
VP = float(args.VP)

enable_temperature = args.enable_temperature

final_time = args.final_time

T_ramp_trap = args.T_ramp_trap
T_ramp_TP = args.T_ramp_TP

t_delay_temp1 = args.t_delay_temp1

box_length = args.box_length
final_length = args.final_length

dt = 1e-6
J = 0
detuning = -10e6
imaginary_steps = 500

seed = np.random.randint(1_000_000)

config = parse_config("config.yaml")

contact = Contact(a_s=100)

_process = psutil.Process(os.getpid())


# ============================================================
# Utility functions
# ============================================================

def print_mem(label):
    rss = _process.memory_info().rss / 1024**3
    print(
        f"[MEM] {label}: {rss:.3f} GB",
        flush=True,
    )


def lattice_ramp(
    t,
    T_ramp=8e-3,
    t_delay=0,
    VP=4,
):
    """
    Smooth quintic ramp:
        0 -> VP
    """

    if t is None:
        t = 0.0

    if t <= t_delay:
        return 0.0

    if t >= t_delay + T_ramp:
        return VP

    x = (t - t_delay) / T_ramp

    s = (
        10 * x**3
        - 15 * x**4
        + 6 * x**5
    )

    return VP * s


def lattice_static(t, VP=4):
    return VP

def omega_of_t(t, omega_initial, omega_final, T_ramp, t_delay=0):
    if t is None:
        t = 0.0
    if t <= t_delay:
        return 0.0
    if t >= t_delay + T_ramp:
        return omega_final
    x = (t - t_delay) / T_ramp
    s = 10*x**3 - 15*x**4 + 6*x**5
    diff = omega_final - omega_initial
    return  diff * s + omega_initial


# Ramp starts immediately.
lattice_ramp_ = lambda t: lattice_ramp(
    t,
    T_ramp=T_ramp_TP,
    t_delay=0,
    VP=VP,
)


# Initial box
trap_initial_ = BoxTrap(
    box_length=box_length
)
#trap_initial_ = Trap(omegax=box_length, omegay=box_length)


# ============================================================
# Thermal-state cache
# ============================================================

def get_or_make_thermal_state(
    temperature,
    N_particles,
    gamma,
    thermalization_time,
    grid_size,
    box_length,
    trap,
    cavity_monitor,
    seed,
    imaginary_steps=500,
    cache_dir="thermal_states",
):
    """
    Load the thermal state if it already exists.

    Otherwise:
        1. thermalize
        2. take final state
        3. save final state
        4. return it
    """

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    cache_path = cache_dir / (
        f"thermal_"
        f"T{temperature:g}_"
        f"N{N_particles}_"
        f"gamma{gamma:g}_"
        f"therm{thermalization_time * 1e3:g}ms_"
        f"grid{grid_size * 1e6:g}um_"
        f"box{box_length * 1e6:g}um.pt"
    )

    # --------------------------------------------------------
    # Cached state exists
    # --------------------------------------------------------

    if cache_path.exists():

        print(
            f"[THERMAL] Loading cached state:\n"
            f"          {cache_path}",
            flush=True,
        )

        state = torch.load(
            cache_path,
            map_location="cpu",
            weights_only=False,
        )

        return state

    # --------------------------------------------------------
    # Otherwise generate thermal state
    # --------------------------------------------------------

    print(
        "[THERMAL] No cached state found.",
        flush=True,
    )

    print(
        f"[THERMAL] Generating thermal state:\n"
        f"          {cache_path}",
        flush=True,
    )

    print(
        f"[THERMAL] Seed = {seed}",
        flush=True,
    )

    result = get_thermal_state(
        temperature,
        thermalization_time=thermalization_time,

        grid_size=grid_size,
        N_particles=N_particles,

        monitor_cavity=cavity_monitor,
        monitor_every=monitor_every_th,

        gamma=gamma,
        contact_as=100,

        trap=trap,

        seed=seed,

        imaginary_steps=imaginary_steps,

        J=0,
    )

    state = result["states"][-1]

    # Store on CPU so cache does not depend on GPU/device.
    torch.save(
        state.detach().cpu(),
        cache_path,
    )

    print(
        f"[THERMAL] Saved thermal state:\n"
        f"          {cache_path}",
        flush=True,
    )

    return state


# ============================================================
# SO cache path
# ============================================================

def get_SO_cache_path(
    temp1,
    N_particles,
    gamma,
    VP,
    T_ramp_TP,
    t_delay_temp1,
    grid_size,
    box_length,
    cache_dir,
):
    """
    Construct filename for the self-organized preparation.

    Notice that temp2 is deliberately NOT part of the filename:
    the same SO initial state can therefore be used for many
    different temperature quenches.
    """

    cache_dir = Path(cache_dir)

    cache_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    return cache_dir / (
        f"SO_"
        f"T{temp1:g}_"
        f"N{N_particles}_"
        f"gamma{gamma:g}_"
        f"VP{VP:g}_"
        f"Tramp{T_ramp_TP * 1e3:g}ms_"
        f"tSO{t_delay_temp1 * 1e3:g}ms_"
        f"grid{grid_size * 1e6:g}um_"
        f"box{box_length * 1e6:g}um.pt"
    )


# ============================================================
# Helper for caching arrays/tensors
# ============================================================

def cpu_copy(x):
    """
    Make something safe for torch.save without leaving
    GPU tensors in the cache.
    """

    if x is None:
        return None

    if torch.is_tensor(x):
        return x.detach().cpu()

    return np.asarray(x)


# ============================================================
# Temperature quench:
#
#       thermal state
#            |
#            v
#        SO at T1
#            |
#         T1 -> T2
#
# ============================================================

def qTemp_SO(
    enable_temperature,
    gamma,

    temp1=75,
    temp2=30,

    t_delay_temp1=22e-3,

    final_time=50e-3,

    thermalization_time=50e-3,

    N_particles=int(20e3),

    init_state=None,
):

    # ========================================================
    # Build SO-cache filename
    # ========================================================

    SO_cache_path = get_SO_cache_path(
        temp1=temp1,
        N_particles=N_particles,
        gamma=gamma,
        VP=VP,
        T_ramp_TP=T_ramp_TP,
        t_delay_temp1=t_delay_temp1,
        grid_size=grid_size,
        box_length=box_length,
        cache_dir=args.SO_cache_dir,
    )

    print(
        f"[SO] Cache path:\n"
        f"     {SO_cache_path}",
        flush=True,
    )


    # ========================================================
    # Box stays constant during the temperature quench
    # ========================================================

    trap_dyn = BoxTrap(
        box_length=box_length
    )
    #trap_dyn = Trap(
        #omegax=lambda t: omega_of_t(t, box_length, 
            #final_length, 0, t_delay=0),
        #omegay=lambda t: omega_of_t(t, box_length, 
            #final_length, 0, t_delay=0),
    #)


    # ========================================================
    # Cavity for first part:
    #
    #     VP : 0 -> VP
    #
    # ========================================================

    cavity1 = DispersiveCavity(
        lattice_depth=lattice_ramp_,
        cavity_detuning=detuning,
        **config["potentials"]["cavity"],
    )

    cavity_monitor1 = CavityMonitor(
        cavity1
    )


    # ========================================================
    # Decide where initial SO state comes from
    #
    # Priority:
    #
    #   1. explicit init_state
    #   2. --preload_SO cache
    #   3. thermal state -> propagate SO
    #
    # ========================================================

    res1 = None
    state = None


    # ========================================================
    # CASE 1:
    # Explicitly supplied self-organized state
    # ========================================================

    if init_state is not None:

        print(
            "[SO] Using explicitly supplied init_state.",
            flush=True,
        )

        print(
            "[SO] Skipping thermalization.",
            flush=True,
        )

        print(
            "[SO] Skipping first propagate_bilayer_sgpe.",
            flush=True,
        )

        state = init_state

        # Minimal res1 for compatibility with downstream code.
        res1 = {
            "states": [state],
            "alpha": np.array([]),
            "times": np.array([]),
        }


    # ========================================================
    # CASE 2:
    # Load cached SO state
    #
    # This skips BOTH:
    #   thermalization
    #   first propagate_bilayer_sgpe
    # ========================================================

    elif args.preload_SO and SO_cache_path.exists():

        print(
            f"[SO] Loading cached self-organized state:\n"
            f"     {SO_cache_path}",
            flush=True,
        )

        print(
            "[SO] Skipping thermalization.",
            flush=True,
        )

        print(
            "[SO] Skipping first propagate_bilayer_sgpe.",
            flush=True,
        )

        cached_SO = torch.load(
            SO_cache_path,
            map_location="cpu",
            weights_only=False,
        )

        state = cached_SO["state"]

        alpha1 = cached_SO.get(
            "alpha",
            np.array([]),
        )

        times1 = cached_SO.get(
            "times",
            np.array([]),
        )

        # Minimal reconstruction of result1.
        res1 = {
            "states": [state],
            "alpha": (
                alpha1
                if alpha1 is not None
                else np.array([])
            ),
            "times": (
                times1
                if times1 is not None
                else np.array([])
            ),
        }

        print(
            "[SO] Cached SO state loaded.",
            flush=True,
        )


    # ========================================================
    # CASE 3:
    # Need to prepare SO state from thermal state
    # ========================================================

    else:

        if args.preload_SO and not SO_cache_path.exists():

            print(
                "[SO] --preload_SO requested, "
                "but no matching cache exists.",
                flush=True,
            )

            print(
                "[SO] Preparing SO state and saving it "
                "for the next run.",
                flush=True,
            )


        # ----------------------------------------------------
        # Get thermal state
        # ----------------------------------------------------

        print_mem(
            "before thermal"
        )

        if enable_temperature:

            # Cavity object is only being supplied to the
            # thermal routine in the same way as your previous
            # implementation.
            thermal_cavity = DispersiveCavity(
                lattice_depth=lattice_ramp_,
                cavity_detuning=detuning,
                **config["potentials"]["cavity"],
            )

            thermal_cavity_monitor = CavityMonitor(
                thermal_cavity
            )

            state_thermal = get_or_make_thermal_state(
                temperature=temp1,

                N_particles=N_particles,

                gamma=gamma,

                thermalization_time=thermalization_time,

                grid_size=grid_size,
                box_length=box_length,

                trap=trap_initial_,

                cavity_monitor=thermal_cavity_monitor,

                seed=seed,

                imaginary_steps=imaginary_steps,

                cache_dir=args.thermal_cache_dir,
            )

        else:

            state_thermal = get_BEC(
                0,
                500,

                trap=trap_initial_,

                N_particles=N_particles,

                grid_size=grid_size,
            )[1]


        print_mem(
            "after thermal"
        )


        # ----------------------------------------------------
        # Construct bilayer from thermal state
        # ----------------------------------------------------

        bilayer, pots1, pots2, P1, P2 = make_bilayer(
            state_thermal,
            state_thermal,

            1,

            trap=trap_dyn,

            N_particles=N_particles,

            grid_size=grid_size,
        )


        # ----------------------------------------------------
        # Chemical potential for first evolution
        # ----------------------------------------------------

        mu1 = estimate_mu(
            bilayer.layer1,
            [
                trap_dyn,
                contact,
            ],
        )


        # ----------------------------------------------------
        # First propagation:
        #
        #       self-organization at T1
        #
        # ----------------------------------------------------

        print(
            "[SO] Running first propagate_bilayer_sgpe...",
            flush=True,
        )

        res1 = propagate_bilayer_sgpe(
            bilayer,

            final_time=t_delay_temp1,

            time_step=dt,

            J=0,

            temperature=(
                temp1
                if enable_temperature
                else 0
            ),

            gamma=(
                gamma
                if enable_temperature
                else 0
            ),

            chemical_potential=mu1,

            potentials1=[
                trap_dyn,
                contact,
                cavity1,
            ],

            potentials2=[
                trap_dyn,
                contact,
                cavity1,
            ],

            projector1=P1,
            projector2=P2,

            leave_progress_bar=False,

            monitor_cavity=cavity1,

            monitor_every=monitor_every,
        )


        # ----------------------------------------------------
        # Final SO state
        # ----------------------------------------------------

        state = res1["states"][-1]


        # ====================================================
        # ALWAYS save the SO preparation after it was generated
        #
        # Only final state is stored, not every intermediate
        # GPE state.
        #
        # alpha + times are kept so we can reconstruct a
        # minimal res1 on later runs.
        # ====================================================

        SO_data = {
            "state": cpu_copy(state),

            "alpha": cpu_copy(
                res1.get("alpha", None)
            ),

            "times": cpu_copy(
                res1.get("times", None)
            ),

            # Useful metadata
            "temperature1": temp1,
            "N_particles": N_particles,
            "gamma": gamma,
            "VP": VP,

            "T_ramp_TP": T_ramp_TP,
            "t_delay_temp1": t_delay_temp1,

            "grid_size": grid_size,
            "box_length": box_length,

            "thermalization_time": thermalization_time,

            "seed": seed,
        }


        torch.save(
            SO_data,
            SO_cache_path,
        )


        print(
            f"[SO] Saved final SO state + alpha + times:\n"
            f"     {SO_cache_path}",
            flush=True,
        )

        print_mem(
            "after SO preparation"
        )


    # ========================================================
    # At this point `state` is always the state immediately
    # before the T1 -> T2 quench.
    # ========================================================

    if state is None:
        raise RuntimeError(
            "Failed to prepare/load the self-organized state."
        )


    # ========================================================
    # Second propagation:
    #
    #         T1 -> T2
    #
    # Lattice is now fixed at VP.
    # ========================================================

    lattice_static_ = lambda t: lattice_static(
        t,
        VP=VP,
    )


    cavity2 = DispersiveCavity(
        lattice_depth=lattice_static_,
        cavity_detuning=detuning,
        **config["potentials"]["cavity"],
    )


    cavity_monitor2 = CavityMonitor(
        cavity2
    )


    # --------------------------------------------------------
    # Rebuild bilayer from final SO state
    # --------------------------------------------------------

    bilayer, pots1, pots2, P1, P2 = make_bilayer(
        state,
        state,

        1,

        trap=trap_dyn,

        N_particles=N_particles,

        grid_size=grid_size,
    )


    # --------------------------------------------------------
    # Chemical potential for second evolution
    #
    # This preserves the behavior of your original script.
    # --------------------------------------------------------

    mu2 = estimate_mu(
        bilayer.layer1,
        [
            trap_dyn,
            contact,
        ],
    )


    print(
        f"[QUENCH] Starting T = {temp1:g} -> {temp2:g}",
        flush=True,
    )

    print_mem(
        "before temperature quench"
    )


    res2 = propagate_bilayer_sgpe(
        bilayer,

        final_time=final_time - t_delay_temp1,

        time_step=dt,

        J=0,

        temperature=(
            temp2
            if enable_temperature
            else 0
        ),

        gamma=(
            gamma
            if enable_temperature
            else 0
        ),

        chemical_potential=mu2,

        potentials1=[
            trap_dyn,
            contact,
            cavity2,
        ],

        potentials2=[
            trap_dyn,
            contact,
            cavity2,
        ],

        projector1=P1,
        projector2=P2,

        leave_progress_bar=False,

        monitor_cavity=cavity2,

        monitor_every=monitor_every,
    )


    print_mem(
        "after temperature quench"
    )


    return (
        res1,
        res2,
        cavity_monitor1,
        cavity_monitor2,
    )


# ============================================================
# Run
# ============================================================

res1, res2, cm1, cm2 = qTemp_SO(
    enable_temperature=enable_temperature,

    gamma=gamma,

    temp1=temperature1,
    temp2=temperature2,

    t_delay_temp1=t_delay_temp1,

    final_time=final_time,

    thermalization_time=thermalization_time,

    N_particles=N_particles1,
)


# ============================================================
# Save full quench
# ============================================================

save_path = save_quench_run(
    output_dir="results",

    temperature=temperature2,

    result1=res1,
    result2=res2,

    cavity_monitor1=cm1,
    cavity_monitor2=cm2,

    N_particles1=N_particles1,
    N_particles2=N_particles2,

    gamma=gamma,

    dt=dt,

    thermalization_time=thermalization_time,

    grid_size=grid_size,

    final_time1=t_delay_temp1,
    final_time2=final_time - t_delay_temp1,

    J=J,

    detuning=detuning,

    VP=VP,

    a_s=100,

    prefix=f"Temp_Quench_T1_{temperature1:g}_",
)


print(
    f"[DONE] Saved quench run to:\n"
    f"       {save_path}",
    flush=True,
)