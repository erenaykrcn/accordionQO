import argparse
from pathlib import Path
import re
import h5py
import torch
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--temperature", type=float, required=True)
parser.add_argument("--N_particles1", type=int, required=True)
parser.add_argument("--N_particles2", type=int, required=True)
parser.add_argument("--grid_size", type=float, required=True)
parser.add_argument("--thermalization_time", type=float, required=True)
parser.add_argument("--gamma", type=float, required=True)
parser.add_argument("--VP", type=float, required=True)
parser.add_argument("--enable_temperature", type=bool, default=True)
parser.add_argument("--final_time", type=float, default=60e-3)
parser.add_argument("--T_ramp_trap", type=float, default=4e-3)
parser.add_argument("--T_ramp_TP", type=float, default=8e-3)
parser.add_argument("--omega_initial", type=float, default=None)
parser.add_argument("--omega_final", type=float, default=None)
parser.add_argument("--box_length", type=float, default=None)
args = parser.parse_args()


from thermal import get_thermal_state
from SGPE_SO import get_SO_SGPE_state
from torchgpe.utils import parse_config
from utils import save_quench_run, save_state
from torchgpe.bec2D.potentials import Contact, DispersiveCavity, Trap
from torchgpe.bec2D.callbacks import CavityMonitor

def omega_of_t(t, omega_initial, omega_final, T_ramp):
    if t is None:
        t = 0.0
    if t < T_ramp:
        return omega_initial + (omega_final - omega_initial) * t / T_ramp
    return omega_final

def lattice_ramp(t, T_ramp=8e-3, t_delay=0, VP=4):
    if t <= t_delay:
        return 0.0
    if t >= t_delay + T_ramp:
        return VP
    x = (t - t_delay) / T_ramp
    s = 10*x**3 - 15*x**4 + 6*x**5
    return VP * s


config = parse_config("config.yaml")

temperature = args.temperature
N_particles1 = args.N_particles1
N_particles2 = args.N_particles2
grid_size = args.grid_size
VP = args.VP
gamma = args.gamma
thermalization_time = args.thermalization_time
enable_temperature = args.enable_temperature
box_length = args.box_length
final_time = args.final_time
omega_initial = args.omega_initial
omega_final = args.omega_final
T_ramp_trap = args.T_ramp_trap
T_ramp_TP = args.T_ramp_TP
seed = np.random.randint(1e6)


dt = 1e-6
t_ramp = 2e-3
J, detuning, imaginary_steps = 0, -10e6, int(500)
monitor_every = 100


VP = float(VP)
def lattice_static(t):
    return VP

trap = Trap(
    omegax=lambda t: omega_of_t(t, omega_initial, omega_final, T_ramp_trap),
    omegay=lambda t: omega_of_t(t, omega_initial, omega_final, T_ramp_trap),
)
trap_initial = Trap(omegax=omega_initial,omegay=omega_initial)
lattice_ramp_ = lambda t: lattice_ramp(t, T_ramp=T_ramp_TP, t_delay=T_ramp_trap, VP=VP)


### Thermal State, Begin
psi_thermal = get_thermal_state(temperature, gamma=gamma, J=J, dt=dt, 
	thermalization_time=thermalization_time, monitor_every=2000,
    trap=trap_initial, grid_size=grid_size, N_particles=N_particles1,
	imaginary_steps=imaginary_steps, seed=seed
    ) if enable_temperature else get_BEC(0, int(500), trap=trap_init,
        N_particles=N_particles, grid_size=grid_size)[1]
### Thermal State, END


# Quench, re-organization and equilibration of vortices.
result1, cavity_monitor1 = get_SO_SGPE_state(
    psi_thermal, temperature if enable_temperature else 0, 
    N_particles2, lattice_ramp, final_time, trap=trap, detuning = detuning, J=J, a_s=100,
    grid_size=grid_size, dt=dt, gamma=gamma if enable_temperature else 0,
    monitor_every=monitor_every)


save_path = save_quench_run(
    output_dir="results",
    temperature=temperature,
    result1=result1,
    result2=result1, 
    cavity_monitor1=cavity_monitor1,
    cavity_monitor2=cavity_monitor1,
    N_particles1=N_particles1,
    N_particles2=N_particles2,
    gamma=gamma,
    dt=dt,
    thermalization_time=thermalization_time,
    grid_size=grid_size,
    final_time1=final_time,
    final_time2=final_time,
    J=J,
    detuning=detuning,
    VP=VP,
    a_s=100,
    prefix='omegaQ_Z_Scaling_',
)
