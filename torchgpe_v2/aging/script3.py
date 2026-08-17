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
group = parser.add_mutually_exclusive_group(required=True)
group.add_argument("--omegar", type=float, default=None)
group.add_argument("--box_length", type=float, default=None)
args = parser.parse_args()


from thermal import get_thermal_state
from SGPE_SO import get_SO_SGPE_state
from torchgpe.utils import parse_config
from utils import save_quench_run, save_state
from torchgpe.bec2D.potentials import Contact, DispersiveCavity, Trap
from torchgpe.bec2D.callbacks import CavityMonitor

config = parse_config("config.yaml")

temperature = args.temperature
N_particles1 = args.N_particles1
N_particles2 = args.N_particles2
grid_size = args.grid_size
VP = args.VP
gamma = args.gamma
thermalization_time = args.thermalization_time
enable_temperature = args.enable_temperature
omegar = args.omegar
box_length = args.box_length
final_time = args.final_time
seed = np.random.randint(1e6)


dt = 1e-6
t_ramp = 2e-3
J, detuning, imaginary_steps = 0, -10e6, int(500)
monitor_every = 20


VP = float(VP)
def lattice_ramp(t):
    if t >= t_ramp*VP:
        return VP
    return VP * (t / (t_ramp*VP))
def lattice_static(t):
    return VP


### Thermal State, Begin
psi_thermal = get_thermal_state(temperature, gamma=gamma, J=J, dt=dt, 
	thermalization_time=thermalization_time, monitor_every=2000,
    omega_r=omegar, box_length=box_length, grid_size=grid_size, N_particles=N_particles1,
	imaginary_steps=imaginary_steps, seed=seed
    ) if enable_temperature else get_BEC(0, int(500), box_length=box_length, omega_r=omegar,
        N_particles=N_particles, grid_size=grid_size)[1]
### Thermal State, END


# Quench, re-organization and equilibration of vortices.
result1, cavity_monitor1 = get_SO_SGPE_state(
    psi_thermal, temperature if enable_temperature else 0, 
    N_particles2, lattice_ramp, final_time, detuning = detuning, J=J, a_s=100,
    omega_r=omegar, box_length=box_length, 
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
    prefix='Z_Scaling_',
)
