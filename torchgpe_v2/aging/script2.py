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
parser.add_argument("--omegar", type=float, required=True)
parser.add_argument("--thermalization_time", type=float, required=True)
parser.add_argument("--gamma1", type=float, required=True)
parser.add_argument("--gamma2", type=float, required=True)
parser.add_argument("--VP", type=float, required=True)
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
omegar = args.omegar
VP = args.VP
gamma1, gamma2 = args.gamma1, args.gamma2
thermalization_time = args.thermalization_time

gamma = gamma1
dt = 1e-6
final_time1, final_time2 = 100e-6, 30e-6
J, detuning, imaginary_steps = 0, -10e6, 500
monitor_every1 = 500
monitor_every2 = 10
J, detuning, imaginary_steps = 0, -10e6, int(500)


VP = float(VP)
def lattice_ramp(t):
    t_ramp = VP * 1e-3
    if t >= t_ramp:
        return VP
    return VP * (t / t_ramp)
def lattice_static(t):
    return VP


### Thermal State, Begin
psi_thermal = get_thermal_state(temperature, gamma=gamma, J=J, dt=dt, 
	thermalization_time=thermalization_time,
    omegar=omegar, grid_size=grid_size, N_particles=N_particles1,
	imaginary_steps=imaginary_steps
    )
save_state(
    output_dir="results",
    temperature=temperature,
    state=psi_thermal,
    thermalization_time=thermalization_time,
    omegar=omegar,
    grid_size=grid_size,
    N_particles=N_particles1,
)
### Thermal State, END


# Pump to induce weak SO.
result1, cavity_monitor1 = get_SO_SGPE_state(psi_thermal, temperature, N_particles1, 
	lattice_ramp, final_time1, detuning = detuning, J=J, a_s=100,
    omegar=omegar, grid_size=grid_size, dt=dt, gamma=gamma, monitor_every=monitor_every1)
psi_SO = result1['states'][-1]


gamma = gamma2
# Quench, re-organization and equilibration of vortices.
result2, cavity_monitor2 = get_SO_SGPE_state(psi_SO, temperature, N_particles2, 
	lattice_static, final_time2, detuning = detuning, a_s=100*np.sqrt(3),
    omegar=omegar, grid_size=grid_size, dt=dt, gamma=gamma, monitor_every=monitor_every2)


save_path = save_quench_run(
    output_dir="results",
    temperature=temperature,
    result1=result1,
    result2=result2, 
    cavity_monitor1=cavity_monitor1,
    cavity_monitor2=cavity_monitor2, 
    N_particles1=N_particles1,
    N_particles2=N_particles2,
    gamma=gamma,
    dt=dt,
    thermalization_time=thermalization_time,
    omegar=omegar,
    grid_size=grid_size,
    final_time1=final_time1,
    final_time2=final_time2,
    J=J,
    detuning=detuning,
    VP=VP
)
