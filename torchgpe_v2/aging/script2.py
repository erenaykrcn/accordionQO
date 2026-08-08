import argparse
from pathlib import Path
import re
import h5py
import torch

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
final_time1, final_time2 = 50e-3, 50e-3
J, detuning, imaginary_steps = 0, -10e6, 500
monitor_every = 100
J, detuning, imaginary_steps = 0, -10e6, int(500)


VP = float(VP)
def lattice_ramp(t):
    t_ramp = VP * 1e-3
    if t >= t_ramp:
        return VP
    return VP * (t / t_ramp)
def lattice_static(t):
    return VP

"""
cavity = DispersiveCavity(
    lattice_depth=lattice_static,
    cavity_detuning=detuning,
    **config["potentials"]["cavity"]
)
cavity_monitor = CavityMonitor(cavity)
result = get_thermal_state(temperature, gamma=gamma, J=J, dt=dt, 
    thermalization_time=thermalization_time,
    omegar=omegar, grid_size=grid_size, N_particles=N_particles1,
    imaginary_steps=imaginary_steps, monitor_cavity=cavity_monitor
    )
save_path = save_quench_run(
    output_dir="results",
    temperature=temperature,
    result1=result,
    result2=result, 
    cavity_monitor1=cavity_monitor,
    cavity_monitor2=cavity_monitor, 
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
    VP=VP,
    prefix='Thermal_states'
)"""


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


"""results_dir = Path("./results")
latest_path = max(
    results_dir.glob(
        f"thermal_state_T{temperature:g}"
        f"_tth{thermalization_time:g}"
        f"_wr{omegar:g}"
        f"_L{grid_size:g}"
        f"_N{N_particles1:g}"
        "_id*.hdf5"
    ),"""
#    key=lambda p: int(re.search(r"_id(\d+)\.hdf5$", p.name).group(1))
""")
path =  str(latest_path)
with h5py.File(path, "r") as f:
    psi = f['state'][:]
psi_thermal = torch.from_numpy(psi)"""


"""results_dir = Path("./results")
latest_path = max(
    results_dir.glob(
        f"SO_quench"
        f"_T{temperature:g}"
        f"_N1{N_particles1:g}"
        f"_N2{N_particles2:g}"
        f"_tth{thermalization_time:g}"
        f"_wr{omegar:g}"
        f"_L{grid_size:g}"
        f"_J{J:g}"
        f"_D{detuning:g}"
        "_id*.hdf5"
    ),
    key=lambda p: int("""
#        re.search(r"_id(\d+)\.hdf5$", p.name).group(1)
"""    )
)
with h5py.File(latest_path, "r") as f:
    psi_SO = torch.from_numpy(
        f["states/pump_ramp"][-1]
    )
"""

# Pump to induce weak SO.
result1, cavity_monitor1 = get_SO_SGPE_state(psi_thermal, temperature, N_particles1, 
	lattice_ramp, final_time1, detuning = detuning, J=J,
    omegar=omegar, grid_size=grid_size, dt=dt, gamma=gamma, monitor_every=monitor_every)
psi_SO = result1['states'][-1]

gamma = gamma2
# N-> N/2 Quench, re-organization and equilibration of vortices.
result2, cavity_monitor2 = get_SO_SGPE_state(psi_SO, temperature, N_particles2, 
	lattice_static, final_time2, detuning = detuning,
    omegar=omegar, grid_size=grid_size, dt=dt, gamma=gamma, monitor_every=monitor_every)


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
