from torchgpe.bec2D.potentials import DispersiveCavity, Trap, Contact
from torchgpe.bec2D.callbacks import CavityMonitor
from torchgpe.utils import parse_config

import sys
sys.path.append("../../../accordionQO/")
from torchgpe_v2.bec2D.gas import Gas
from torchgpe_v2.aging.thermal import make_bilayer, estimate_mu, BoxTrap, get_BEC, get_thermal_state
import numpy as np
import torch
import matplotlib.pyplot as plt

from torchgpe_v2.bec2D.bilayer_v5 import (
    BilayerGas, propagate_bilayer,
    propagate_bilayer_sgpe,
    make_momentum_projector,
)
from utils import save_quench_run

gamma = 0.01
thermalization_time = 50e-3
final_time = 50e-3
box_length = 30e-6
grid_size=45e-6
N_particles = 50e3
enable_T = True
temperature = 30
seed = np.random.randint(1e6)

config = parse_config("config.yaml")
ramp = config["boundaries"]["lattice_ramp"]
rt = config["propagation"]["real_time"]
dt = rt["time_step"]
trap = BoxTrap(box_length=box_length)
contact = Contact(a_s=100)
detuning = -10e6

cavity = DispersiveCavity(
        lattice_depth=config["boundaries"]["lattice_ramp"],
        cavity_detuning=detuning,
        **config["potentials"]["cavity"]
)
cavity_monitor = CavityMonitor(cavity)
    
bec, psi_final = get_BEC(0, int(200), True, grid_size=grid_size)
x_um = (bec.x * bec.adim_length * 1e6).detach().cpu().numpy()
y_um = (bec.y * bec.adim_length * 1e6).detach().cpu().numpy()
X_um, Y_um = np.meshgrid(x_um, y_um, indexing="ij")
    
    
res = get_thermal_state(temperature, thermalization_time=thermalization_time,
             grid_size=grid_size, N_particles=int(50e3), monitor_cavity=cavity_monitor,
            monitor_every=2000, gamma=gamma, contact_as=100, box_length=30e-6,
            seed=seed
            )



cavity = DispersiveCavity(
                lattice_depth=config["boundaries"]["lattice_ramp"],
                cavity_detuning=detuning,
                **config["potentials"]["cavity"]
)
cavity_monitor = CavityMonitor(cavity)

N_particles=100e3
bilayer, pots1, pots2, P1, P2 = make_bilayer(res['states'][-1], res['states'][-1], 
           box_length=box_length, N_particles=N_particles, grid_size=grid_size, seed=seed)
mu = estimate_mu(bilayer.layer1, [trap, contact])
result = propagate_bilayer_sgpe(
            bilayer,
            final_time=final_time,
            time_step=1e-6,
            J=0,
    
            temperature = temperature if enable_T else 0,
            gamma = gamma if enable_T else 0,
            
            chemical_potential=mu,
            potentials1=[
                trap,
                contact,
                cavity,
            ],
            potentials2=[
                trap,
                contact,
                
                cavity,
            ],
            projector1=P1,
            projector2=P2,
            leave_progress_bar=False,
            
            monitor_cavity=cavity,
            monitor_every=1000,
)




save_path = save_quench_run(
    output_dir="results",
    temperature=temperature,
    result1=result,
    result2=result,
    cavity_monitor1=cavity_monitor,
    cavity_monitor2=cavity_monitor, 
    N_particles1=N_particles,
    N_particles2=N_particles,
    gamma=gamma,
    dt=dt,
    thermalization_time=thermalization_time,
    omegar=0,
    grid_size=grid_size,
    final_time1=final_time,
    final_time2=final_time,
    J=0,
    detuning=detuning,
    VP=0,
    a_s=0,
    prefix='TC_VARIATIONS'
)