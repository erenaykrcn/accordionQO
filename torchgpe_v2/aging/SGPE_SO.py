from thermal import get_thermal_state, make_bilayer, estimate_mu, get_BEC, BoxTrap

import numpy as np
import torch
import matplotlib.pyplot as plt
from torchgpe.bec2D import Gas
from torchgpe.bec2D.potentials import Trap, Contact
from torchgpe.utils import parse_config
import torch

from torchgpe.bec2D.potentials import Contact, DispersiveCavity, Trap
from torchgpe.bec2D.callbacks import CavityMonitor
import sys
sys.path.append("../../")
from torchgpe_v2.bec2D.gas import Gas
import numpy as np
import torch
import matplotlib.pyplot as plt

from torchgpe_v2.bec2D.bilayer_v5 import (
    BilayerGas, propagate_bilayer,
    propagate_bilayer_sgpe,
    make_momentum_projector,
)
from torchgpe_v2.bec2D.potentials import Trap, Contact


config = parse_config("config.yaml")

def get_SO_SGPE_state(init_state, temperature, N_particles, lattice, 
    final_time, trap, detuning=-10e6, grid_size=40e-6, dt=1e-6, gamma=0.01, J=0, 
    monitor_every=10, a_s=100):

    #trap = Trap(omegax=omega_r, omegay=omega_r) if omega_r is not None else BoxTrap(box_length=box_length)
    contact = Contact(a_s=a_s)
    cavity = DispersiveCavity(
                lattice_depth=lattice,
                cavity_detuning=detuning,
                **config["potentials"]["cavity"]
    )
    cavity_monitor = CavityMonitor(cavity)
    
    bilayer, pots1, pots2, P1, P2 = make_bilayer(init_state, init_state, 1, 
               trap=trap, N_particles=N_particles, grid_size=grid_size)
    mu = estimate_mu(bilayer.layer1, [trap, contact],)
    result = propagate_bilayer_sgpe(
                bilayer,
                final_time=final_time,
                time_step=dt,
                J=J,
                temperature=temperature,
    
                gamma=gamma,
                
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
                monitor_every=monitor_every,
    ) 

    return result, cavity_monitor
