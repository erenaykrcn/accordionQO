from thermal import get_thermal_state, make_bilayer, estimate_mu, get_BEC, BoxTrap

import numpy as np
import torch
import matplotlib.pyplot as plt
from torchgpe.bec2D import Gas
from torchgpe.bec2D.potentials import Trap, Contact
from torchgpe.utils import parse_config
import torch


config = parse_config("config.yaml")



def get_SO_SGPE_state(init_state, temperature, N_particles, lattice, 
    final_time, trap, detuning=-10e6, grid_size=40e-6, dt=1e-6, gamma=0.01, J=0, 
    monitor_every=10, a_s=100):

    def make_multi_vortex_state(
        X,
        Y,
        sigma_adim,
        vortices,
        adim_length=1.0,
        background="gaussian",
        eps=1e-12,
    ):
        device = X.device
        real_dtype = X.dtype
        complex_dtype = torch.complex64 if real_dtype == torch.float32 else torch.complex128

        # Background envelope
        if background == "gaussian":
            r2 = X**2 + Y**2
            amplitude = torch.exp(-r2 / (2 * sigma_adim**2))
        elif background == "uniform":
            amplitude = torch.ones_like(X)
        else:
            raise ValueError("background must be 'gaussian' or 'uniform'")

        psi0 = amplitude.to(complex_dtype)

        for v in vortices:
            X0 = v.get("X0", 0.0) / adim_length
            Y0 = v.get("Y0", 0.0) / adim_length
            charge = int(v.get("charge", 1))
            core_adim = float(v.get("core_adim", 1e-6))

            Xs = X - X0
            Ys = Y - Y0
            r2_local = Xs**2 + Ys**2
            r_local = torch.sqrt(r2_local + core_adim**2)

            # Unit-charge vortex phase factor
            z = (Xs + 1j * Ys) / (r_local + eps)

            # Optional core suppression
            core_amp = r_local

            if charge > 0:
                vortex_factor = (core_amp * z) ** charge
            elif charge < 0:
                vortex_factor = (core_amp * torch.conj(z)) ** (-charge)
            else:
                vortex_factor = torch.ones_like(psi0)

            psi0 = psi0 * vortex_factor.to(complex_dtype)

        # Normalize
        norm = torch.sqrt(torch.sum(torch.abs(psi0)**2))
        psi0 = psi0 / (norm + eps)

        return psi0

    from torchgpe.bec2D.potentials import Contact, DispersiveCavity, Trap
    from torchgpe.bec2D.callbacks import CavityMonitor
    import sys
    sys.path.append("../../")
    from torchgpe_v2.bec2D.gas import Gas
    import numpy as np
    import torch
    import matplotlib.pyplot as plt

    from torchgpe_v2.bec2D.bilayer_v4 import (
        BilayerGas, propagate_bilayer,
        propagate_bilayer_sgpe,
        make_momentum_projector,
    )
    from torchgpe_v2.bec2D.potentials import Trap, Contact

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
