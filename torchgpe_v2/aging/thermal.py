import numpy as np
import torch
import matplotlib.pyplot as plt

from torchgpe.bec2D import Gas
from torchgpe.bec2D.potentials import Trap, Contact
from torchgpe.utils import parse_config

import torch


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

import numpy as np
import torch
import matplotlib.pyplot as plt

#from torchgpe.bec2D import Gas
from torchgpe.bec2D.potentials import Contact
from torchgpe.utils.potentials import LinearPotential


# ============================================================
# Box potential
# ============================================================

"""class BoxTrap(LinearPotential):

    def __init__(
        self,
        box_length,
        wall_height=1000.0,
        wall_width=0.5e-6,
    ):
        super().__init__()

        self.box_length = box_length
        self.wall_height = wall_height
        self.wall_width = wall_width

    def get_potential(self, X, Y):
        # X,Y are dimensionless TorchGPE coordinates,
        # so convert physical lengths -> dimensionless lengths
        L = self.box_length / self.gas.adim_length
        w = self.wall_width / self.gas.adim_length

        half_L = L / 2

        # Smooth walls using tanh
        wall_x = 0.5 * (
            1.0 + torch.tanh((torch.abs(X) - half_L) / w)
        )

        wall_y = 0.5 * (
            1.0 + torch.tanh((torch.abs(Y) - half_L) / w)
        )

        # union of x and y walls
        wall = 1.0 - (1.0 - wall_x) * (1.0 - wall_y)

        return self.wall_height * wall"""

from torchgpe.utils.potentials import (
    LinearPotential,
    time_dependent_variable,
    any_time_dependent_variable,
)

class BoxTrap(LinearPotential):

    def __init__(
        self,
        box_length,
        wall_height=1000.0,
        wall_width=0.5e-6,
    ):
        super().__init__()

        self.box_length = time_dependent_variable(box_length)
        self.wall_height = time_dependent_variable(wall_height)
        self.wall_width = time_dependent_variable(wall_width)

        self.is_time_dependent = any_time_dependent_variable(
            box_length,
            wall_height,
            wall_width,
        )

    def get_potential(self, X, Y, time=None):

        L_phys = self.box_length(time)
        wall_height = self.wall_height(time)
        wall_width_phys = self.wall_width(time)


        L = L_phys / self.gas.adim_length
        w = wall_width_phys / self.gas.adim_length

        half_L = L / 2

        wall_x = 0.5 * (
            1.0 + torch.tanh((torch.abs(X) - half_L) / w)
        )

        wall_y = 0.5 * (
            1.0 + torch.tanh((torch.abs(Y) - half_L) / w)
        )

        wall = 1.0 - (1.0 - wall_x) * (1.0 - wall_y)

        return wall_height * wall


def get_BEC(
    N_vortices,
    N_iterations,
    co_rot=False,
    grid_size=40e-6,
    trap=None,
    N_particles=int(5e4),
    init_state=None,
    wall_height=1000.0,
    wall_width=0.5e-6,
    background = "gaussian",

):

    bec = Gas(
        N_particles=N_particles,
        grid_size=grid_size,
    )
    sigma_adim = 6e-6 / bec.adim_length
    vortex_length = 10e-6

    contact = Contact(a_s=100)

    if init_state is not None:
        bec.psi = init_state
        psi_final = init_state

    else:
        vortices = []

        for i in range(N_vortices):
            vortices.append({
                "X0": (
                    np.random.random() * vortex_length
                    - vortex_length / 2
                ) * 1e6,
                "Y0": (
                    np.random.random() * vortex_length
                    - vortex_length / 2
                ) * 1e6,
                "charge": (
                    +1
                    if np.random.random() > 0.5
                    else (+1 if co_rot else -1)
                ),
                "core_adim": 1e-3,
            })

        bec.psi = make_multi_vortex_state(
            bec.X,
            bec.Y,
            sigma_adim=sigma_adim,
            vortices=vortices,
            adim_length=bec.adim_length,
            background=background,
        )

        bec.ground_state(
            potentials=[trap, contact],
            N_iterations=N_iterations,
        )

        psi_final = bec.psi.clone()

    return bec, psi_final


def make_bilayer(
    psi1,
    psi2,
    seed=0,
    trap=None,
    grid_size=40e-6,
    N_particles=int(5e4),
    contact_as=100,
    wall_height=1000.0,
    wall_width=0.5e-6,
):
    torch.manual_seed(seed)
    np.random.seed(seed)

    gas_kwargs = dict(
        N_particles=N_particles,
        N_grid=256,
        grid_size=grid_size,
        normalize_on_assignment=False,
    )

    gas1 = Gas(**gas_kwargs)
    gas2 = Gas(**gas_kwargs)

    gas1.psi = psi1.to(gas1.complex_dtype)
    gas2.psi = psi2.to(gas2.complex_dtype)

    bilayer = BilayerGas(gas1, gas2)

    potentials1 = [
        trap,
        Contact(
            a_s=contact_as,
            a_orth=1e-6,
        ),
    ]

    potentials2 = [
        trap,
        Contact(
            a_s=contact_as,
            a_orth=1e-6,
        ),
    ]

    projector1 = make_momentum_projector(gas1)
    projector2 = make_momentum_projector(gas2)

    return (
        bilayer,
        potentials1,
        potentials2,
        projector1,
        projector2,
    )


from torchgpe.utils.potentials import LinearPotential, NonLinearPotential

def J_ramp(t, ramp_time=1e-3, J_initial = 2 * np.pi * 40, J_final = 2 * np.pi * 2):
    s = np.clip(t / ramp_time, 0.0, 1.0)
    smooth = 3 * s**2 - 2 * s**3

    return J_initial + smooth * (
        J_final - J_initial
    )

def estimate_mu(gas, potentials):
    psi = gas.psi
    nx, ny = psi.shape

    kx = 2 * torch.pi * torch.fft.fftfreq(
        nx, d=float(gas.dx),
        device=gas.device, dtype=gas.float_dtype
    )
    ky = 2 * torch.pi * torch.fft.fftfreq(
        ny, d=float(gas.dy),
        device=gas.device, dtype=gas.float_dtype
    )
    KX, KY = torch.meshgrid(kx, ky, indexing="ij")

    kinetic = torch.fft.ifftn(
        0.5 * (KX**2 + KY**2) * torch.fft.fftn(psi)
    )

    Vpsi = torch.zeros_like(psi)

    for p in potentials:
        p.set_gas(gas)
        p.on_propagation_begin()

        if isinstance(p, LinearPotential):
            V = p.get_potential(*gas.coordinates)
        elif isinstance(p, NonLinearPotential):
            V = p.potential_function(*gas.coordinates, psi)
        else:
            continue

        Vpsi += V * psi

    Hpsi = kinetic + Vpsi

    norm = torch.sum(torch.abs(psi)**2) * gas.dx * gas.dy
    mu = (
        torch.sum(torch.conj(psi) * Hpsi).real
        * gas.dx * gas.dy
        / norm
    )

    return mu.item()


def phase_wrap(x):
    return (x + np.pi) % (2 * np.pi) - np.pi

def detect_vortices_from_phase(phase):
    # Plaquettes only; no periodic wrap across boundaries
    th00 = phase[:-1, :-1]
    th10 = phase[1:, :-1]
    th11 = phase[1:, 1:]
    th01 = phase[:-1, 1:]

    winding = (
        phase_wrap(th10 - th00)
        + phase_wrap(th11 - th10)
        + phase_wrap(th01 - th11)
        + phase_wrap(th00 - th01)
    ) / (2 * np.pi)

    winding_int = np.rint(winding).astype(int)

    vort = np.argwhere(winding_int == 1)
    antiv = np.argwhere(winding_int == -1)

    # Return plaquette-center coordinates in array-index units
    return vort + 0.5, antiv + 0.5

def detect_vortices_masked(psi, density_mask):
    psi = psi.detach().cpu().numpy() if torch.is_tensor(psi) else np.asarray(psi)

    phase = np.angle(psi)
    vort, antiv = detect_vortices_from_phase(phase)

    vort_i = np.floor(vort).astype(int)
    antiv_i = np.floor(antiv).astype(int)

    if len(vort):
        valid = (
            (vort_i[:, 0] >= 0)
            & (vort_i[:, 0] < density_mask.shape[0])
            & (vort_i[:, 1] >= 0)
            & (vort_i[:, 1] < density_mask.shape[1])
        )
        vort = vort[valid]
        vort_i = vort_i[valid]
        vort = vort[density_mask[vort_i[:, 0], vort_i[:, 1]]]

    if len(antiv):
        valid = (
            (antiv_i[:, 0] >= 0)
            & (antiv_i[:, 0] < density_mask.shape[0])
            & (antiv_i[:, 1] >= 0)
            & (antiv_i[:, 1] < density_mask.shape[1])
        )
        antiv = antiv[valid]
        antiv_i = antiv_i[valid]
        antiv = antiv[density_mask[antiv_i[:, 0], antiv_i[:, 1]]]

    return vort, antiv


def get_thermal_state(
    T,
    gamma=0.01,
    J=0,
    dt=1e-6,
    thermalization_time=30e-3,
    trap=None,
    grid_size=40e-6,
    N_particles=int(100e3),
    imaginary_steps=int(500),
    monitor_cavity=None,
    monitor_alpha=False,
    monitor_every=10,
    init_state=None,
    contact_as=100,
    wall_height=1000.0,
    wall_width=0.5e-6,
    seed=None,
):

    bec, psi_init = get_BEC(
        0,
        imaginary_steps,
        True,
        trap=trap,
        N_particles=N_particles,
        init_state=init_state,
        wall_height=wall_height,
        wall_width=wall_width,
        grid_size=grid_size
    )

    bilayer, pots1, pots2, P1, P2 = make_bilayer(
        psi_init,
        psi_init,
        seed=seed,
        trap=trap,
        N_particles=N_particles,
        contact_as=contact_as,
        wall_height=wall_height,
        wall_width=wall_width,
    )

    mu = estimate_mu(
        bilayer.layer1,
        pots1,
    )

    result = propagate_bilayer_sgpe(
        bilayer,
        thermalization_time,
        dt,
        J,
        T,
        gamma,
        mu,
        pots1,
        pots2,
        P1,
        P2,
        leave_progress_bar=False,
        monitor_cavity=monitor_cavity,
        monitor_alpha=monitor_alpha,
        monitor_every=monitor_every,
    )

    psi_thermal = bilayer.layer1.psi.clone()

    if monitor_cavity is None:
        return psi_thermal
    else:
        return result
