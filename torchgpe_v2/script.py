import numpy as np
import torch
import matplotlib.pyplot as plt

from torchgpe.bec2D import Gas
from torchgpe.bec2D.potentials import Trap, Contact
from torchgpe.utils.potentials import LinearPotential, NonLinearPotential

import torch
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--T", type=float, required=True)
parser.add_argument("--gamma", type=float, required=True)
parser.add_argument("--seed", type=int, required=True)
parser.add_argument("--treshold", type=float, required=True)
parser.add_argument("--thermalization_time", type=float, required=True)
parser.add_argument("--J", type=float, required=True)
args = parser.parse_args()
T, gamma, seed, density_threshold, thermalization_time, J = args.T, args.gamma, args.seed, args.treshold, args.thermalization_time, args.J
omegar = 20

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


# -----------------------------
# System setup
# -----------------------------
def get_BEC(N_vortices, N_iterations, co_rot=False, omegar=20):
    bec = Gas(
            N_particles=2e5,
            grid_size=20e-6,          # 30 microns box size
            #n_points=2**9,            # try 2**10 if you want more resolution
    )
    # Harmonic trap + contact interactions
    trap = Trap(omegax=omegar, omegay=omegar)
    contact = Contact(a_s=100)
    
    vortices = []
    for i in range(N_vortices):
        vortices.append({"X0": np.random.random()*10-5, 
                         "Y0": np.random.random()*10-5, 
                         "charge": +1 if np.random.random()>0.5 else (+1 if co_rot else -1), "core_adim": 1e-3})
    bec.psi = make_multi_vortex_state(bec.X, bec.Y, sigma_adim=6e-6 / bec.adim_length, vortices=vortices)
    psi_final = bec.psi.clone()
    bec.ground_state(
            potentials=[trap, contact],
            N_iterations=N_iterations,
    )
    psi_final = bec.psi.clone()
    return bec, psi_final


bec, psi_final = get_BEC(0, int(200), True, omegar=omegar)

from bec2D.gas import Gas
import numpy as np
import torch
import matplotlib.pyplot as plt

from bec2D.bilayer import (
    BilayerGas, propagate_bilayer,
    propagate_bilayer_sgpe,
    make_momentum_projector,
)

from bec2D.potentials import Trap, Contact

def make_bilayer(psi1, psi2, seed=0, omegar=20):
    torch.manual_seed(seed)
    np.random.seed(seed)

    gas_kwargs = dict(
        N_particles=int(5e4),
        N_grid=256,
        grid_size=40e-6,
        normalize_on_assignment=False,
    )

    gas1 = Gas(**gas_kwargs)
    gas2 = Gas(**gas_kwargs)

    gas1.psi = psi1.to(gas1.complex_dtype)
    gas2.psi = psi2.to(gas2.complex_dtype)

    bilayer = BilayerGas(gas1, gas2)

    potentials1 = [
        Trap(omegax=omegar, omegay=omegar),
        Contact(a_s=100, a_orth=1e-6),
    ]
    potentials2 = [
        Trap(omegax=omegar, omegay=omegar),
        Contact(a_s=100, a_orth=1e-6),
    ]

    projector1 = make_momentum_projector(gas1)
    projector2 = make_momentum_projector(gas2)

    return bilayer, potentials1, potentials2, projector1, projector2


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

def detect_vortices_masked(psi, density_threshold=0.01):
    psi = psi.detach().cpu().numpy() if torch.is_tensor(psi) else np.asarray(psi)

    density = np.abs(psi)**2
    phase = np.angle(psi)

    vort, antiv = detect_vortices_from_phase(phase)
    # Density at each plaquette, normalized to peak density
    plaquette_density = 0.25 * (
        density[:-1, :-1]
        + density[1:, :-1]
        + density[1:, 1:]
        + density[:-1, 1:]
    )
    mask = plaquette_density > density_threshold * density.max()

    vort_i = np.floor(vort).astype(int)
    antiv_i = np.floor(antiv).astype(int)

    vort = vort[mask[vort_i[:, 0], vort_i[:, 1]]] if len(vort) else vort
    antiv = antiv[mask[antiv_i[:, 0], antiv_i[:, 1]]] if len(antiv) else antiv

    return vort, antiv


bilayer, pots1, pots2, P1, P2 = make_bilayer(psi_final, psi_final, seed)  # create fresh gases
mu = estimate_mu(bilayer.layer1, pots1)
# Thermalize
propagate_bilayer_sgpe(
            bilayer, thermalization_time, 1e-6, J, T, gamma, mu,
            pots1, pots2, P1, P2, leave_progress_bar=False,
)
bec, psi = bilayer.layer1, bec.psi



# ----------------------------

# Sampling

# ----------------------------
sample_time = 5 * thermalization_time
sample_interval = thermalization_time / 10      # 10 samples per thermalization time
n_samples = int(sample_time / sample_interval)
vortex_counts = []
antivortex_counts = []
for _ in range(n_samples):
    propagate_bilayer_sgpe(
        bilayer,
        sample_interval,
        1e-6,
        J,
        T,
        gamma,
        mu,
        pots1,
        pots2,
        P1,
        P2,
        leave_progress_bar=False,
    )
    psi = bilayer.layer1.psi
    vort, antiv = detect_vortices_masked(
        psi,
        density_threshold=density_threshold,

    )
    vortex_counts.append(len(vort))
    antivortex_counts.append(len(antiv))

mean_vort = np.mean(vortex_counts)
std_vort = np.std(vortex_counts)
mean_antiv = np.mean(antivortex_counts)
std_antiv = np.std(antivortex_counts)

with open("log.txt", "a") as f:
    f.write(
        f"T={T:.3f} , J={J:.3f}"
        f"<Nv>={mean_vort:.2f}±{std_vort:.2f} "
        f"<Na>={mean_antiv:.2f}±{std_antiv:.2f} "
        f"Nsamples={n_samples}\n"
    )

