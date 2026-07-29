from typing import Callable, Union

import numpy as np
import torch
from tqdm.auto import trange

from torchgpe.utils.potentials import (
    LinearPotential,
    NonLinearPotential,
    time_dependent_variable,
)


class BilayerGas:
    def __init__(self, layer1, layer2):
        self.layer1 = layer1
        self.layer2 = layer2

        if layer1.N_grid != layer2.N_grid:
            raise ValueError("Both layers must have the same N_grid.")

        if not np.isclose(layer1.grid_size_x, layer2.grid_size_x):
            raise ValueError("Both layers must have the same grid size.")

        if not np.isclose(layer1.adim_length, layer2.adim_length):
            raise ValueError("Both layers must use the same units.")

        if layer1.device != layer2.device:
            raise ValueError("Both layers must use the same device.")


def josephson_step(
    psi1: torch.Tensor,
    psi2: torch.Tensor,
    J: float,
    dt: float,
):
    """
    Exact Josephson evolution.

    J is J/hbar in rad/s.
    dt is in seconds.
    """
    theta = torch.as_tensor(
        J * dt,
        device=psi1.device,
        dtype=psi1.real.dtype,
    )

    c = torch.cos(theta)
    s = torch.sin(theta)

    psi1_old = psi1.clone()
    psi2_old = psi2.clone()

    psi1_new = c * psi1_old + 1j * s * psi2_old
    psi2_new = c * psi2_old + 1j * s * psi1_old

    return psi1_new, psi2_new


def _prepare_potentials(gas, potentials):
    for potential in potentials:
        potential.set_gas(gas)
        potential.on_propagation_begin()

    static_linear = [
        p for p in potentials
        if isinstance(p, LinearPotential) and not p.is_time_dependent
    ]

    dynamic_linear = [
        p for p in potentials
        if isinstance(p, LinearPotential) and p.is_time_dependent
    ]

    static_nonlinear = [
        p for p in potentials
        if isinstance(p, NonLinearPotential) and not p.is_time_dependent
    ]

    dynamic_nonlinear = [
        p for p in potentials
        if isinstance(p, NonLinearPotential) and p.is_time_dependent
    ]

    static_linear_total = sum(
        (p.get_potential(*gas.coordinates) for p in static_linear),
        torch.zeros_like(gas.X),
    )

    return (
        static_linear_total,
        dynamic_linear,
        static_nonlinear,
        dynamic_nonlinear,
    )


def _potential_step(
    gas,
    dt_adim,
    time,
    static_linear,
    dynamic_linear,
    static_nonlinear,
    dynamic_nonlinear,
):
    total = static_linear.clone()

    for potential in dynamic_linear:
        total += potential.get_potential(*gas.coordinates, time)

    for potential in static_nonlinear:
        total += potential.potential_function(
            *gas.coordinates,
            gas.psi,
        )

    for potential in dynamic_nonlinear:
        total += potential.potential_function(
            *gas.coordinates,
            gas.psi,
            time,
        )

    gas.psi = gas.psi * torch.exp(-1j * total * dt_adim)


@torch.no_grad()
def propagate_bilayer(
    bilayer: BilayerGas,
    final_time: float,
    time_step: float,
    J: Union[float, Callable],
    potentials1=None,
    potentials2=None,
    leave_progress_bar=True,
):
    potentials1 = [] if potentials1 is None else potentials1
    potentials2 = [] if potentials2 is None else potentials2

    gas1 = bilayer.layer1
    gas2 = bilayer.layer2

    J_t = time_dependent_variable(J)

    p1 = _prepare_potentials(gas1, potentials1)
    p2 = _prepare_potentials(gas2, potentials2)

    n_steps = round(final_time / time_step)
    dt = final_time / n_steps
    dt_adim = dt * gas1.adim_pulse

    kinetic = 0.5 * (gas1.Kx**2 + gas1.Ky**2)
    kinetic_half = torch.exp(-0.5j * kinetic * dt_adim)

    for step in trange(
        n_steps,
        leave=leave_progress_bar,
        desc="Bilayer propagation",
    ):
        time_mid = (step + 0.5) * dt
        J_mid = J_t(time_mid)

        # Half Josephson step
        psi1, psi2 = josephson_step(
            gas1.psi,
            gas2.psi,
            J_mid,
            dt / 2,
        )
        gas1.psi = psi1
        gas2.psi = psi2

        # Half kinetic step
        gas1.psik = gas1.psik * kinetic_half
        gas2.psik = gas2.psik * kinetic_half

        # Full local potential step
        _potential_step(gas1, dt_adim, time_mid, *p1)
        _potential_step(gas2, dt_adim, time_mid, *p2)

        # Half kinetic step
        gas1.psik = gas1.psik * kinetic_half
        gas2.psik = gas2.psik * kinetic_half

        # Half Josephson step
        psi1, psi2 = josephson_step(
            gas1.psi,
            gas2.psi,
            J_mid,
            dt / 2,
        )
        gas1.psi = psi1
        gas2.psi = psi2



import math


def complex_wiener_increment(
    shape,
    device,
    dtype_real,
    dt_adim,
):
    """
    Complex Wiener increment satisfying

        <dW*(r) dW(r')> ~ dt

    before spatial-grid normalization.
    """
    re = torch.randn(shape, device=device, dtype=dtype_real)
    im = torch.randn(shape, device=device, dtype=dtype_real)

    return (
        re + 1j * im
    ) * math.sqrt(dt_adim / 2.0)


def apply_projector(psi, projector):
    if projector is None:
        return psi

    return torch.fft.ifftn(
        torch.fft.fftn(psi) * projector
    )


def make_momentum_projector(gas, k_cut=None):
    """
    Projector defined on the unpadded real-space grid.
    k_cut is in TorchGPE dimensionless inverse-length units.
    """
    nx, ny = gas.psi.shape

    kx = 2 * torch.pi * torch.fft.fftfreq(
        nx,
        d=float(gas.dx),
        device=gas.device,
        dtype=gas.float_dtype,
    )

    ky = 2 * torch.pi * torch.fft.fftfreq(
        ny,
        d=float(gas.dy),
        device=gas.device,
        dtype=gas.float_dtype,
    )

    KX, KY = torch.meshgrid(kx, ky, indexing="ij")
    K = torch.sqrt(KX**2 + KY**2)

    if k_cut is None:
        k_nyquist = min(
            torch.pi / gas.dx,
            torch.pi / gas.dy,
        )
        k_cut = 0.7 * float(k_nyquist)

    return (K <= k_cut).to(gas.complex_dtype)


def _evaluate_total_potential(
    gas,
    time,
    prepared_potentials,
):
    (
        static_linear,
        dynamic_linear,
        static_nonlinear,
        dynamic_nonlinear,
    ) = prepared_potentials

    total = static_linear.clone()

    for potential in dynamic_linear:
        total += potential.get_potential(
            *gas.coordinates,
            time,
        )

    for potential in static_nonlinear:
        total += potential.potential_function(
            *gas.coordinates,
            gas.psi,
        )

    for potential in dynamic_nonlinear:
        total += potential.potential_function(
            *gas.coordinates,
            gas.psi,
            time,
        )

    return total


def _sgpe_local_step(
    psi,
    total_potential,
    gamma,
    temperature,
    chemical_potential,
    dt_adim,
    dxdy,
    noise_mask=None,
):
    """
    Euler-Maruyama local SGPE step:

        dpsi = -(i + gamma)(H - mu) psi dt + dW

    Parameters are dimensionless in TorchGPE units.
    """
    deterministic = (
        -(1j + gamma)
        * (total_potential - chemical_potential)
        * psi
        * dt_adim
    )

    noise_prefactor = math.sqrt(
        2.0 * gamma * temperature / dxdy
    )

    dW = complex_wiener_increment(
        psi.shape,
        psi.device,
        psi.real.dtype,
        dt_adim,
    ).to(psi.dtype)

    if noise_mask is not None:
        dW = noise_mask * dW

    return psi + deterministic + noise_prefactor * dW



@torch.no_grad()
def propagate_bilayer_sgpe(
    bilayer,
    final_time,
    time_step,
    J,
    temperature,
    gamma,
    chemical_potential,
    potentials1,
    potentials2,
    projector1=None,
    projector2=None,
    leave_progress_bar=True,
    density_relax_strength=0.015,
    phase_noise_boost=1.0,
    amplitude_noise_factor=0.15,
    normalize_every=50,
):
    import math
    import numpy as np
    import torch
    from tqdm.auto import tqdm

    gas1 = bilayer.layer1
    gas2 = bilayer.layer2

    psi1 = gas1.psi.clone()
    psi2 = gas2.psi.clone()

    # Public API uses seconds; GPE evolution uses dimensionless time.
    dt_SI = float(time_step)
    dt = dt_SI * float(gas1.adim_pulse)
    n_steps = int(np.ceil(float(final_time) / dt_SI))

    dx = float(gas1.dx)
    dy = float(gas1.dy)
    dxdy = dx * dy

    device = psi1.device
    dtype_real = psi1.real.dtype
    dtype_cplx = psi1.dtype

    Nx, Ny = psi1.shape

    # Momentum grid matching the actual wavefunction.
    kx = 2 * torch.pi * torch.fft.fftfreq(
        Nx, d=dx, device=device, dtype=dtype_real
    )
    ky = 2 * torch.pi * torch.fft.fftfreq(
        Ny, d=dy, device=device, dtype=dtype_real
    )
    KX, KY = torch.meshgrid(kx, ky, indexing="ij")
    K2 = KX**2 + KY**2

    kinetic_half = torch.exp(-0.5j * K2 * dt)

    gamma = float(gamma)
    temperature = float(temperature)
    chemical_potential = float(chemical_potential)

    # Preserve the initial relaxed density profile as the reservoir target.
    n_target1 = torch.abs(psi1) ** 2
    n_target2 = torch.abs(psi2) ** 2

    nmax1 = n_target1.max().clamp_min(1e-30)
    nmax2 = n_target2.max().clamp_min(1e-30)

    reservoir_mask1 = torch.sqrt(
        torch.clamp(n_target1 / nmax1, 0.0, 1.0)
    )
    reservoir_mask2 = torch.sqrt(
        torch.clamp(n_target2 / nmax2, 0.0, 1.0)
    )

    N_target1 = torch.sum(n_target1) * dxdy
    N_target2 = torch.sum(n_target2) * dxdy

    phase_noise_pref = (
        phase_noise_boost
        * math.sqrt(2.0 * gamma * temperature * dt)
    )

    amp_noise_pref = (
        amplitude_noise_factor
        * math.sqrt(2.0 * gamma * temperature * dt / dxdy)
    )

    for potential in potentials1:
        potential.set_gas(gas1)
        potential.on_propagation_begin()

    for potential in potentials2:
        potential.set_gas(gas2)
        potential.on_propagation_begin()

    def kinetic_step(psi):
        return torch.fft.ifftn(
            kinetic_half * torch.fft.fftn(psi)
        )

    def apply_projector_local(psi, projector):
        if projector is None:
            return psi

        return torch.fft.ifftn(
            projector.to(device=device, dtype=dtype_cplx)
            * torch.fft.fftn(psi)
        )

    def complex_noise(amplitude):
        return amplitude * (
            torch.randn(
                (Nx, Ny),
                device=device,
                dtype=dtype_real,
            )
            + 1j
            * torch.randn(
                (Nx, Ny),
                device=device,
                dtype=dtype_real,
            )
        ) / math.sqrt(2.0)

    def normalize_to_number(psi, target_number):
        number = torch.sum(torch.abs(psi) ** 2) * dxdy

        return psi * torch.sqrt(
            target_number / number.clamp_min(1e-30)
        )

    def density_relax(psi, target_density, strength):
        if strength <= 0:
            return psi

        phase = torch.angle(psi)
        amplitude = torch.abs(psi)

        relaxed_amplitude = (
            (1.0 - strength) * amplitude
            + strength * torch.sqrt(target_density + 1e-30)
        )

        return relaxed_amplitude * torch.exp(1j * phase)

    def local_energy(gas, psi, potentials, time_SI):
        # Some TorchGPE potentials use gas.psi internally.
        gas.psi = psi

        energy = torch.zeros(
            psi.shape,
            device=device,
            dtype=dtype_real,
        )

        for potential in potentials:
            value = None

            if hasattr(potential, "potential_function"):
                try:
                    value = potential.potential_function(
                        *gas.coordinates,
                        psi,
                        time=time_SI,
                    )
                except TypeError:
                    try:
                        value = potential.potential_function(
                            *gas.coordinates,
                            psi,
                        )
                    except TypeError:
                        pass

            if value is None and hasattr(potential, "get_potential"):
                try:
                    value = potential.get_potential(
                        *gas.coordinates,
                        time=time_SI,
                    )
                except TypeError:
                    value = potential.get_potential(
                        *gas.coordinates
                    )

            if value is not None:
                energy = energy + value.real

        return energy - chemical_potential

    iterator = range(n_steps)

    if leave_progress_bar:
        iterator = tqdm(iterator)

    for step in iterator:
        time_SI = step * dt_SI

        # Kinetic half-step
        psi1 = kinetic_step(psi1)
        psi2 = kinetic_step(psi2)

        H1 = local_energy(
            gas1, psi1, potentials1, time_SI
        )
        H2 = local_energy(
            gas2, psi2, potentials2, time_SI
        )

        J_now = J(time_SI) if callable(J) else J
        J_now = float(J_now)

        psi1_old = psi1
        psi2_old = psi2

        # Same explicit dissipative/Josephson step as SGPE_v2.
        psi1 = psi1 + (
            -(1j + gamma) * H1 * psi1_old
            + (1j + gamma) * J_now * psi2_old
        ) * dt

        psi2 = psi2 + (
            -(1j + gamma) * H2 * psi2_old
            + (1j + gamma) * J_now * psi1_old
        ) * dt

        # Phase-dominated thermal noise localized inside the cloud.
        xi1 = torch.randn(
            (Nx, Ny), device=device, dtype=dtype_real
        )
        xi2 = torch.randn(
            (Nx, Ny), device=device, dtype=dtype_real
        )

        psi1 = psi1 * torch.exp(
            1j * reservoir_mask1 * phase_noise_pref * xi1
        )
        psi2 = psi2 * torch.exp(
            1j * reservoir_mask2 * phase_noise_pref * xi2
        )

        # Weaker additive noise allows density holes and vortex cores.
        psi1 = (
            psi1
            + reservoir_mask1
            * complex_noise(amp_noise_pref)
        )
        psi2 = (
            psi2
            + reservoir_mask2
            * complex_noise(amp_noise_pref)
        )

        # Second kinetic half-step
        psi1 = kinetic_step(psi1)
        psi2 = kinetic_step(psi2)

        psi1 = apply_projector_local(psi1, projector1)
        psi2 = apply_projector_local(psi2, projector2)

        # Stabilize the cloud envelope while preserving its phase.
        psi1 = density_relax(
            psi1,
            n_target1,
            density_relax_strength,
        )
        psi2 = density_relax(
            psi2,
            n_target2,
            density_relax_strength,
        )

        if normalize_every and step % normalize_every == 0:
            psi1 = normalize_to_number(psi1, N_target1)
            psi2 = normalize_to_number(psi2, N_target2)

        if not torch.isfinite(psi1).all():
            raise RuntimeError(
                f"Layer 1 became non-finite at step {step}"
            )

        if not torch.isfinite(psi2).all():
            raise RuntimeError(
                f"Layer 2 became non-finite at step {step}"
            )

    gas1.psi = psi1
    gas2.psi = psi2

    for potential in potentials1:
        if hasattr(potential, "on_propagation_end"):
            potential.on_propagation_end()

    for potential in potentials2:
        if hasattr(potential, "on_propagation_end"):
            potential.on_propagation_end()

    return bilayer


