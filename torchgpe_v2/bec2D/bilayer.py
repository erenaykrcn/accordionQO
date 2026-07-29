from __future__ import annotations

from typing import Callable, Union
import math

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

    Parameters
    ----------
    psi1, psi2
        Layer wavefunctions.
    J
        J / hbar in rad/s.
    dt
        Physical time interval in seconds.
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
    """
    Deterministic bilayer split-step propagation.
    """
    potentials1 = [] if potentials1 is None else potentials1
    potentials2 = [] if potentials2 is None else potentials2

    gas1 = bilayer.layer1
    gas2 = bilayer.layer2

    J_t = time_dependent_variable(J)

    p1 = _prepare_potentials(gas1, potentials1)
    p2 = _prepare_potentials(gas2, potentials2)

    n_steps = max(1, round(final_time / time_step))
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

        psi1, psi2 = josephson_step(
            gas1.psi,
            gas2.psi,
            J_mid,
            dt / 2,
        )
        gas1.psi = psi1
        gas2.psi = psi2

        gas1.psik = gas1.psik * kinetic_half
        gas2.psik = gas2.psik * kinetic_half

        _potential_step(gas1, dt_adim, time_mid, *p1)
        _potential_step(gas2, dt_adim, time_mid, *p2)

        gas1.psik = gas1.psik * kinetic_half
        gas2.psik = gas2.psik * kinetic_half

        psi1, psi2 = josephson_step(
            gas1.psi,
            gas2.psi,
            J_mid,
            dt / 2,
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


def complex_wiener_increment(
    shape,
    device,
    dtype_real,
    dt_adim,
):
    """
    Complex Wiener increment satisfying approximately

        <dW*(r) dW(r')> = dt_adim

    before spatial-grid normalization.
    """
    re = torch.randn(shape, device=device, dtype=dtype_real)
    im = torch.randn(shape, device=device, dtype=dtype_real)

    return (re + 1j * im) * math.sqrt(dt_adim / 2.0)


def apply_projector(psi, projector):
    if projector is None:
        return psi

    projector = projector.to(device=psi.device, dtype=psi.dtype)
    return torch.fft.ifftn(torch.fft.fftn(psi) * projector)


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


def _scaled_relaxation_strength(
    full_step_strength: float,
    dt_sub: float,
    dt_full: float,
) -> float:
    """
    Rescale an amplitude-relaxation fraction for adaptive substeps.
    """
    if full_step_strength <= 0:
        return 0.0
    if full_step_strength >= 1:
        return 1.0

    return 1.0 - (1.0 - full_step_strength) ** (dt_sub / dt_full)


def _state_is_valid(
    psi: torch.Tensor,
    reference_density: torch.Tensor,
    density_guard_factor: float,
) -> bool:
    if not bool(torch.isfinite(psi).all()):
        return False

    max_density = torch.max(torch.abs(psi) ** 2)
    density_scale = reference_density.max().clamp_min(1e-30)

    return bool(max_density <= density_guard_factor * density_scale)


@torch.no_grad()
def propagate_bilayer_sgpe(
    bilayer,
    final_time,
    time_step,
    J,
    temperature,
    gamma,
    chemical_potential,
    potentials1=None,
    potentials2=None,
    projector1=None,
    projector2=None,
    leave_progress_bar=True,
    density_relax_strength=0.015,
    phase_noise_boost=1.0,
    amplitude_noise_factor=0.15,
    normalize_every=50,
    max_retry_levels=8,
    density_guard_factor=50.0,
    warn_on_retry=True,
):
    """
    Robust bilayer SGPE propagation.

    The public time arguments are in seconds. The GPE/SGPE update is carried
    out in TorchGPE dimensionless time using

        dt_adim = dt_SI * gas.adim_pulse.

    The routine preserves the phenomenological structure of the earlier SGPE:
    phase-dominated noise, weaker additive complex noise, momentum projection,
    density-envelope relaxation, and periodic number stabilization.

    Unstable steps are rolled back and retried with 2, 4, 8, ... substeps.
    The stochastic amplitudes are recomputed using sqrt(dt_sub), as required.

    Parameters
    ----------
    temperature
        Dimensionless temperature in the same energy units as the
        dimensionless Hamiltonian and chemical potential.
    J
        Either a constant J/hbar in rad/s or a callable J(t_SI).
    max_retry_levels
        Maximum number of binary timestep refinements. The smallest trial
        timestep is dt / 2**max_retry_levels.
    density_guard_factor
        Reject a trial if its maximum density exceeds this factor times the
        initial target peak density.
    """
    potentials1 = [] if potentials1 is None else potentials1
    potentials2 = [] if potentials2 is None else potentials2

    gas1 = bilayer.layer1
    gas2 = bilayer.layer2

    psi1 = gas1.psi.clone()
    psi2 = gas2.psi.clone()

    dt_SI = float(time_step)
    if dt_SI <= 0:
        raise ValueError("time_step must be positive.")
    if final_time <= 0:
        raise ValueError("final_time must be positive.")

    n_steps = max(1, int(np.ceil(float(final_time) / dt_SI)))
    dt_SI = float(final_time) / n_steps
    dt = dt_SI * float(gas1.adim_pulse)

    dx = float(gas1.dx)
    dy = float(gas1.dy)
    dxdy = dx * dy

    device = psi1.device
    dtype_real = psi1.real.dtype
    dtype_cplx = psi1.dtype

    nx, ny = psi1.shape

    kx = 2 * torch.pi * torch.fft.fftfreq(
        nx,
        d=dx,
        device=device,
        dtype=dtype_real,
    )
    ky = 2 * torch.pi * torch.fft.fftfreq(
        ny,
        d=dy,
        device=device,
        dtype=dtype_real,
    )
    KX, KY = torch.meshgrid(kx, ky, indexing="ij")
    kinetic = 0.5 * (KX**2 + KY**2)

    gamma = float(gamma)
    temperature = float(temperature)
    chemical_potential = float(chemical_potential)

    if gamma < 0:
        raise ValueError("gamma must be non-negative.")
    if temperature < 0:
        raise ValueError("temperature must be non-negative.")

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

    prepared1 = _prepare_potentials(gas1, potentials1)
    prepared2 = _prepare_potentials(gas2, potentials2)

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

    def complex_noise(amplitude):
        eta = (
            torch.randn(
                (nx, ny),
                device=device,
                dtype=dtype_real,
            )
            + 1j
            * torch.randn(
                (nx, ny),
                device=device,
                dtype=dtype_real,
            )
        ) / math.sqrt(2.0)

        return amplitude * eta.to(dtype_cplx)

    def kinetic_half_step(psi, dt_sub):
        propagator = torch.exp(-0.5j * kinetic * dt_sub)
        return torch.fft.ifftn(
            propagator * torch.fft.fftn(psi)
        )

    def local_energy(gas, psi, prepared, time_SI):
        gas.psi = psi
        return (
            _evaluate_total_potential(
                gas,
                time_SI,
                prepared,
            ).real
            - chemical_potential
        )

    def sgpe_microstep(
        psi1_in,
        psi2_in,
        dt_sub,
        dt_SI_sub,
        time_SI,
    ):
        psi1_sub = kinetic_half_step(psi1_in, dt_sub)
        psi2_sub = kinetic_half_step(psi2_in, dt_sub)

        H1 = local_energy(
            gas1,
            psi1_sub,
            prepared1,
            time_SI,
        )
        H2 = local_energy(
            gas2,
            psi2_sub,
            prepared2,
            time_SI,
        )

        J_now = J(time_SI) if callable(J) else J
        J_now = float(J_now)

        psi1_old = psi1_sub
        psi2_old = psi2_sub

        psi1_sub = psi1_old + (
            -(1j + gamma) * H1 * psi1_old
            + (1j + gamma) * J_now * psi2_old
        ) * dt_sub

        psi2_sub = psi2_old + (
            -(1j + gamma) * H2 * psi2_old
            + (1j + gamma) * J_now * psi1_old
        ) * dt_sub

        phase_noise_pref = (
            phase_noise_boost
            * math.sqrt(
                2.0 * gamma * temperature * dt_sub
            )
        )

        amp_noise_pref = (
            amplitude_noise_factor
            * math.sqrt(
                2.0 * gamma * temperature * dt_sub / dxdy
            )
        )

        xi1 = torch.randn(
            (nx, ny),
            device=device,
            dtype=dtype_real,
        )
        xi2 = torch.randn(
            (nx, ny),
            device=device,
            dtype=dtype_real,
        )

        psi1_sub = psi1_sub * torch.exp(
            1j * reservoir_mask1 * phase_noise_pref * xi1
        )
        psi2_sub = psi2_sub * torch.exp(
            1j * reservoir_mask2 * phase_noise_pref * xi2
        )

        psi1_sub = (
            psi1_sub
            + reservoir_mask1 * complex_noise(amp_noise_pref)
        )
        psi2_sub = (
            psi2_sub
            + reservoir_mask2 * complex_noise(amp_noise_pref)
        )

        psi1_sub = kinetic_half_step(psi1_sub, dt_sub)
        psi2_sub = kinetic_half_step(psi2_sub, dt_sub)

        psi1_sub = apply_projector(psi1_sub, projector1)
        psi2_sub = apply_projector(psi2_sub, projector2)

        alpha_sub = _scaled_relaxation_strength(
            density_relax_strength,
            dt_sub,
            dt,
        )

        psi1_sub = density_relax(
            psi1_sub,
            n_target1,
            alpha_sub,
        )
        psi2_sub = density_relax(
            psi2_sub,
            n_target2,
            alpha_sub,
        )

        return psi1_sub, psi2_sub

    iterator = trange(
        n_steps,
        leave=leave_progress_bar,
        desc="Bilayer SGPE propagation",
    )

    retry_count = 0
    max_retry_used = 0

    for step in iterator:
        time_SI = step * dt_SI

        psi1_checkpoint = psi1.clone()
        psi2_checkpoint = psi2.clone()

        accepted = False

        for retry_level in range(max_retry_levels + 1):
            n_substeps = 2**retry_level
            dt_sub = dt / n_substeps
            dt_SI_sub = dt_SI / n_substeps

            psi1_trial = psi1_checkpoint.clone()
            psi2_trial = psi2_checkpoint.clone()

            valid = True

            for substep in range(n_substeps):
                substep_time_SI = (
                    time_SI
                    + (substep + 0.5) * dt_SI_sub
                )

                psi1_trial, psi2_trial = sgpe_microstep(
                    psi1_trial,
                    psi2_trial,
                    dt_sub,
                    dt_SI_sub,
                    substep_time_SI,
                )

                valid1 = _state_is_valid(
                    psi1_trial,
                    n_target1,
                    density_guard_factor,
                )
                valid2 = _state_is_valid(
                    psi2_trial,
                    n_target2,
                    density_guard_factor,
                )

                if not valid1 or not valid2:
                    valid = False
                    break

            if valid:
                psi1 = psi1_trial
                psi2 = psi2_trial
                accepted = True

                if retry_level > 0:
                    retry_count += 1
                    max_retry_used = max(
                        max_retry_used,
                        retry_level,
                    )

                    if warn_on_retry:
                        print(
                            f"SGPE warning: step {step} accepted "
                            f"after {n_substeps} substeps."
                        )

                break

        if not accepted:
            max_n1 = float(
                torch.nan_to_num(
                    torch.abs(psi1_checkpoint) ** 2
                ).max().detach().cpu()
            )
            max_n2 = float(
                torch.nan_to_num(
                    torch.abs(psi2_checkpoint) ** 2
                ).max().detach().cpu()
            )

            raise RuntimeError(
                "SGPE step remained unstable after all retry levels. "
                f"step={step}, "
                f"time={time_SI:.6e} s, "
                f"smallest dt_adim={dt / 2**max_retry_levels:.3e}, "
                f"checkpoint max densities=({max_n1:.3e}, {max_n2:.3e})."
            )

        if normalize_every and (step + 1) % normalize_every == 0:
            psi1 = normalize_to_number(psi1, N_target1)
            psi2 = normalize_to_number(psi2, N_target2)

        gas1.psi = psi1
        gas2.psi = psi2

    gas1.psi = psi1
    gas2.psi = psi2

    for potential in potentials1:
        if hasattr(potential, "on_propagation_end"):
            potential.on_propagation_end()

    for potential in potentials2:
        if hasattr(potential, "on_propagation_end"):
            potential.on_propagation_end()

    if warn_on_retry and retry_count > 0:
        print(
            f"SGPE completed with {retry_count} retried main steps; "
            f"largest refinement was 2**{max_retry_used} substeps."
        )

    return bilayer
