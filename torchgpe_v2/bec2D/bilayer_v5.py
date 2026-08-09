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
    atom_number1=None,
    atom_number2=None,
    monitor_cavity=None,
    monitor_every=10,
    evolve_layer2=False,
    monitor_alpha=True,
):
    """
    TorchGPE-compatible bilayer SGPE.

    Hamiltonian evolution is performed with the same split-step structure
    as TorchGPE:

        kinetic half-step
        local potential full-step
        kinetic half-step

    Josephson coupling is Strang split around that evolution.

    Thermal damping/noise is then added as a separate projected stochastic
    correction.

    Supports nonlinear potentials with an `update_time(t)` method,
    e.g. DynamicContact with a time-dependent scattering length.

    Important limiting case
    -----------------------
    gamma = 0
    temperature = 0
    J = 0

    -> reduces to ordinary TorchGPE-style real-time propagation.
    """

    import math
    import numpy as np
    import torch
    from tqdm.auto import trange

    potentials1 = [] if potentials1 is None else potentials1
    potentials2 = [] if potentials2 is None else potentials2

    gas1 = bilayer.layer1
    gas2 = bilayer.layer2

    psi1 = gas1.psi.clone()
    psi2 = gas2.psi.clone()

    gamma = float(gamma)
    temperature = float(temperature)
    chemical_potential = float(chemical_potential)

    if final_time <= 0:
        raise ValueError("final_time must be positive.")

    if time_step <= 0:
        raise ValueError("time_step must be positive.")

    if gamma < 0:
        raise ValueError("gamma must be non-negative.")

    if temperature < 0:
        raise ValueError("temperature must be non-negative.")

    # ---------------------------------------------------------
    # Physical atom numbers represented by unit-normalized psi
    # ---------------------------------------------------------
    def infer_atom_number(gas, supplied):
        if supplied is not None:
            return float(supplied)

        if hasattr(gas, "N_particles"):
            return float(gas.N_particles)

        raise ValueError(
            "Could not infer atom number. "
            "Pass atom_number1/atom_number2 explicitly."
        )

    N_atoms1 = infer_atom_number(gas1, atom_number1)
    N_atoms2 = infer_atom_number(gas2, atom_number2)

    # ---------------------------------------------------------
    # Time
    # ---------------------------------------------------------
    n_steps = max(
        1,
        int(round(float(final_time) / float(time_step))),
    )

    dt_SI = float(final_time) / n_steps
    dt = dt_SI * float(gas1.adim_pulse)

    dx = float(gas1.dx)
    dy = float(gas1.dy)
    cell_area = dx * dy

    device = psi1.device
    real_dtype = psi1.real.dtype
    complex_dtype = psi1.dtype

    nx, ny = psi1.shape

    # ---------------------------------------------------------
    # Prepare TorchGPE potentials
    # ---------------------------------------------------------
    prepared1 = _prepare_potentials(
        gas1,
        potentials1,
    )

    prepared2 = (
        _prepare_potentials(
            gas2,
            potentials2,
        )
        if evolve_layer2
        else None
    )

    (
        static_linear1,
        dynamic_linear1,
        static_nonlinear1,
        dynamic_nonlinear1,
    ) = prepared1

    if evolve_layer2:
        (
            static_linear2,
            dynamic_linear2,
            static_nonlinear2,
            dynamic_nonlinear2,
        ) = prepared2

    # ---------------------------------------------------------
    # NEW:
    # update internal parameters of potentials such as
    # DynamicContact before evaluating their potential.
    #
    # This does NOT affect ordinary Contact, Trap, Cavity, etc.
    # ---------------------------------------------------------
    def update_internal_time_dependence(potentials, time_SI):
        for potential in potentials:
            if hasattr(potential, "update_time"):
                potential.update_time(time_SI)

    # ---------------------------------------------------------
    # Momentum grid matching the unpadded psi
    # ---------------------------------------------------------
    kx = 2 * torch.pi * torch.fft.fftfreq(
        nx,
        d=dx,
        device=device,
        dtype=real_dtype,
    )

    ky = 2 * torch.pi * torch.fft.fftfreq(
        ny,
        d=dy,
        device=device,
        dtype=real_dtype,
    )

    KX, KY = torch.meshgrid(
        kx,
        ky,
        indexing="ij",
    )

    kinetic = 0.5 * (KX**2 + KY**2)

    kinetic_half = torch.exp(
        -0.5j * kinetic * dt
    )

    # ---------------------------------------------------------
    # Basic operations
    # ---------------------------------------------------------
    def kinetic_half_step(psi):
        return torch.fft.ifftn(
            kinetic_half * torch.fft.fftn(psi)
        )

    def project(field, projector):
        if projector is None:
            return field

        projector = projector.to(
            device=field.device,
            dtype=field.dtype,
        )

        return torch.fft.ifftn(
            torch.fft.fftn(field) * projector
        )

    def inner(a, b):
        return (
            torch.sum(torch.conj(a) * b)
            * cell_area
        )

    def tangent_project(psi, field):
        norm = inner(psi, psi).real.clamp_min(1e-30)
        overlap = inner(psi, field)

        return field - psi * overlap / norm

    def kinetic_action(psi):
        return torch.fft.ifftn(
            kinetic * torch.fft.fftn(psi)
        )

    # ---------------------------------------------------------
    # TorchGPE-style potential step
    # ---------------------------------------------------------
    def potential_step(
        gas,
        psi,
        time_SI,
        static_linear,
        dynamic_linear,
        static_nonlinear,
        dynamic_nonlinear,
    ):
        gas.psi = psi

        total = static_linear.clone()

        # Ordinary TorchGPE dynamic linear potentials
        for potential in dynamic_linear:
            total = total + potential.get_potential(
                *gas.coordinates,
                time_SI,
            )

        # -------------------------------------------------
        # NEW:
        # DynamicContact remains classified as a static
        # nonlinear TorchGPE potential, because its
        # potential_function signature itself is unchanged.
        #
        # We only update its internal coupling g(t).
        # -------------------------------------------------
        update_internal_time_dependence(
            static_nonlinear,
            time_SI,
        )

        for potential in static_nonlinear:
            total = total + potential.potential_function(
                *gas.coordinates,
                psi,
            )

        # Existing genuinely dynamic nonlinear potentials
        for potential in dynamic_nonlinear:
            total = total + potential.potential_function(
                *gas.coordinates,
                psi,
                time_SI,
            )

        return psi * torch.exp(
            -1j * total * dt
        )

    def torchgpe_step_layer1(psi, time_SI):
        psi = kinetic_half_step(psi)

        psi = potential_step(
            gas1,
            psi,
            time_SI,
            static_linear1,
            dynamic_linear1,
            static_nonlinear1,
            dynamic_nonlinear1,
        )

        psi = kinetic_half_step(psi)

        return psi

    def torchgpe_step_layer2(psi, time_SI):
        psi = kinetic_half_step(psi)

        psi = potential_step(
            gas2,
            psi,
            time_SI,
            static_linear2,
            dynamic_linear2,
            static_nonlinear2,
            dynamic_nonlinear2,
        )

        psi = kinetic_half_step(psi)

        return psi

    # ---------------------------------------------------------
    # Josephson evolution
    # ---------------------------------------------------------
    def josephson_half_step(
        psi1_now,
        psi2_now,
        J_now,
    ):
        if J_now == 0:
            return psi1_now, psi2_now

        theta = torch.as_tensor(
            J_now * dt / 2,
            device=device,
            dtype=real_dtype,
        )

        c = torch.cos(theta)
        s = torch.sin(theta)

        p1 = psi1_now
        p2 = psi2_now

        return (
            c * p1 + 1j * s * p2,
            c * p2 + 1j * s * p1,
        )

    # ---------------------------------------------------------
    # Full GP action for dissipative SGPE part
    # ---------------------------------------------------------
    def gp_action(
        gas,
        psi,
        time_SI,
        prepared,
    ):
        gas.psi = psi

        (
            static_linear,
            dynamic_linear,
            static_nonlinear,
            dynamic_nonlinear,
        ) = prepared

        # -------------------------------------------------
        # NEW:
        # make sure dissipative Hamiltonian uses the same
        # instantaneous interaction strength.
        # -------------------------------------------------
        update_internal_time_dependence(
            static_nonlinear,
            time_SI,
        )

        update_internal_time_dependence(
            dynamic_nonlinear,
            time_SI,
        )

        local = _evaluate_total_potential(
            gas,
            time_SI,
            prepared,
        ).real

        return (
            kinetic_action(psi)
            + local * psi
        )

    def dissipative_drift(
        gas,
        psi,
        time_SI,
        prepared,
        projector,
    ):
        if gamma == 0:
            return torch.zeros_like(psi)

        Hpsi = gp_action(
            gas,
            psi,
            time_SI,
            prepared,
        )

        Kpsi = (
            Hpsi
            - chemical_potential * psi
        )

        damping = -gamma * tangent_project(
            psi,
            Kpsi,
        )

        return project(
            damping,
            projector,
        )

    # ---------------------------------------------------------
    # Thermal noise
    # ---------------------------------------------------------
    def complex_normal():
        re = torch.randn(
            (nx, ny),
            device=device,
            dtype=real_dtype,
        )

        im = torch.randn(
            (nx, ny),
            device=device,
            dtype=real_dtype,
        )

        return (
            re + 1j * im
        ).to(complex_dtype) / math.sqrt(2)

    def stochastic_increment(
        psi,
        atom_number,
        projector,
    ):
        if gamma == 0 or temperature == 0:
            return torch.zeros_like(psi)

        amplitude = math.sqrt(
            2.0
            * gamma
            * temperature
            * dt
            / (
                atom_number
                * cell_area
            )
        )

        noise = amplitude * complex_normal()

        noise = project(
            noise,
            projector,
        )

        return tangent_project(
            psi,
            noise,
        )

    # ---------------------------------------------------------
    # Norm correction
    # ---------------------------------------------------------
    norm1_target = (
        torch.sum(torch.abs(psi1)**2)
        * cell_area
    )

    norm2_target = (
        torch.sum(torch.abs(psi2)**2)
        * cell_area
    )

    total_norm_target = (
        norm1_target + norm2_target
    )

    def normalize_one(psi, target):
        norm = (
            torch.sum(torch.abs(psi)**2)
            * cell_area
        )

        return psi * torch.sqrt(
            target
            / norm.real.clamp_min(1e-30)
        )

    def normalize_total(p1, p2):
        norm = (
            torch.sum(torch.abs(p1)**2)
            + torch.sum(torch.abs(p2)**2)
        ) * cell_area

        scale = torch.sqrt(
            total_norm_target
            / norm.real.clamp_min(1e-30)
        )

        return p1 * scale, p2 * scale

    # ---------------------------------------------------------
    # Monitoring
    # ---------------------------------------------------------
    states = []
    cavity_times = []
    cavity_alpha = []

    iterator = trange(
        n_steps,
        leave=leave_progress_bar,
        desc="Bilayer TorchGPE-SGPE propagation",
    )

    for step in iterator:

        time_SI = step * dt_SI

        # -----------------------------------------------------
        # NEW:
        # Use midpoint for internally time-dependent couplings.
        #
        # This is important for a_s(t):
        # if ramp_time << dt, it correctly approaches an
        # instantaneous quench instead of spending the whole
        # first timestep at a_s_initial.
        # -----------------------------------------------------
        time_mid_SI = time_SI + 0.5 * dt_SI

        J_now = (
            float(J(time_mid_SI))
            if callable(J)
            else float(J)
        )

        # =====================================================
        # 1. Hamiltonian evolution
        # =====================================================

        if evolve_layer2 and J_now != 0:
            psi1, psi2 = josephson_half_step(
                psi1,
                psi2,
                J_now,
            )

        # Use midpoint for local Hamiltonian.
        psi1 = torchgpe_step_layer1(
            psi1,
            time_mid_SI,
        )

        if evolve_layer2:
            psi2 = torchgpe_step_layer2(
                psi2,
                time_mid_SI,
            )

        if evolve_layer2 and J_now != 0:
            psi1, psi2 = josephson_half_step(
                psi1,
                psi2,
                J_now,
            )

        # =====================================================
        # 2. Dissipative SGPE correction
        # =====================================================

        if gamma != 0:

            damp1 = dissipative_drift(
                gas1,
                psi1,
                time_mid_SI,
                prepared1,
                projector1,
            )

            psi1 = psi1 + damp1 * dt

            if evolve_layer2:
                damp2 = dissipative_drift(
                    gas2,
                    psi2,
                    time_mid_SI,
                    prepared2,
                    projector2,
                )

                psi2 = psi2 + damp2 * dt

        # =====================================================
        # 3. Thermal stochastic correction
        # =====================================================

        if gamma != 0 and temperature != 0:

            psi1 = psi1 + stochastic_increment(
                psi1,
                N_atoms1,
                projector1,
            )

            if evolve_layer2:
                psi2 = psi2 + stochastic_increment(
                    psi2,
                    N_atoms2,
                    projector2,
                )

        # =====================================================
        # 4. Numerical norm correction
        # =====================================================

        if gamma != 0 or temperature != 0:

            if evolve_layer2:
                psi1, psi2 = normalize_total(
                    psi1,
                    psi2,
                )
            else:
                psi1 = normalize_one(
                    psi1,
                    norm1_target,
                )

        # =====================================================
        # 5. Update actual Gas objects BEFORE monitoring
        # =====================================================

        gas1.psi = psi1

        if evolve_layer2:
            gas2.psi = psi2

        # =====================================================
        # 6. Check numerical stability
        # =====================================================

        finite = (
            torch.isfinite(psi1.real).all()
            and torch.isfinite(psi1.imag).all()
        )

        if evolve_layer2:
            finite = (
                finite
                and torch.isfinite(psi2.real).all()
                and torch.isfinite(psi2.imag).all()
            )

        if not bool(finite):
            raise RuntimeError(
                f"SGPE became non-finite at step {step}, "
                f"time={time_SI:.6e} s."
            )

        # =====================================================
        # 7. Monitoring
        # =====================================================

        if (
            monitor_cavity is not None
            and step % monitor_every == 0
        ):
            states.append(
                psi1.clone()
            )

            if monitor_alpha:
                t_now = (step + 1) * dt_SI

                cavity_times.append(
                    t_now
                )

                cavity_alpha.append(
                    monitor_cavity.get_alpha(
                        gas1.psi,
                        time=t_now,
                    ).detach().cpu()
                )

    # ---------------------------------------------------------
    # Potential cleanup
    # ---------------------------------------------------------
    for potential in potentials1:
        if hasattr(
            potential,
            "on_propagation_end",
        ):
            potential.on_propagation_end()

    if evolve_layer2:
        for potential in potentials2:
            if hasattr(
                potential,
                "on_propagation_end",
            ):
                potential.on_propagation_end()

    # ---------------------------------------------------------
    # Return
    # ---------------------------------------------------------
    if monitor_cavity is None:
        return bilayer

    return {
        "bilayer": bilayer,
        "times": torch.tensor(
            cavity_times,
            dtype=torch.float64,
        ),
        "alpha": (
            torch.stack(cavity_alpha)
            if cavity_alpha
            else torch.empty(
                0,
                dtype=torch.complex128,
            )
        ),
        "states": states,
    }