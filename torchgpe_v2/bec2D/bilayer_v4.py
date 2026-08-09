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
    monitor_alpha=True
):
    if not evolve_layer2:
        print('propagating one layer')

    """
    Number-conserving projected SGPE compatible with TorchGPE's
    unit-normalized wavefunction convention.

    TorchGPE convention
    -------------------
        integral |psi_j|^2 d^2r = 1

    while the physical atom number N_j is already included in the
    nonlinear interaction potential.

    The normalized stochastic field approximately obeys

        dpsi_j =
            -i K_j psi_j dt
            -gamma Q_j[K_j psi_j] dt
            +Q_j[dW_j],

    where Q_j projects perpendicular to psi_j and

        <dW_j*(r) dW_j(r')>
            = 2 gamma T / N_j
              delta(r-r') dt.

    The Hamiltonian operator is

        K_1 psi_1 =
            (H_GP,1 - mu) psi_1 - J psi_2,

        K_2 psi_2 =
            (H_GP,2 - mu) psi_2 - J psi_1.

    Notes
    -----
    - No density-envelope relaxation is used.
    - No phenomenological multiplicative phase noise is used.
    - No grand-canonical growth of the field norm is allowed.
    - A small total-norm correction is applied every step only to
      remove numerical integration error.
    - Josephson population transfer between layers remains possible
      because only the total bilayer norm is corrected.
    """

    potentials1 = [] if potentials1 is None else potentials1
    potentials2 = [] if potentials2 is None else potentials2

    gas1 = bilayer.layer1
    gas2 = bilayer.layer2

    psi1 = gas1.psi.clone()
    psi2 = gas2.psi.clone()

    if final_time <= 0:
        raise ValueError("final_time must be positive.")

    if time_step <= 0:
        raise ValueError("time_step must be positive.")

    gamma = float(gamma)
    temperature = float(temperature)
    chemical_potential = float(chemical_potential)

    if gamma < 0:
        raise ValueError("gamma must be non-negative.")

    if temperature < 0:
        raise ValueError("temperature must be non-negative.")

    # -------------------------------------------------------------
    # Infer physical atom numbers represented by the normalized
    # TorchGPE fields.
    # -------------------------------------------------------------
    def infer_atom_number(gas, supplied_value, layer_name):
        if supplied_value is not None:
            number = float(supplied_value)
        else:
            candidate_names = (
                "N_particles",
                "n_particles",
                "N_atoms",
                "n_atoms",
                "atom_number",
                "number_of_atoms",
            )

            number = None

            for name in candidate_names:
                if hasattr(gas, name):
                    candidate = getattr(gas, name)

                    if candidate is not None:
                        number = float(candidate)
                        break

            if number is None and hasattr(bilayer, "N_particles"):
                candidate = getattr(bilayer, "N_particles")

                if isinstance(candidate, (tuple, list)):
                    index = 0 if layer_name == "layer1" else 1
                    number = float(candidate[index])
                else:
                    number = float(candidate)

        if number is None:
            raise ValueError(
                f"Could not infer the physical atom number for "
                f"{layer_name}. Pass atom_number1 and atom_number2 "
                f"explicitly, or store N_particles on each gas object."
            )

        if number <= 0:
            raise ValueError(
                f"Physical atom number for {layer_name} must be positive."
            )

        return number

    N_atoms1 = infer_atom_number(
        gas1,
        atom_number1,
        "layer1",
    )

    N_atoms2 = infer_atom_number(
        gas2,
        atom_number2,
        "layer2",
    )

    # -------------------------------------------------------------
    # Time and grid parameters
    # -------------------------------------------------------------
    n_steps = max(
        1,
        int(np.ceil(float(final_time) / float(time_step))),
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

    # Initial total TorchGPE normalization, normally approximately 2.
    target_total_norm = (
        torch.sum(torch.abs(psi1) ** 2)
        + torch.sum(torch.abs(psi2) ** 2)
    ) * cell_area


    target_norm1 = (
        torch.sum(torch.abs(psi1) ** 2) * cell_area
    )

    def normalize_layer1(psi):
        current_norm = (
            torch.sum(torch.abs(psi) ** 2) * cell_area
        )

        scale = torch.sqrt(
            target_norm1
            / current_norm.real.clamp_min(1e-30)
        )

        return psi * scale



    # -------------------------------------------------------------
    # Momentum-space kinetic operator
    # -------------------------------------------------------------
    kx = 2.0 * torch.pi * torch.fft.fftfreq(
        nx,
        d=dx,
        device=device,
        dtype=real_dtype,
    )

    ky = 2.0 * torch.pi * torch.fft.fftfreq(
        ny,
        d=dy,
        device=device,
        dtype=real_dtype,
    )

    KX, KY = torch.meshgrid(kx, ky, indexing="ij")
    kinetic_energy = 0.5 * (KX**2 + KY**2)

    prepared1 = _prepare_potentials(
        gas1,
        potentials1,
    )

    prepared2 = _prepare_potentials(gas2, potentials2) if evolve_layer2 else None

    def project_momentum(field, projector):
        if projector is None:
            return field

        return apply_projector(field, projector)

    def inner_product(field1, field2):
        return (
            torch.sum(torch.conj(field1) * field2)
            * cell_area
        )

    def tangent_projection(field, direction):
        """
        Project direction perpendicular to field:

            Q_psi f = f - psi <psi|f>/<psi|psi>.

        This removes changes parallel to psi, which would change
        the field norm.
        """
        norm = inner_product(field, field).real.clamp_min(1e-30)
        overlap = inner_product(field, direction)

        return direction - field * overlap / norm

    def kinetic_action(field):
        return torch.fft.ifftn(
            kinetic_energy * torch.fft.fftn(field)
        )

    def gp_action(
        gas,
        field,
        prepared_potentials,
        time_SI,
    ):
        """
        Return H_GP[field] field.

        The nonlinear potentials are those already used by TorchGPE.
        Their coefficients should therefore already contain the
        physical atom number represented by the unit-normalized field.
        """
        gas.psi = field

        local_potential = _evaluate_total_potential(
            gas,
            time_SI,
            prepared_potentials,
        ).real

        return (
            kinetic_action(field)
            + local_potential * field
        )

    def grand_potential_gradient(
        psi1_now,
        psi2_now,
        time_SI,
    ):
        J_now = J(time_SI) if callable(J) else J
        J_now = float(J_now)

        Hpsi1 = gp_action(
            gas1,
            psi1_now,
            prepared1,
            time_SI,
        )

        Hpsi2 = gp_action(
            gas2,
            psi2_now,
            prepared2,
            time_SI,
        )

        Kpsi1 = (
            Hpsi1
            - chemical_potential * psi1_now
            - J_now * psi2_now
        )

        Kpsi2 = (
            Hpsi2
            - chemical_potential * psi2_now
            - J_now * psi1_now
        )

        return Kpsi1, Kpsi2

    def deterministic_drift(
        psi1_now,
        psi2_now,
        time_SI,
    ):
        J_now = J(time_SI) if callable(J) else J
        J_now = float(J_now)

        Hpsi1 = gp_action(
            gas1,
            psi1_now,
            prepared1,
            time_SI,
        )

        if evolve_layer2:
            Hpsi2 = gp_action(
                gas2,
                psi2_now,
                prepared2,
                time_SI,
            )

            Kpsi1 = (
                Hpsi1
                - chemical_potential * psi1_now
                - J_now * psi2_now
            )

            Kpsi2 = (
                Hpsi2
                - chemical_potential * psi2_now
                - J_now * psi1_now
            )
        else:
            # Single-layer evolution: no Josephson term
            Kpsi1 = Hpsi1 - chemical_potential * psi1_now
            Kpsi2 = None

        drift1 = (
            -1j * Kpsi1
            - gamma * tangent_projection(psi1_now, Kpsi1)
        )

        drift1 = project_momentum(drift1, projector1)

        if not evolve_layer2:
            return drift1, None

        drift2 = (
            -1j * Kpsi2
            - gamma * tangent_projection(psi2_now, Kpsi2)
        )

        drift2 = project_momentum(drift2, projector2)

        return drift1, drift2

    def complex_normal():
        """
        Complex Gaussian field eta satisfying

            <|eta_i|^2> = 1
        """
        real_part = torch.randn(
            (nx, ny),
            device=device,
            dtype=real_dtype,
        )

        imaginary_part = torch.randn(
            (nx, ny),
            device=device,
            dtype=real_dtype,
        )

        return (
            real_part + 1j * imaginary_part
        ).to(complex_dtype) / math.sqrt(2.0)

    def stochastic_increment(
        field,
        atom_number,
        projector,
        gaussian_field,
    ):
        """
        Noise for the unit-normalized TorchGPE field.

        Starting from Psi = sqrt(N) psi gives

            dpsi = dPsi / sqrt(N),

        hence the 1/sqrt(N) factor.
        """
        amplitude = math.sqrt(
            2.0
            * gamma
            * temperature
            * dt
            / (atom_number * cell_area)
        )

        raw_increment = amplitude * gaussian_field

        raw_increment = project_momentum(
            raw_increment,
            projector,
        )

        return tangent_projection(
            field,
            raw_increment,
        )

    def normalize_total(psi1_now, psi2_now):
        """
        Correct only accumulated numerical error in the total norm.

        This does not independently renormalize the two layers, so
        Josephson population transfer is retained.
        """
        current_total_norm = (
            torch.sum(torch.abs(psi1_now) ** 2)
            + torch.sum(torch.abs(psi2_now) ** 2)
        ) * cell_area

        scale = torch.sqrt(
            target_total_norm
            / current_total_norm.real.clamp_min(1e-30)
        )

        return (
            psi1_now * scale,
            psi2_now * scale,
        )

    # Apply the momentum projector to the initial fields.
    psi1 = project_momentum(psi1, projector1)

    if evolve_layer2:
        psi2 = project_momentum(psi2, projector2)
        psi1, psi2 = normalize_total(psi1, psi2)
    else:
        psi1 = normalize_layer1(psi1)

    iterator = trange(
        n_steps,
        leave=leave_progress_bar,
        desc="Bilayer canonical SGPE propagation",
    )

    states = []
    cavity_times = []
    cavity_alpha = []
    for step in iterator:
        time_SI = step * dt_SI
        next_time_SI = (step + 1) * dt_SI

        # Independent reservoir noise before tangent projection.
        gaussian1 = complex_normal()
        gaussian2 = complex_normal()

        noise1 = stochastic_increment(
            psi1,
            N_atoms1,
            projector1,
            gaussian1,
        )

        noise2 = stochastic_increment(
            psi2,
            N_atoms2,
            projector2,
            gaussian2,
        )


        drift1, drift2 = deterministic_drift(
            psi1,
            psi2,
            time_SI,
        )

        psi1_predictor = project_momentum(
            psi1 + drift1 * dt + noise1,
            projector1,
        )

        if evolve_layer2:
            psi2_predictor = project_momentum(
                psi2 + drift2 * dt + noise2,
                projector2,
            )
        else:
            psi2_predictor = psi2

        drift1_predictor, drift2_predictor = deterministic_drift(
            psi1_predictor,
            psi2_predictor,
            next_time_SI,
        )

        # Since tangent-projected noise depends on psi, recompute its
        # projection at the predictor while using the same Gaussian
        # realization.
        noise1_predictor = stochastic_increment(
            psi1_predictor,
            N_atoms1,
            projector1,
            gaussian1,
        )

        psi1 = project_momentum(
            psi1
            + 0.5 * (drift1 + drift1_predictor) * dt
            + 0.5 * (noise1 + noise1_predictor),
            projector1,
        )

        if evolve_layer2:
            noise2_predictor = stochastic_increment(
                psi2_predictor,
                N_atoms2,
                projector2,
                gaussian2,
            )

            psi2 = project_momentum(
                psi2
                + 0.5 * (drift2 + drift2_predictor) * dt
                + 0.5 * (noise2 + noise2_predictor),
                projector2,
            )
            
        # Numerical norm correction only.
        if evolve_layer2:
            psi1, psi2 = normalize_total(psi1, psi2)
        else:
            psi1 = normalize_layer1(psi1)

        # numerical norm correction
        if evolve_layer2:
            psi1, psi2 = normalize_total(psi1, psi2)
        else:
            psi1 = normalize_layer1(psi1)

        # IMPORTANT: update Gas BEFORE monitoring
        gas1.psi = psi1

        if evolve_layer2:
            gas2.psi = psi2

        if monitor_cavity is not None and step % monitor_every == 0:
            states.append(psi1.clone())

            if monitor_alpha:
                t_now = (step + 1) * dt_SI
                cavity_times.append(t_now)

                cavity_alpha.append(
                    monitor_cavity.get_alpha(
                        gas1.psi,
                        time=t_now,
                    ).detach().cpu()
                )

    for potential in potentials1:
        if hasattr(potential, "on_propagation_end"):
            potential.on_propagation_end()

    for potential in potentials2:
        if hasattr(potential, "on_propagation_end"):
            potential.on_propagation_end()

    if monitor_cavity is None:
        return bilayer
    else:
        return {
            "bilayer": bilayer,
            "times": torch.tensor(cavity_times),
            "alpha": (
                torch.stack(cavity_alpha)
                if cavity_alpha
                else torch.empty(0, dtype=torch.complex128)
            ),
            "states": states
        }
