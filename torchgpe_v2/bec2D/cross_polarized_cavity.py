"""
Transversally pumped dispersive cavity with a polarization-resolved,
retro-reflected pump (e.g. the lin-perp-lin scheme of the Brantut group).

Model
-----
Pump (incoming + retro-reflected beam), each with real unit polarization:

    E_p(r) ∝ e^{+i k_p·r} e_in + e^{-i k_p·r} e_out ,   θ(r) = k_p·r

Polarizations are parametrized by an angle χ measured from the axis
perpendicular to the 2D plane (ẑ) towards the in-plane direction ẑ×p̂
(p̂ = pump propagation direction):

    e = cos(χ) ẑ + sin(χ) (ẑ×p̂)

    standing_wave (old TorchGPE):  χ_in = χ_out = 0        (lin ∥ lin)
    lin_perp_lin  (Brantut):       χ_in = -45°, χ_out = +45°

The cavity supports two degenerate standing-wave modes along ĉ with
polarizations ẑ ("out_of_plane") and ẑ×ĉ ("in_plane"). Overlaps:

    out_of_plane:  c_in = cos χ_in,               c_out = cos χ_out
    in_plane:      c_in = sin χ_in · (p̂·ĉ),       c_out = sin χ_out · (p̂·ĉ)

Scalar light shift |E_p + E_c|^2 / Δ_a gives (all in TorchGPE units)

    V_pump = sgn(Δa) V Er · ½ [1 + cos(χ_in-χ_out) cos 2θ]          (no lattice for ⊥)
    V_cav  = U0 cos²(k_c·r) Σ_m |α_m|²
    V_int  = 2 η Σ_m Re[ α_m f_m*(r) ]

with complex mode functions

    f_m(r) = cos(k_c·r) · (c_in,m e^{iθ} + c_out,m e^{-iθ}) / 2

and, after adiabatic elimination of each mode,

    α_m = (N η Θ_m + probe_m) / (Δ̃_c + iκ),   Θ_m = ∫ |ψ|² f_m  (complex)

For χ_in = χ_out = 0 this reduces exactly to
torchgpe.bec2D.potentials.DispersiveCavity (f = cos cos, real).

For lin-perp-lin:
  * the pump intensity is uniform -> only a constant offset V Er / 2,
  * the out-of-plane mode couples to cos(k_c r) cos(k_p r)        (Θ_z),
  * the in-plane mode couples to  cos(k_c r) sin(k_p r) · (p̂·ĉ)    (Θ_y),
  matching Eq. (8)-(10) of Zwettler et al., arXiv:2405.18204 (Methods).

`lattice_depth` keeps its old meaning: the depth V (in Er) the same pump
power would produce as a lin∥lin standing wave. So the same VP corresponds
to the same laser power in both schemes; with lin⊥lin the uniform offset
is V/2 (Brantut's "V0 = V_KD / 2").

Vector light shift (optional)
-----------------------------
A lin⊥lin pump has a polarization gradient (σ+ / lin / σ- with period
λ/2). Far from the fine-structure splitting this is irrelevant, but for
87Rb at tens of GHz from D2 (Δ ≪ Δ_FS ≈ 7 THz) it is NOT suppressed for
m_F ≠ 0. A pump-only vector term

    V_vec = sgn(Δa) V Er · ½ · r_v · sin(χ_out-χ_in) · sin 2θ

can be switched on with `vector_shift_ratio = r_v`, where (scalar-normalized)

    r_v ≈ g_F m_F (B̂·p̂) (1/Δ2 - 1/Δ1) / (2/Δ2 + 1/Δ1)

(≈ 0.25 for 87Rb |F=1,m_F=-1>, Δ2 = -76.6 GHz, B ∥ pump). The vector
pump-cavity cross terms are NOT included.
"""

from __future__ import annotations

from typing import Callable, Sequence, Union

import numpy as np
import scipy.constants as spconsts
import torch

from torchgpe.utils.potentials import (
    NonLinearPotential,
    any_time_dependent_variable,
    time_dependent_variable,
)


PUMP_SCHEMES = {
    # name: (chi_in, chi_out) in radians
    "standing_wave": (0.0, 0.0),
    "lin_perp_lin": (-np.pi / 4, np.pi / 4),
}

CAVITY_MODES = ("out_of_plane", "in_plane")


class CrossPolarizedCavity(NonLinearPotential):
    """Dispersive cavity pumped by a retro-reflected beam with arbitrary
    linear polarizations of the incoming and reflected beams.

    Args:
        lattice_depth: Pump strength V in units of Er, defined as the depth
            of the equivalent lin∥lin standing wave. Float or callable of time.
        atomic_detuning: Pump-atom detuning (Hz).
        cavity_detuning: Pump-cavity detuning (Hz). Float or callable.
        cavity_decay: Cavity field decay rate κ/2π (Hz).
        cavity_coupling: Single-atom coupling g0/2π (Hz).
        cavity_angle: Cavity axis angle in the 2D plane (rad).
        pump_angle: Pump propagation angle in the 2D plane (rad).
        pump_scheme: "standing_wave" or "lin_perp_lin". Ignored for any
            polarization explicitly given below.
        pump_polarization_in, pump_polarization_out: χ of incoming /
            reflected beam (rad), measured from the out-of-plane axis.
        cavity_modes: "auto", "out_of_plane", "in_plane", "both", or a
            sequence of mode names. "auto" keeps the out-of-plane mode and
            adds the in-plane one only if it actually couples.
        waist: Pump waist (m). Defaults to infinity.
        cavity_probe: On-axis probe, float or callable (same units as
            DispersiveCavity).
        probe_mode: Name of the mode the probe drives.
        vector_shift_ratio: r_v of the pump-only vector light shift (see
            module docstring). Defaults to 0.
    """

    def __init__(
        self,
        lattice_depth: Union[float, Callable],
        atomic_detuning: float,
        cavity_detuning: Union[float, Callable],
        cavity_decay: float,
        cavity_coupling: float,
        cavity_angle: float = 0.0,
        pump_angle: float = np.pi / 2,
        pump_scheme: str = "lin_perp_lin",
        pump_polarization_in: float = None,
        pump_polarization_out: float = None,
        cavity_modes: Union[str, Sequence[str]] = "auto",
        waist: float = np.inf,
        cavity_probe: Union[float, Callable] = 0,
        probe_mode: str = "out_of_plane",
        vector_shift_ratio: float = 0.0,
    ):
        super().__init__()

        if pump_scheme not in PUMP_SCHEMES:
            raise ValueError(
                f"pump_scheme must be one of {list(PUMP_SCHEMES)}, got {pump_scheme!r}"
            )

        chi_in, chi_out = PUMP_SCHEMES[pump_scheme]
        if pump_polarization_in is not None:
            chi_in = float(pump_polarization_in)
        if pump_polarization_out is not None:
            chi_out = float(pump_polarization_out)

        self.lattice_depth = lattice_depth
        self.atomic_detuning = atomic_detuning
        self.cavity_detuning = cavity_detuning
        self.cavity_decay = cavity_decay
        self.cavity_coupling = cavity_coupling
        self.cavity_angle = cavity_angle
        self.pump_angle = pump_angle
        self.pump_scheme = pump_scheme
        self.chi_in = chi_in
        self.chi_out = chi_out
        self.waist = waist
        self.cavity_probe = cavity_probe
        self.probe_mode = probe_mode
        self.vector_shift_ratio = float(vector_shift_ratio)

        self.modes = self._resolve_modes(cavity_modes)

        if probe_mode not in CAVITY_MODES:
            raise ValueError(f"probe_mode must be one of {CAVITY_MODES}")

    # ------------------------------------------------------------------
    # Geometry helpers
    # ------------------------------------------------------------------
    @property
    def in_plane_overlap(self) -> float:
        """p̂·ĉ: projection of the pump's in-plane polarization on the
        cavity's in-plane polarization."""
        return float(np.cos(self.pump_angle - self.cavity_angle))

    def mode_overlaps(self, mode: str):
        """(c_in, c_out) polarization overlaps of the two pump beams with
        a cavity mode."""
        if mode == "out_of_plane":
            return np.cos(self.chi_in), np.cos(self.chi_out)
        if mode == "in_plane":
            proj = self.in_plane_overlap
            return np.sin(self.chi_in) * proj, np.sin(self.chi_out) * proj
        raise ValueError(f"Unknown cavity mode {mode!r}")

    def _resolve_modes(self, cavity_modes):
        if isinstance(cavity_modes, str):
            if cavity_modes == "auto":
                modes = ["out_of_plane"]
                c_in, c_out = self.mode_overlaps("in_plane")
                if max(abs(c_in), abs(c_out)) > 1e-9:
                    modes.append("in_plane")
                return tuple(modes)
            if cavity_modes == "both":
                return CAVITY_MODES
            cavity_modes = (cavity_modes,)

        modes = tuple(cavity_modes)
        for m in modes:
            if m not in CAVITY_MODES:
                raise ValueError(f"Unknown cavity mode {m!r}; use {CAVITY_MODES}")
        if len(set(modes)) != len(modes):
            raise ValueError("Duplicate cavity modes.")
        return modes

    # ------------------------------------------------------------------
    # TorchGPE interface
    # ------------------------------------------------------------------
    def on_propagation_begin(self):
        gas = self.gas

        self.is_time_dependent = any_time_dependent_variable(
            self.cavity_detuning, self.lattice_depth, self.cavity_probe
        )

        self._cavity_detuning = time_dependent_variable(self.cavity_detuning)
        self._lattice_depth = time_dependent_variable(self.lattice_depth)
        self._cavity_probe = time_dependent_variable(self.cavity_probe)

        # --- constants (identical to DispersiveCavity) ---
        self.g0 = 2 * np.pi * self.cavity_coupling
        self._atomic_detuning = 2 * np.pi * self.atomic_detuning
        self.kappa = 2 * np.pi * self.cavity_decay
        self.freq_d2 = gas.d2_pulse
        self.lambda_pump = 2 * np.pi * spconsts.c / (self.freq_d2 + self._atomic_detuning)
        self.adim_lambda_pump = self.lambda_pump / gas.adim_length
        self.k_pump = 2 * np.pi / self.lambda_pump
        self.adim_k_pump = 2 * np.pi / self.adim_lambda_pump
        self.Er = 0.5 * (spconsts.hbar * self.k_pump) ** 2 / gas.mass
        self.U0 = self.g0**2 / self._atomic_detuning
        self._adim_waist = self.waist / gas.adim_length

        self.c1 = gas.N_particles * gas.dx * gas.dy
        self.c3 = self.c1 * self.U0
        self.eta_prefactor = (
            np.sqrt(self.Er * np.abs(self._atomic_detuning) / spconsts.hbar)
            * self.g0 / self._atomic_detuning
        )

        # --- coordinates ---
        R_pump = gas.X * np.cos(self.pump_angle) + gas.Y * np.sin(self.pump_angle)
        R_pump_orth = -gas.X * np.sin(self.pump_angle) + gas.Y * np.cos(self.pump_angle)
        R_cavity = gas.X * np.cos(self.cavity_angle) + gas.Y * np.sin(self.cavity_angle)

        theta = self.adim_k_pump * R_pump
        cos_c = torch.cos(self.adim_k_pump * R_cavity)

        self._gaussian_profile = (
            1 / (1 + (self.adim_lambda_pump * R_pump / (np.pi * self._adim_waist**2)) ** 2)
            * torch.exp(
                -2 * R_pump_orth**2
                / (self._adim_waist**2 + (self.adim_lambda_pump * R_pump / (np.pi * self._adim_waist)) ** 2)
            )
        )
        sqrt_profile = torch.sqrt(self._gaussian_profile)

        self.COS2 = cos_c**2

        # --- pump-only potential per unit V (adimensional) ---
        delta_chi = self.chi_out - self.chi_in
        pump_shape = 0.5 * (1 + np.cos(delta_chi) * torch.cos(2 * theta))
        if self.vector_shift_ratio != 0:
            pump_shape = pump_shape + 0.5 * self.vector_shift_ratio * np.sin(delta_chi) * torch.sin(2 * theta)
        self._pump_lattice = (
            np.sign(self._atomic_detuning) * self.Er * pump_shape
            / (spconsts.hbar * gas.adim_pulse) * self._gaussian_profile
        )

        self._cavity_lattice = self.COS2 * self.U0 / gas.adim_pulse

        # --- complex mode functions f_m(r), shape (n_modes, Nx, Ny) ---
        e_plus = torch.exp(1j * theta)
        e_minus = torch.conj(e_plus)
        fs = []
        for mode in self.modes:
            c_in, c_out = self.mode_overlaps(mode)
            fs.append(cos_c * sqrt_profile * (c_in * e_plus + c_out * e_minus) / 2)
        self.F = torch.stack(fs).to(gas.complex_dtype)
        self._F_conj = torch.conj(self.F)

        self._probe_vector = torch.zeros(len(self.modes), dtype=gas.complex_dtype, device=gas.device)
        if self.probe_mode in self.modes:
            self._probe_vector[self.modes.index(self.probe_mode)] = 1.0

    def get_order(self, psi: torch.Tensor):
        """Complex order parameters Θ_m = Σ |ψ|² f_m, one per mode."""
        density = torch.abs(psi) ** 2
        return (density.unsqueeze(0) * self.F).sum(dim=(-2, -1))

    def get_alpha_modes(self, psi: torch.Tensor, time: float = None):
        """Intracavity field of every active mode, shape (n_modes,)."""
        order = self.get_order(psi)
        bunching = (torch.abs(psi) ** 2 * self.COS2).sum()

        self.c2 = 2 * np.pi * self._cavity_detuning(time) + 1j * self.kappa
        self._cavity_detuning_tilde = 2 * np.pi * self._cavity_detuning(time) - self.c3 * bunching
        self.c6 = self.c2 - self.c3 * bunching
        self.eta = np.sqrt(self._lattice_depth(time)) * self.eta_prefactor

        probe = self._cavity_probe(time) * (self.Er / spconsts.hbar) * self._probe_vector
        return (self.c1 * self.eta * order + probe) / self.c6

    def get_alpha(self, psi: torch.Tensor, time: float = None):
        """Intracavity field. A scalar if a single mode is active (drop-in
        compatible with DispersiveCavity), otherwise shape (n_modes,)."""
        alpha = self.get_alpha_modes(psi, time)
        return alpha[0] if alpha.numel() == 1 else alpha

    def potential_function(self, X: torch.Tensor, Y: torch.Tensor, psi: torch.Tensor, time: float = None):
        alpha = self.get_alpha_modes(psi, time)

        self.pump_lattice = self._lattice_depth(time) * self._pump_lattice
        cavity_lattice = (torch.abs(alpha) ** 2).sum() * self._cavity_lattice
        interaction = (
            2 / self.gas.adim_pulse * self.eta
            * torch.real((alpha[:, None, None] * self._F_conj).sum(dim=0))
        )

        return self.pump_lattice + cavity_lattice + interaction

    def describe(self) -> str:
        lines = [
            f"CrossPolarizedCavity: scheme={self.pump_scheme}, "
            f"chi_in={np.degrees(self.chi_in):.1f} deg, chi_out={np.degrees(self.chi_out):.1f} deg, "
            f"pump_angle={np.degrees(self.pump_angle):.1f} deg, cavity_angle={np.degrees(self.cavity_angle):.1f} deg",
            f"  pump-lattice contrast cos(chi_out-chi_in) = {np.cos(self.chi_out - self.chi_in):+.3f}",
        ]
        for m in self.modes:
            c_in, c_out = self.mode_overlaps(m)
            lines.append(
                f"  mode {m:13s}: cos-coupling (c_in+c_out)/2 = {(c_in + c_out) / 2:+.3f}, "
                f"sin-coupling (c_out-c_in)/2 = {(c_out - c_in) / 2:+.3f}"
            )
        return "\n".join(lines)
