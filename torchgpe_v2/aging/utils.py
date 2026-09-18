from pathlib import Path

import h5py
import numpy as np
import torch
import matplotlib.pyplot as plt

def to_numpy(x):
    """Convert Torch tensors or array-like objects to NumPy arrays."""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


    
def save_quench_run(
    output_dir,
    temperature,
    result1,
    result2,
    cavity_monitor1,
    cavity_monitor2,
    *,
    N_particles1,
    N_particles2,
    gamma,
    dt,
    thermalization_time,
    grid_size,
    final_time1,
    final_time2,
    J,
    detuning,
    VP,
    a_s,
    N_grid,
    prefix,
):
    """
    Save the SGPE states, cavity fields, and run parameters in HDF5 format.
    """
    output_path, run_id = get_next_quench_path(
        output_dir=output_dir,
        temperature=temperature,
        gamma=gamma,
        N_particles1=N_particles1,
        N_particles2=N_particles2,
        thermalization_time=thermalization_time,
        grid_size=grid_size,
        J=J,
        detuning=detuning,
        VP=VP,
        a_s=a_s,
        N_grid=N_grid,
        prefix=prefix,
    )

    states1 = to_numpy(result1["states"])
    states2 = to_numpy(result2["states"])

    if "alpha" not in result1:
        raise KeyError(
            "result1 does not contain 'alpha'. "
            "Check monitor_cavity and monitor_every in propagate_bilayer_sgpe."
        )

    if "alpha" not in result2:
        raise KeyError(
            "result2 does not contain 'alpha'. "
            "Check monitor_cavity and monitor_every in propagate_bilayer_sgpe."
        )

    alpha1 = to_numpy(result1["alpha"])
    alpha2 = to_numpy(result2["alpha"])

    with h5py.File(output_path, "x") as h5file:
        # "x" creates a new file and raises an error if it already exists,
        # providing an additional safeguard against overwriting.

        states_group = h5file.create_group("states")

        states_group.create_dataset(
            "pump_ramp",
            data=states1,
            compression="gzip",
            compression_opts=4,
            shuffle=True,
        )

        states_group.create_dataset(
            "post_quench",
            data=states2,
            compression="gzip",
            compression_opts=4,
            shuffle=True,
        )

        cavity_group = h5file.create_group("cavity")

        cavity_group.create_dataset(
            "alpha_pump_ramp",
            data=alpha1,
            compression="gzip",
            compression_opts=4,
            shuffle=True,
        )

        cavity_group.create_dataset(
            "alpha_post_quench",
            data=alpha2,
            compression="gzip",
            compression_opts=4,
            shuffle=True,
        )

        # Store scalar run parameters as file attributes.
        h5file.attrs["run_id"] = run_id
        h5file.attrs["temperature"] = temperature
        h5file.attrs["N_particles1"] = N_particles1
        h5file.attrs["N_particles2"] = N_particles2
        h5file.attrs["gamma"] = gamma
        h5file.attrs["dt"] = dt
        h5file.attrs["thermalization_time"] = thermalization_time
        h5file.attrs["grid_size"] = grid_size
        h5file.attrs["final_time1"] = final_time1
        h5file.attrs["final_time2"] = final_time2
        h5file.attrs["J"] = J
        h5file.attrs["detuning"] = detuning

    print(f"Saved run {run_id} to: {output_path}")

    return output_path



def get_next_state_path(
    output_dir,
    temperature,
    thermalization_time,
    omegar,
    grid_size,
    N_particles,
    prefix="thermal_state",
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    run_id = 0

    while True:
        filename = (
            f"{prefix}"
            f"_T{temperature:g}"
            f"_tth{thermalization_time:g}"
            f"_wr{omegar:g}"
            f"_L{grid_size:g}"
            f"_N{N_particles:g}"
            f"_id{run_id:03d}.hdf5"
        )

        path = output_dir / filename

        if not path.exists():
            return path, run_id

        run_id += 1

def get_next_quench_path(
    output_dir,
    temperature,
    gamma,
    N_particles1,
    N_particles2,
    thermalization_time,
    grid_size,
    J,
    detuning,
    VP,
    a_s,
    N_grid,
    prefix="SO_quench",
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    run_id = 0

    while True:
        filename = (
            f"{prefix}"
            f"_T{temperature:g}"
            f"_g{gamma:g}"
            f"_N1{N_particles1:g}"
            f"_N2{N_particles2:g}"
            f"_tth{thermalization_time:g}"
            f"_L{grid_size:g}"
            f"_J{J:g}"
            f"_D{detuning:g}"
            f"_VP{VP:g}"
            f"_a_s{a_s:g}"
            f"N_grid{N_grid:g}"
            f"_id{run_id:03d}.hdf5"
        )

        path = output_dir / filename

        if not path.exists():
            return path, run_id

        run_id += 1


def save_state(
    output_dir,
    temperature,
    state,
    *,
    thermalization_time,
    omegar,
    grid_size,
    N_particles,
    prefix="thermal_state",
):
    output_path, run_id = get_next_state_path(
        output_dir=output_dir,
        temperature=temperature,
        thermalization_time=thermalization_time,
        omegar=omegar,
        grid_size=grid_size,
        N_particles=N_particles,
        prefix=prefix,
    )

    state = to_numpy(state)

    with h5py.File(output_path, "x") as h5file:
        h5file.create_dataset(
            "state",
            data=state,
            compression="gzip",
            compression_opts=4,
            shuffle=True,
        )

        h5file.attrs["run_id"] = run_id
        h5file.attrs["temperature"] = temperature
        h5file.attrs["thermalization_time"] = thermalization_time
        h5file.attrs["omegar"] = omegar
        h5file.attrs["grid_size"] = grid_size
        h5file.attrs["N_particles"] = N_particles

    print(f"Saved run {run_id} to: {output_path}")

    return output_path


def plot_psi(
    psi, x_um, y_um,
    density_mask=None,
    density_threshold=0.05,
    show=True,
    fig=None
):
    X_um, Y_um = np.meshgrid(x_um, y_um, indexing="ij")

    dens_f = (torch.abs(psi) ** 2).detach().cpu().numpy()
    phase_f = torch.angle(psi).detach().cpu().numpy()

    extent = [x_um.min(), x_um.max(), y_um.min(), y_um.max()]

    if fig is None:
        fig0, axes0 = plt.subplots(1, 2, figsize=(10, 4))

        im2 = axes0[0].imshow(
            dens_f, origin="lower", extent=extent, aspect="equal"
        )
        axes0[0].set_title("Relaxed density")
        axes0[0].set_xlabel("x (µm)")
        axes0[0].set_ylabel("y (µm)")
        axes0[0].grid(False)
        plt.colorbar(im2, ax=axes0[0])

        im3 = axes0[1].imshow(
            phase_f, origin="lower", extent=extent, aspect="equal"
        )
        axes0[1].set_title("Relaxed phase")
        axes0[1].set_xlabel("x (µm)")
        axes0[1].set_ylabel("y (µm)")
        axes0[1].grid(False)
        plt.colorbar(im3, ax=axes0[1])

        plt.tight_layout()

        if show:
            plt.show()

    psi_np = psi.detach().cpu().numpy()
    phase = np.angle(psi_np)
    density = np.abs(psi_np)**2

    vort, antiv = detect_vortices_masked(
        psi,
        density_mask=density_mask,
        density_threshold=density_threshold
    )

    x_v = np.interp(vort[:, 1], np.arange(len(x_um)), x_um)
    y_v = np.interp(vort[:, 0], np.arange(len(y_um)), y_um)

    x_av = np.interp(antiv[:, 1], np.arange(len(x_um)), x_um)
    y_av = np.interp(antiv[:, 0], np.arange(len(y_um)), y_um)

    if fig is None:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    else:
        fig.clear()
        axes = fig.subplots(1, 2)

    im0 = axes[0].imshow(
        density,
        origin="lower",
        extent=extent,
        aspect="equal",
    )

    axes[0].scatter(
        x_v, y_v,
        s=12,
        facecolors="none",
        edgecolors="red",
        label=f"vortex ({len(vort)})",
    )
    axes[0].scatter(
        x_av, y_av,
        s=12,
        marker="x",
        color="yellow",
        label=f"antivortex ({len(antiv)})",
    )

    axes[0].set_title("Density with vortices")
    axes[0].set_xlabel("x (µm)")
    axes[0].set_ylabel("y (µm)")
    axes[0].legend()

    im1 = axes[1].imshow(
        phase_f,
        origin="lower",
        extent=extent,
        aspect="equal",
        vmin=-np.pi,
        vmax=np.pi,
    )

    axes[1].scatter(
        x_v, y_v,
        s=12,
        facecolors="none",
        edgecolors="red",
    )
    axes[1].scatter(
        x_av, y_av,
        s=12,
        marker="x",
        color="yellow",
    )

    axes[1].set_title("Phase with vortices")
    axes[1].set_xlabel("x (µm)")
    axes[1].set_ylabel("y (µm)")

    axes[0].grid(False)
    axes[1].grid(False)

    plt.colorbar(im0, ax=axes[0])
    plt.colorbar(im1, ax=axes[1], label="Phase")

    fig.tight_layout()

    if show:
        plt.show()

    return fig


from matplotlib.colors import LogNorm
from dataclasses import dataclass
 
 
@dataclass
class PsiKData:
    kx: np.ndarray          # (Nx,) signed k-axis, 1/um, ascending
    ky: np.ndarray          # (Ny,) signed k-axis, 1/um, ascending
    psi_k: torch.Tensor     # complex, (Nx, Ny), fftshifted, continuum-normalized
    dens_k: np.ndarray      # |psi_k|^2, (Nx, Ny)  == n(k)
    phase_k: np.ndarray     # angle(psi_k), (Nx, Ny)
 
 
# ---------------------------------------------------------------------------
# Core FFT: separated out so it can be reused by plotting AND by anything
# else (slices, fits, T-sweeps) without re-plotting every time.
# ---------------------------------------------------------------------------
def compute_psi_k(psi, x_um, y_um):
    """
    FFT psi(x,y) -> psi_k(kx,ky) with the phase correction for a grid not
    centered on index 0, and continuum-FT normalization (dx*dy prefactor),
    so sum(dens_k)*dkx*dky/(2*pi)**2 == sum(|psi|^2)*dx*dy  (Parseval).
 
    Returns a PsiKData bundle -- no plotting.
    """
    device = psi.device
    dtype = torch.float64 if psi.dtype == torch.complex128 else torch.float32
 
    nx, ny = psi.shape
    dx = float(x_um[1] - x_um[0])
    dy = float(y_um[1] - y_um[0])
 
    kx_raw = torch.as_tensor(2 * np.pi * np.fft.fftfreq(nx, d=dx), device=device, dtype=dtype)
    ky_raw = torch.as_tensor(2 * np.pi * np.fft.fftfreq(ny, d=dy), device=device, dtype=dtype)
    KX_raw, KY_raw = torch.meshgrid(kx_raw, ky_raw, indexing="ij")
    phase_corr = torch.exp(-1j * (KX_raw * float(x_um[0]) + KY_raw * float(y_um[0])))
 
    psi_k = torch.fft.fft2(psi) * phase_corr * dx * dy
    psi_k = torch.fft.fftshift(psi_k)
 
    kx = np.fft.fftshift(2 * np.pi * np.fft.fftfreq(nx, d=dx))
    ky = np.fft.fftshift(2 * np.pi * np.fft.fftfreq(ny, d=dy))
 
    dens_k = (torch.abs(psi_k) ** 2).detach().cpu().numpy()
    phase_k = torch.angle(psi_k).detach().cpu().numpy()
 
    return PsiKData(kx=kx, ky=ky, psi_k=psi_k, dens_k=dens_k, phase_k=phase_k)
 
 
def _radial_profile(dens_k, KX, KY, nbins=80, kmax=None):
    Kmag = np.sqrt(KX**2 + KY**2)
    if kmax is None:
        kmax = Kmag.max()
    bins = np.linspace(0.0, kmax, nbins + 1)
    idx = np.digitize(Kmag.ravel(), bins) - 1
    flat = dens_k.ravel()
    prof = np.full(nbins, np.nan)
    for i in range(nbins):
        m = idx == i
        if m.any():
            prof[i] = flat[m].mean()
    kcenters = 0.5 * (bins[1:] + bins[:-1])
    return kcenters, prof
 
 
# ---------------------------------------------------------------------------
# Plotting (now just a thin wrapper around compute_psi_k) -- always returns
# the data too, so `fig, data = plot_psi_k(...)` gives you both.
# ---------------------------------------------------------------------------
def plot_psi_k(
    psi, x_um, y_um,
    log_scale=True,
    vmin_frac=1e-6,
    recoil_Q=None,          # scalar |Q|, or list of (Qx,Qy) tuples, or list of scalars
    radial_bins=80,
    show=True,
    fig=None,
):
    """
    Momentum-space companion to plot_psi. Computes psi_k via compute_psi_k,
    plots n(k) (log color scale) + radial profile (+ a quick preview with
    phase(k)), and returns (fig, PsiKData) so you can reuse the data
    downstream (slices, T-sweeps, fits) without re-running the FFT.
    """
    data = compute_psi_k(psi, x_um, y_um)
    kx, ky, dens_k, phase_k = data.kx, data.ky, data.dens_k, data.phase_k
    KX, KY = np.meshgrid(kx, ky, indexing="ij")
 
    extent_k = [kx.min(), kx.max(), ky.min(), ky.max()]
    recoils = [] if recoil_Q is None else (
        [recoil_Q] if np.isscalar(recoil_Q) else list(recoil_Q)
    )
 
    def _norm():
        if not log_scale:
            return None
        vmax = dens_k.max()
        pos = dens_k[dens_k > 0]
        vmin = max(vmax * vmin_frac, pos.min() if pos.size else 1e-30)
        return LogNorm(vmin=vmin, vmax=vmax)
 
    # ---------- quick preview: n(k) and phase(k) ----------
    if fig is None:
        fig0, axes0 = plt.subplots(1, 2, figsize=(10, 4))
 
        im0 = axes0[0].imshow(dens_k.T, origin="lower", extent=extent_k, aspect="equal", norm=_norm())
        axes0[0].set_title("Momentum density n(k)")
        axes0[0].set_xlabel(r"$k_x$ ($\mu m^{-1}$)")
        axes0[0].set_ylabel(r"$k_y$ ($\mu m^{-1}$)")
        axes0[0].grid(False)
        plt.colorbar(im0, ax=axes0[0])
 
        im1 = axes0[1].imshow(phase_k.T, origin="lower", extent=extent_k, aspect="equal",
                               vmin=-np.pi, vmax=np.pi)
        axes0[1].set_title(r"Phase of $\tilde\psi(k)$")
        axes0[1].set_xlabel(r"$k_x$ ($\mu m^{-1}$)")
        axes0[1].set_ylabel(r"$k_y$ ($\mu m^{-1}$)")
        axes0[1].grid(False)
        plt.colorbar(im1, ax=axes0[1])
 
        plt.tight_layout()
        if show:
            plt.show()
 
    # ---------- detailed: n(k) with recoil markers + radial profile ----------
    if fig is None:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    else:
        fig.clear()
        axes = fig.subplots(1, 2)
 
    im2 = axes[0].imshow(dens_k.T, origin="lower", extent=extent_k, aspect="equal", norm=_norm())
    axes[0].scatter([0], [0], s=40, facecolors="none", edgecolors="w", linewidths=1.2, label="k=0")
    theta = np.linspace(0, 2 * np.pi, 200)
    for Q in recoils:
        if isinstance(Q, (tuple, list, np.ndarray)):
            axes[0].scatter([Q[0]], [Q[1]], s=50, facecolors="none", edgecolors="w", marker="s")
        else:
            axes[0].plot(Q * np.cos(theta), Q * np.sin(theta), "--", color="w", linewidth=1)
    axes[0].set_title("n(k)  (log scale)" if log_scale else "n(k)")
    axes[0].set_xlabel(r"$k_x$ ($\mu m^{-1}$)")
    axes[0].set_ylabel(r"$k_y$ ($\mu m^{-1}$)")
    axes[0].legend(loc="upper right", fontsize=8)
    plt.colorbar(im2, ax=axes[0])
 
    kcenters, prof = _radial_profile(dens_k, KX, KY, nbins=radial_bins)
    axes[1].plot(kcenters, prof, "-o", ms=3)
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].set_xlabel(r"$|k|$ ($\mu m^{-1}$)")
    axes[1].set_ylabel(r"$n(|k|)$ (azimuthal avg.)")
    axes[1].set_title("Radial momentum profile")
    for Q in recoils:
        if not isinstance(Q, (tuple, list, np.ndarray)):
            axes[1].axvline(Q, color="gray", linestyle="--", linewidth=1)
 
    axes[0].grid(False)
    fig.tight_layout()
 
    if show:
        plt.show()
 
    return fig, data
 
 
# ---------------------------------------------------------------------------
# 1D slice through the 2D momentum density (both +k and -k), and a helper
# to overlay that slice across several T's on one axes.
# ---------------------------------------------------------------------------
def get_k_slice(data: PsiKData, axis="x", at=0.0):
    """
    1D cut through n(k): along kx at ky~=`at` (axis='x'), or along ky at
    kx~=`at` (axis='y'). Returns (k_signed, n_k), both arrays running from
    negative to positive k so you can plot +/- momenta directly.
    """
    if axis == "x":
        j0 = int(np.argmin(np.abs(data.ky - at)))
        k_signed, n_k = data.kx, data.dens_k[:, j0]
    elif axis == "y":
        i0 = int(np.argmin(np.abs(data.kx - at)))
        k_signed, n_k = data.ky, data.dens_k[i0, :]
    else:
        raise ValueError("axis must be 'x' or 'y'")
    order = np.argsort(k_signed)
    return k_signed[order], n_k[order]
 
def plot_k_slice_vs_T(
    psi_by_T, x_um, y_um, axis="x",
    log_y=True, log_x=True,
    linthresh=0.05, ax=None
):
    """
    Plot momentum slices n(k) vs k for different temperatures.

    log_x=True uses a symmetric-log x-axis, allowing both
    positive and negative k while behaving logarithmically
    away from k=0.

    linthresh controls the width of the linear region around k=0.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6.5, 4.5))
    else:
        fig = ax.figure

    def _T_key(item):
        T = item[0]
        try:
            return float(T)
        except (TypeError, ValueError):
            return T

    slices = {}

    for T, psi in sorted(psi_by_T.items(), key=_T_key, reverse=True):
        data = compute_psi_k(psi, x_um, y_um)
        k, n_k = get_k_slice(data, axis=axis)

        slices[T] = (k, n_k)

        ax.plot(
            k, n_k,
            marker="o", ms=3, lw=1,
            label=f"T = {T}"
        )

    # symmetric log: negative k <-- 0 --> positive k
    if log_x:
        ax.set_xscale("symlog", linthresh=linthresh)

    if log_y:
        ax.set_yscale("log")

    ax.axvline(0, color="gray", lw=0.8, ls="--")

    label = fr"$k_{axis}$ ($\mu m^{{-1}}$)"
    ax.set_xlabel(label)
    ax.set_ylabel(
        fr"$n(k_{axis}, 0)$"
        if axis == "x"
        else fr"$n(0, k_{axis})$"
    )

    ax.set_title(f"Momentum slice along {axis}-axis vs. T")
    ax.legend(fontsize=8)

    fig.tight_layout()
    return fig, ax, slices
    

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

    # Plaquette-center coordinates in array-index units
    return vort + 0.5, antiv + 0.5


def detect_vortices_masked(
    psi,
    density_mask=None,
    density_threshold=0.05,
):

    if torch.is_tensor(psi):
        psi = psi.detach().cpu().numpy()
    else:
        psi = np.asarray(psi)

    density = np.abs(psi)**2
    phase = np.angle(psi)

    # ------------------------------------------------------------
    # Construct density mask if none was supplied
    # ------------------------------------------------------------
    if density_mask is None:
        density_mask = density > density_threshold * density.max()
    else:
        density_mask = np.asarray(density_mask, dtype=bool)

        if density_mask.shape != density.shape:
            raise ValueError(
                f"density_mask shape {density_mask.shape} "
                f"does not match psi shape {density.shape}"
            )

    # ------------------------------------------------------------
    # Convert site mask -> plaquette mask
    #
    # Require all four corners of the plaquette to lie inside
    # the accepted density region.
    # ------------------------------------------------------------
    plaquette_mask = (
        density_mask[:-1, :-1]
        & density_mask[1:, :-1]
        & density_mask[1:, 1:]
        & density_mask[:-1, 1:]
    )

    # ------------------------------------------------------------
    # Detect phase winding
    # ------------------------------------------------------------
    vort, antiv = detect_vortices_from_phase(phase)

    # argwhere() coordinates correspond directly to the
    # lower-left index of each plaquette
    vort_i = np.floor(vort).astype(int)
    antiv_i = np.floor(antiv).astype(int)

    if len(vort):
        vort = vort[plaquette_mask[vort_i[:, 0], vort_i[:, 1]]]

    if len(antiv):
        antiv = antiv[plaquette_mask[antiv_i[:, 0], antiv_i[:, 1]]]

    return vort, antiv


import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from dataclasses import dataclass
 
 
@dataclass
class PsiKData:
    kx: np.ndarray          # (Nx,) signed k-axis, 1/um, ascending
    ky: np.ndarray          # (Ny,) signed k-axis, 1/um, ascending
    psi_k: torch.Tensor     # complex, (Nx, Ny), fftshifted, continuum-normalized
    dens_k: np.ndarray      # |psi_k|^2, (Nx, Ny)  == n(k)
    phase_k: np.ndarray     # angle(psi_k), (Nx, Ny)
 
 
# ---------------------------------------------------------------------------
# Core FFT: separated out so it can be reused by plotting AND by anything
# else (slices, fits, T-sweeps) without re-plotting every time.
# ---------------------------------------------------------------------------
def compute_psi_k(psi, x_um, y_um):
    """
    FFT psi(x,y) -> psi_k(kx,ky) with the phase correction for a grid not
    centered on index 0, and continuum-FT normalization (dx*dy prefactor),
    so sum(dens_k)*dkx*dky/(2*pi)**2 == sum(|psi|^2)*dx*dy  (Parseval).
 
    Returns a PsiKData bundle -- no plotting.
    """
    device = psi.device
    dtype = torch.float64 if psi.dtype == torch.complex128 else torch.float32
 
    nx, ny = psi.shape
    dx = float(x_um[1] - x_um[0])
    dy = float(y_um[1] - y_um[0])
 
    kx_raw = torch.as_tensor(2 * np.pi * np.fft.fftfreq(nx, d=dx), device=device, dtype=dtype)
    ky_raw = torch.as_tensor(2 * np.pi * np.fft.fftfreq(ny, d=dy), device=device, dtype=dtype)
    KX_raw, KY_raw = torch.meshgrid(kx_raw, ky_raw, indexing="ij")
    phase_corr = torch.exp(-1j * (KX_raw * float(x_um[0]) + KY_raw * float(y_um[0])))
 
    psi_k = torch.fft.fft2(psi) * phase_corr * dx * dy
    psi_k = torch.fft.fftshift(psi_k)
 
    kx = np.fft.fftshift(2 * np.pi * np.fft.fftfreq(nx, d=dx))
    ky = np.fft.fftshift(2 * np.pi * np.fft.fftfreq(ny, d=dy))
 
    dens_k = (torch.abs(psi_k) ** 2).detach().cpu().numpy()
    phase_k = torch.angle(psi_k).detach().cpu().numpy()
 
    return PsiKData(kx=kx, ky=ky, psi_k=psi_k, dens_k=dens_k, phase_k=phase_k)
 
 
def _radial_profile(dens_k, KX, KY, nbins=80, kmax=None):
    Kmag = np.sqrt(KX**2 + KY**2)
    if kmax is None:
        kmax = Kmag.max()
    bins = np.linspace(0.0, kmax, nbins + 1)
    idx = np.digitize(Kmag.ravel(), bins) - 1
    flat = dens_k.ravel()
    prof = np.full(nbins, np.nan)
    for i in range(nbins):
        m = idx == i
        if m.any():
            prof[i] = flat[m].mean()
    kcenters = 0.5 * (bins[1:] + bins[:-1])
    return kcenters, prof
 
 
# ---------------------------------------------------------------------------
# Plotting (now just a thin wrapper around compute_psi_k) -- always returns
# the data too, so `fig, data = plot_psi_k(...)` gives you both.
# ---------------------------------------------------------------------------
def plot_psi_k(
    psi, x_um, y_um,
    log_scale=True,
    vmin_frac=1e-6,
    recoil_Q=None,          # scalar |Q|, or list of (Qx,Qy) tuples, or list of scalars
    radial_bins=80,
    show=True,
    fig=None,
):
    """
    Momentum-space companion to plot_psi. Computes psi_k via compute_psi_k,
    plots n(k) (log color scale) + radial profile (+ a quick preview witha
    phase(k)), and returns (fig, PsiKData) so you can reuse the data
    downstream (slices, T-sweeps, fits) without re-running the FFT.
    """
    data = compute_psi_k(psi, x_um, y_um)
    kx, ky, dens_k, phase_k = data.kx, data.ky, data.dens_k, data.phase_k
    KX, KY = np.meshgrid(kx, ky, indexing="ij")
 
    extent_k = [kx.min(), kx.max(), ky.min(), ky.max()]
    recoils = [] if recoil_Q is None else (
        [recoil_Q] if np.isscalar(recoil_Q) else list(recoil_Q)
    )
 
    def _norm():
        if not log_scale:
            return None
        vmax = dens_k.max()
        pos = dens_k[dens_k > 0]
        vmin = max(vmax * vmin_frac, pos.min() if pos.size else 1e-30)
        return LogNorm(vmin=vmin, vmax=vmax)
 
    # ---------- quick preview: n(k) and phase(k) ----------
    if fig is None:
        fig0, axes0 = plt.subplots(1, 2, figsize=(10, 4))
 
        im0 = axes0[0].imshow(dens_k.T, origin="lower", extent=extent_k, aspect="equal", norm=_norm())
        axes0[0].set_title("Momentum density n(k)")
        axes0[0].set_xlabel(r"$k_x$ ($\mu m^{-1}$)")
        axes0[0].set_ylabel(r"$k_y$ ($\mu m^{-1}$)")
        axes0[0].grid(False)
        plt.colorbar(im0, ax=axes0[0])
 
        im1 = axes0[1].imshow(phase_k.T, origin="lower", extent=extent_k, aspect="equal",
                               vmin=-np.pi, vmax=np.pi)
        axes0[1].set_title(r"Phase of $\tilde\psi(k)$")
        axes0[1].set_xlabel(r"$k_x$ ($\mu m^{-1}$)")
        axes0[1].set_ylabel(r"$k_y$ ($\mu m^{-1}$)")
        axes0[1].grid(False)
        plt.colorbar(im1, ax=axes0[1])
 
        plt.tight_layout()
        if show:
            plt.show()
 
    # ---------- detailed: n(k) with recoil markers + radial profile ----------
    if fig is None:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    else:
        fig.clear()
        axes = fig.subplots(1, 2)
 
    im2 = axes[0].imshow(dens_k.T, origin="lower", extent=extent_k, aspect="equal", norm=_norm())
    axes[0].scatter([0], [0], s=40, facecolors="none", edgecolors="w", linewidths=1.2, label="k=0")
    theta = np.linspace(0, 2 * np.pi, 200)
    for Q in recoils:
        if isinstance(Q, (tuple, list, np.ndarray)):
            axes[0].scatter([Q[0]], [Q[1]], s=50, facecolors="none", edgecolors="w", marker="s")
        else:
            axes[0].plot(Q * np.cos(theta), Q * np.sin(theta), "--", color="w", linewidth=1)
    axes[0].set_title("n(k)  (log scale)" if log_scale else "n(k)")
    axes[0].set_xlabel(r"$k_x$ ($\mu m^{-1}$)")
    axes[0].set_ylabel(r"$k_y$ ($\mu m^{-1}$)")
    axes[0].legend(loc="upper right", fontsize=8)
    plt.colorbar(im2, ax=axes[0])
 
    kcenters, prof = _radial_profile(dens_k, KX, KY, nbins=radial_bins)
    axes[1].plot(kcenters, prof, "-o", ms=3)
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].set_xlabel(r"$|k|$ ($\mu m^{-1}$)")
    axes[1].set_ylabel(r"$n(|k|)$ (azimuthal avg.)")
    axes[1].set_title("Radial momentum profile")
    for Q in recoils:
        if not isinstance(Q, (tuple, list, np.ndarray)):
            axes[1].axvline(Q, color="gray", linestyle="--", linewidth=1)
 
    axes[0].grid(False)
    fig.tight_layout()
 
    if show:
        plt.show()
 
    return fig, data
 
 
# ---------------------------------------------------------------------------
# 1D slice through the 2D momentum density (both +k and -k), and a helper
# to overlay that slice across several T's on one axes.
# ---------------------------------------------------------------------------
def get_k_slice(data: PsiKData, axis="x", at=0.0):
    """
    1D cut through n(k): along kx at ky~=`at` (axis='x'), or along ky at
    kx~=`at` (axis='y'). Returns (k_signed, n_k), both arrays running from
    negative to positive k so you can plot +/- momenta directly.
    """
    if axis == "x":
        j0 = int(np.argmin(np.abs(data.ky - at)))
        k_signed, n_k = data.kx, data.dens_k[:, j0]
    elif axis == "y":
        i0 = int(np.argmin(np.abs(data.kx - at)))
        k_signed, n_k = data.ky, data.dens_k[i0, :]
    else:
        raise ValueError("axis must be 'x' or 'y'")
    order = np.argsort(k_signed)
    return k_signed[order], n_k[order]
 
 
def plot_k_slice_vs_T(psi_by_T, x_um, y_um, axis="x", log_y=True, ax=None):
    """
    psi_by_T : dict {T_label: psi_tensor} (T_label can be a number or string;
               plotted in descending T order so the legend reads hot->cold).
    Plots n(k) along the chosen axis (through k=0 on the other axis) for
    every T on one set of axes -- exactly what you want to watch the k=0
    peak grow as T drops. Returns (fig, ax, {T_label: (k, n_k)}).
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6.5, 4.5))
    else:
        fig = ax.figure
 
    def _T_key(item):
        T = item[0]
        try:
            return float(T)
        except (TypeError, ValueError):
            return T
 
    slices = {}
    for T, psi in sorted(psi_by_T.items(), key=_T_key, reverse=True):
        data = compute_psi_k(psi, x_um, y_um)
        k, n_k = get_k_slice(data, axis=axis)
        slices[T] = (k, n_k)
        ax.plot(k, n_k, marker="o", ms=3, lw=1, label=f"T = {T}")
 
    ax.axvline(0, color="gray", lw=0.8, ls="--")
    if log_y:
        ax.set_yscale("log")
    label = fr"$k_{axis}$ ($\mu m^{{-1}}$)"
    ax.set_xlabel(label)
    ax.set_ylabel(fr"$n(k_{axis}, 0)$" if axis == "x" else fr"$n(0, k_{axis})$")
    ax.set_title(f"Momentum slice along {axis}-axis vs. T")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig, ax, slices
 
 
# ---------------------------------------------------------------------------
# Track n(k=0) across a time-evolution trajectory (e.g. states_T10), to watch
# the coherent peak build up in real time -- the direct test of N_0(t) ~ t^{d/z}.
# ---------------------------------------------------------------------------
def k0_population(data: PsiKData, k_cutoff):
    """
    Integrate n(k) over a disk |k| <= k_cutoff (1/um) around the origin:
 
        N0 = int_{|k|<k_cutoff} n(k) d^2k / (2*pi)^2
 
    Rotationally symmetric (doesn't care about grid orientation/checkerboard
    axes the way a square window would), and gives a properly normalized
    population estimate consistent with the Parseval check in compute_psi_k
    -- much less noisy than reading the single k=0 pixel, since it averages
    over every grid point inside the disk.
    """
    KX, KY = np.meshgrid(data.kx, data.ky, indexing="ij")
    mask = (KX**2 + KY**2) <= k_cutoff**2
    dkx = data.kx[1] - data.kx[0]
    dky = data.ky[1] - data.ky[0]
    return data.dens_k[mask].sum() * dkx * dky / (2 * np.pi) ** 2
 
 
def plot_k0_vs_time(
    states, x_um, y_um,
    dt=None, times=None,
    k_cutoff=None,
    k0_window=0,
    log_x=False, log_y=True,
    fit_power_law=False,
    ax=None,
):
    """
    states : sequence of psi tensors, one per saved time step (e.g. states_T10)
    dt     : uniform spacing between saved states, used to build the time
             axis if `times` isn't given (times = arange(len(states)) * dt).
             If neither dt nor times is given, the x-axis is just step index.
    times  : explicit array of times (same length as `states`), overrides dt
 
    k_cutoff : radius (1/um) of a small circle around k=0 -- integrates n(k)
               over |k|<=k_cutoff to get N0(t) (see k0_population). This is
               the recommended, numerically stable option: a single grid
               pixel is shot-noise-dominated, a disk averages over many
               pixels while still only counting the "core." If left None,
               defaults to 2 k-space grid spacings (2*dkx), i.e. a small
               but multi-pixel disk -- tune it larger if still noisy, or
               smaller if it starts eating into the surrounding Lorentzian
               tail / recoil peaks.
    k0_window : legacy fallback -- 0 uses the single k=0 pixel, N>0 averages
                a (2N+1)x(2N+1) *square* window instead. Ignored whenever
                `k_cutoff` is given; kept only for backward compatibility.
 
    fit_power_law : fit N0(t) ~ A*t^p (log-log, skipping t=0) and overlay it,
                    printing the fitted exponent -- compare to the d/z you
                    expect (e.g. diffusive coarsening z=2, d=2 -> p=1).
 
    Returns (fig, ax, times, n0).
    """
    n_steps = len(states)
    if times is not None:
        times = np.asarray(times, dtype=float)
    elif dt is not None:
        times = np.arange(n_steps) * dt
    else:
        times = np.arange(n_steps, dtype=float)
 
    use_circle = k_cutoff is not None or k0_window == 0
    if use_circle and k_cutoff is None:
        # default: a small disk 2 k-space grid spacings in radius
        dx = float(x_um[1] - x_um[0])
        nx = len(x_um)
        dkx0 = 2 * np.pi * np.fft.fftfreq(nx, d=dx)[1]  # smallest nonzero |k| step
        k_cutoff = 2 * abs(dkx0)
 
    n0 = np.empty(n_steps)
    for i, psi in enumerate(states):
        data = compute_psi_k(psi, x_um, y_um)
 
        if use_circle:
            n0[i] = k0_population(data, k_cutoff)
        else:
            i0 = int(np.argmin(np.abs(data.kx)))
            j0 = int(np.argmin(np.abs(data.ky)))
            w = k0_window
            n0[i] = data.dens_k[i0 - w:i0 + w + 1, j0 - w:j0 + w + 1].mean()
 
    if ax is None:
        fig, ax = plt.subplots(figsize=(6.5, 4.5))
    else:
        fig = ax.figure
 
    label = r"$N_0=\int_{|k|<k_c} n(k)\,d^2k/(2\pi)^2$" if use_circle else "n(k=0)"
    ax.plot(times, n0, "-o", ms=4, lw=1.2, color="#2a78d6", label=label)
 
    if fit_power_law:
        mask = times > 0
        if mask.sum() >= 2:
            p, logA = np.polyfit(np.log(times[mask]), np.log(n0[mask]), 1)
            fit = np.exp(logA) * times[mask] ** p
            ax.plot(times[mask], fit, "--", color="gray", lw=1,
                     label=fr"fit: $t^{{{p:.2f}}}$")
            print(f"fitted power-law exponent p = {p:.3f}  (N0 ~ t^p)")
 
    if log_x:
        ax.set_xscale("log")
    if log_y:
        ax.set_yscale("log")
    ax.set_xlabel("time" if (dt is not None or times is not None) else "step index")
    ax.set_ylabel(r"$N_0(t)$")
    ax.set_title("Growth of the coherent k~0 population vs. time")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig, ax, times, n0


import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


def fit_coarsening_growth(datasets, times, ylabel=r'$\xi$',
    start_frac=0.01, report_z=True, xlabelsize=20, ylabelsize=20,
    pal=None, figsize=(12, 6.5), return_data=False, model='power'):

    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.optimize import curve_fit

    plt.figure(figsize=figsize)
    __ = 0

    plotted_data = []

    for y, label in datasets:
        y = np.asarray(y)
        t_abs = np.asarray(times[:len(y)])

        t = t_abs - t_abs[0]

        n0 = max(1, int(start_frac * len(y)))
        xi0 = np.mean(y[:n0])

        mask = t > 0
        tf = t[mask]
        yf = y[mask]

        # ---------- simple power law ----------
        def power_model(t, A, z):
            return xi0 + A * t**(1/z)

        popt_p, pcov_p = curve_fit(
            power_model,
            tf,
            yf,
            p0=[(yf[-1] - xi0) / tf[-1]**0.5, 2.0],
            bounds=([0, 0.2], [np.inf, 20])
        )

        A_p, z_p = popt_p
        z_p_err = np.sqrt(pcov_p[1, 1])

        pred_p = power_model(tf, *popt_p)
        rss_p = np.sum((yf - pred_p)**2)
        tss = np.sum((yf - np.mean(yf))**2)
        r2_p = 1 - rss_p/tss

        n = len(yf)
        k_p = 2
        aic_p = n * np.log(rss_p/n) + 2*k_p

        # ---------- log-corrected power law ----------
        def log_model(t, A, z, t0):
            return xi0 + A * (t / np.log(t/t0))**(1/z)

        t0_max = tf.min() * 0.99
        t0_guess = tf.min() * 0.1

        popt_l, pcov_l = curve_fit(
            log_model,
            tf,
            yf,
            p0=[(yf[-1] - xi0) / tf[-1]**0.5, 2.0, t0_guess],
            bounds=(
                [0, 0.2, 1e-12],
                [np.inf, 20, t0_max]
            ),
            maxfev=50000
        )

        A_l, z_l, t0_l = popt_l
        z_l_err = np.sqrt(pcov_l[1, 1])
        t0_l_err = np.sqrt(pcov_l[2, 2])

        pred_l = log_model(tf, *popt_l)
        rss_l = np.sum((yf - pred_l)**2)
        r2_l = 1 - rss_l/tss

        k_l = 3
        aic_l = n * np.log(rss_l/n) + 2*k_l

        # ---------- print ----------
        print(f'\n{label}')
        print('------------------------------')
        print(
            f'Power law:       z = {z_p:.3f} ± {z_p_err:.3f}, '
            f'R² = {r2_p:.5f}, AIC = {aic_p:.2f}'
        )
        print(
            f'Log corrected:   z = {z_l:.3f} ± {z_l_err:.3f}, '
            f't0 = {t0_l*1e3:.4f} ± {t0_l_err*1e3:.4f} ms, '
            f'R² = {r2_l:.5f}, AIC = {aic_l:.2f}'
        )

        delta_aic = aic_l - aic_p

        if delta_aic < -2:
            print(f'→ Log-corrected preferred (ΔAIC = {delta_aic:.2f})')
        elif delta_aic > 2:
            print(f'→ Power law preferred (ΔAIC = {delta_aic:.2f})')
        else:
            print(f'→ No clear preference (ΔAIC = {delta_aic:.2f})')

        # ---------- plot ----------
        tt = np.linspace(tf.min(), tf.max(), 1000)

        if pal is None:
            plt.plot(t * 1e3, y, alpha=0.5)
        else:
            plt.plot(t * 1e3, y, alpha=0.5, color=pal[__])

        p = round(1/z_p, 2)

        if model == 'power':
            fit_curve = power_model(tt, *popt_p)
            fit_label = (
                fr'{label} power: $z={z_p:.2f}$'
                if report_z
                else f'{label}, t^({str(p)}), z={round(1.75/p, 2)}'
            )
        elif model == 'log':
            fit_curve = log_model(tt, *popt_l)
            fit_label = (
                fr'{label} log: $z={z_l:.2f}$'
                if report_z
                else f'{label}, log-corrected, z={z_l:.2f}'
            )
        else:
            raise ValueError("model must be 'power' or 'log'")

        if pal is None:
            plt.plot(
                tt * 1e3,
                fit_curve,
                '--',
                label=fit_label
            )
        else:
            plt.plot(
                tt * 1e3,
                fit_curve,
                '--',
                color=pal[__],
                label=fit_label
            )

        # ---------- save plotted data ----------
        if return_data:
            plotted_data.append({
                'label': label,
                't_ms': t * 1e3,
                'y': y,
                'tf_ms': tf * 1e3,
                'yf': yf,
                'tt_ms': tt * 1e3,
                'power_fit': power_model(tt, *popt_p),
                'log_fit': log_model(tt, *popt_l),
                'A': A_p,
                'z': z_p,
                'z_err': z_p_err,
                'xi0': xi0,
                'A_log': A_l,
                'z_log': z_l,
                'z_log_err': z_l_err,
                't0': t0_l,
                't0_err': t0_l_err,
                'r2_power': r2_p,
                'aic_power': aic_p,
                'r2_log': r2_l,
                'aic_log': aic_l,
                'delta_aic': delta_aic,
            })

        __ += 1

    plt.xlabel('Time since start of fit (ms)', size=xlabelsize)
    plt.ylabel(ylabel, size=ylabelsize)
    plt.legend(loc='upper left', fontsize=xlabelsize)
    plt.tight_layout()

    if ylabel == r'$|\alpha|$':
        plt.savefig('../figs/z(T)_scaling_TQuenches_ALPHA.pdf')
    if ylabel == r'$\xi$':
        plt.savefig('../figs/z(T)_scaling_TQuenches_XI.pdf')
    if ylabel == r'$N_0$':
        plt.savefig('../figs/z(T)_scaling_TQuenches_N0.pdf')

    plt.show()

    if return_data:
        return plotted_data


import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def scan_z_vs_fit_end(
    datasets,
    times,
    start_frac=0.01,
    end_fracs=np.linspace(0.05, 0.30, 30),
    y0_frac=0.01
):
    results = {}

    for y, label in datasets:
        y = np.asarray(y)
        times = np.asarray(times)

        N = len(y)
        start = int(N * start_frac)

        zs = []
        z_errs = []
        end_times = []
        r2s = []

        for end_frac in end_fracs:
            end = int(N * end_frac)

            if end <= start + 5:
                continue

            yy = y[start:end]
            tt_abs = times[start:end]

            tt = tt_abs - tt_abs[0]

            n0 = max(1, int(y0_frac * len(yy)))
            y0 = np.mean(yy[:n0])

            mask = tt > 0
            tf = tt[mask]
            yf = yy[mask]

            def model(t, A, z):
                return y0 + A * t**(1/z)

            try:
                p0 = [
                    max((yf[-1] - y0) / tf[-1]**0.5, 1e-12),
                    2.0
                ]

                popt, pcov = curve_fit(
                    model,
                    tf,
                    yf,
                    p0=p0,
                    bounds=([0, 0.2], [np.inf, 20]),
                    maxfev=20000
                )

                A, z = popt
                z_err = np.sqrt(pcov[1, 1])

                pred = model(tf, *popt)

                rss = np.sum((yf - pred)**2)
                tss = np.sum((yf - np.mean(yf))**2)
                r2 = 1 - rss/tss

                zs.append(z)
                z_errs.append(z_err)
                end_times.append(tt_abs[-1])
                r2s.append(r2)

            except (RuntimeError, ValueError):
                continue

        results[label] = {
            'end_times': np.array(end_times),
            'z': np.array(zs),
            'z_err': np.array(z_errs),
            'r2': np.array(r2s)
        }

    return results


import numpy as np
import torch

def radial_density_correlation(psi, density_mask=None, dx=1e-6):
    # density
    n = (torch.abs(psi)**2).detach().cpu().numpy()

    if density_mask is None:
        density_mask = np.ones_like(n, dtype=bool)

    # connected density fluctuations
    n_mean = np.mean(n[density_mask])
    dn = (n - n_mean) * density_mask

    # mask autocorrelation for proper normalization near boundaries
    M = density_mask.astype(float)

    F_dn = np.fft.fft2(dn)
    F_M  = np.fft.fft2(M)

    corr = np.fft.ifft2(F_dn * np.conj(F_dn)).real
    norm = np.fft.ifft2(F_M  * np.conj(F_M)).real

    corr = np.fft.fftshift(corr)
    norm = np.fft.fftshift(norm)

    corr = np.divide(
        corr, norm,
        out=np.zeros_like(corr),
        where=norm > 0
    )

    # normalize C_n(0)=1
    cy, cx = np.array(corr.shape) // 2
    corr /= corr[cy, cx]

    # radial average
    yy, xx = np.indices(corr.shape)
    rr_pix = np.sqrt((xx - cx)**2 + (yy - cy)**2)

    rbin = rr_pix.astype(int)
    maxbin = min(cx, cy)

    C_rad = np.array([
        np.mean(corr[rbin == i])
        for i in range(maxbin)
    ])

    r = np.arange(maxbin) * dx

    return r, C_rad

import numpy as np
import torch
import matplotlib.pyplot as plt

def correlation_length_threshold(r, C, threshold=0.3):
    valid = np.isfinite(C)
    r = r[valid]
    C = C[valid]

    idx = np.where(C <= threshold)[0]

    if len(idx) == 0:
        return np.nan

    i = idx[0]

    if i == 0:
        return r[0]

    r1, r2 = r[i-1], r[i]
    C1, C2 = C[i-1], C[i]

    return r1 + (threshold - C1) * (r2 - r1) / (C2 - C1)

def radial_phase_correlation(state, density_mask=None, dx=1.0):
    psi = state.detach().cpu()

    phase_field = psi / (torch.abs(psi) + 1e-12)

    if density_mask is None:
        mask = torch.ones_like(torch.abs(psi), dtype=torch.float32)
    else:
        mask = torch.as_tensor(
            density_mask,
            dtype=torch.float32
        ).cpu()

    z = phase_field * mask

    Zk = torch.fft.fft2(z)
    Mk = torch.fft.fft2(mask)

    corr = torch.fft.ifft2(Zk * torch.conj(Zk)).real
    norm = torch.fft.ifft2(Mk * torch.conj(Mk)).real

    corr = corr / (norm + 1e-12)

    corr = torch.fft.fftshift(corr).numpy()

    Ny, Nx = corr.shape

    x = (np.arange(Nx) - Nx // 2) * dx
    y = (np.arange(Ny) - Ny // 2) * dx

    X, Y = np.meshgrid(x, y)
    R = np.sqrt(X**2 + Y**2)

    dr = dx
    r_bins = np.arange(0, R.max() + dr, dr)

    r = 0.5 * (r_bins[:-1] + r_bins[1:])
    C = np.zeros_like(r)

    for i in range(len(r)):
        shell = (R >= r_bins[i]) & (R < r_bins[i + 1])

        if np.any(shell):
            C[i] = np.nanmean(corr[shell])
        else:
            C[i] = np.nan

    C = np.abs(C)

    C /= C[0]

    return r, C