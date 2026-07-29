# bkt_bilayer_trapped_sgpe_2d.py

import argparse
import os
import math
import numpy as np
import torch
import matplotlib.pyplot as plt


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def phase_wrap(x):
    return (x + np.pi) % (2 * np.pi) - np.pi


def complex_noise(shape, device, dtype_real, scale=1.0):
    # Wiener increment.
    re = torch.randn(shape, device=device, dtype=dtype_real)
    im = torch.randn(shape, device=device, dtype=dtype_real)
    return scale * (re + 1j * im) / math.sqrt(2.0)


def make_grids(Nx, Ny, Lx, Ly, device, dtype):
    dx = Lx / Nx
    dy = Ly / Ny

    x = (torch.arange(Nx, device=device, dtype=dtype) - Nx // 2) * dx
    y = (torch.arange(Ny, device=device, dtype=dtype) - Ny // 2) * dy
    X, Y = torch.meshgrid(x, y, indexing="ij")

    kx = 2 * math.pi * torch.fft.fftfreq(Nx, d=dx, device=device, dtype=dtype)
    ky = 2 * math.pi * torch.fft.fftfreq(Ny, d=dy, device=device, dtype=dtype)
    KX, KY = torch.meshgrid(kx, ky, indexing="ij")
    K2 = KX**2 + KY**2

    return X, Y, K2, dx, dy


def apply_projector(psi, Pk):
    return torch.fft.ifftn(torch.fft.fftn(psi) * Pk)


def kinetic_step(psi, K2, dt):
    return torch.fft.ifftn(torch.fft.fftn(psi) * torch.exp(-0.5j * K2 * dt))


def normalize_to_number(psi, N_target, dxdy):
    N_now = (torch.abs(psi) ** 2).sum().real * dxdy
    return psi * torch.sqrt(N_target / (N_now + 1e-30))


def density_relax(psi, n_target, strength):
    """
    Phenomenological density stabilization.
    Keeps the trapped cloud as a dense central blob while preserving phase.
    """
    if strength <= 0:
        return psi

    amp = torch.abs(psi)
    phase = psi / (amp + 1e-30)

    target_amp = torch.sqrt(torch.clamp(n_target, min=0.0))
    new_amp = (1.0 - strength) * amp + strength * target_amp

    return new_amp * phase


def detect_vortices_masked(psi, density_cut=0.08):
    theta = torch.angle(psi)
    dens = torch.abs(psi) ** 2
    dens_np = dens.detach().cpu().numpy()

    th00 = theta
    th10 = torch.roll(theta, shifts=-1, dims=0)
    th11 = torch.roll(theta, shifts=(-1, -1), dims=(0, 1))
    th01 = torch.roll(theta, shifts=-1, dims=1)

    d1 = phase_wrap((th10 - th00).detach().cpu().numpy())
    d2 = phase_wrap((th11 - th10).detach().cpu().numpy())
    d3 = phase_wrap((th01 - th11).detach().cpu().numpy())
    d4 = phase_wrap((th00 - th01).detach().cpu().numpy())

    winding = (d1 + d2 + d3 + d4) / (2 * np.pi)
    winding_int = np.rint(winding).astype(int)

    plaquette_density = 0.25 * (
        dens_np
        + np.roll(dens_np, -1, axis=0)
        + np.roll(dens_np, -1, axis=1)
        + np.roll(np.roll(dens_np, -1, axis=0), -1, axis=1)
    )

    mask = plaquette_density > density_cut * dens_np.max()

    vort = np.argwhere((winding_int == 1) & mask)
    antiv = np.argwhere((winding_int == -1) & mask)

    return vort, antiv


def relative_phase_order(psi1, psi2):
    denom = torch.abs(psi1) * torch.abs(psi2)
    mask = denom > 1e-12

    z = torch.zeros_like(psi1)
    z[mask] = psi1[mask] * torch.conj(psi2[mask]) / denom[mask]

    return torch.abs(torch.mean(z[mask])).item() if torch.any(mask) else 0.0


def phase_lock_fraction(psi1, psi2, threshold=np.pi / 4):
    dphi = torch.angle(psi1) - torch.angle(psi2)
    dphi = (dphi + np.pi) % (2 * np.pi) - np.pi
    return torch.mean((torch.abs(dphi) < threshold).float()).item()


def radial_average(field, dx, dy):
    Nx, Ny = field.shape
    x = (np.arange(Nx) - Nx // 2) * dx
    y = (np.arange(Ny) - Ny // 2) * dy
    X, Y = np.meshgrid(x, y, indexing="ij")
    R = np.sqrt(X**2 + Y**2)

    rmax = min(x.max(), y.max())
    nbins = min(Nx, Ny) // 2
    bins = np.linspace(0, rmax, nbins + 1)
    centers = 0.5 * (bins[:-1] + bins[1:])

    inds = np.digitize(R.ravel(), bins) - 1
    valid = (inds >= 0) & (inds < nbins)

    vals = np.zeros(nbins)
    counts = np.zeros(nbins)

    np.add.at(vals, inds[valid], field.ravel()[valid])
    np.add.at(counts, inds[valid], 1)

    vals /= np.maximum(counts, 1)
    return centers, vals


def compute_g1_radial(psi, dx, dy):
    psi_np = psi.detach().cpu().numpy()
    F = np.fft.fftn(psi_np)
    C = np.fft.ifftn(np.abs(F) ** 2).real / psi_np.size
    C = np.fft.fftshift(C)
    C /= C.max() + 1e-30
    return radial_average(C, dx, dy)


@torch.no_grad()
def bilayer_sgpe_run(
    T=0.35,
    J=0.05,
    gamma=0.01,
    g=0.15,
    mu=1.0,
    n_peak=6.0,
    Nx=192,
    Ny=192,
    Lx=80.0,
    Ly=80.0,
    dt=1e-3,
    n_steps=50000,
    save_every=200,
    thermalize_steps=10000,
    seed=0,
    device="cpu",
    dtype_real=torch.float32,
    trap_omega=0.035,
    density_relax_strength=0.015,
    amp_noise_fraction=0.03,
    phase_noise_boost=1.0,
    density_cut=0.08,
    k_cut=None,
    injected_vortices_1=None,
    injected_vortices_2=None,
    injected_core_size=0.4,
):
    set_seed(seed)

    device = torch.device(device)
    dtype_cplx = torch.complex64 if dtype_real == torch.float32 else torch.complex128

    X, Y, K2, dx, dy = make_grids(Nx, Ny, Lx, Ly, device, dtype_real)
    dxdy = dx * dy
    r2 = X**2 + Y**2

    if k_cut is None:
        k_ny = math.pi / max(dx, dy)
        k_cut = 0.70 * k_ny

    Pk = (torch.sqrt(K2) <= k_cut).to(dtype_cplx)

    # Harmonic trap and Thomas-Fermi-like target density
    Vtrap = 0.5 * trap_omega**2 * r2
    n_tf = torch.clamp((mu - Vtrap) / g, min=0.0)

    # Rescale target density to requested central peak.
    n_tf = n_tf * (n_peak / (n_tf.max() + 1e-30))

    # Smooth soft mask for where the reservoir acts.
    reservoir_mask = torch.sqrt(torch.clamp(n_tf / (n_tf.max() + 1e-30), 0.0, 1.0))
    N_target = float((n_tf.sum() * dxdy).detach().cpu())

    # Initial trapped cloud
    amp0 = torch.sqrt(n_tf + 1e-30)
    psi1 = amp0 * (
        1.0 + amp_noise_fraction * complex_noise((Nx, Ny), device, dtype_real).to(dtype_cplx)
    )
    psi2 = amp0 * (
        1.0 + amp_noise_fraction * complex_noise((Nx, Ny), device, dtype_real).to(dtype_cplx)
    )

    psi1 = apply_projector(psi1.to(dtype_cplx), Pk)
    psi2 = apply_projector(psi2.to(dtype_cplx), Pk)

    psi1 = normalize_to_number(psi1, N_target, dxdy)
    psi2 = normalize_to_number(psi2, N_target, dxdy)

    if injected_vortices_1 is not None:
        psi1 = imprint_vortices_sgpe(
            psi1,
            X,
            Y,
            injected_vortices_1,
            core_size=injected_core_size,
        )

    if injected_vortices_2 is not None:
        psi2 = imprint_vortices_sgpe(
            psi2,
            X,
            Y,
            injected_vortices_2,
            core_size=injected_core_size,
        )

    times = []
    vortex_counts_1 = []
    antivortex_counts_1 = []
    vortex_counts_2 = []
    antivortex_counts_2 = []
    rel_order_series = []
    lock_fraction_series = []
    number_series_1 = []
    number_series_2 = []
    density_var_series_1 = []
    density_var_series_2 = []

    frames_density_1 = []
    frames_density_2 = []
    frames_phase_1 = []
    frames_phase_2 = []
    frames_rel_phase = []

    vortex_positions_1 = []
    antivortex_positions_1 = []
    vortex_positions_2 = []
    antivortex_positions_2 = []

    # Important: phase-noise dominated. This creates thermal phase disorder
    # without blasting particles out of the trap.
    phase_noise_pref = phase_noise_boost * math.sqrt(2.0 * gamma * T * dt)

    # Much smaller additive amplitude noise, localized to cloud.
    amp_noise_pref = 0.15 * math.sqrt(2.0 * gamma * T * dt / dxdy)

    for step in range(n_steps):
        psi1 = kinetic_step(psi1, K2, dt / 2)
        psi2 = kinetic_step(psi2, K2, dt / 2)

        n1 = torch.abs(psi1) ** 2
        n2 = torch.abs(psi2) ** 2

        H1 = Vtrap + g * n1 - mu
        H2 = Vtrap + g * n2 - mu

        psi1_old = psi1
        psi2_old = psi2

        psi1 = psi1 + (-(1j + gamma) * H1 * psi1_old + (1j + gamma) * J * psi2_old) * dt
        psi2 = psi2 + (-(1j + gamma) * H2 * psi2_old + (1j + gamma) * J * psi1_old) * dt

        # Phase thermalization inside cloud
        xi1 = torch.randn((Nx, Ny), device=device, dtype=dtype_real)
        xi2 = torch.randn((Nx, Ny), device=device, dtype=dtype_real)

        psi1 = psi1 * torch.exp(1j * reservoir_mask * phase_noise_pref * xi1)
        psi2 = psi2 * torch.exp(1j * reservoir_mask * phase_noise_pref * xi2)

        # Small localized complex noise to let vortex cores breathe
        eta1 = complex_noise((Nx, Ny), device, dtype_real, amp_noise_pref).to(dtype_cplx)
        eta2 = complex_noise((Nx, Ny), device, dtype_real, amp_noise_pref).to(dtype_cplx)

        psi1 = psi1 + reservoir_mask * eta1
        psi2 = psi2 + reservoir_mask * eta2

        psi1 = kinetic_step(psi1, K2, dt / 2)
        psi2 = kinetic_step(psi2, K2, dt / 2)

        psi1 = apply_projector(psi1, Pk)
        psi2 = apply_projector(psi2, Pk)

        # Stabilize trapped blob, but preserve phase defects.
        psi1 = density_relax(psi1, n_tf, density_relax_strength)
        psi2 = density_relax(psi2, n_tf, density_relax_strength)

        if step % 50 == 0:
            psi1 = normalize_to_number(psi1, N_target, dxdy)
            psi2 = normalize_to_number(psi2, N_target, dxdy)

        if step % save_every == 0:
            t = step * dt

            dens1 = (torch.abs(psi1) ** 2).detach().cpu().numpy()
            dens2 = (torch.abs(psi2) ** 2).detach().cpu().numpy()
            ph1 = torch.angle(psi1).detach().cpu().numpy()
            ph2 = torch.angle(psi2).detach().cpu().numpy()
            relph = phase_wrap(ph1 - ph2)

            v1, av1 = detect_vortices_masked(psi1, density_cut=density_cut)
            v2, av2 = detect_vortices_masked(psi2, density_cut=density_cut)

            times.append(t)
            vortex_counts_1.append(len(v1))
            antivortex_counts_1.append(len(av1))
            vortex_counts_2.append(len(v2))
            antivortex_counts_2.append(len(av2))
            rel_order_series.append(relative_phase_order(psi1, psi2))
            lock_fraction_series.append(phase_lock_fraction(psi1, psi2))
            number_series_1.append(float(dens1.sum() * dxdy))
            number_series_2.append(float(dens2.sum() * dxdy))
            density_var_series_1.append(float(dens1.var()))
            density_var_series_2.append(float(dens2.var()))
            vortex_positions_1.append(vortex_indices_to_xy(v1, X, Y))
            antivortex_positions_1.append(vortex_indices_to_xy(av1, X, Y))
            vortex_positions_2.append(vortex_indices_to_xy(v2, X, Y))
            antivortex_positions_2.append(vortex_indices_to_xy(av2, X, Y))

            if step >= thermalize_steps:
                frames_density_1.append(dens1.astype(np.float32))
                frames_density_2.append(dens2.astype(np.float32))
                frames_phase_1.append(ph1.astype(np.float32))
                frames_phase_2.append(ph2.astype(np.float32))
                frames_rel_phase.append(relph.astype(np.float32))

    r_g1_1, g1_1 = compute_g1_radial(psi1, dx, dy)
    r_g1_2, g1_2 = compute_g1_radial(psi2, dx, dy)

    return {
        "psi1_final": psi1.detach().cpu().numpy(),
        "psi2_final": psi2.detach().cpu().numpy(),
        "times": np.array(times),
        "vortex_counts_1": np.array(vortex_counts_1),
        "antivortex_counts_1": np.array(antivortex_counts_1),
        "vortex_counts_2": np.array(vortex_counts_2),
        "antivortex_counts_2": np.array(antivortex_counts_2),
        "rel_order_series": np.array(rel_order_series),
        "lock_fraction_series": np.array(lock_fraction_series),
        "number_series_1": np.array(number_series_1),
        "number_series_2": np.array(number_series_2),
        "density_var_series_1": np.array(density_var_series_1),
        "density_var_series_2": np.array(density_var_series_2),
        "frames_density_1": np.array(frames_density_1),
        "frames_density_2": np.array(frames_density_2),
        "frames_phase_1": np.array(frames_phase_1),
        "frames_phase_2": np.array(frames_phase_2),
        "frames_rel_phase": np.array(frames_rel_phase),
        "r_g1_1": np.array(r_g1_1),
        "g1_1": np.array(g1_1),
        "r_g1_2": np.array(r_g1_2),
        "g1_2": np.array(g1_2),
        "vortex_positions_1": np.array(vortex_positions_1, dtype=object),
        "antivortex_positions_1": np.array(antivortex_positions_1, dtype=object),
        "vortex_positions_2": np.array(vortex_positions_2, dtype=object),
        "antivortex_positions_2": np.array(antivortex_positions_2, dtype=object),
        "n_target": n_tf.detach().cpu().numpy(),
        "params": {
            "T": T,
            "J": J,
            "gamma": gamma,
            "g": g,
            "mu": mu,
            "n_peak": n_peak,
            "Nx": Nx,
            "Ny": Ny,
            "Lx": Lx,
            "Ly": Ly,
            "dt": dt,
            "n_steps": n_steps,
            "trap_omega": trap_omega,
            "density_relax_strength": density_relax_strength,
            "phase_noise_boost": phase_noise_boost,
            "amp_noise_fraction": amp_noise_fraction,
            "density_cut": density_cut,
            "k_cut": k_cut,
            "save_every": save_every,
            "thermalize_steps": thermalize_steps,
            "seed": seed,
            "device": str(device),
        },
    }


def save_run(result, outdir):
    os.makedirs(outdir, exist_ok=True)

    np.savez_compressed(
        os.path.join(outdir, "summary.npz"),
        times=result["times"],
        vortex_counts_1=result["vortex_counts_1"],
        antivortex_counts_1=result["antivortex_counts_1"],
        vortex_counts_2=result["vortex_counts_2"],
        antivortex_counts_2=result["antivortex_counts_2"],
        rel_order_series=result["rel_order_series"],
        lock_fraction_series=result["lock_fraction_series"],
        number_series_1=result["number_series_1"],
        number_series_2=result["number_series_2"],
        density_var_series_1=result["density_var_series_1"],
        density_var_series_2=result["density_var_series_2"],
        r_g1_1=result["r_g1_1"],
        g1_1=result["g1_1"],
        r_g1_2=result["r_g1_2"],
        g1_2=result["g1_2"],
        psi1_final=result["psi1_final"],
        psi2_final=result["psi2_final"],
        n_target=result["n_target"],
        params=np.array([result["params"]], dtype=object),
    )

    np.savez_compressed(
        os.path.join(outdir, "frames.npz"),
        frames_density_1=result["frames_density_1"],
        frames_density_2=result["frames_density_2"],
        frames_phase_1=result["frames_phase_1"],
        frames_phase_2=result["frames_phase_2"],
        frames_rel_phase=result["frames_rel_phase"],
    )

    times = result["times"]

    vc1 = result["vortex_counts_1"]
    av1 = result["antivortex_counts_1"]
    vc2 = result["vortex_counts_2"]
    av2 = result["antivortex_counts_2"]

    dens1 = np.abs(result["psi1_final"]) ** 2
    dens2 = np.abs(result["psi2_final"]) ** 2
    relph = phase_wrap(np.angle(result["psi1_final"]) - np.angle(result["psi2_final"]))

    r1 = result["r_g1_1"]
    g1_1 = result["g1_1"]
    r2 = result["r_g1_2"]
    g1_2 = result["g1_2"]

    plt.figure(figsize=(14, 8))

    plt.subplot(2, 3, 1)
    plt.plot(times, vc1, label="v L1")
    plt.plot(times, av1, label="av L1")
    plt.plot(times, vc2, "--", label="v L2")
    plt.plot(times, av2, "--", label="av L2")
    plt.xlabel("time")
    plt.ylabel("count")
    plt.title("Masked vortex counts")
    plt.legend(fontsize=8)

    plt.subplot(2, 3, 2)
    plt.plot(times, result["rel_order_series"], label=r"$|\langle e^{i\Delta\phi}\rangle|$")
    plt.plot(times, result["lock_fraction_series"], label="lock fraction")
    plt.xlabel("time")
    plt.ylabel("interlayer coherence")
    plt.title("Relative phase locking")
    plt.legend(fontsize=8)

    plt.subplot(2, 3, 3)
    plt.semilogy(r1[1:], np.maximum(g1_1[1:], 1e-12), label="g1 L1")
    plt.semilogy(r2[1:], np.maximum(g1_2[1:], 1e-12), label="g1 L2")
    plt.xlabel("r")
    plt.ylabel("g1(r)")
    plt.title("Intralayer coherence")
    plt.legend(fontsize=8)

    plt.subplot(2, 3, 4)
    plt.imshow(dens1.T, origin="lower")
    plt.title("Final density L1")
    plt.colorbar(shrink=0.8)

    plt.subplot(2, 3, 5)
    plt.imshow(dens2.T, origin="lower")
    plt.title("Final density L2")
    plt.colorbar(shrink=0.8)

    plt.subplot(2, 3, 6)
    plt.imshow(relph.T, origin="lower", cmap="twilight", vmin=-np.pi, vmax=np.pi)
    plt.title("Final relative phase")
    plt.colorbar(shrink=0.8)

    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "diagnostics.png"), dpi=160)
    plt.close()


def summarize_run(T, J, result):
    mean_v1 = 0.5 * (
        result["vortex_counts_1"].mean() + result["antivortex_counts_1"].mean()
    )
    mean_v2 = 0.5 * (
        result["vortex_counts_2"].mean() + result["antivortex_counts_2"].mean()
    )

    print(f"T = {T:.3f}, J = {J:.3f}")
    print(f"  mean vortex count layer 1 : {mean_v1:.2f}")
    print(f"  mean vortex count layer 2 : {mean_v2:.2f}")
    print(f"  mean relative order       : {np.mean(result['rel_order_series']):.4f}")
    print(f"  mean lock fraction        : {np.mean(result['lock_fraction_series']):.4f}")
    print()


def run_and_save(args, T, J, seed_offset=0):
    result = bilayer_sgpe_run(
        T=T,
        J=J,
        gamma=args.gamma,
        g=args.g,
        mu=args.mu,
        n_peak=args.n_peak,
        Nx=args.Nx,
        Ny=args.Ny,
        Lx=args.Lx,
        Ly=args.Ly,
        dt=args.dt,
        n_steps=args.n_steps,
        save_every=args.save_every,
        thermalize_steps=args.thermalize_steps,
        seed=args.seed + seed_offset,
        device=args.device,
        trap_omega=args.trap_omega,
        density_relax_strength=args.density_relax_strength,
        amp_noise_fraction=args.amp_noise_fraction,
        phase_noise_boost=args.phase_noise_boost,
        density_cut=args.density_cut,
    )

    outdir = f"./out_T_{T:.3f}_J_{J:.3f}".replace(".", "p")
    save_run(result, outdir)
    summarize_run(T, J, result)

    return result


def vortex_indices_to_xy(vort_idx, X, Y):
    """
    Convert plaquette indices from detect_vortices_masked into physical SGPE coordinates.
    """
    if len(vort_idx) == 0:
        return np.empty((0, 2))

    X_np = X.detach().cpu().numpy()
    Y_np = Y.detach().cpu().numpy()

    coords = []

    Nx, Ny = X_np.shape

    for i, j in vort_idx:
        ip = (i + 1) % Nx
        jp = (j + 1) % Ny

        x = 0.25 * (
            X_np[i, j]
            + X_np[ip, j]
            + X_np[i, jp]
            + X_np[ip, jp]
        )

        y = 0.25 * (
            Y_np[i, j]
            + Y_np[ip, j]
            + Y_np[i, jp]
            + Y_np[ip, jp]
        )

        coords.append([x, y])

    return np.array(coords)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--Nx", type=int, default=192)
    parser.add_argument("--Ny", type=int, default=192)
    parser.add_argument("--Lx", type=float, default=80.0)
    parser.add_argument("--Ly", type=float, default=80.0)

    parser.add_argument("--g", type=float, default=0.15)
    parser.add_argument("--mu", type=float, default=1.0)
    parser.add_argument("--n_peak", type=float, default=6.0)

    parser.add_argument("--gamma", type=float, default=0.01)
    parser.add_argument("--T", type=float, default=0.35)
    parser.add_argument("--J", type=float, default=0.05)

    parser.add_argument("--trap_omega", type=float, default=0.035)
    parser.add_argument("--density_relax_strength", type=float, default=0.015)
    parser.add_argument("--phase_noise_boost", type=float, default=1.0)
    parser.add_argument("--amp_noise_fraction", type=float, default=0.03)
    parser.add_argument("--density_cut", type=float, default=0.08)

    parser.add_argument("--dt", type=float, default=1e-3)
    parser.add_argument("--n_steps", type=int, default=50000)
    parser.add_argument("--thermalize_steps", type=int, default=10000)
    parser.add_argument("--save_every", type=int, default=200)

    parser.add_argument("--T_sweep", action="store_true")
    parser.add_argument("--J_sweep", action="store_true")

    args = parser.parse_args()

    if args.T_sweep:
        Ts = [0.10, 0.18, 0.26, 0.34, 0.44, 0.56]
        summary = []

        for i, T in enumerate(Ts):
            result = run_and_save(args, T=T, J=args.J, seed_offset=i)

            summary.append({
                "T": T,
                "J": args.J,
                "mean_vortex_count_1": float(
                    0.5 * (
                        result["vortex_counts_1"].mean()
                        + result["antivortex_counts_1"].mean()
                    )
                ),
                "mean_vortex_count_2": float(
                    0.5 * (
                        result["vortex_counts_2"].mean()
                        + result["antivortex_counts_2"].mean()
                    )
                ),
                "mean_rel_order": float(np.mean(result["rel_order_series"])),
                "mean_lock_fraction": float(np.mean(result["lock_fraction_series"])),
            })

        np.savez_compressed("./T_sweep_summary.npz", summary=np.array(summary, dtype=object))
        print("Saved T_sweep_summary.npz")

    elif args.J_sweep:
        Js = [0.00, 0.02, 0.05, 0.10, 0.20]
        summary = []

        for i, J in enumerate(Js):
            result = run_and_save(args, T=args.T, J=J, seed_offset=i)

            summary.append({
                "T": args.T,
                "J": J,
                "mean_vortex_count_1": float(
                    0.5 * (
                        result["vortex_counts_1"].mean()
                        + result["antivortex_counts_1"].mean()
                    )
                ),
                "mean_vortex_count_2": float(
                    0.5 * (
                        result["vortex_counts_2"].mean()
                        + result["antivortex_counts_2"].mean()
                    )
                ),
                "mean_rel_order": float(np.mean(result["rel_order_series"])),
                "mean_lock_fraction": float(np.mean(result["lock_fraction_series"])),
            })

        np.savez_compressed("./J_sweep_summary.npz", summary=np.array(summary, dtype=object))
        print("Saved J_sweep_summary.npz")

    else:
        run_and_save(args, T=args.T, J=args.J)

def imprint_vortices_sgpe(psi, X, Y, vortices, core_size=0.4, eps=1e-30):
    """
    Imprint vortices into an SGPE wavefunction.

    X, Y, core_size, x0, y0 are in SGPE simulation units.
    """
    psi0 = psi.clone()
    psi_new = psi.clone()

    total_phase = torch.zeros_like(X)
    total_core = torch.ones_like(X)

    for v in vortices:
        x0 = float(v["x"])
        y0 = float(v["y"])
        q = int(v.get("charge", +1))

        dx = X - x0
        dy = Y - y0
        r = torch.sqrt(dx**2 + dy**2 + eps)

        theta = torch.atan2(dy, dx)

        total_phase = total_phase + q * theta
        total_core = total_core * torch.tanh(r / (core_size + eps))

    psi_new = psi_new * total_core * torch.exp(1j * total_phase)

    # Restore norm
    n0 = torch.sum(torch.abs(psi0) ** 2)
    n1 = torch.sum(torch.abs(psi_new) ** 2)
    psi_new = psi_new * torch.sqrt(n0 / (n1 + eps))

    return psi_new

if __name__ == "__main__":
    main()



