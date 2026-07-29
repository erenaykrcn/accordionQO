# bkt_bilayer_sgpe_2d.py
#
# Minimal bilayer 2D SGPE simulation for BKT-like dynamics with tunneling J.
#
# Dimensionless units:
#   ħ = 1, m = 1, k_B = 1
#
# Fields:
#   psi1(x, y, t), psi2(x, y, t)
#
# Evolved equations:
#   dψ1 = [-(i+γ)( -1/2 ∇² + g|ψ1|² - μ )ψ1 + (i+γ) J ψ2] dt + sqrt(2γT) dW1
#   dψ2 = [-(i+γ)( -1/2 ∇² + g|ψ2|² - μ )ψ2 + (i+γ) J ψ1] dt + sqrt(2γT) dW2
#
# This is a simple SGPE-style classical-field model for exploring:
#   - vortex / antivortex creation and annihilation
#   - interlayer phase locking
#   - effect of tunneling J on BKT-like disordering
#
# Usage examples:
#   python bkt_bilayer_sgpe_2d.py --device cpu --T 0.4 --J 0.05
#   python bkt_bilayer_sgpe_2d.py --device cpu --J_sweep
#   python bkt_bilayer_sgpe_2d.py --device cpu --T_sweep
#

import argparse
import os
import math
import numpy as np
import torch
import matplotlib.pyplot as plt


# ----------------------------
# Utilities
# ----------------------------

def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def complex_noise(shape, device, dtype_real, scale):
    re = torch.randn(shape, device=device, dtype=dtype_real)
    im = torch.randn(shape, device=device, dtype=dtype_real)
    return scale * (re + 1j * im) / math.sqrt(2.0)


def make_k_grid(Nx, Ny, Lx, Ly, device, dtype):
    dx = Lx / Nx
    dy = Ly / Ny
    kx = 2.0 * math.pi * torch.fft.fftfreq(Nx, d=dx, device=device, dtype=dtype)
    ky = 2.0 * math.pi * torch.fft.fftfreq(Ny, d=dy, device=device, dtype=dtype)
    KX, KY = torch.meshgrid(kx, ky, indexing="ij")
    K2 = KX**2 + KY**2
    return KX, KY, K2


def make_xy_grid(Nx, Ny, Lx, Ly, device, dtype):
    dx = Lx / Nx
    dy = Ly / Ny
    x = (torch.arange(Nx, device=device, dtype=dtype) - Nx // 2) * dx
    y = (torch.arange(Ny, device=device, dtype=dtype) - Ny // 2) * dy
    X, Y = torch.meshgrid(x, y, indexing="ij")
    return X, Y


def projector_mask(K2, k_cut):
    return (torch.sqrt(K2) <= k_cut)


def apply_projector(psi, Pk):
    psik = torch.fft.fftn(psi)
    psik = psik * Pk
    return torch.fft.ifftn(psik)

def normalize_to_number(psi, N_target, dxdy):
    N_now = (torch.abs(psi) ** 2).sum().real * dxdy
    return psi * torch.sqrt(N_target / (N_now + 1e-30))


def kinetic_step(psi, K2, dt):
    phase = torch.exp(-0.5j * K2 * dt)
    psik = torch.fft.fftn(psi)
    psik = psik * phase
    return torch.fft.ifftn(psik)


def normalize_to_mean_density(psi, n0, dxdy):
    N_target = n0 * psi.numel() * dxdy
    N_now = (torch.abs(psi) ** 2).sum().real * dxdy
    return psi * torch.sqrt(N_target / (N_now + 1e-30))


def radial_average(field_2d, dx, dy):
    Nx, Ny = field_2d.shape
    x = (np.arange(Nx) - Nx // 2) * dx
    y = (np.arange(Ny) - Ny // 2) * dy
    X, Y = np.meshgrid(x, y, indexing="ij")
    R = np.sqrt(X**2 + Y**2)

    rmax = min(x.max(), y.max())
    nbins = min(Nx, Ny) // 2
    bins = np.linspace(0, rmax, nbins + 1)
    centers = 0.5 * (bins[:-1] + bins[1:])

    vals = np.zeros(nbins, dtype=np.float64)
    counts = np.zeros(nbins, dtype=np.int64)

    flat_r = R.ravel()
    flat_f = field_2d.ravel()

    inds = np.digitize(flat_r, bins) - 1
    valid = (inds >= 0) & (inds < nbins)
    inds = inds[valid]
    flat_f = flat_f[valid]

    for i, v in zip(inds, flat_f):
        vals[i] += v
        counts[i] += 1

    vals /= np.maximum(counts, 1)
    return centers, vals


def compute_g1_radial(psi, dx, dy):
    psi_np = psi.detach().cpu().numpy()
    F = np.fft.fftn(psi_np)
    C = np.fft.ifftn(np.abs(F) ** 2) / psi_np.size
    C = np.fft.fftshift(C).real
    C /= C.max() + 1e-30
    return radial_average(C, dx, dy)


def phase_wrap(x):
    return (x + np.pi) % (2.0 * np.pi) - np.pi


def detect_vortices(psi):
    theta = torch.angle(psi)

    th00 = theta
    th10 = torch.roll(theta, shifts=-1, dims=0)
    th11 = torch.roll(theta, shifts=(-1, -1), dims=(0, 1))
    th01 = torch.roll(theta, shifts=-1, dims=1)

    d1 = phase_wrap((th10 - th00).cpu().numpy())
    d2 = phase_wrap((th11 - th10).cpu().numpy())
    d3 = phase_wrap((th01 - th11).cpu().numpy())
    d4 = phase_wrap((th00 - th01).cpu().numpy())

    winding = (d1 + d2 + d3 + d4) / (2.0 * np.pi)
    winding_int = np.rint(winding).astype(int)

    vort = np.argwhere(winding_int == 1)
    antiv = np.argwhere(winding_int == -1)
    return vort, antiv


def relative_phase_order(psi1, psi2):
    """
    C_rel = | < exp(i (phi1 - phi2)) > |
          = | < psi1 * psi2* / (|psi1||psi2|) > |
    """
    a1 = torch.abs(psi1)
    a2 = torch.abs(psi2)
    denom = a1 * a2
    mask = denom > 1e-12
    z = torch.zeros_like(psi1)
    z[mask] = psi1[mask] * torch.conj(psi2[mask]) / denom[mask]
    return torch.abs(torch.mean(z)).item()


def phase_lock_fraction(psi1, psi2, threshold=np.pi / 4):
    """
    Fraction of pixels where |phi1 - phi2| < threshold, modulo 2pi.
    """
    dphi = torch.angle(psi1) - torch.angle(psi2)
    dphi = ((dphi + np.pi) % (2.0 * np.pi)) - np.pi
    return torch.mean((torch.abs(dphi) < threshold).float()).item()


def nearest_same_sign_alignment(v1, v2, Lx, Ly, Nx, Ny):
    """
    Rough interlayer alignment metric:
    mean nearest-neighbor distance in physical units between same-sign vortices
    in layer 1 and layer 2. Returns np.nan if one list is empty.
    """
    if len(v1) == 0 or len(v2) == 0:
        return np.nan

    dx = Lx / Nx
    dy = Ly / Ny

    a = v1.astype(np.float64)
    b = v2.astype(np.float64)

    dmin_all = []
    for p in a:
        di = np.abs(b[:, 0] - p[0])
        dj = np.abs(b[:, 1] - p[1])

        # periodic wrap
        di = np.minimum(di, Nx - di)
        dj = np.minimum(dj, Ny - dj)

        dist = np.sqrt((di * dx) ** 2 + (dj * dy) ** 2)
        dmin_all.append(np.min(dist))

    return float(np.mean(dmin_all))


# ----------------------------
# Bilayer SGPE
# ----------------------------

@torch.no_grad()
def bilayer_sgpe_run(
    T=0.35,
    J=0.05,
    gamma=0.02,
    g=0.15,
    mu=0.0,
    n0=6.0,
    Nx=192,
    Ny=192,
    Lx=80.0,
    Ly=80.0,
    dt=2e-3,
    n_steps=50000,
    k_cut=None,
    save_every=200,
    thermalize_steps=10000,
    seed=0,
    device="cpu",
    dtype_real=torch.float32,
):
    set_seed(seed)

    device = torch.device(device)
    dtype_cplx = torch.complex64 if dtype_real == torch.float32 else torch.complex128

    dx = Lx / Nx
    dy = Ly / Ny
    dxdy = dx * dy

    _, _, K2 = make_k_grid(Nx, Ny, Lx, Ly, device=device, dtype=dtype_real)

    if k_cut is None:
        k_ny = math.pi / max(dx, dy)
        k_cut = 0.65 * k_ny

    Pk = projector_mask(K2, k_cut).to(dtype_real)
    Pk_c = Pk.to(dtype_cplx)

    amp0 = math.sqrt(max(n0, 1e-8))
    
    # Start from weak noisy fields with a small relative phase perturbation
    #psi1 = amp0 * (
    #    1.0 + 0.05 * complex_noise((Nx, Ny), device, dtype_real, scale=1.0).to(dtype_cplx)
    #)
    #psi2 = amp0 * (
    #    1.0 + 0.05 * complex_noise((Nx, Ny), device, dtype_real, scale=1.0).to(dtype_cplx)
    #)

    # Real-space grid
    x = (torch.arange(Nx, device=device, dtype=dtype_real) - Nx // 2) * dx
    y = (torch.arange(Ny, device=device, dtype=dtype_real) - Ny // 2) * dy
    X, Y = torch.meshgrid(x, y, indexing="ij")
    sigma = 0.25 * min(Lx, Ly)   # tune this
    N_target = n0 * 2.0 * math.pi * sigma**2

    r2 = X**2 + Y**2
    envelope = torch.exp(-r2 / (2.0 * sigma**2))

    noise_amp = 0.03
    psi1 = amp0 * envelope * (
        1.0 + noise_amp * complex_noise((Nx, Ny), device, dtype_real, scale=1.0).to(dtype_cplx)
    )

    psi2 = amp0 * envelope * (
        1.0 + noise_amp * complex_noise((Nx, Ny), device, dtype_real, scale=1.0).to(dtype_cplx)
    )


    psi1 = apply_projector(psi1, Pk_c)
    psi2 = apply_projector(psi2, Pk_c)

    #psi1 = normalize_to_mean_density(psi1, n0=n0, dxdy=dxdy)
    #psi2 = normalize_to_mean_density(psi2, n0=n0, dxdy=dxdy)
    psi1 = normalize_to_number(psi1, N_target, dxdy)
    psi2 = normalize_to_number(psi2, N_target, dxdy)

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
    align_vortex_series = []
    align_antivortex_series = []

    frames_density_1 = []
    frames_density_2 = []
    frames_phase_1 = []
    frames_phase_2 = []
    frames_rel_phase = []

    noise_pref = math.sqrt(2.0 * gamma * T * dt / dxdy)
    noise_mask = torch.exp(-r2 / (2.0 * sigma**2))
    trap_omega = 0.03
    Vtrap = 0.5 * trap_omega**2 * (X**2 + Y**2)

    for step in range(n_steps):
        # Evolves in real time by Strang splitting.
        # Half kinetic step
        psi1 = kinetic_step(psi1, K2, dt / 2)
        psi2 = kinetic_step(psi2, K2, dt / 2)

        # Contact Int. + Chem. Pot. + Interlayer Tunneling
        n1 = torch.abs(psi1) ** 2
        n2 = torch.abs(psi2) ** 2
        V1 = Vtrap + g * n1 - mu
        V2 = Vtrap + g * n2 - mu
        psi1 = psi1 + (-(1j + gamma) * V1 * psi1 + (1j + gamma) * J * psi2) * dt
        psi2 = psi2 + (-(1j + gamma) * V2 * psi2 + (1j + gamma) * J * psi1) * dt

        # Independent thermal noise in each layer - Wiener Process.
        eta1 = complex_noise((Nx, Ny), device, dtype_real, scale=noise_pref).to(dtype_cplx)
        eta2 = complex_noise((Nx, Ny), device, dtype_real, scale=noise_pref).to(dtype_cplx)
        psi1 = psi1 + noise_mask * eta1
        psi2 = psi2 + noise_mask * eta2

        # Half kinetic step
        psi1 = kinetic_step(psi1, K2, dt / 2)
        psi2 = kinetic_step(psi2, K2, dt / 2)

        # Project to coherent region
        psi1 = apply_projector(psi1, Pk_c)
        psi2 = apply_projector(psi2, Pk_c)

        # Weak number control
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

            v1, av1 = detect_vortices(psi1)
            v2, av2 = detect_vortices(psi2)

            rel_order = relative_phase_order(psi1, psi2)
            lock_frac = phase_lock_fraction(psi1, psi2)

            align_v = nearest_same_sign_alignment(v1, v2, Lx, Ly, Nx, Ny)
            align_av = nearest_same_sign_alignment(av1, av2, Lx, Ly, Nx, Ny)

            times.append(t)
            vortex_counts_1.append(len(v1))
            antivortex_counts_1.append(len(av1))
            vortex_counts_2.append(len(v2))
            antivortex_counts_2.append(len(av2))
            rel_order_series.append(rel_order)
            lock_fraction_series.append(lock_frac)
            number_series_1.append(float(dens1.sum() * dxdy))
            number_series_2.append(float(dens2.sum() * dxdy))
            density_var_series_1.append(float(dens1.var()))
            density_var_series_2.append(float(dens2.var()))
            align_vortex_series.append(align_v)
            align_antivortex_series.append(align_av)

            if step >= thermalize_steps:
                frames_density_1.append(dens1.astype(np.float32))
                frames_density_2.append(dens2.astype(np.float32))
                frames_phase_1.append(ph1.astype(np.float32))
                frames_phase_2.append(ph2.astype(np.float32))
                frames_rel_phase.append(relph.astype(np.float32))

    r_g1_1, g1_1 = compute_g1_radial(psi1, dx=dx, dy=dy)
    r_g1_2, g1_2 = compute_g1_radial(psi2, dx=dx, dy=dy)

    result = {
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
        "align_vortex_series": np.array(align_vortex_series),
        "align_antivortex_series": np.array(align_antivortex_series),
        "frames_density_1": np.array(frames_density_1),
        "frames_density_2": np.array(frames_density_2),
        "frames_phase_1": np.array(frames_phase_1),
        "frames_phase_2": np.array(frames_phase_2),
        "frames_rel_phase": np.array(frames_rel_phase),
        "r_g1_1": np.array(r_g1_1),
        "g1_1": np.array(g1_1),
        "r_g1_2": np.array(r_g1_2),
        "g1_2": np.array(g1_2),
        "params": {
            "T": T,
            "J": J,
            "gamma": gamma,
            "g": g,
            "mu": mu,
            "n0": n0,
            "Nx": Nx,
            "Ny": Ny,
            "Lx": Lx,
            "Ly": Ly,
            "dt": dt,
            "n_steps": n_steps,
            "k_cut": k_cut,
            "save_every": save_every,
            "thermalize_steps": thermalize_steps,
            "seed": seed,
            "device": str(device),
        },
    }
    return result


# ----------------------------
# Saving / plotting
# ----------------------------

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
        align_vortex_series=result["align_vortex_series"],
        align_antivortex_series=result["align_antivortex_series"],
        r_g1_1=result["r_g1_1"],
        g1_1=result["g1_1"],
        r_g1_2=result["r_g1_2"],
        g1_2=result["g1_2"],
        psi1_final=result["psi1_final"],
        psi2_final=result["psi2_final"],
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

    rel_order = result["rel_order_series"]
    lock_fraction = result["lock_fraction_series"]

    r1 = result["r_g1_1"]
    g1_1 = result["g1_1"]
    r2 = result["r_g1_2"]
    g1_2 = result["g1_2"]

    dens1 = np.abs(result["psi1_final"]) ** 2
    dens2 = np.abs(result["psi2_final"]) ** 2
    relph = np.angle(result["psi1_final"]) - np.angle(result["psi2_final"])
    relph = (relph + np.pi) % (2 * np.pi) - np.pi

    plt.figure(figsize=(14, 8))

    plt.subplot(2, 3, 1)
    plt.plot(times, vc1, label="vortices L1")
    plt.plot(times, av1, label="antivortices L1")
    plt.plot(times, vc2, "--", label="vortices L2")
    plt.plot(times, av2, "--", label="antivortices L2")
    plt.xlabel("time")
    plt.ylabel("count")
    plt.title("Vortex counts")
    plt.legend(fontsize=8)

    plt.subplot(2, 3, 2)
    plt.plot(times, rel_order, label="|<exp(i dphi)>|")
    plt.plot(times, lock_fraction, label="lock fraction")
    plt.xlabel("time")
    plt.ylabel("interlayer coherence")
    plt.title("Relative-phase locking")
    plt.legend(fontsize=8)

    plt.subplot(2, 3, 3)
    plt.semilogy(r1[1:], np.maximum(g1_1[1:], 1e-12), label="g1 layer 1")
    plt.semilogy(r2[1:], np.maximum(g1_2[1:], 1e-12), label="g1 layer 2")
    plt.xlabel("r")
    plt.ylabel("g1(r)")
    plt.title("Intralayer coherence")
    plt.legend(fontsize=8)

    plt.subplot(2, 3, 4)
    plt.imshow(dens1.T, origin="lower", aspect="equal")
    plt.title("Final density L1")
    plt.colorbar(shrink=0.8)

    plt.subplot(2, 3, 5)
    plt.imshow(dens2.T, origin="lower", aspect="equal")
    plt.title("Final density L2")
    plt.colorbar(shrink=0.8)

    plt.subplot(2, 3, 6)
    plt.imshow(relph.T, origin="lower", aspect="equal", vmin=-np.pi, vmax=np.pi)
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
    rel_order_mean = float(np.mean(result["rel_order_series"]))
    lock_fraction_mean = float(np.mean(result["lock_fraction_series"]))
    g1_tail_1 = float(np.mean(result["g1_1"][len(result["g1_1"]) // 3:]))
    g1_tail_2 = float(np.mean(result["g1_2"][len(result["g1_2"]) // 3:]))

    print(f"T = {T:.3f}, J = {J:.3f}")
    print(f"  mean vortex count layer 1 : {mean_v1:.2f}")
    print(f"  mean vortex count layer 2 : {mean_v2:.2f}")
    print(f"  mean relative order       : {rel_order_mean:.4f}")
    print(f"  mean lock fraction        : {lock_fraction_mean:.4f}")
    print(f"  g1 tail mean layer 1      : {g1_tail_1:.4e}")
    print(f"  g1 tail mean layer 2      : {g1_tail_2:.4e}")

    if rel_order_mean > 0.8:
        print("  interlayer regime         : strongly phase-locked")
    elif rel_order_mean > 0.4:
        print("  interlayer regime         : partially locked")
    else:
        print("  interlayer regime         : weakly locked / decoupled")

    if 0.5 * (mean_v1 + mean_v2) < 5:
        print("  vortex regime             : few free vortices")
    elif 0.5 * (mean_v1 + mean_v2) < 25:
        print("  vortex regime             : intermediate / crossover")
    else:
        print("  vortex regime             : many free vortices")
    print()


# ----------------------------
# Main
# ----------------------------

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--Nx", type=int, default=192)
    parser.add_argument("--Ny", type=int, default=192)
    parser.add_argument("--Lx", type=float, default=80.0)
    parser.add_argument("--Ly", type=float, default=80.0)

    parser.add_argument("--g", type=float, default=0.15)
    parser.add_argument("--mu", type=float, default=0.0)
    parser.add_argument("--n0", type=float, default=6.0)

    parser.add_argument("--gamma", type=float, default=0.02)
    parser.add_argument("--T", type=float, default=0.35)
    parser.add_argument("--J", type=float, default=0.05)

    parser.add_argument("--dt", type=float, default=2e-3)
    parser.add_argument("--n_steps", type=int, default=50000)
    parser.add_argument("--thermalize_steps", type=int, default=10000)
    parser.add_argument("--save_every", type=int, default=200)

    parser.add_argument("--T_sweep", action="store_true")
    parser.add_argument("--J_sweep", action="store_true")

    args = parser.parse_args()

    if args.T_sweep:
        Ts = [0.20, 0.28, 0.34, 0.40, 0.48, 0.58]
        summary = []

        for i, T in enumerate(Ts):
            result = bilayer_sgpe_run(
                T=T,
                J=args.J,
                gamma=args.gamma,
                g=args.g,
                mu=args.mu,
                n0=args.n0,
                Nx=args.Nx,
                Ny=args.Ny,
                Lx=args.Lx,
                Ly=args.Ly,
                dt=args.dt,
                n_steps=args.n_steps,
                thermalize_steps=args.thermalize_steps,
                save_every=args.save_every,
                seed=args.seed + i,
                device=args.device,
            )

            outdir = f"./out_T_{T:.3f}_J_{args.J:.3f}".replace(".", "p")
            save_run(result, outdir)
            summarize_run(T, args.J, result)

            summary.append({
                "T": T,
                "J": args.J,
                "mean_vortex_count_1": float(0.5 * (result["vortex_counts_1"].mean() + result["antivortex_counts_1"].mean())),
                "mean_vortex_count_2": float(0.5 * (result["vortex_counts_2"].mean() + result["antivortex_counts_2"].mean())),
                "mean_rel_order": float(np.mean(result["rel_order_series"])),
                "mean_lock_fraction": float(np.mean(result["lock_fraction_series"])),
            })

        np.savez_compressed("./T_sweep_summary.npz", summary=np.array(summary, dtype=object))
        print("Saved T_sweep_summary.npz")

    elif args.J_sweep:
        Js = [0.00, 0.02, 0.05, 0.10, 0.20]
        summary = []

        for i, J in enumerate(Js):
            result = bilayer_sgpe_run(
                T=args.T,
                J=J,
                gamma=args.gamma,
                g=args.g,
                mu=args.mu,
                n0=args.n0,
                Nx=args.Nx,
                Ny=args.Ny,
                Lx=args.Lx,
                Ly=args.Ly,
                dt=args.dt,
                n_steps=args.n_steps,
                thermalize_steps=args.thermalize_steps,
                save_every=args.save_every,
                seed=args.seed + i,
                device=args.device,
            )

            outdir = f"./out_T_{args.T:.3f}_J_{J:.3f}".replace(".", "p")
            save_run(result, outdir)
            summarize_run(args.T, J, result)

            summary.append({
                "T": args.T,
                "J": J,
                "mean_vortex_count_1": float(0.5 * (result["vortex_counts_1"].mean() + result["antivortex_counts_1"].mean())),
                "mean_vortex_count_2": float(0.5 * (result["vortex_counts_2"].mean() + result["antivortex_counts_2"].mean())),
                "mean_rel_order": float(np.mean(result["rel_order_series"])),
                "mean_lock_fraction": float(np.mean(result["lock_fraction_series"])),
            })

        np.savez_compressed("./J_sweep_summary.npz", summary=np.array(summary, dtype=object))
        print("Saved J_sweep_summary.npz")

    else:
        result = bilayer_sgpe_run(
            T=args.T,
            J=args.J,
            gamma=args.gamma,
            g=args.g,
            mu=args.mu,
            n0=args.n0,
            Nx=args.Nx,
            Ny=args.Ny,
            Lx=args.Lx,
            Ly=args.Ly,
            dt=args.dt,
            n_steps=args.n_steps,
            thermalize_steps=args.thermalize_steps,
            save_every=args.save_every,
            seed=args.seed,
            device=args.device,
        )

        outdir = f"./out_T_{args.T:.3f}_J_{args.J:.3f}".replace(".", "p")
        save_run(result, outdir)
        summarize_run(args.T, args.J, result)


if __name__ == "__main__":
    main()