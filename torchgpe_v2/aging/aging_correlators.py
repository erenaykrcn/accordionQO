"""
Two-time correlators C(t, t_w) from stochastic-GPE states after a temperature quench,
plus the standard aging diagnostics.

Aging criterion
---------------
Time-translation invariance (TTI, equilibrium):   C(t_w + tau, t_w) = f(tau)          for all t_w
Aging (coarsening, out of equilibrium):           C(t_w + tau, t_w) depends on t_w;
    the older the system, the slower it decorrelates. Simple aging: C = F(t / t_w);
    domain-coarsening scaling: C = F(L(t) / L(t_w)), with L ~ 1/sqrt(n_vortex).

Observables (all gauge invariant: the global phase rotation e^{-i mu t} drops out)
-----------------------------------------------------------------------------------
"psi"     : field overlap      |sum_r psi*(r,t) psi(r,t_w)|            (amplitude-weighted)
"phase"   : order parameter    |sum_r w e^{-i th(r,t)} e^{i th(r,t_w)}| (canonical O(2) coarsening
            autocorrelation; w = cloud mask & density above threshold)
"vortex"  : vortex-charge field q(r,t) (plaquette winding number, smoothed by a Gaussian of
            width vort_sigma pixels so that slightly drifted vortices still overlap)
            C = <sum q(t) q(t_w)>
"density" : connected density fluctuations dn = n - <n>_ensemble,  C = <sum dn(t) dn(t_w)>

Normalisation is done on ensemble averages:  C(t1,t2) = <G12> / sqrt(<G11><G22>)
(so C(t,t)=1, and "vortex" stays defined even if a given run has no vortex at some time).
Error bars: leave-one-out jackknife over realizations.

Data layout assumed (as in your script):
    traces = states_post_traces_E[T]      # sequence over realizations
    traces[k][i]                          # complex 2D field of realization k at time index i
    t = linspace(0, t_total, n_states)    # t = 0 is the quench
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

ALL_OBS = ("psi", "phase", "vortex", "density")


# ---------------------------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------------------------
def log_time_indices(n_states, t_total, t_min, t_max=None, n_samples=120):
    """Log-spaced (unique) frame indices between t_min and t_max. t=0 is the quench."""
    t_states = np.linspace(0, t_total, n_states)
    t_max = t_states[-1] if t_max is None else t_max
    i_min = max(1, int(np.searchsorted(t_states, t_min)))
    i_max = min(n_states - 1, int(np.searchsorted(t_states, t_max)))
    idx = np.unique(np.geomspace(i_min, i_max, n_samples).astype(int))
    return idx, t_states[idx]


def _wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def vortex_charge(psi):
    """Integer winding number on every plaquette (same shape as psi, last row/col = 0)."""
    th = np.angle(psi)
    dx = _wrap(np.diff(th, axis=1))
    dy = _wrap(np.diff(th, axis=0))
    circ = dx[:-1, :] + dy[:, 1:] - dx[1:, :] - dy[:, :-1]
    q = np.zeros(psi.shape, np.int8)
    q[:-1, :-1] = np.rint(circ / (2 * np.pi)).astype(np.int8)
    return q


def _frames(trace, idx, shape):
    return np.stack([np.asarray(trace[i]).reshape(shape) for i in idx]).astype(np.complex64)


def _features(psi_t, mask, observables, stride, vort_sigma, n_min_frac,
              bg_sigma, psi_smooth, count_fn):
    """Per-realization feature matrices (n_times x n_pixels) for each observable."""
    n = np.abs(psi_t) ** 2
    # "dense" = inside the cloud mask AND background density above n_min_frac * max.
    # The threshold is applied to a density blurred over bg_sigma pixels (a few core
    # sizes), so vortex cores (n -> 0 at the singularity) are NOT cut away, while the
    # dilute cloud edge (where phase noise makes spurious windings) still is.
    n_bg = np.stack([gaussian_filter(nt, bg_sigma) for nt in n]) if bg_sigma > 0 else n
    nmax = np.array([nt[mask].max() for nt in n_bg])
    dense = mask[None] & (n_bg > n_min_frac * nmax[:, None, None])
    s = (slice(None), slice(None, None, stride), slice(None, None, stride))
    m = mask[s[1:]]
    F = {}
    if "psi" in observables:
        F["psi"] = psi_t[s][:, m]
    if "phase" in observables:
        F["phase"] = (dense * np.exp(1j * np.angle(psi_t))).astype(np.complex64)[s][:, m]
    if "density" in observables:
        F["density"] = n[s][:, m].astype(np.float32)

    # vortex charges; optional coarse-graining of psi first removes short-wavelength
    # thermal phase noise (spurious vortex-antivortex pairs at the grid scale)
    psi_v = psi_t
    if psi_smooth > 0:
        psi_v = np.stack([gaussian_filter(p.real, psi_smooth) + 1j * gaussian_filter(p.imag, psi_smooth)
                          for p in psi_t])
    q = np.stack([vortex_charge(p) for p in psi_v]) * dense
    if count_fn is not None:                                   # user's own counter for n_v(t)
        nv = np.array([count_fn(p) for p in psi_t], float)
    else:
        nv = np.abs(q).sum(axis=(1, 2)).astype(float)          # vortices + antivortices
    if "vortex" in observables:
        qs = np.stack([gaussian_filter(qi.astype(np.float32), vort_sigma) for qi in q])
        F["vortex"] = qs[s][:, m]
    return F, nv, q


def check_vortex_detection(psi, mask, n_min_frac=0.05, bg_sigma=4.0, psi_smooth=0.0,
                           extent=None):
    """Plot density with detected +/- charges for ONE frame, to tune the parameters."""
    F, nv, q = _features(psi[None].astype(np.complex64), np.asarray(mask, bool), (),
                         1, 1.0, n_min_frac, bg_sigma, psi_smooth, None)
    q = q[0]
    raw = vortex_charge(psi)
    fig, ax = plt.subplots(1, 2, figsize=(11, 5))
    for a, qq, ttl in ((ax[0], raw, "all windings (no mask)"),
                       (ax[1], q, f"kept: {int(np.abs(q).sum())}  (+{int((q>0).sum())}/-{int((q<0).sum())})")):
        a.imshow(np.abs(psi) ** 2, origin="lower", cmap="Greys_r")
        a.contour(np.asarray(mask, float), levels=[0.5], colors="y", linewidths=0.8)
        yp, xp = np.where(qq > 0); ym, xm = np.where(qq < 0)
        a.plot(xp + 0.5, yp + 0.5, "r+", ms=9, mew=1.5)
        a.plot(xm + 0.5, ym + 0.5, "cx", ms=8, mew=1.5)
        a.set_title(ttl)
    fig.tight_layout()
    return fig


def _gram(Fk, complex_mod):
    G = Fk.conj() @ Fk.T
    return np.abs(G) if complex_mod else G.real


def _normalise(G):
    """C_ij = G_ij / sqrt(G_ii G_jj); NaN where a time has zero signal (e.g. no vortices)."""
    d = np.sqrt(np.diag(G).astype(float))
    d[d <= 0] = np.nan
    return G / np.outer(d, d)


# ---------------------------------------------------------------------------------------------
# main computation
# ---------------------------------------------------------------------------------------------
def two_time_correlators(traces, idx, mask, observables=ALL_OBS, realizations=None,
                         stride=1, vort_sigma=2.0, n_min_frac=0.05, bg_sigma=4.0,
                         psi_smooth=0.0, count_fn=None, verbose=True):
    """
    Returns dict with
        C[obs]      : (Nt, Nt) ensemble two-time correlator, C(t,t)=1
        C_err[obs]  : (Nt, Nt) jackknife error
        n_vortex    : (R, Nt) plaquette vortex count per realization (inside dense mask)
    """
    mask = np.asarray(mask, bool)
    realizations = range(len(traces)) if realizations is None else realizations
    realizations = list(realizations)
    R = len(realizations)
    G = {o: [] for o in observables if o != "density"}
    nvs = []

    # density needs the ensemble mean for the connected part -> keep S = sum_k D_k
    dens = "density" in observables
    D_list = []

    for c, k in enumerate(realizations):
        psi_t = _frames(traces[k], idx, mask.shape)
        F, nv, _ = _features(psi_t, mask, observables, stride, vort_sigma, n_min_frac,
                             bg_sigma, psi_smooth, count_fn)
        nvs.append(nv)
        for o in G:
            G[o].append(_gram(F[o], complex_mod=(o in ("psi", "phase"))))
        if dens:
            D_list.append(F["density"])
        if verbose:
            print(f"\rrealization {c+1}/{R}", end="")
    if verbose:
        print()

    out = {"C": {}, "C_err": {}, "n_vortex": np.array(nvs)}

    for o, Gk in G.items():
        Gk = np.array(Gk)                                # (R, Nt, Nt)
        Gs = Gk.sum(0)
        out["C"][o] = _normalise(Gs / R)
        jk = np.array([_normalise((Gs - Gk[k]) / (R - 1)) for k in range(R)])
        out["C_err"][o] = np.sqrt((R - 1) * jk.var(0))

    if dens:
        D = np.array(D_list)                             # (R, Nt, P)
        Dbar = D.mean(0)

        def conn(Dsub):
            m = Dsub.mean(0)
            dD = Dsub - m
            return _normalise(np.einsum("kip,kjp->ij", dD, dD) / len(Dsub))

        out["C"]["density"] = conn(D)
        jk = np.array([conn(np.delete(D, k, 0)) for k in range(R)])
        out["C_err"]["density"] = np.sqrt((R - 1) * jk.var(0))
    return out


# ---------------------------------------------------------------------------------------------
# aging diagnostics
# ---------------------------------------------------------------------------------------------
def aging_cuts(t, C, C_err, t_ws):
    """For each waiting time t_w: (tau, C(t_w+tau, t_w), err)."""
    cuts = []
    for tw in t_ws:
        iw = int(np.argmin(np.abs(t - tw)))
        j = np.arange(iw, len(t))
        cuts.append((t[iw], t[j] - t[iw], C[iw, j], C_err[iw, j], j))
    return cuts


def plot_aging(t, res, obs, t_ws, area=None, title="", fname=None):
    C, E = res["C"][obs], res["C_err"][obs]
    nv = res["n_vortex"].mean(0)
    if not np.isfinite(C).any():
        print(f"[{obs}] correlator is undefined at all times (zero signal, e.g. no vortices "
              f"detected) -> nothing to plot. <n_v>(t) = {np.round(nv, 2)}")
        return None
    cuts = aging_cuts(t, C, E, t_ws)
    cols = plt.cm.viridis(np.linspace(0, 0.9, len(cuts)))

    fig, ax = plt.subplots(1, 4, figsize=(20, 4.6))
    # (a) two-time map: TTI -> contours parallel to the diagonal
    T1, T2 = np.meshgrid(t, t, indexing="ij")
    pc = ax[0].pcolormesh(T2, T1, C, shading="nearest", cmap="magma", vmin=np.nanmin(C), vmax=1)
    if np.nanmax(C) - np.nanmin(C) > 1e-6:
        ax[0].contour(T2, T1, np.ma.masked_invalid(C), levels=8, colors="w", linewidths=0.6)
    ax[0].set(xscale="log", yscale="log", xlabel=r"$t$", ylabel=r"$t'$",
              title=r"$C(t,t')$ (TTI: contours $\parallel$ diagonal)")
    fig.colorbar(pc, ax=ax[0])

    has = {1: False, 2: False, 3: False}
    for (tw, tau, c, e, j), col in zip(cuts, cols):
        if len(j) < 2 or tw <= 0:                        # t_w at/after the last sample, or t_w=0
            print(f"[{obs}] skipping t_w={tw:.3g}: no later samples or t_w<=0")
            continue
        lab = rf"$t_w={tw:.3g}$"
        ax[1].plot(tau[1:], c[1:], color=col, label=lab)
        ax[1].fill_between(tau[1:], (c - e)[1:], (c + e)[1:], color=col, alpha=0.25, lw=0)
        ax[2].plot((tw + tau) / tw, c, color=col, label=lab)
        ax[2].fill_between((tw + tau) / tw, c - e, c + e, color=col, alpha=0.25, lw=0)
        has[1] = has[2] = True
        # coarsening scaling L ~ n_v^{-1/2}; only where vortices are actually present
        if nv[j[0]] > 0.5:
            with np.errstate(divide="ignore", invalid="ignore"):
                Lr = np.sqrt(nv[j[0]] / nv[j])
            ok = np.isfinite(Lr) & (Lr > 0) & (nv[j] > 0.5)
            if ok.sum() > 1:
                ax[3].plot(Lr[ok], c[ok], color=col, label=lab)
                has[3] = True
        else:
            print(f"[{obs}] t_w={tw:.3g}: <n_v>={nv[j[0]]:.2f} -> left out of the L-scaling panel")

    ax[1].set(xlabel=r"$\tau = t - t_w$", ylabel=r"$C(t_w+\tau,\,t_w)$",
              title="TTI test: collapse here = no aging")
    ax[2].set(xlabel=r"$t/t_w$", title="simple aging: collapse vs $t/t_w$")
    ax[3].set(xlabel=r"$L(t)/L(t_w)=\sqrt{n_v(t_w)/n_v(t)}$", title="coarsening scaling")
    for a in (1, 2, 3):
        if has[a]:
            ax[a].set_xscale("log")
        else:
            ax[a].text(0.5, 0.5, "no usable data", ha="center", va="center",
                       transform=ax[a].transAxes)
    if has[1]:
        ax[1].legend(fontsize=8)
    fig.suptitle(f"{title}  [{obs}]")
    fig.tight_layout()
    if fname:
        fig.savefig(fname, dpi=120)
    return fig


def plot_vortex_number(t, res, fname=None):
    nv = res["n_vortex"]
    m, s = nv.mean(0), nv.std(0) / np.sqrt(len(nv))
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.errorbar(t, m, s, fmt="o-", ms=3)
    ax.set(xscale="log", xlabel="t", ylabel=r"$\langle N_v\rangle$ (plaquette count)")
    if (m > 0).any():
        ax.set_yscale("log")
    fig.tight_layout()
    if fname:
        fig.savefig(fname, dpi=120)
    return fig