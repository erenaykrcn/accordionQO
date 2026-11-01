"""
Aging tests on the cavity signal |alpha(t)|^2, which is all we see in the experiment.

Model (your Sec. "Phase ordering kinetics and cavity superradiance"):
    |alpha|^2  ~  Theta^2 = 1 - phi^2 (z t / g)^2 ,   phi = density-weighted bond coherence
1 - phi = (spin waves: fast, stationary)  +  (vortices: slow, step-like, aging)
=> expected: connected two-time correlator of |alpha|^2 decays in two steps: a fast drop
   that does not depend on t_w, then a slow decay that does.

Test 0  calibration:  |alpha|^2 vs phi^2 (computed from the states) and vs N_v;
        detected alpha steps vs N_v drops (timing).
Test 1  connected two-time correlator of the filtered |alpha|^2 (ensemble over realizations).
Test 2  steps -> event counts -> count_aging (persistence, wait at fixed level, surrogate
        that remembers nothing beyond the current count).

Data assumed:  A_raw : (R, Nt_alpha) array of |alpha|^2, one row per realization, on t_alpha
               (t = 0 at the quench).
"""


import sys
sys.path.append("../../")
from torchgpe_v2.aging.count_aging import connected_correlator, plot_count_aging


import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import median_filter, uniform_filter1d
from scipy.signal import find_peaks



# ---------------------------------------------------------------------------------------------
# preprocessing
# ---------------------------------------------------------------------------------------------
def filter_alpha(A, med_pts=501, avg_pts=51):
    """Same filter as your plotting script, applied row-wise (on the full trace)."""
    A = np.atleast_2d(np.asarray(A, float))
    F = median_filter(A, size=(1, med_pts), mode="nearest")
    return uniform_filter1d(F, avg_pts, axis=1, mode="nearest") if avg_pts > 1 else F


def crop(t, A, t_min=None, t_max=None):
    """Keep t_min <= t <= t_max. Filter BEFORE cropping (no edge artifacts) and do NOT shift
    t: it must stay the time since the quench, because t_w is the age of the system."""
    t = np.asarray(t)
    k = np.ones(len(t), bool)
    if t_min is not None:
        k &= t >= t_min
    if t_max is not None:
        k &= t <= t_max
    return t[k], np.atleast_2d(A)[:, k]


def downsample(t, A, n_out):
    k = max(1, A.shape[1] // n_out)
    m = (A.shape[1] // k) * k
    return t[:m].reshape(-1, k).mean(1), A[:, :m].reshape(A.shape[0], -1, k).mean(2)


# ---------------------------------------------------------------------------------------------
# Test 0: phi from the states (bond coherence at the lattice spacing)
# ---------------------------------------------------------------------------------------------
def bond_coherence(psi, mask, shift_px):
    """
    Grid analogue of phi: average of Re[psi*(r) psi(r+a)] / (|psi(r)||psi(r+a)|) weighted by
    sqrt(n(r) n(r+a)), over a = +-shift_px along x and y (nearest neighbour of the
    checkerboard lattice). Equivalent to sum Re[psi* psi'] / sum |psi||psi'| over the cloud.
    shift_px: lattice spacing in pixels (pick it from the period of |psi|^2 if unsure).
    """
    psi = np.asarray(psi)
    mask = np.asarray(mask, bool)
    num = den = 0.0
    for ax in (0, 1):
        p2 = np.roll(psi, -shift_px, axis=ax)
        m2 = mask & np.roll(mask, -shift_px, axis=ax)
        prod = np.conj(psi) * p2
        num += prod.real[m2].sum()
        den += np.abs(prod)[m2].sum()
    return num / den


def plot_calibration(t_a, A_f, t_s, Nv, phi=None, steps=None, fname=None):
    """
    t_a, A_f : alpha time grid and filtered |alpha|^2 (R, Nt_a)
    t_s, Nv  : state time grid and vortex counts (R, Ns);  phi: (R, Ns) or None
    steps    : output of detect_steps (list of event-time arrays) to compare with N_v drops
    """
    Ai = np.array([np.interp(t_s, t_a, a) for a in A_f])           # alpha on the state grid
    n = 3 if steps is not None else 2
    fig, ax = plt.subplots(1, n, figsize=(6 * n, 4.6))
    sc = ax[0].scatter(Nv.ravel() + 0.15 * np.random.randn(Nv.size), Ai.ravel(),
                       c=np.tile(t_s, len(Nv)), s=3, cmap="viridis", alpha=0.5)
    ax[0].set(xlabel=r"$N_v$ (jittered)", ylabel=r"$|\alpha|^2$ filtered", title=r"$|\alpha|^2$ vs $N_v$")
    fig.colorbar(sc, ax=ax[0], label="t")
    if phi is not None:
        x = phi.ravel() ** 2
        ax[1].scatter(x, Ai.ravel(), c=np.tile(t_s, len(Nv)), s=3, cmap="viridis", alpha=0.5)
        ok = np.isfinite(x) & np.isfinite(Ai.ravel())
        p = np.polyfit(x[ok], Ai.ravel()[ok], 1)
        xx = np.linspace(x[ok].min(), x[ok].max(), 50)
        r = np.corrcoef(x[ok], Ai.ravel()[ok])[0, 1]
        ax[1].plot(xx, np.polyval(p, xx), "r-", label=f"linear fit, r={r:.3f}")
        ax[1].set(xlabel=r"$\varphi^2$", ylabel=r"$|\alpha|^2$",
                  title=r"model: $|\alpha|^2 \propto 1-\varphi^2 (zt/g)^2$  (linear, slope<0)")
        ax[1].legend()
    else:
        ax[1].text(0.5, 0.5, "pass phi to test the model", ha="center", transform=ax[1].transAxes)
    if steps is not None:
        # time lag from each N_v drop to the nearest detected alpha step
        lags = []
        for k, ev in enumerate(steps):
            drops = t_s[1:][np.diff(Nv[k]) < 0]
            if len(ev) and len(drops):
                lags += [ev[np.argmin(np.abs(ev - d))] - d for d in drops]
        lags = np.array(lags)
        ax[2].hist(lags, bins=60)
        n_drops = sum(int((np.diff(Nv[k]) < 0).sum()) for k in range(len(Nv)))
        n_steps = sum(len(e) for e in steps)
        ax[2].set(xlabel=r"$t_{\alpha\,step} - t_{N_v\,drop}$", ylabel="count",
                  title=f"timing: {n_steps} alpha steps vs {n_drops} N_v drops")
    fig.tight_layout()
    if fname:
        fig.savefig(fname, dpi=120)
    return fig


# ---------------------------------------------------------------------------------------------
# step detection on |alpha|^2
# ---------------------------------------------------------------------------------------------
def detect_steps(t, A_f, win, h, min_sep=None, t_min=None, t_max=None):
    """
    Downward steps in each filtered trace: drop D(t) = <a>_{[t, t+win]} - <a>_{[t-win, t]}.
    An event is a local minimum of D below -h, at least min_sep apart.
        t_min, t_max : only events in this window are kept (pass the FULL filtered trace,
                  so the comparison windows at the edges still see real data)
        win     : averaging window on each side (samples). Use win ~ med_pts of filter_alpha:
                  when noise > step height the median smears a step over ~med_pts samples,
                  and a shorter win underestimates the step (and misses some). Must stay
                  shorter than the typical spacing between steps.
        h       : step-height threshold (units of |alpha|^2); start at ~ half the smallest step
                  you see by eye, then check against N_v in plot_calibration
    Returns list of event-time arrays and list of step heights.
    """
    min_sep = win if min_sep is None else min_sep
    times, heights = [], []
    for a in np.atleast_2d(A_f):
        c = np.concatenate([[0.0], np.cumsum(a)])
        i = np.arange(win, len(a) - win)
        right = (c[i + win] - c[i]) / win
        left = (c[i] - c[i - win]) / win
        D = right - left
        pk, _ = find_peaks(-D, height=h, distance=min_sep)
        te = t[i[pk]]
        keep = np.ones(len(te), bool)
        if t_min is not None:
            keep &= te >= t_min
        if t_max is not None:
            keep &= te <= t_max
        times.append(te[keep])
        heights.append(-D[pk][keep])
    return times, heights


def events_to_counts(t_grid, events):
    """Event times -> monotone 'remaining steps' count N(t) = (#events after t), shape (R, Nt).
    Valid as an absolute level only if every run reaches its final plateau within the trace."""
    K = np.array([np.searchsorted(np.sort(ev), t_grid, side="right") for ev in events])
    return K[:, -1:] - K


def plot_step_check(t, A_raw, A_f, events, k=0, fname=None):
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(t, A_raw[k], color="C0", alpha=0.3, lw=0.7)
    ax.plot(t, A_f[k], color="C0", lw=2)
    for e in events[k]:
        ax.axvline(e, color="C3", lw=1, ls="--")
    ax.set(xlabel="t", ylabel=r"$|\alpha|^2$", title=f"realization {k}: detected steps (red)")
    fig.tight_layout()
    if fname:
        fig.savefig(fname, dpi=120)
    return fig


# ---------------------------------------------------------------------------------------------
# Test 1: connected two-time correlator of |alpha|^2
# ---------------------------------------------------------------------------------------------
def plot_alpha_correlator(t, A, t_ws, title="", fname=None):
    """
    t, A : (downsampled) time grid and |alpha|^2 (R, Nt). Use the filtered trace for the slow
           part; repeat with a lightly filtered one to see the fast part (stationary, no aging).
    """
    rho, err = connected_correlator(A)
    cols = plt.cm.viridis(np.linspace(0, 0.9, len(t_ws)))
    fig, ax = plt.subplots(1, 3, figsize=(18, 4.6))
    T1, T2 = np.meshgrid(t, t, indexing="ij")
    pc = ax[0].pcolormesh(T2, T1, rho, shading="nearest", cmap="magma")
    fig.colorbar(pc, ax=ax[0])
    ax[0].set(xlabel="t", ylabel="t'", title=r"$\rho_\alpha(t,t')$")
    for tw, c in zip(t_ws, cols):
        iw = int(np.argmin(np.abs(t - tw)))
        j = np.arange(iw + 1, len(t))
        if len(j) < 2 or t[iw] <= 0:
            continue
        tau = t[j] - t[iw]
        ax[1].plot(tau, rho[iw, j], color=c, label=rf"$t_w={t[iw]:.3g}$")
        ax[1].fill_between(tau, rho[iw, j] - err[iw, j], rho[iw, j] + err[iw, j], color=c,
                           alpha=0.2, lw=0)
        ax[2].plot(t[j] / t[iw], rho[iw, j], color=c)
    ax[1].set(xscale="log", xlabel=r"$\tau$", ylabel=r"$\rho_\alpha(t_w+\tau,t_w)$",
              title="TTI test (collapse = no aging)")
    ax[1].legend(fontsize=8)
    ax[2].set(xscale="log", xlabel=r"$t/t_w$", title="simple aging test")
    fig.suptitle(title)
    fig.tight_layout()
    if fname:
        fig.savefig(fname, dpi=120)
    return fig, rho, err


# ---------------------------------------------------------------------------------------------
# Test 2: steps -> count-aging pipeline (re-exported for convenience)
# ---------------------------------------------------------------------------------------------
def alpha_count_aging(t_grid, events, t_ws, tau_max, **kw):
    N = events_to_counts(t_grid, events)
    return plot_count_aging(t_grid, N, t_ws, tau_max, **kw)