"""
Aging from vortex NUMBER only:  N_v(t) per realization, no positions, no signs.

The same functions work on any integer step signal, e.g. the number of steps read off
the cavity trace |alpha(t)|^2 later on.

Three levels of evidence
------------------------
1. Connected two-time count correlator
       dN(t) = N(t) - <N(t)>_ensemble          (ensemble mean at each t, NOT a time average)
       rho(t, t_w) = <dN(t) dN(t_w)> / sqrt(Var N(t) Var N(t_w))
   -> plot rho(t_w + tau, t_w) vs tau and vs t/t_w.

2. Event persistence  Pi(tau | t_w) = P(no vortex loss in (t_w, t_w + tau] | N(t_w) > 0)
   Aging: older systems wait longer for the next event; simple aging: Pi = f(tau / t_w).

3. NULL TEST (the important one).  N(t) decreases, and the loss rate depends on N
   (annihilation ~ N^2).  A process with no memory whose rates depend only on n already
   gives t_w-dependent curves, because older systems hold fewer vortices.  Genuine aging =
   memory BEYOND N, carried by hidden degrees of freedom (separations, positions).
     a) memoryless surrogate: fit n -> n' jump rates from the data (pooled over all times),
        simulate an ensemble from the same N(t_start), compare every curve with the data.
     b) conditional waiting time at fixed n: time to next event given N(t_w) = n.
        Memoryless process  -> independent of t_w.
        Hidden aging        -> grows with t_w (survivors at fixed n are "older": farther apart).
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import median_filter


# ---------------------------------------------------------------------------------------------
# preprocessing
# ---------------------------------------------------------------------------------------------
def clean_counts(N, med=1, monotone=True):
    """
    N : (R, Nt) raw counts on a common, preferably uniform and dense, time grid.
    med      : median window (samples) along time to remove single-frame detector flicker.
    monotone : enforce non-increasing N (running minimum) -> no spurious 'creation' events.
               Switch off if thermal pair creation is physical at your T.
    """
    N = np.asarray(N, float)
    if med > 1:
        N = median_filter(N, size=(1, med), mode="nearest")
    if monotone:
        N = np.minimum.accumulate(N, axis=1)
    return np.rint(N).astype(int)


# ---------------------------------------------------------------------------------------------
# 1. connected two-time correlator
# ---------------------------------------------------------------------------------------------
def _rho(N):
    dN = N - N.mean(0)
    cov = dN.T @ dN / len(N)
    d = np.sqrt(np.diag(cov))
    d[d <= 0] = np.nan
    return cov / np.outer(d, d)


def connected_correlator(N, jackknife=True):
    N = np.asarray(N, float)
    rho = _rho(N)
    if not jackknife:
        return rho, np.zeros_like(rho)
    R = len(N)
    jk = np.array([_rho(np.delete(N, k, 0)) for k in range(R)])
    return rho, np.sqrt((R - 1) * np.nanvar(jk, 0))


# ---------------------------------------------------------------------------------------------
# 2. persistence and conditional waiting times
# ---------------------------------------------------------------------------------------------
def persistence(t, N, t_w, n=None):
    """
    Pi(tau | t_w [, N(t_w)=n]) on tau = t[j] - t_w, for realizations with N(t_w) > 0.
    'No event' = N unchanged (assumes monotone counts).  Returns tau, Pi, err, n_used.
    """
    iw = int(np.argmin(np.abs(t - t_w)))
    Nw = N[:, iw]
    sel = Nw > 0 if n is None else Nw == n
    if sel.sum() == 0:
        return t[iw:] - t[iw], np.full(len(t) - iw, np.nan), np.full(len(t) - iw, np.nan), 0
    same = N[sel, iw:] == Nw[sel, None]
    p = same.mean(0)
    return t[iw:] - t[iw], p, np.sqrt(p * (1 - p) / sel.sum()), int(sel.sum())


def restricted_mean_wait(t, N, t_w, tau_max, n=None):
    """Mean time to next event, truncated at tau_max (same horizon for every t_w -> fair)."""
    tau, p, _, m = persistence(t, N, t_w, n)
    k = tau <= tau_max
    if m == 0 or tau[k][-1] < 0.99 * tau_max:
        return np.nan, np.nan, m
    # per-realization truncated waiting times for an error bar
    iw = int(np.argmin(np.abs(t - t_w)))
    Nw = N[:, iw]
    sel = Nw > 0 if n is None else Nw == n
    same = (N[sel, iw:] == Nw[sel, None])[:, k]
    first = np.where(same.all(1), len(tau[k]) - 1, np.argmax(~same, 1))   # no event -> censored at tau_max
    w = tau[k][first]
    return w.mean(), w.std() / np.sqrt(len(w)), m


# ---------------------------------------------------------------------------------------------
# 3. memoryless surrogate (rates depend on n only)
# ---------------------------------------------------------------------------------------------
def fit_markov_rates(t, N, i_start=0):
    """
    Empirical jump rates  k[n -> n'] = (# observed n->n' jumps) / (total time spent at n),
    pooled over all realizations and all times >= t[i_start].
    Multi-step drops within one sample are counted as one jump n -> n'.
    """
    dt = np.diff(t[i_start:])
    A = N[:, i_start:]
    occ, jumps = {}, {}
    for a in A:
        for k in range(len(dt)):
            n0, n1 = a[k], a[k + 1]
            occ[n0] = occ.get(n0, 0.0) + dt[k]
            if n1 != n0:
                jumps[(n0, n1)] = jumps.get((n0, n1), 0) + 1
    rates = {}
    for (n0, n1), c in jumps.items():
        rates.setdefault(n0, []).append((n1, c / occ[n0]))
    return rates, occ


def simulate_markov(rates, N0, t, n_rep=10, seed=0):
    """Gillespie simulation of the fitted memoryless process from initial counts N0, sampled on t."""
    rng = np.random.default_rng(seed)
    N0 = np.repeat(np.asarray(N0), n_rep)
    out = np.empty((len(N0), len(t)), int)
    for r, n in enumerate(N0):
        tc, k = t[0], 0
        while k < len(t):
            out_rates = rates.get(n, [])
            ktot = sum(x[1] for x in out_rates)
            t_next = tc + rng.exponential(1 / ktot) if ktot > 0 else np.inf
            while k < len(t) and t[k] < t_next:
                out[r, k] = n
                k += 1
            if ktot > 0:
                probs = np.array([x[1] for x in out_rates]) / ktot
                n = out_rates[rng.choice(len(out_rates), p=probs)][0]
                tc = t_next
    return out


# ---------------------------------------------------------------------------------------------
# plotting
# ---------------------------------------------------------------------------------------------
def plot_count_aging(t, N, t_ws, tau_max, n_values=None, i_start=0, n_rep=10, title="",
                     fname=None):
    """
    t : (Nt,) uniform time grid (t = 0 at the quench);  N : (R, Nt) cleaned counts.
    t_ws : waiting times for the cuts;  tau_max : horizon for the conditional mean wait
           (t_ws[-1] + tau_max must be <= t[-1]).
    """
    t = np.asarray(t, float)
    rates, _ = fit_markov_rates(t, N, i_start)
    S = simulate_markov(rates, N[:, i_start], t[i_start:], n_rep=n_rep)
    S = np.concatenate([np.repeat(N[:, :i_start], n_rep, 0), S], 1) if i_start else S

    rho, rho_e = connected_correlator(N)
    rho_s, _ = connected_correlator(S, jackknife=False)
    cols = plt.cm.viridis(np.linspace(0, 0.9, len(t_ws)))

    fig, ax = plt.subplots(1, 4, figsize=(21, 4.6))
    # (a) mean count: does the memoryless model even reproduce <N>(t)?
    ax[0].plot(t, N.mean(0), "k", label="data")
    ax[0].plot(t, S.mean(0), "r--", label="memoryless surrogate")
    ax[0].set(xlabel="t", ylabel=r"$\langle N_v\rangle$", title="mean count")
    ax[0].legend()

    for tw, c in zip(t_ws, cols):
        iw = int(np.argmin(np.abs(t - tw)))
        j = np.arange(iw + 1, len(t))
        tau = t[j] - t[iw]
        # (b) connected correlator
        ax[1].plot(tau, rho[iw, j], color=c, label=rf"$t_w={t[iw]:.3g}$")
        ax[1].fill_between(tau, rho[iw, j] - rho_e[iw, j], rho[iw, j] + rho_e[iw, j],
                           color=c, alpha=0.2, lw=0)
        ax[1].plot(tau, rho_s[iw, j], "--", color=c, lw=1)
        # (c) persistence vs tau / t_w
        tt, p, pe, m = persistence(t, N, tw)
        tts, ps, _, _ = persistence(t, S, tw)
        if m:
            ax[2].plot(tt[1:] / t[iw], p[1:], color=c)
            ax[2].fill_between(tt[1:] / t[iw], (p - pe)[1:], (p + pe)[1:], color=c, alpha=0.2, lw=0)
            ax[2].plot(tts[1:] / t[iw], ps[1:], "--", color=c, lw=1)

    ax[1].set(xscale="log", xlabel=r"$\tau$", ylabel=r"$\rho(t_w+\tau, t_w)$",
              title="connected count correlator  (-- surrogate)")
    ax[1].legend(fontsize=8)
    ax[2].set(xscale="log", xlabel=r"$\tau / t_w$", ylabel=r"$\Pi(\tau|t_w)$",
              title="persistence (no event)  (-- surrogate)")

    # (d) conditional mean wait at fixed n: the clean memory-beyond-N test
    if n_values is None:
        n_values = sorted({int(v) for v in np.unique(N[:, i_start]) if v > 0})[:4]
    tw_grid = np.linspace(t[max(i_start, 1)], t[-1] - tau_max, 15)
    for n, mk in zip(n_values, "osd^v<>"):
        d = np.array([restricted_mean_wait(t, N, tw, tau_max, n) for tw in tw_grid])
        s = np.array([restricted_mean_wait(t, S, tw, tau_max, n) for tw in tw_grid])
        ok = d[:, 2] >= 10                         # need >= 10 realizations at that (t_w, n)
        l = ax[3].errorbar(tw_grid[ok], d[ok, 0], d[ok, 1], fmt=mk + "-", ms=4, label=f"n={n}")
        oks = s[:, 2] >= 10
        ax[3].plot(tw_grid[oks], s[oks, 0], "--", color=l[0].get_color(), lw=1)
    ax[3].set(xlabel=r"$t_w$", ylabel=rf"$\langle T_{{next}}\rangle$ (truncated at {tau_max:g})",
              title=r"wait at fixed $N(t_w)=n$: flat = no memory beyond $N$")
    ax[3].legend(fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    if fname:
        fig.savefig(fname, dpi=120)
    return fig, dict(rates=rates, surrogate=S, rho=rho, rho_err=rho_e)