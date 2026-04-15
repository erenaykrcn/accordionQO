import numpy as np
import scipy.constants as const
import scipy.sparse as sp

rng = np.random.default_rng()


T1_vals = np.logspace(-3, -.5, 25)
T2_vals = np.logspace(-3, -.5, 25)
T_total = 70e-3
theta_i = np.deg2rad(2)
theta_f = np.deg2rad(10)
s_final = 10.0   # lattice depth in recoil units

s_dip, w0_SI = 5, 10e-6 # Dipole trap depth and width

dt_SI = 10e-6 # For accordion

# constants
hbar = const.hbar
kB = 8.617e-5 # eV/K
c = const.c
u = const.atomic_mass
m_Rb = 86.9091805310 * u
a_s = 5.3e-9 # s-wave scattering length.
hbar_SI = 1.054571817e-34
amu = 1.66053906660e-27
m_SI = 87 * amu
lamL_SI = 1064e-9

# recoil scales
kL = 2 * np.pi / lamL_SI
ER = hbar_SI**2 * kL**2 / (2 * m_SI)
tR = hbar_SI / ER
dt = dt_SI / tR

# D1 line (795 nm)
lam_D1 = 794.978850e-9
omega_D1 = 2 * np.pi * c / lam_D1
Gamma_D1 = 2 * np.pi * 5.746e6  # rad/s

# D2 line (780 nm) — you already have this
lam_D2 = 780.241209686e-9
omega_D2 = 2 * np.pi * c / lam_D2
Gamma_D2 = 2 * np.pi * 6.0666e6
omega0 = omega_D2

# example beam parameters
theta = 10
P = 200e-3
waist = 100e-6
lamL = 1064e-9
kL = 2 * np.pi / lamL
a = lamL/(2*np.sin(np.deg2rad(theta)))
omegaL = 2 * np.pi * c / lamL
det = np.abs(omegaL - omega0)          # rad/s
Delta_D1 = np.abs(omegaL - omega_D1)
Delta_D2 = np.abs(omegaL - omega_D2)
I0 = 2 * P / (np.pi * waist**2)
V0 = (np.pi * c**2 / 2) * I0 * (
    Gamma_D1 / (omega_D1**3 * Delta_D1) +
    Gamma_D2 / (omega_D2**3 * Delta_D2)
)
E_rec = (hbar *  np.pi / a)**2 / (2 * m_Rb)
a90 = lamL/(2*np.sin(np.deg2rad(90)))
E_rec90 = (hbar *  np.pi / a90)**2 / (2 * m_Rb)
print("Accordion's Depth, V0/Erec90 =", V0 / E_rec90)

k_lattice = np.pi / a
omega_osc = 2 * (E_rec / hbar) * np.sqrt(V0 / E_rec)   # trap frequency at each site
a_ho = np.sqrt(hbar / (m_Rb * omega_osc))               # harmonic oscillator length
U = np.sqrt(8/np.pi) * (hbar**2 / m_Rb) * a_s / a_ho**3  # in Joules
U_eV = U / 1.6e-19
T = 100e-9


# --- Losses! ---
a_s = 5.3e-9          # m
K3D_SI = 1e-41         # m^6 / s
g3D = 4 * np.pi * hbar_SI**2 * a_s / m_SI
omega_r = 2*np.pi*50 # Hz
N_atoms = 1e5
a_r = np.sqrt(hbar_SI / (m_Rb * omega_r))
# --- dimensionless baseline coefficients for psi normalized to 1 ---
g1D_0 = (g3D / (2*np.pi*a_r**2)) / ER * (N_atoms * kL)
Gamma3_1D_0 = (K3D_SI / (3*np.pi**2 * a_r**4)) * tR * (N_atoms * kL)**2
def nonlinear_coeffs(psi):
    rho = np.abs(psi)**2
    n_eff = np.sum(rho**2) * dx / np.sum(rho)
    n_eff_SI = N_atoms * kL * n_eff   # since psi normalized to 1
    swell = (1 + 2 * a_s * n_eff_SI)**0.25
    g_eff = g1D_0 / swell**2
    gamma3_eff = Gamma3_1D_0 / swell**4
    return g_eff, gamma3_eff



# ITE for GS.
d_tau_SI = 0.5e-6
nsteps = 30000

Nx = 2**10
Lx_SI = 80e-6
Lx = kL * Lx_SI
x = np.linspace(-Lx/2, Lx/2, Nx, endpoint=False)
dx = x[1] - x[0]
x_SI = x / kL
k = 2 * np.pi * np.fft.fftfreq(Nx, d=dx)
kinetic_phase = np.exp(-1j * (k**2) * dt)
main = -2.0 * np.ones(Nx)
off = 1.0 * np.ones(Nx - 1)
lap = sp.diags([off, main, off], offsets=[-1, 0, 1], format="csr") / dx**2
Vd0_SI = s_dip*ER
Vdip_SI = -Vd0_SI * np.exp(-2 * x_SI**2 / w0_SI**2) 
V = Vdip_SI / ER
d_tau = d_tau_SI / tR
tol = 1e-12
sigma0_SI = 5e-6
sigma0 = kL * sigma0_SI
psi = np.exp(-x**2 / (2 * sigma0**2)).astype(complex)
def normalize_continuum(psi, dx):
    return psi / np.sqrt(np.sum(np.abs(psi)**2) * dx)
psi = normalize_continuum(psi, dx)
def energy_dimless(psi, lap, V, g, dx):
    kinetic = np.real(np.vdot(psi, (-lap @ psi))) * dx
    potential = np.sum(V * np.abs(psi)**2) * dx
    interaction = 0.5 * g * np.sum(np.abs(psi)**4) * dx
    return kinetic + potential + interaction
energies = []
psi0 = psi.copy()
for step in range(nsteps):
    g, gamma = nonlinear_coeffs(psi)
    psi_old = psi.copy()
    rho = np.abs(psi)**2
    psi = psi - d_tau * ((-lap @ psi) + V * psi + g1D_0 * rho * psi)
    psi = normalize_continuum(psi, dx)
    err = np.sqrt(np.sum(np.abs(psi - psi_old)**2) * dx)

    if step % 100 == 0:
        energies.append(energy_dimless(psi, lap, V, g, dx))

    if err < tol:
        print("Converged at step", step)
        break


# Accordion's evolution

Nx = 2**10
Lx_SI = 80e-6
Lx = kL * Lx_SI
x = np.linspace(-Lx/2, Lx/2, Nx, endpoint=False)
dx = x[1] - x[0]
x_SI = x / kL
k = 2 * np.pi * np.fft.fftfreq(Nx, d=dx)
kinetic_phase = np.exp(-1j * (k**2) * dt)
Vdip = Vdip_SI / ER
def theta_of_t(t, T_ramp1, T_ramp2, theta_i, theta_f, t_delay=0):
    T_ramp1 = T_ramp1+t_delay
    if t < T_ramp1:
        return theta_i
    elif t < T_ramp1 + T_ramp2:
        tau = (t - T_ramp1) / T_ramp2
        return theta_i + (theta_f - theta_i) * tau
    else:
        return theta_f

def s_of_t(t, T_ramp1, s_final):
    if t < T_ramp1:
        return s_final * (t / T_ramp1)
    else:
        return s_final

def phi_of_t(t, sigma):
    return rng.normal(0, sigma)

def Vlat(t, T_ramp1, T_ramp2, s_final, theta_i, theta_f, sigma=0, t_delay=0):
    s = s_of_t(t, T_ramp1, s_final)
    theta = theta_of_t(t, T_ramp1, T_ramp2, theta_i, theta_f, t_delay=t_delay)
    phi = phi_of_t(t, sigma=sigma)
    return -s * np.cos(np.sin(theta) * x + phi)**2


def Vtotal(t,  T_ramp1, T_ramp2, s_final, theta_i, theta_f, sigma=0, t_delay=0):
    return Vdip + Vlat(t,  T_ramp1, T_ramp2,  s_final, theta_i, theta_f, sigma=sigma, t_delay=t_delay)


def step_gpe(psi, t, dt,  T_ramp1, T_ramp2,  s_final, theta_i, theta_f, sigma=0, 
             return_En=False, t_delay=0):
    rho = np.abs(psi)**2
    g, gamma3 = nonlinear_coeffs(psi)
    phase1 = np.exp(-1j * (Vtotal(t, T_ramp1, T_ramp2,  s_final, theta_i, theta_f, sigma=sigma, t_delay=t_delay) + g * rho) * dt / 2
                   - (gamma3 / 2) * rho**2 * dt / 2
                   )
    psi = phase1 * psi

    psi_k = np.fft.fft(psi)
    psi_k *= kinetic_phase
    psi = np.fft.ifft(psi_k)

    rho = np.abs(psi)**2
    phase2 = np.exp(-1j * (Vtotal(t + dt, T_ramp1, T_ramp2,  s_final, theta_i, theta_f, sigma=sigma, t_delay=t_delay) + g * rho) * dt / 2
                   - (gamma3 / 2) * rho**2 * dt / 2
                   )
    psi = phase2 * psi

    if return_En:
        V_now = Vtotal(t + dt, T_ramp1, T_ramp2, s_final, theta_i, theta_f, sigma=sigma, t_delay=t_delay)
        E_pot = np.sum(V_now * rho) * dx
        d2psi = np.fft.ifft(-(k**2) * np.fft.fft(psi))
        E_kin = np.real(np.sum(np.conj(psi) * (-d2psi)) * dx)
        rho = np.abs(psi)**2
        E_int = 0.5 * g * np.sum(rho**2) * dx
        return psi, E_kin + E_pot + E_int
    return psi


def evolve(psi0,  T_ramp1_SI, T_ramp2_SI, T_total_SI,  s_final, theta_i, theta_f, sigma=0, t_delay=0):
    T_ramp1, T_ramp2 = T_ramp1_SI / tR, T_ramp2_SI / tR
    T_total = T_total_SI / tR
    Nt = int(T_total / dt)
    
    psi = psi0.copy()
    times_SI = []
    times = []
    center_pop = []
    states = []
    energies = []
    for n in range(Nt):
        t = n * dt
        if n % 200 == 0:
            psi, en = step_gpe(psi, t, dt, T_ramp1, T_ramp2, s_final, theta_i, theta_f, sigma=sigma, return_En=True, t_delay=t_delay,
                              )
            energies.append(en)
        else:
            psi = step_gpe(psi, t, dt, T_ramp1, T_ramp2, s_final, theta_i, theta_f, sigma=sigma, t_delay=t_delay,
                          )
        if n % 100 == 0:
            t_SI = t * tR
            times.append(t)
            times_SI.append(t_SI)
            theta = theta_of_t(t, T_ramp1, T_ramp2, theta_i, theta_f, t_delay=t_delay)
            a_dimless = 1 / np.sin(theta)   # since a_SI = 1/(kL sin theta)
            mask = np.abs(x) < a_dimless / 2
            P0 = np.sum(np.abs(psi[mask])**2) * dx
            center_pop.append(P0)
            states.append(psi.copy())

    psi_final = psi.copy()
    rho_final_SI = kL * np.abs(psi_final)**2

    return states, times_SI, times, energies

# --- lattice spacing from final angle ---
a_lat = lamL_SI / (2 * np.sin(theta_f))
L = x_SI.max() - x_SI.min()
n_sites = int(np.ceil(L / a_lat)) + 2
centers = np.arange(-n_sites//2, n_sites//2 + 1) * a_lat
centers = centers[(centers >= x_SI.min()) & (centers <= x_SI.max())]
edges = np.zeros(len(centers) + 1)
edges[1:-1] = 0.5 * (centers[:-1] + centers[1:])
edges[0] = centers[0] - a_lat / 2
edges[-1] = centers[-1] + a_lat / 2
energiess = []
final_states = []
for T_ramp1 in T1_vals:
    for T_ramp2 in T2_vals:
        states, times_SI, times, energies = evolve(psi0, T_ramp1, T_ramp2, 1,
                                                        s_final, theta_i, theta_f
                                                    )
        energiess.append(energies[-1])
        final_states.append(states[-1])
losses = []
fracs = []
for state in final_states:
    rho = np.abs(state)**2
    pops = np.array([
        rho[(x_SI >= edges[j]) & (x_SI < edges[j+1])].sum()
        for j in range(len(centers))
    ], dtype=float)
    pops /= pops.sum()
    fracs.append(1-np.max(pops))
    losses.append(1-np.sqrt(np.sum(np.abs(state)**2) * dx))
data = {
    'energies': energiess, 'losses': losses, 'fidelities': fracs
}

import json
with open(f"./results/T1_T2.json", "w") as f:
    json.dump(data, f)

