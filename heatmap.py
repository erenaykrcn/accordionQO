import numpy as np
import scipy.constants as const
import scipy.sparse as sp


T1_vals = np.logspace(-3, -1, 25) # s
T2_vals = np.logspace(-3, -1, 25) # s

lambda0=780e-3
# constants
hbar = const.hbar
kB = 8.617e-5 # eV/K
c = const.c
u = const.atomic_mass
m_Rb = 86.9091805310 * u
a_s = 5.3e-9 # s-wave scattering length.

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
theta = 14
P_total = 4
waist = 100e-6 # radius
lamL = 1064e-9
kL = 2 * np.pi / lamL
a = lamL/(2*np.sin(np.deg2rad(theta)))

omega_z = 2*np.pi*200
omega_r = 2*np.pi*35 # Hz
d_tau_SI = 1e-6 # ITE
dt_SI = 1e-6 # RTE

# recoil scales
ER = hbar**2 * kL**2 / (2 * m_Rb)
tR = hbar / ER
dt = dt_SI / tR

omegaL = 2 * np.pi * c / lamL
det = np.abs(omegaL - omega0)          # rad/s
Delta_D1 = np.abs(omegaL - omega_D1)
Delta_D2 = np.abs(omegaL - omega_D2)

P_arm = P_total / 2
I_arm = 2 * P_arm / (np.pi * waist**2)
# equal-power, fully interfering arms
I0 = 4 * I_arm
V0 = (np.pi * c**2 / 2) * I0 * (
    Gamma_D1 / (omega_D1**3 * Delta_D1) +
    Gamma_D2 / (omega_D2**3 * Delta_D2)
)
E_rec = (hbar *  np.pi / a)**2 / (2 * m_Rb)
a90 = lamL/(2*np.sin(np.deg2rad(90)))
E_rec90 = (hbar *  np.pi / a90)**2 / (2 * m_Rb)
k_lattice = np.pi / a
omega_osc = 2 * (E_rec / hbar) * np.sqrt(V0 / E_rec)   # trap frequency at each site
a_ho = np.sqrt(hbar / (m_Rb * omega_osc))               # harmonic oscillator length
U = np.sqrt(8/np.pi) * (hbar**2 / m_Rb) * a_s / a_ho**3  # in Joules
U_eV = U / 1.6e-19
s = V0 / E_rec  # dimensionless depth
J = (4 / np.sqrt(np.pi)) * E_rec * s**0.75 * np.exp(-2 * np.sqrt(s))  # in Joules
J_eV = J / 1.6e-19
T = 100e-9

# ----------------------------
# physical constants
# ----------------------------
hbar_SI = 1.054571817e-34
amu = 1.66053906660e-27
m_SI = 87 * amu
lamL_SI = 1064e-9



# --- Losses! ---
a_s = 5.3e-9          # m
K3D_SI = 1e-41         # m^6 / s
g3D = 4 * np.pi * hbar_SI**2 * a_s / m_SI
N_atoms = 1e5
a_r = np.sqrt(hbar_SI / (m_Rb * omega_r))
# --- dimensionless baseline coefficients for psi normalized to 1 ---
g1D_0 = (g3D / (2*np.pi*a_r**2)) / ER * (N_atoms * kL)
Gamma3_1D_0 = (K3D_SI / (3*np.pi**2 * a_r**4)) * tR * (N_atoms * kL)**2


def nonlinear_coeffs(psi):
    rho = np.abs(psi)**2
    norm = np.sum(rho) * dx
    n_eff = np.sum(rho**2) * dx / norm
    n_eff_SI = N_atoms * kL * n_eff
    swell = (1 + 2 * a_s * n_eff_SI)**0.25
    g_eff = g1D_0 / swell**2
    gamma3_eff = Gamma3_1D_0 / swell**4
    return g_eff, gamma3_eff



# ----------------------------
# dimensionless grid: x = kL * x_SI
# ----------------------------
Nx = 2**10
Lx_SI = 80e-6
Lx = kL * Lx_SI
x = np.linspace(-Lx/2, Lx/2, Nx, endpoint=False)
dx = x[1] - x[0]
x_SI = x / kL
# FFT momentum grid in dimensionless units
k = 2 * np.pi * np.fft.fftfreq(Nx, d=dx)
kinetic_phase = np.exp(-1j * (k**2) * dt)

# ----------------------------
# kinetic operator in recoil units
# H/ER = -d^2/dx^2 + V + g|psi|^2
# ----------------------------
main = -2.0 * np.ones(Nx)
off = 1.0 * np.ones(Nx - 1)
lap = sp.diags([off, main, off], offsets=[-1, 0, 1], format="csr") / dx**2

V_SI = 0.5 * m_SI * omega_z**2 * x_SI**2
V = V_SI / ER
# ----------------------------
# dimensionless interaction strength
# g' = g_SI * kL / ER
# For now pick a value directly as a knob
# ----------------------------
d_tau = d_tau_SI / tR
nsteps = 30000
tol = 1e-12
# ----------------------------
# initial guess in dimensionless x
# ----------------------------
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
    psi = psi - d_tau * ((-lap @ psi) + V * psi + g * rho * psi)
    psi = normalize_continuum(psi, dx)
    err = np.sqrt(np.sum(np.abs(psi - psi_old)**2) * dx)

    if step % 100 == 0:
        energies.append(energy_dimless(psi, lap, V, g, dx))

    if err < tol:
        print("Converged at step", step)
        break
rho_SI = kL * np.abs(psi)**2
psi0 = psi


rng = np.random.default_rng(seed=12)
# ----------------------------
# physical constants
# ----------------------------
hbar_SI = 1.054571817e-34
amu = 1.66053906660e-27
m_SI = 87 * amu
lamL_SI = 1064e-9

# recoil scales
kL = 2 * np.pi / lamL_SI
ER = hbar_SI**2 * kL**2 / (2 * m_SI)
tR = hbar_SI / ER

# ----------------------------
# dimensionless grid: x = kL * x_SI
# ----------------------------
Nx = 2**10
Lx_SI = 80e-6
Lx = kL * Lx_SI
x = np.linspace(-Lx/2, Lx/2, Nx, endpoint=False)
dx = x[1] - x[0]
x_SI = x / kL
# FFT momentum grid in dimensionless units
k = 2 * np.pi * np.fft.fftfreq(Nx, d=dx)
kinetic_phase = np.exp(-1j * (k**2) * dt)

# ----------------------------
# dipole trap: Gaussian beam
# shifted so min is zero
# ----------------------------
#Vdip = Vdip_SI / ER
#V_SI = 0.5 * m_SI * omega_r**2 * x_SI**2
#Vdip = V_SI / ER
Vdip = V


def theta_of_t(t, T_ramp1, T_ramp2, theta_i, theta_f, t_delay=0, dS=False):
    T_ramp1 = T_ramp1+t_delay
    if t < T_ramp1:
        return theta_i
    elif t < T_ramp1 + T_ramp2:
        tau = (t - T_ramp1) / T_ramp2
        return theta_i + (theta_f - theta_i) * tau
    else:
        return theta_f

def s_of_t(t, T_ramp1, s_final, dS=False):
    if t < T_ramp1:
        return s_final * (t / T_ramp1)
    else:
        return s_final

def phi_of_t(t, sigma, dS=False):
    return (np.pi/2 if dS else 0) + rng.normal(0, sigma)


def Vlat(t, T_ramp1, T_ramp2, s_final, theta_i, theta_f, sigma=0, t_delay=0, dS=False):
    s = s_of_t(t, T_ramp1, s_final, dS=dS)
    theta = theta_of_t(t, T_ramp1, T_ramp2, theta_i, theta_f, t_delay=t_delay, dS=dS)
    phi = phi_of_t(t, sigma=sigma, dS=dS)
    return -s * np.cos(np.sin(theta) * x + phi)**2


def Vtotal(t,  T_ramp1, T_ramp2, s_final, theta_i, theta_f, sigma=0, t_delay=0, dS=False):
    return Vdip + Vlat(t,  T_ramp1, T_ramp2,  s_final, theta_i, theta_f, sigma=sigma, t_delay=t_delay, dS=dS)


def step_gpe(psi, t, dt,  T_ramp1, T_ramp2,  s_final, theta_i, theta_f, sigma=0, 
              t_delay=0, dS=False):
    rho = np.abs(psi)**2
    g, gamma3 = nonlinear_coeffs(psi)
    phase1 = np.exp(-1j * (Vtotal(t, T_ramp1, T_ramp2,  s_final, theta_i, theta_f, sigma=sigma, t_delay=t_delay, dS=dS) + g * rho) * dt / 2
                   - (gamma3 / 2) * rho**2 * dt / 2
                   )
    psi = phase1 * psi

    psi_k = np.fft.fft(psi)
    psi_k *= kinetic_phase
    psi = np.fft.ifft(psi_k)

    rho = np.abs(psi)**2
    phase2 = np.exp(-1j * (Vtotal(t + dt, T_ramp1, T_ramp2,  s_final, theta_i, theta_f, sigma=sigma, t_delay=t_delay, dS=dS) + g * rho) * dt / 2
                   - (gamma3 / 2) * rho**2 * dt / 2
                   )
    psi = phase2 * psi
    return psi


def evolve(psi0,  T_ramp1_SI, T_ramp2_SI, T_total_SI,  s_final, theta_i, theta_f, sigma=0, t_delay=0, dS=False):
    T_ramp1, T_ramp2 = T_ramp1_SI / tR, T_ramp2_SI / tR
    T_total = T_total_SI / tR
    Nt = int(T_total / dt)
    
    psi = psi0.copy()
    times_SI = []
    times = []
    center_pop = []
    states = []
    for n in range(Nt):
        t = n * dt
        psi = step_gpe(psi, t, dt, T_ramp1, T_ramp2, s_final, theta_i, theta_f, sigma=sigma, t_delay=t_delay, dS=dS)

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

    return states, times_SI, times

sigma = 0.0 #radians
dS = False # double Sheet
theta_i = np.deg2rad(4)
theta_f = np.deg2rad(14)
s_final = 400   # lattice depth in recoil units

energies, losses = ([], [])
for T_ramp1 in T1_vals:
	for T_ramp2 in T2_vals:
		t_delay,  T_total = 0, (T_ramp1+T_ramp2)*1.05
		states, times_SI, times = evolve(psi0.copy(), T_ramp1, T_ramp2, T_total,
		                                                s_final, theta_i, theta_f, t_delay=t_delay, dS=dS, sigma=sigma)
		rho = np.abs(states[-1])**2
		psi = states[-1]
		g, gamma3 = nonlinear_coeffs(states[-1])
		V_now = Vtotal(T_total, T_ramp1, T_ramp2, s_final, theta_i, theta_f, sigma=sigma, t_delay=t_delay, dS=dS)
		E_pot = np.sum(V_now * rho) * dx
		d2psi = np.fft.ifft(-(k**2) * np.fft.fft(psi))
		E_kin = np.real(np.sum(np.conj(states[-1]) * (-d2psi)) * dx)
		E_int = 0.5 * g * np.sum(rho**2) * dx
		energy = (E_kin + E_pot + E_int)/np.sqrt(np.sum(np.abs(states[-1])**2) * dx)
		energies.append(energy)
		losses.append(1-np.sqrt(np.sum(np.abs(states[-1])**2) * dx))

data = {
    'energies': energies, 'losses': losses
}
import json
with open(f"./results/T1_T2.json", "w") as f:
    json.dump(data, f)