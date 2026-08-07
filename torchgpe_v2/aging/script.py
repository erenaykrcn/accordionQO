from thermal import get_thermal_state
from SGPE_SO import get_SO_SGPE_state
from torchgpe.utils import parse_config
from utils import save_quench_run, save_state

config = parse_config("config.yaml")

temperature, N_particles1, N_particles2 = 40, int(200e3), int(50e3)
gamma, grid_size, omegar  = 0.01, 60e-3, 30
dt, thermalization_time, final_time1, final_time2 = 1e-6, 30e-3, 20e-3, 30e-3
monitor_every = 500

lattice_ramp = config["boundaries"]["lattice_ramp"]
lattice_static = config["boundaries"]["lattice_static"]

J, detuning, imaginary_steps = 0, -10e6, int(500)


# Initial, thermal state before pump.
psi_thermal = get_thermal_state(temperature, gamma=gamma, J=J, dt=dt, 
	thermalization_time=thermalization_time,
    omegar=omegar, grid_size=grid_size, N_particles=N_particles1,
	imaginary_steps=imaginary_steps
    )

save_state(
    output_dir="results",
    temperature=temperature,
    state=psi_thermal,
    thermalization_time=thermalization_time,
    omegar=omegar,
    grid_size=grid_size,
    N_particles=N_particles1,
)

# Pump to induce weak SO.
"""result1, cavity_monitor1 = get_SO_SGPE_state(psi_thermal, temperature, N_particles1, 
	lattice_ramp, final_time1, detuning = detuning, J=J,
    omegar=omegar, grid_size=grid_size, dt=dt, gamma=gamma, monitor_every=monitor_every)
psi_SO = result1['states'][-1]

# N-> N/2 Quench, re-organization and equilibration of vortices.
result2, cavity_monitor2 = get_SO_SGPE_state(psi_SO, temperature, N_particles2, 
	lattice_static, final_time2, detuning = detuning,
    omegar=omegar, grid_size=grid_size, dt=dt, gamma=gamma, monitor_every=monitor_every)


save_path = save_quench_run(
    output_dir="results",
    temperature=temperature,
    result1=result1,
    result2=result2,
    cavity_monitor1=cavity_monitor1,
    cavity_monitor2=cavity_monitor2,
    N_particles1=N_particles1,
    N_particles2=N_particles2,
    gamma=gamma,
    dt=dt,
    thermalization_time=thermalization_time,
    omegar=omegar,
    grid_size=grid_size,
    final_time1=final_time1,
    final_time2=final_time2,
    J=J,
    detuning=detuning,
)"""