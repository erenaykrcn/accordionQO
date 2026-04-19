from torchgpe.bec2D import Gas
from torchgpe.bec2D.callbacks import CavityMonitor
from torchgpe.bec2D.potentials import Contact, DispersiveCavity, Trap
from torchgpe.utils import parse_config

import os
import numpy as np
import torch
from tqdm.auto import tqdm
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context

_G = {}

def init_worker(config_path, psi_final):
    # Parse config INSIDE each worker process
    config = parse_config(config_path)

    _G["config"] = config
    _G["psi_final"] = psi_final

    # Optional but usually helpful on CPU
    torch.set_num_threads(1)

def run_one_detuning(job):
    d_idx, detuning = job

    config = _G["config"]
    psi_final = _G["psi_final"]

    contact = Contact()
    trap = Trap(**config["potentials"]["trap"])

    bec = Gas(
        **config["gas"],
        float_dtype=torch.float32,
        complex_dtype=torch.complex64
    )

    bec.psi = psi_final.clone()

    cavity = DispersiveCavity(
        lattice_depth=config["boundaries"]["lattice_ramp"],   # safe now: created inside worker
        cavity_detuning=detuning,
        **config["potentials"]["cavity"]
    )

    cavity_monitor = CavityMonitor(cavity)
    with open(f"./logs.txt", "a") as file:
        file.write(f"Propagating Det. {detuning}\n")

    bec.propagate(
        potentials=[trap, contact, cavity],
        callbacks=[cavity_monitor],
        **config["propagation"]["real_time"]
    )

    alpha_row = cavity_monitor.alpha[0]

    if isinstance(alpha_row, torch.Tensor):
        print('here')
        alpha_row = alpha_row.detach().cpu()

    del cavity_monitor
    del bec   # ← also important
    return d_idx, alpha_row


if __name__ == "__main__":
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    torch.set_num_threads(1)

    config_path = "config.yaml"
    config = parse_config(config_path)

    np.random.seed(config["random_seed"])
    torch.manual_seed(config["random_seed"])

    contact = Contact()
    trap = Trap(**config["potentials"]["trap"])
    bec_PD = Gas(
        **config["gas"],
        float_dtype=torch.float32,
        complex_dtype=torch.complex64
    )

    sigma = config["initial_wavefunction"]["gaussian_sigma"] / bec_PD.adim_length
    bec_PD.psi = torch.exp(
        -(bec_PD.X**2 + bec_PD.Y**2) / (2 * sigma**2)
    )

    bec_PD.ground_state(
        potentials=[trap, contact],
        N_iterations=config["propagation"]["imaginary_time"]["N_iterations"],
    )

    psi_final = bec_PD.psi.detach().cpu().clone()

    detunings = torch.linspace(*config["boundaries"]["cavity_detuning"])

    ramp = config["boundaries"]["lattice_ramp"]

    rt = config["propagation"]["real_time"]
    dt = rt["time_step"]
    T = rt["final_time"]
    n_steps = int(round(T / dt))
    times = torch.arange(n_steps) * dt
    #times = torch.arange(0, rt["final_time"], rt["time_step"])
    depths = torch.tensor([ramp(t) for t in times], dtype=torch.float32)

    alphas = torch.empty(
        (len(detunings), len(depths)),
        dtype=torch.complex64
    )

    jobs = [(i, float(det)) for i, det in enumerate(detunings)]
    nproc = int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))
    nproc = min(nproc, len(detunings))
    ctx = get_context("fork")

    with ProcessPoolExecutor(
        max_workers=nproc,
        mp_context=ctx,
        initializer=init_worker,
        initargs=(config_path, psi_final),
    ) as ex:
        for d_idx, alpha_row in tqdm(
            ex.map(run_one_detuning, jobs),
            total=len(jobs),
            desc="Phase diagram",
            smoothing=0,
        ):
            alphas[d_idx] = alpha_row

    torch.save(
        {
            "alphas": alphas,
            "detunings": detunings,
            "depths": depths,
        },
        "phase_diagram.pt"
    )

    np.savez(
        "phase_diagram.npz",
        alphas=alphas.numpy(),
        detunings=detunings.numpy(),
        depths=depths.numpy(),
    )
