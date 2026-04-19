from torchgpe.bec2D import Gas
from torchgpe.bec2D.callbacks import CavityMonitor
from torchgpe.bec2D.potentials import Contact, DispersiveCavity, Trap
from torchgpe.utils.potentials import linear_ramp

from datetime import datetime
import numpy as np
import torch
import matplotlib.pyplot as plt 
from matplotlib import ticker
from scipy.constants import hbar
from tqdm.auto import tqdm
from torchgpe.utils import parse_config

config = parse_config("config.yaml")
np.random.seed(config["random_seed"])
torch.manual_seed(config["random_seed"])

contact = Contact()
trap = Trap(**config["potentials"]["trap"])
bec_PD = Gas(**config["gas"], float_dtype=torch.float32, complex_dtype=torch.complex64)

bec_PD.psi = torch.exp(-(bec_PD.X**2 + bec_PD.Y**2)/(2*(config["initial_wavefunction"]["gaussian_sigma"] / bec_PD.adim_length)**2))
bec_PD.ground_state(potentials=[trap, contact], N_iterations=int(1e4))
psi_final = bec_PD.psi.clone()


ramp = config["boundaries"]["lattice_ramp"]
detunings = torch.linspace(*config["boundaries"]["cavity_detuning"])
depths = torch.tensor([ramp(t) for t in torch.arange(0, config["propagation"]["real_time"]["final_time"], config["propagation"]["real_time"]["time_step"])])
alphas = torch.tensor(np.empty((detunings.shape[0], depths.shape[0]), dtype=complex))

for d_idx, detuning in enumerate(tqdm(detunings, smoothing=0, desc = "Phase diagram", bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')):
    cavity = DispersiveCavity(lattice_depth=ramp, cavity_detuning=detuning, **config["potentials"]["cavity"])
    cavityMonitor = CavityMonitor(cavity)
    bec_PD.psi = psi_final.clone()
    bec_PD.propagate(potentials = [trap, contact, cavity], callbacks=[cavityMonitor], **config["propagation"]["real_time"])
    alphas[d_idx] = cavityMonitor.alpha[0]
