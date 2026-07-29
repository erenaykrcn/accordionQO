import torch
import math


def imprint_vortices_v2(
    psi,
    bec,
    vortices,
    core_size_um=0.3,
    renormalize=True,
    eps=1e-30,
):
    device = psi.device
    dtype_real = torch.float32 if psi.dtype == torch.complex64 else torch.float64

    X, Y = get_bec_xy_grid(bec, psi)

    adim_length_um = bec.adim_length * 1e6
    core_size = core_size_um / adim_length_um

    psi0 = psi.clone()
    psi_new = psi.clone()

    total_phase = torch.zeros_like(psi.real)
    total_core = torch.ones_like(psi.real)

    for v in vortices:
        x0 = v["x_um"] / adim_length_um
        y0 = v["y_um"] / adim_length_um
        q = int(v.get("charge", +1))

        dx = X - x0
        dy = Y - y0

        r = torch.sqrt(dx**2 + dy**2 + eps)
        theta = torch.atan2(dy, dx)

        total_phase = total_phase + q * theta

        # Localized vortex core.
        # This should now be 2D, not a broadcasted column.
        core_profile = torch.tanh(r / (core_size + eps))
        total_core = total_core * core_profile

    psi_new = psi_new * total_core * torch.exp(1j * total_phase)

    if renormalize:
        norm0 = torch.sum(torch.abs(psi0) ** 2)
        norm1 = torch.sum(torch.abs(psi_new) ** 2)
        psi_new = psi_new * torch.sqrt(norm0 / (norm1 + eps))

    return psi_new


def get_bec_xy_grid(bec, psi):
    """
    Return X, Y coordinate grids with the same 2D shape as psi.
    Coordinates are in torchgpe dimensionless units.
    """
    device = psi.device
    dtype_real = torch.float32 if psi.dtype == torch.complex64 else torch.float64

    x = bec.x.to(device=device, dtype=dtype_real)
    y = bec.y.to(device=device, dtype=dtype_real)

    # Case 1: bec.x and bec.y are already 2D grids
    if x.shape == psi.shape and y.shape == psi.shape:
        return x, y

    # Case 2: bec.x and bec.y are 1D coordinate vectors
    if x.ndim == 1 and y.ndim == 1:
        X, Y = torch.meshgrid(x, y, indexing="ij")
        return X, Y

    # Case 3: bec.x is Nx x 1 and bec.y is 1 x Ny, or similar
    X, Y = torch.broadcast_tensors(x, y)

    if X.shape != psi.shape:
        raise ValueError(
            f"Coordinate grid shape {X.shape} does not match psi shape {psi.shape}. "
            f"bec.x shape = {bec.x.shape}, bec.y shape = {bec.y.shape}"
        )

    return X, Y