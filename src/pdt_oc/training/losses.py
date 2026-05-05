import math
from typing import Dict, Tuple

import torch
import torch.nn.functional as F


def diagonal_gaussian_log_prob(
    z: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
) -> torch.Tensor:
    return -0.5 * (
        math.log(2.0 * math.pi)
        + logvar
        + (z - mu) ** 2 / logvar.exp()
    ).sum(dim=-1)


def vae_elbo(
    batch: Dict[str, torch.Tensor],
    output: Dict[str, torch.Tensor],
    prior: torch.nn.Module,
    recon_weight: float = 1.0,
    kl_weight: float = 1.0,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    recon = output["recon"]
    mu = output["mu"]
    logvar = output["logvar"]
    z = output["z"]

    recon_loss = sum(
        F.mse_loss(recon[m], batch[m], reduction="mean") for m in batch
    )

    log_q = diagonal_gaussian_log_prob(z, mu, logvar)
    log_p = prior.log_prob(z)
    kl = (log_q - log_p).mean()

    total = recon_weight * recon_loss + kl_weight * kl
    return total, {"recon": recon_loss.detach(), "kl": kl.detach()}
