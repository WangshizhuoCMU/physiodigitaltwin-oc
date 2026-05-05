import math
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import torch
from torch import nn

from .encoders import ModalityEncoder
from .fusion import CrossAttentionFusion


@dataclass(frozen=True)
class VAEConfig:
    rna_dim: int
    cnv_dim: int
    mut_dim: int
    clin_dim: int
    rna_hidden: Tuple[int, ...] = (32,)
    kv_hidden: Tuple[int, ...] = (16,)
    clin_hidden: Tuple[int, ...] = ()
    rna_out: int = 32
    kv_out: int = 16
    clin_out: int = 8
    attn_heads: int = 4
    decoder_hidden: Tuple[int, ...] = (32, 32)
    z_dim: int = 16
    n_components: int = 4
    logvar_min: float = -8.0
    logvar_max: float = 8.0


class GMMPrior(nn.Module):
    def __init__(self, n_components: int, z_dim: int, init_scale: float = 0.5) -> None:
        super().__init__()
        self.means = nn.Parameter(torch.randn(n_components, z_dim) * init_scale)
        self.logits = nn.Parameter(torch.zeros(n_components))

    def log_prob(self, z: torch.Tensor) -> torch.Tensor:
        diff = z.unsqueeze(1) - self.means.unsqueeze(0)
        log_normal = -0.5 * (diff ** 2).sum(-1) - 0.5 * z.shape[-1] * math.log(2.0 * math.pi)
        log_w = nn.functional.log_softmax(self.logits, dim=0)
        return torch.logsumexp(log_w + log_normal, dim=-1)


class VAE(nn.Module):
    def __init__(self, cfg: VAEConfig) -> None:
        super().__init__()
        self.cfg = cfg

        self.enc_rna = ModalityEncoder(cfg.rna_dim, cfg.rna_hidden, cfg.rna_out)
        self.enc_cnv = ModalityEncoder(cfg.cnv_dim, cfg.kv_hidden, cfg.kv_out)
        self.enc_mut = ModalityEncoder(cfg.mut_dim, cfg.kv_hidden, cfg.kv_out)
        self.enc_clin = ModalityEncoder(cfg.clin_dim, cfg.clin_hidden, cfg.clin_out)

        self.fusion = CrossAttentionFusion(
            embed_dim=cfg.rna_out,
            kv_dim=cfg.kv_out,
            num_heads=cfg.attn_heads,
        )

        fused_dim = cfg.rna_out + cfg.kv_out + cfg.kv_out + cfg.clin_out
        self.mu_head = nn.Linear(fused_dim, cfg.z_dim)
        self.logvar_head = nn.Linear(fused_dim, cfg.z_dim)

        self.prior = GMMPrior(cfg.n_components, cfg.z_dim)

        trunk: list[nn.Module] = []
        last = cfg.z_dim
        for h in cfg.decoder_hidden:
            trunk.append(nn.Linear(last, h))
            trunk.append(nn.ReLU())
            last = h
        self.decoder_trunk = nn.Sequential(*trunk)
        self.dec_rna = nn.Linear(last, cfg.rna_dim)
        self.dec_cnv = nn.Linear(last, cfg.cnv_dim)
        self.dec_mut = nn.Linear(last, cfg.mut_dim)
        self.dec_clin = nn.Linear(last, cfg.clin_dim)

    def encode(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        h_rna = self.enc_rna(batch["rna"])
        h_cnv = self.enc_cnv(batch["cnv"])
        h_mut = self.enc_mut(batch["mut"])
        h_clin = self.enc_clin(batch["clin"])
        kv = torch.stack([h_cnv, h_mut], dim=1)
        h_rna_post = self.fusion(h_rna, kv)
        fused = torch.cat([h_rna_post, h_cnv, h_mut, h_clin], dim=-1)
        mu = self.mu_head(fused)
        logvar = torch.clamp(self.logvar_head(fused), self.cfg.logvar_min, self.cfg.logvar_max)
        return mu, logvar

    def reparameterize(
        self,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        eps: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if eps is None:
            eps = torch.randn_like(mu)
        return mu + torch.exp(0.5 * logvar) * eps

    def decode(self, z: torch.Tensor) -> Dict[str, torch.Tensor]:
        h = self.decoder_trunk(z)
        return {
            "rna": self.dec_rna(h),
            "cnv": self.dec_cnv(h),
            "mut": self.dec_mut(h),
            "clin": self.dec_clin(h),
        }

    def forward(
        self,
        batch: Dict[str, torch.Tensor],
        eps: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        mu, logvar = self.encode(batch)
        z = self.reparameterize(mu, logvar, eps)
        recon = self.decode(z)
        return {"recon": recon, "mu": mu, "logvar": logvar, "z": z}


def build_vae(cfg: VAEConfig) -> VAE:
    return VAE(cfg)
