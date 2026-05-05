from dataclasses import dataclass
from typing import Dict

import torch


MODALITIES = ("rna", "cnv", "mut", "clin")


@dataclass(frozen=True)
class OmicsDims:
    rna: int
    cnv: int
    mut: int
    clin: int

    def as_dict(self) -> Dict[str, int]:
        return {"rna": self.rna, "cnv": self.cnv, "mut": self.mut, "clin": self.clin}


def default_dims() -> OmicsDims:
    return OmicsDims(rna=200, cnv=200, mut=200, clin=10)


def small_dims() -> OmicsDims:
    return OmicsDims(rna=64, cnv=64, mut=64, clin=8)


def generate_batch(
    batch_size: int,
    dims: OmicsDims,
    seed: int = 0,
    mut_rate: float = 0.05,
    device: str = "cpu",
) -> Dict[str, torch.Tensor]:
    g = torch.Generator(device=device).manual_seed(seed)
    rna = torch.randn(batch_size, dims.rna, generator=g, device=device)
    cnv = torch.randint(-2, 3, (batch_size, dims.cnv), generator=g, device=device).float()
    mut_probs = torch.full((batch_size, dims.mut), mut_rate, device=device)
    mut = torch.bernoulli(mut_probs, generator=g)
    clin = torch.randn(batch_size, dims.clin, generator=g, device=device)
    return {"rna": rna, "cnv": cnv, "mut": mut, "clin": clin}
