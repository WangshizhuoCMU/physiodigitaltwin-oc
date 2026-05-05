import pytest
import torch

from pdt_oc.data import generate_batch, small_dims
from pdt_oc.models import VAEConfig, build_vae
from pdt_oc.training import diagonal_gaussian_log_prob, vae_elbo


def _build_small_cfg(z_dim: int = 16) -> VAEConfig:
    dims = small_dims()
    return VAEConfig(
        rna_dim=dims.rna,
        cnv_dim=dims.cnv,
        mut_dim=dims.mut,
        clin_dim=dims.clin,
        z_dim=z_dim,
    )


def test_encoder_output_shapes_match_z_dim() -> None:
    torch.manual_seed(0)
    cfg = _build_small_cfg(z_dim=24)
    model = build_vae(cfg)
    batch = generate_batch(batch_size=4, dims=small_dims(), seed=0)
    mu, logvar = model.encode(batch)
    assert mu.shape == (4, 24)
    assert logvar.shape == (4, 24)


def test_decoder_reconstructs_each_modality_at_input_dim() -> None:
    torch.manual_seed(0)
    cfg = _build_small_cfg()
    model = build_vae(cfg)
    batch = generate_batch(batch_size=3, dims=small_dims(), seed=1)
    output = model(batch)
    assert output["recon"]["rna"].shape == batch["rna"].shape
    assert output["recon"]["cnv"].shape == batch["cnv"].shape
    assert output["recon"]["mut"].shape == batch["mut"].shape
    assert output["recon"]["clin"].shape == batch["clin"].shape


def test_reparameterize_with_zero_eps_returns_mean() -> None:
    torch.manual_seed(0)
    cfg = _build_small_cfg()
    model = build_vae(cfg)
    batch = generate_batch(batch_size=2, dims=small_dims(), seed=0)
    mu, logvar = model.encode(batch)
    z = model.reparameterize(mu, logvar, eps=torch.zeros_like(mu))
    assert torch.allclose(z, mu)


def test_diagonal_log_prob_matches_torch_normal() -> None:
    torch.manual_seed(0)
    mu = torch.randn(5, 8)
    logvar = torch.randn(5, 8) * 0.5
    z = torch.randn(5, 8)
    custom = diagonal_gaussian_log_prob(z, mu, logvar)
    sigma = torch.exp(0.5 * logvar)
    reference = torch.distributions.Normal(mu, sigma).log_prob(z).sum(dim=-1)
    assert torch.allclose(custom, reference, atol=1e-6)


def test_vae_loss_components_are_finite_at_init() -> None:
    torch.manual_seed(0)
    cfg = _build_small_cfg()
    model = build_vae(cfg)
    batch = generate_batch(batch_size=4, dims=small_dims(), seed=0)
    output = model(batch)
    total, parts = vae_elbo(batch, output, model.prior)
    assert torch.isfinite(total)
    assert torch.isfinite(parts["recon"])
    assert torch.isfinite(parts["kl"])


def test_vae_overfits_single_batch_recon_drops_below_ten_percent() -> None:
    torch.manual_seed(0)
    cfg = _build_small_cfg()
    model = build_vae(cfg)
    batch = generate_batch(batch_size=4, dims=small_dims(), seed=0)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    initial_recon: list[float] = []
    final_recon: list[float] = []
    initial_total: list[float] = []
    final_total: list[float] = []

    for step in range(500):
        optimizer.zero_grad()
        out = model(batch)
        total, parts = vae_elbo(batch, out, model.prior, kl_weight=0.0)
        total.backward()
        optimizer.step()
        if step < 5:
            initial_recon.append(float(parts["recon"]))
            initial_total.append(float(total.detach()))
        if step >= 480:
            final_recon.append(float(parts["recon"]))
            final_total.append(float(total.detach()))

    init_r = sum(initial_recon) / len(initial_recon)
    fin_r = sum(final_recon) / len(final_recon)
    init_t = sum(initial_total) / len(initial_total)
    fin_t = sum(final_total) / len(final_total)

    assert fin_r < 0.1 * init_r, (init_r, fin_r)
    assert fin_t < 0.1 * init_t, (init_t, fin_t)
