# physiodigitaltwin-oc

Pharmacology-constrained scaffold for an HGSOC chemotherapy-response digital twin.

## Scope

- Gompertzian tumour growth with cell-kill coupling
- Two-compartment carboplatin pharmacokinetics
- Hill dose–response (algebraic; trace helper for post-hoc reporting)
- Combined `f_physics` ready for `solve_ivp`
- Dormand–Prince RK45 integration
- Linear λ(t) warmup schedule
- Collocation point sampling
- VAE multi-omics encoder with cross-attention fusion and 4-component GMM prior
- Synthetic multi-omics batch generator

## Run

```
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
pytest -q
```
