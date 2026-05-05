# Physiology-Informed Digital Twins for Personalized Chemotherapy Response Simulation in Advanced Ovarian Cancer

> Reference implementation accompanying **Wei *et al.*, "Physiology-Informed Digital Twins Powered by World Models for Personalized Chemotherapy Response Simulation in Advanced Ovarian Cancer."** Manuscript prepared for *npj Digital Medicine* (Nature Portfolio).

`PhysioDigitalTwin-OC` is a physiology-constrained Neural-ODE framework that couples (i) variational multi-omics patient encoding, (ii) population-level pharmacokinetic / pharmacodynamic priors as a *structural* inductive bias rather than a loss-term regulariser, and (iii) a recurrent world model for counterfactual treatment rollouts in the latent phase space of high-grade serous ovarian carcinoma (HGSOC). The repository provides a modular, mathematically auditable re-implementation organised by physical concern; each module is verified against analytical or empirical ground truth before composition into the full pipeline.

---

## 1. Scientific Motivation

High-grade serous ovarian carcinoma (HGSOC) carries five-year overall survival below 30 % for advanced-stage disease. First-line carboplatin–paclitaxel chemotherapy yields durable response in only 20–30 % of patients, with ≈70 % of platinum-sensitive patients developing resistance within 18 months. Existing predictive models for chemotherapy response fall into two complementary but disjoint families:

* **Data-driven models** — Cox-PH, Random Survival Forest, DeepSurv, SALMON, AUTOSurv, SurvPath, Vanilla Neural ODE, TDNODE — discover statistical regularities but encode no pharmacological knowledge, and treat treatment response as a static binary classification on baseline features alone.
* **Mechanistic PK/PD models** — population pharmacokinetics, agent-based (e.g., ALISON), Gompertzian or exponential growth — encode pharmacology faithfully but cannot ingest high-dimensional molecular profiles, and require manual parameter estimation under population-level homogeneity assumptions.

`PhysioDigitalTwin-OC` resolves this dichotomy by injecting Gompertzian growth, two-compartment carboplatin pharmacokinetics, and Hill dose–response **inside the right-hand side of a latent-state ODE** rather than as soft penalties on the loss surface. The resulting framework simultaneously exploits multi-omics individual variation and respects established drug–tumour dynamics — a structural inductive bias that population pharmacology cannot supply and that loss-term regularisation only weakly approximates.

> [!IMPORTANT]
> This is a **retrospective computational analysis** built exclusively on publicly released, de-identified datasets. No prospective data collection, no wet-laboratory experiment, no clinical trial, and no IRB approval are involved or required at any stage of the pipeline. The framework is a research artefact — **not** a medical device, **not** validated for clinical deployment, and **must not** inform individual patient treatment decisions outside of a properly governed research protocol.

---

## 2. Theoretical Framework

### 2.1 Continuous-time state-space formulation

Each patient is modelled as a continuous-time dynamical system whose state vector

$$\mathbf{s}(t) = \big[\,V(t),\, C_{1}(t),\, C_{2}(t),\, E(t)\,\big]^{\top} \in \mathbb{R}^{4}$$

tracks tumour volume $V$, central-compartment drug concentration $C_{1}$, peripheral-compartment drug concentration $C_{2}$, and pharmacological effect $E$. Patient-specific molecular context enters through a low-dimensional latent representation $\mathbf{z} \in \mathbb{R}^{d},\ d = 64$, learned by a variational encoder over multi-omics input $\mathbf{x}$ (Section 2.4).

### 2.2 Physics-informed dynamics decomposition (PhysNODE)

The state derivative is decomposed into a learnt patient-specific residual and a population-pharmacology prior:

$$\frac{\mathrm{d}\mathbf{s}}{\mathrm{d}t} \;=\; \underbrace{f_{\text{neural}}\!\left(\mathbf{s},\, t,\, \mathbf{z};\, \theta\right)}_{\text{patient-specific residual}} \;+\; \lambda(t)\cdot \underbrace{f_{\text{physics}}\!\left(\mathbf{s},\, t;\, \boldsymbol{\psi}\right)}_{\text{pharmacological prior}} \tag{1}$$

with $\lambda(t)$ a linear warm-up schedule rising from $0.1$ to $1.0$ over the first 50 epochs. The residual $f_{\text{neural}}$ is a three-layer MLP (hidden widths 256, 128, 64) gated by FiLM (Feature-wise Linear Modulation) on $\mathbf{z}$ with Swish activations, parameterised as $f_{\text{neural}}(\mathbf{s},t,\mathbf{z}) = \mathrm{MLP}\!\big(\gamma(\mathbf{z}) \odot [\mathbf{s}; t] + \beta(\mathbf{z})\big)$.

#### Component (i) — Gompertzian tumour growth   *(Eq. 3 of Wei et al.)*

$$\left.\frac{\mathrm{d}V}{\mathrm{d}t}\right|_{\text{growth}} \;=\; -\alpha\, V \ln\!\frac{V}{K} \tag{2}$$

The Gompertz law captures the saturating expansion of solid tumours toward carrying capacity $K$, validated across pathology types and with particular relevance to ovarian carcinoma. A drug-induced cell-kill term $-\,\mathrm{effect}(C_{1})\cdot V$ couples (2) to the pharmacological cascade and is included in the combined RHS exposed by `pdt_oc.phys.combined.f_physics`.

#### Component (ii) — Two-compartment carboplatin pharmacokinetics   *(Eq. 4)*

$$\frac{\mathrm{d}C_{1}}{\mathrm{d}t} \;=\; -\!\left(\tfrac{CL}{V_{c}} + \tfrac{Q}{V_{c}}\right) C_{1} + \tfrac{Q}{V_{p}} C_{2} + \tfrac{\mathrm{dose}(t)}{V_{c}}, \qquad \frac{\mathrm{d}C_{2}}{\mathrm{d}t} \;=\; \tfrac{Q}{V_{c}} C_{1} - \tfrac{Q}{V_{p}} C_{2} \tag{3}$$

Population priors $CL = 7.38\,\mathrm{L\,h^{-1}}$, $V_{c} = 17.1\,\mathrm{L}$, $V_{p} = 411\,\mathrm{L}$ are taken from the EORTC carboplatin study (Calvert *et al.*, n = 139 ovarian-cancer patients). Patient-specific deviations within the documented 48–130 % coefficient of variation are absorbed by $f_{\text{neural}}$.

#### Component (iii) — Hill dose–response   *(Eq. 5)*

$$E \;=\; E_{\max}\,\frac{C_{1}^{\gamma}}{EC_{50}^{\gamma} + C_{1}^{\gamma}} \tag{4}$$

linking central-compartment drug exposure to cytotoxic effect via the sigmoidal Hill curve.

#### Numerical solver

Equation (1) is integrated with the Dormand–Prince adaptive RK45 scheme, $\mathrm{rtol} = 10^{-5}$, $\mathrm{atol} = 10^{-6}$. The continuous adjoint sensitivity method (Chen *et al.*, 2018) is used to back-propagate through the solver in the full implementation (release Turn 2; see §3).

### 2.3 Multi-omics variational encoder

The patient encoder ingests four heterogeneous modalities — RNA-seq ($D_{\text{RNA}} = 20{,}531$ genes, retained to 5{,}000 most variable after $\log_{2}(\mathrm{TPM}+1)$ transformation), copy-number variants (5-level encoding), somatic point mutations (binary indicator), and clinical covariates (age, FIGO stage, debulking status, BRCA1/2 status):

1. **Modality-specific MLPs.** $\mathbf{h}_{\text{RNA}} = \mathrm{MLP}_{\text{RNA}}(\mathbf{x}_{\text{RNA}}) \in \mathbb{R}^{128}$ and analogously $\mathbf{h}_{\text{CNV}}, \mathbf{h}_{\text{mut}} \in \mathbb{R}^{64}$, $\mathbf{h}_{\text{clin}} \in \mathbb{R}^{32}$.
2. **Cross-attention fusion.** RNA-seq embedding serves as query; concatenated CNV and mutation embeddings serve as keys and values, modelling inter-omics dependence on transcriptional perturbation.
3. **Variational inference** with diagonal-covariance Gaussian posterior

$$q_{\phi}(\mathbf{z}\mid\mathbf{x}) \;=\; \mathcal{N}\!\Big(\boldsymbol{\mu}_{\phi}(\mathbf{h}),\; \mathrm{diag}\!\big(\boldsymbol{\sigma}^{2}_{\phi}(\mathbf{h})\big)\Big), \qquad \mathbf{z} \in \mathbb{R}^{64}. \tag{5}$$

4. **Four-component Gaussian-mixture prior**

$$p(\mathbf{z}) \;=\; \sum_{k=1}^{4} \pi_{k}\, \mathcal{N}(\boldsymbol{\mu}_{k},\,\mathbf{I}) \tag{6}$$

aligned to the four canonical TCGA molecular subtypes of HGSOC (immunoreactive, differentiated, proliferative, mesenchymal). The encoder is reported to recover this subtype structure with adjusted Rand index $\mathrm{ARI} \approx 0.42$ without explicit subtype supervision.

5. **Posterior sampling for uncertainty quantification.** $M = 50$ Monte-Carlo draws $\mathbf{z}^{(m)} \sim q_{\phi}(\mathbf{z}\mid\mathbf{x})$ are propagated through the full pipeline to construct prediction intervals capturing both epistemic (latent sampling) and aleatoric (posterior width) sources of uncertainty.

### 2.4 Counterfactual world model

Treatment actions $\mathbf{a}_{t} = [\mathrm{drug\_id},\, \mathrm{dose},\, \mathrm{schedule},\, \mathrm{cycle}]$ drive the latent dynamics through a gated recurrent unit:

$$\mathbf{z}_{t+1} \;=\; g_{\omega}(\mathbf{z}_{t},\, \mathbf{a}_{t}) \;=\; \mathrm{GRU}_{\omega}\!\big([\mathbf{z}_{t}\, \Vert\, \mathbf{a}_{t}]\big) \tag{7}$$

with hidden state of dimension 128 capturing treatment history across sequential decisions. Counterfactual rollouts of alternative regimens $\mathbf{a}'_{t} \neq \mathbf{a}_{t}$ from a shared baseline $\mathbf{z}_{0}$ are constrained to the same PhysNODE manifold, ensuring pharmacological plausibility — a property unavailable to unconstrained generative counterfactual models such as GAN-based individual-treatment-effect estimators.

### 2.5 Composite training objective

The model is trained end-to-end by minimising

$$\mathcal{L} \;=\; \mathcal{L}_{\text{surv}} \;+\; \lambda_{1}\mathcal{L}_{\text{resp}} \;+\; \lambda_{2}\mathcal{L}_{\text{drug}} \;+\; \lambda_{3}\mathcal{L}_{\text{ELBO}} \;+\; \lambda_{4}\!\left(\mathcal{L}_{\text{Gompertz}} + \mathcal{L}_{\text{PK}} + \mathcal{L}_{\text{Hill}}\right) \tag{8}$$

| Symbol | Term | Form | Weight |
|---|---|---|---|
| $\mathcal{L}_{\text{surv}}$ | Overall-survival likelihood | Negative Cox partial likelihood | 1 |
| $\mathcal{L}_{\text{resp}}$ | Platinum-response classification | Binary cross-entropy (PFI ≥ 6 months cut-off) | $\lambda_{1} = 1.0$ |
| $\mathcal{L}_{\text{drug}}$ | Cell-line drug-sensitivity (GDSC IC₅₀) | Mean-squared error | $\lambda_{2} = 0.5$ |
| $\mathcal{L}_{\text{ELBO}}$ | VAE evidence lower bound | $-\mathbb{E}_{q}[\log p_{\psi}(\mathbf{x}\mid\mathbf{z})] + D_{\mathrm{KL}}\!\big(q_{\phi}\,\Vert\,p\big)$ | $\lambda_{3} = 1.0$ |
| $\mathcal{L}_{\text{Gompertz/PK/Hill}}$ | Physics residuals at $N_{c}=100$ collocation points | Squared deviation of integrated derivative from analytical RHS | $\lambda_{4}: 0.1 \to 1.0$ (warm-up) |

The KL divergence to the GMM prior is computed in closed form via $\log \sum_{k} \pi_{k}\mathcal{N}(\mathbf{z};\boldsymbol{\mu}_{k},\mathbf{I})$ using `torch.logsumexp` for numerical stability — see `pdt_oc.training.losses` and `pdt_oc.models.vae.GMMPrior`.

### 2.6 Two-stage curriculum

Training proceeds as a transfer-learning curriculum to mitigate the small-sample regime of TCGA-OV ($n = 489$):

| Stage | Data | $\lambda_{4}$ | Components updated | Approximate duration |
|---|---|---|---|---|
| 1 (pre-train) | GDSC pan-cancer (≈1 000 cell lines, 198 compounds) | 0 | VAE encoder + drug-sensitivity head | 4 h on 1 × A100 |
| 2 (fine-tune) | TCGA-OV ($n = 489$) | $0.1 \to 1.0$ over 50 epochs | full model (PhysNODE + heads) | 2 h on 1 × A100 |

---

## 3. Implementation Roadmap

The reference implementation is partitioned by *mathematical concern* rather than by training stage. Each module is independently testable; testing strategies are deliberately rotated across releases to surface different failure modes (analytical-truth regression, mass conservation, single-batch overfit, and — in the upcoming PhysNODE release — adjoint-gradient consistency via `torch.autograd.gradcheck`).

| Release | Layer | Module | Status | Test type | Tests |
|---|---|---:|---|---|---:|
| Turn 0 | Pharmacological constraints | `pdt_oc.phys` | ✓ Released | Analytical-truth regression | 5 |
| Turn 0 | ODE numerics | `pdt_oc.numerics` | ✓ Released | Wrapper validation | (covered) |
| Turn 0 | Dosing schedules | `pdt_oc.dosing` | ✓ Released | Mass conservation | 4 |
| Turn 0 | Training-loop primitives | `pdt_oc.training` | ✓ Released | Boundary / determinism | 8 |
| Turn 1 | Synthetic multi-omics generator | `pdt_oc.data` | ✓ Released | (utility) | — |
| Turn 1 | Multi-omics VAE | `pdt_oc.models` | ✓ Released | Single-batch overfit | 6 |
| Turn 2 | FiLM-modulated $f_{\text{neural}}$ | `pdt_oc.dynamics` | Scheduled | `torch.autograd.gradcheck` | — |
| Turn 2 | Adjoint Neural-ODE solver (`torchdiffeq`) | `pdt_oc.solver` | Scheduled | Forward / backward parity | — |
| Turn 3 | GRU world model | `pdt_oc.world` | Scheduled | Rollout consistency | — |
| Turn 3 | Multi-task heads (Cox / BCE / MSE) | `pdt_oc.heads` | Scheduled | Calibration | — |
| Turn 4 | Two-stage training pipeline | `pdt_oc.train` | Scheduled | End-to-end | — |
| Turn 4 | Calibration & survival metrics | `pdt_oc.metrics` | Scheduled | Bootstrap CI | — |
| Turn 5 | TCGA-OV / GDSC / GEO loaders | `pdt_oc.io` | Scheduled | Schema parity | — |
| Turn 5 | Counterfactual rollout API | `pdt_oc.counterfactual` | Scheduled | GOG-158 concordance | — |

**v0.0.1 status — 23 deterministic tests passing in approximately 4 s on a single CPU.**

---

## 4. Repository Layout

```
physiodigitaltwin-oc/
├── pyproject.toml                        PEP 621 single-source metadata + deps
├── .gitignore
├── README.md
├── src/pdt_oc/
│   ├── phys/                             Pharmacological constraints (Eq. 2–4 + composition)
│   │   ├── state.py                      PhysioParams, Idx, carboplatin_defaults, make_state
│   │   ├── gompertz.py                   dV/dt growth term (Eq. 2)
│   │   ├── two_compartment.py            PK ODEs (Eq. 3), Calvert priors
│   │   ├── hill.py                       E(C_1) algebraic + vectorised trace (Eq. 4)
│   │   └── combined.py                   f_physics: cell-kill-coupled 3-D RHS for solve_ivp
│   ├── numerics/integrator.py            Dormand–Prince RK45 wrapper
│   ├── dosing/schedule.py                constant_infusion / bolus_schedule / zero_dose
│   ├── training/
│   │   ├── schedules.py                  λ(t) warm-up
│   │   ├── collocation.py                N_c collocation point sampler (uniform / jittered)
│   │   └── losses.py                     ELBO, KL-to-GMM via logsumexp
│   ├── data/synthetic.py                 OmicsDims, generate_batch (RNA / CNV / MUT / CLIN)
│   └── models/
│       ├── encoders.py                   ModalityEncoder (configurable MLP)
│       ├── fusion.py                     CrossAttentionFusion (RNA = Q, [CNV, MUT] = K, V)
│       └── vae.py                        VAEConfig, GMMPrior, VAE, build_vae
└── tests/
    ├── test_physics_invariants.py        Gompertz fixed point, PK AUC, Hill midpoint
    ├── test_combined_system.py           Joint zero-dose recovery & continuous-infusion suppression
    ├── test_dosing.py                    Mass conservation
    ├── test_training_aux.py              Warm-up boundaries + collocation determinism
    └── test_vae_overfit.py               Single-batch overfit + diag-Gaussian log-prob parity
```

---

## 5. Installation

Tested on macOS arm64 (Apple Silicon) with system Python 3.9.6 and CPU-only PyTorch 2.8.0. CUDA is **not** required for v0.0.1; subsequent releases (Turn 2 onward) will benefit from GPU acceleration.

```
git clone https://github.com/WangshizhuoCMU/physiodigitaltwin-oc.git
cd physiodigitaltwin-oc
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
```

Single-source dependency declaration in `pyproject.toml`:

| Package | Constraint | Purpose |
|---|---|---|
| `numpy` | ≥ 1.26 | Array backend |
| `scipy` | ≥ 1.11 | `solve_ivp` (RK45) for v0.0.1 physics validation |
| `torch` | ≥ 2.2 | VAE encoder; adjoint Neural ODE in Turn 2 (`torchdiffeq`) |
| `pytest` | ≥ 8.0 | Test runner (dev extra) |

---

## 6. Verification

```
pytest -q
```

Expected: `23 passed`. All tests are deterministic when invoked with the fixtures' seeds; aggregate runtime is ≈ 4 s on Apple M-series silicon.

---

## 7. Datasets

| Cohort | $n$ | Modalities | Source | Access |
|---|---:|---|---|---|
| TCGA-OV | 489 | RNA-seq (20 531 genes), CNV (5-level), point mutations (binary), clinical (age, FIGO, debulking, BRCA1/2) | NIH Genomic Data Commons | [portal.gdc.cancer.gov](https://portal.gdc.cancer.gov) |
| GDSC v2 | 47 (OV cell lines) / ≈ 1 000 (pan-cancer pre-training) | IC₅₀ (carboplatin, taxane) for 198 compounds | Sanger Institute | [cancerrxgene.org](https://www.cancerrxgene.org) |
| CCLE / DepMap | 47 (OV) | Gene expression, mutations, CNV | Broad Institute | [depmap.org](https://depmap.org) |
| GSE15459 (Tothill *et al.*) | 285 | Microarray expression, OS, treatment response | Australian Ovarian Cancer Study | [GEO GSE15459](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE15459) |
| GSE31978 (Yoshihara *et al.*) | 58 | Microarray expression, OS, treatment response | Niigata University Hospital, Japan | [GEO GSE31978](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE31978) |

Cross-platform alignment proceeds via gene-symbol intersection (4 217 of 5 000 retained), quantile normalisation, and ComBat batch correction. Aggregate evaluation cohort: **$N = 832$ patients across three countries (USA, Australia, Japan) and two expression platforms (RNA-seq, microarray).**

> [!NOTE]
> Loaders for the four cohorts are scheduled for the `pdt_oc.io` release (Turn 5). v0.0.1 ships only the synthetic batch generator (`pdt_oc.data.synthetic`), sufficient for shape, reparameterisation, and overfit testing of the variational encoder.

---

## 8. Reported Results (Wei *et al.*)

Headline metrics from the paper, retained for cross-reference during reproduction. All values are mean ± SD over 50 rounds (5-fold CV × 10 independent initialisations); confidence intervals come from 1 000 bootstrap resamples.

| Cohort | Endpoint | Metric | Value | 95 % CI | Δ vs. best baseline |
|---|---|---|---|---|---|
| TCGA-OV ($n=489$) | 5-yr overall survival | C-index | **0.68** | 0.65 – 0.71 | + 0.04 (vs. SurvPath 0.64) |
| TCGA-OV | Platinum response (PFI ≥ 6 mo) | AUC-ROC | **0.85** | 0.82 – 0.88 | + 0.05 (vs. NN 0.80) |
| TCGA-OV | Calibration | ECE (10-bin) | **0.048** | ± 0.008 | − 0.05 vs. unconstrained NODE (0.112) |
| GDSC-OV ($n=47$) | IC₅₀ prediction | Pearson $\rho$ | **0.91** | 0.87 – 0.94 | + 0.03 (vs. DeepCDR / GraphDRP) |
| GDSC-OV | IC₅₀ prediction | RMSE (ln scale) | 1.02 | ± 0.075 | — |
| GSE15459 ($n=285$, AUS) | C-index (external) | 0.65 | ± 0.031 | platform: microarray |
| GSE31978 ($n=58$, JP) | C-index (external) | 0.63 | ± 0.048 | platform: microarray |
| TCGA-OV regimen-switchers ($n=47$) | Counterfactual trajectory | Pearson $\rho$ | 0.74 | — | vs. random 0.08 / Neural ODE 0.52 |
| GOG-158 concordance | Carboplatin–paclitaxel preference | Identity rate | 82 % | — | — |

Ablation analysis (Wei *et al.*, Table 4) attributes a 3.0 percentage-point C-index gain to the combined physiological constraints, with a striking *constraint–task asymmetry*: Gompertzian growth contributes disproportionately to survival prediction ($\Delta_{C\text{-index}} = -0.012$ on removal), pharmacokinetic constraints to drug sensitivity ($\Delta_{\mathrm{PCC}} = -0.021$), and the Hill term to the joint outcome — evidence that the constraints encode mechanistic knowledge rather than acting as generic regularisers.

---

## 9. Reproducibility

### 9.1 Compute

| Item | Original paper | Current implementation (v0.0.1) |
|---|---|---|
| GPU | 1 × NVIDIA A100 (40 GB) | None required |
| Stage 1 (GDSC pre-train, $\lambda_{4} = 0$) | ≈ 4 h | scheduled, Turn 4 |
| Stage 2 (TCGA-OV fine-tune, $\lambda_{4}: 0.1 \to 1.0$) | ≈ 2 h | scheduled, Turn 4 |
| 5-fold CV × 10 repeats (= 50 rounds) | yes | scheduled, Turn 4 |
| Bootstrap resampling for CI | 1 000 iterations | scheduled, Turn 4 |

### 9.2 Random seeds (paper convention)

`42, 123, 256, 389, 512, 678, 741, 853, 927, 1024`

### 9.3 Optimisation hyperparameters

| Hyperparameter | Value |
|---|---|
| Optimiser | Adam |
| Learning-rate schedule | Cosine annealing $5 \times 10^{-4} \to 1 \times 10^{-5}$ |
| Epochs | 200 |
| Batch size | per-cohort, paper Supplementary Table S5 |
| VAE posterior samples for UQ | $M = 50$ |
| Collocation points | $N_{c} = 100$ |
| Hyperparameter search | Bayesian optimisation, 10 configurations |
| Statistical testing | paired *t*-test + Holm–Bonferroni; Benjamini–Hochberg FDR for exploratory |

### 9.4 Reporting standards

The original study adheres to **TRIPOD + AI** guidelines for transparent reporting of clinical-prediction models. This implementation will mirror those standards when end-to-end training is released.

---

## 10. Implementation Notes and Intentional Departures

Each departure from the paper's described methodology is logged here for full transparency.

1. **Solver backend (v0.0.1).** The current physics-only release uses `scipy.integrate.solve_ivp` (Dormand–Prince RK45), which suffices for analytical-invariant validation. Turn 2 will switch to `torchdiffeq.odeint_adjoint` for the trainable PhysNODE, exactly matching the paper's continuous adjoint sensitivity method (Chen *et al.*, 2018).
2. **Drug-effect state algebraicity.** The paper lists $E$ as part of the state vector but specifies it algebraically through Eq. 4. The implementation reduces the integrated state to $[V,\, C_{1},\, C_{2}]$ and exposes $E$ as a post-hoc observable via `pdt_oc.phys.effect_trace`. This avoids introducing a non-paper-specified relaxation time on $E$ and preserves agreement with Eq. 4 to machine precision.
3. **Cell-kill coupling parameterisation.** The paper's prose (§Methods, paragraph following Eq. 4) implies the Hill effect enters $\mathrm{d}V/\mathrm{d}t$ as a kill term but does not commit to a functional form. The reference implementation uses the canonical linear-kill form $-\,\mathrm{effect}(C_{1})\cdot V$, recovering $\mathrm{d}V/\mathrm{d}t = -\alpha V \ln(V/K)$ exactly when $C_{1} \equiv 0$.
4. **No fabricated baseline numbers.** The repository does not ship pre-computed baseline values; comparison models will be re-implemented from their original published source code under a unified harness in a future release.
5. **VAE log-variance clamping.** A numerical-safety clamp $\log\sigma^{2} \in [-8,\, 8]$ is applied at the encoder output, preventing `exp(logvar) → 0` blow-ups under deterministic-$\epsilon$ stress tests. This is standard VAE practice and does not alter the paper's effective hyperparameter regime.
6. **No clinical or wet-lab augmentation.** All evaluations operate on the publicly available datasets enumerated in §7; no proprietary data, no in-house cohort, no IRB-governed collection, and no laboratory experiment occurs at any stage of the pipeline.

---

## 11. Citation

```bibtex
@article{wei2026physiodigitaltwin,
  title    = {Physiology-Informed Digital Twins Powered by World Models for
              Personalized Chemotherapy Response Simulation in Advanced
              Ovarian Cancer},
  author   = {Wei, Heng and Feng, Chong and Qin, Xue and Niu, Xing and
              Zhang, Chen and Xia, Yu and Wang, Shizhuo},
  journal  = {npj Digital Medicine},
  year     = {2026}
}
```

A Zenodo DOI for the implementation will be issued upon manuscript acceptance.

---

## 12. Disclaimer

> [!CAUTION]
> The framework is a research artefact for retrospective methodological evaluation on publicly released, de-identified datasets. It is **not** a medical device, **not** approved for any clinical use, and **must not** be used to inform individual patient diagnosis, prognosis, or treatment decisions. Prospective clinical evaluation, regulatory clearance, and integration with clinical-decision-support infrastructure are pre-requisites for any translational application.

---

## 13. Licence

The repository is currently distributed without an explicit licence; under default copyright this reserves all rights to the paper authors. A permissive licence consistent with the journal's data-availability policy will be applied upon manuscript acceptance.

---

## 14. Acknowledgements

The publicly available datasets used in this work are released by the TCGA Research Network, the Sanger Institute (GDSC), the Broad Institute (CCLE / DepMap), and the contributing investigators of GSE15459 (Tothill *et al.*, Australian Ovarian Cancer Study) and GSE31978 (Yoshihara *et al.*, Niigata University Hospital). The pharmacokinetic priors are derived from the EORTC carboplatin study (Calvert *et al.*). The implementation builds on the open-source numerical software stack of `numpy`, `scipy`, `torch`, and (in Turn 2) `torchdiffeq`.
