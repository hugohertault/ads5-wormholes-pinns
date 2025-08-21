# Physics-Informed Neural Networks for Scalar-Stabilized Wormholes in AdS5

This repository provides a **minimal, stable, and reproducible** PINN for a scalar field on a fixed AdS₅-like background, with:

- **Finite-difference (FD) derivatives** for robust PDE residuals (no autograd wrt grid).
- **Bullet-proof asymptotic fit** with ridge regularization + monotone weights and safe fallbacks.
- **Reproducibility**: smoke tests via `pytest`, deterministic configs, GitHub Actions CI.
- **Always-on artifacts**: `phi.png`, `loss.png`, `residual.png`, `asymptotic_fit.png`, `metrics.json` per run.

## 1. Physics background (brief)
- Fixed AdS₅-like background `a(r), b(r)` close to pure AdS with mild perturbations.
- Scalar field with potential `V = -12/L² + (m²L²)/(2L²) φ² + (λL²)/(4 L⁴) φ⁴`.
- PDE: Klein–Gordon on curved background: `φ'' + (a'/a + 3 b'/b) φ' - dV/dφ = 0`.
- BF bound enforced: `m²L² ≥ −4`, with CFT dimensions `Δ± = 2 ± √(4 + m²L²)`.

## 2. Model & training
- **Network**: small MLP (Tanh), stable Xavier init.
- **Grid**: uniform in `r ∈ [r_min, r_max]`, FD for `φ'` and `φ''`.
- **Loss**: PDE residual + throat BC + regularity + decay; optional **mixed BC** (double-trace: `B − fA = 0`).
- **Optim**: Adam → optional LBFGS polish; early-stopping with patience.

## 3. Asymptotics (AdS/CFT) — robust fit
- **Tail fit**: `φ ≈ A e^{−Δ− r/L} + B e^{−Δ+ r/L}` on the last `n_tail` points.
- Robustification: outlier rejection (median/MAD), ridge (`λ=1e−6`), weights increasing with `r`.
- If ill-posed: fallback to safe defaults (never crashes training).

## 4. Quick start
```bash
pip install -r requirements.txt
FAST_SMOKE=1 python main_ads5_pinn_article.py
```
Artifacts go to `runs/<timestamp>/`.

## 5. Tests & CI
```bash
pytest -q
```
CI: `.github/workflows/ci.yml` runs the smoke test on Ubuntu + Python 3.10 (CPU).

## 6. Baseline FD (sanity check)
```bash
python scripts/baseline_fd.py
```
Produces `runs_fd/fd_phi.png` for λ=0.

## 7. File map
- `src/ads5pinns/physics.py`: background, FD operators, KG residual.
- `src/ads5pinns/asymptotics.py`: robust asymptotic fit.
- `src/ads5pinns/losses.py`: composite loss (+ mixed BC / double-trace option).
- `src/ads5pinns/trainer.py`: training loop, plots, metrics.
- `main_ads5_pinn_article.py`: single run + ensemble.
- `scripts/baseline_fd.py`: finite-difference baseline.
- `tests/test_smoke.py`: reproducible smoke test.

## 8. PRD-ready checklist (pratique)
- [x] Résultats reproductibles (tests + CI).
- [x] Fit asymptotique robuste documenté, avec Δ± cohérents.
- [x] Figures standardisées et auto-générées.
- [x] Paramètres BF et cas limites gérés proprement.

## 9. License
MIT — see `LICENSE`.
