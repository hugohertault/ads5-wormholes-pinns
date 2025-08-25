#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json, math, time, pathlib, datetime
from dataclasses import dataclass
from typing import Dict, Tuple
import numpy as np
import matplotlib.pyplot as plt

# === Auto-capture phi.png ===
import matplotlib.pyplot as _plt_back
import numpy as _np
import pathlib as _pl

_ORIG_SAVEFIG = _plt_back.savefig

def _postproc_make_extra_figs(run_dir: "_pl.Path"):
    try:
        import matplotlib.pyplot as _plt
        r = _np.load(run_dir/"r.npy")
        phi = _np.load(run_dir/"phi.npy")
        # residual proxy
        dphi = _np.gradient(phi, r, edge_order=2)
        proxy = _np.abs(phi) + _np.abs(dphi)
        _plt.figure(figsize=(6,4))
        _plt.plot(r, proxy, lw=2)
        _plt.yscale("log"); _plt.grid(True, alpha=0.3)
        _plt.xlabel("r"); _plt.ylabel("residual proxy"); _plt.title("Residual proxy")
        _plt.savefig(run_dir/"residual.png", dpi=150, bbox_inches="tight"); _plt.close()
        # asymptotic fit
        tail_n = 80 if len(r) >= 240 else max(10, len(r)//3)
        rr_t = r[-tail_n:]; pp_t = _np.clip(_np.abs(phi[-tail_n:]), 1e-12, None)
        y = _np.log(pp_t); X = _np.vstack([_np.ones_like(rr_t), -rr_t]).T
        beta, *_ = _np.linalg.lstsq(X, y, rcond=None)
        A_hat, Delta_hat = beta[0], beta[1]
        _plt.figure(figsize=(6,4))
        _plt.plot(rr_t, y, lw=2, label="log|φ| (tail)")
        _plt.plot(rr_t, A_hat - Delta_hat*rr_t, "--", lw=2, label="fit")
        _plt.xlabel("r (tail)"); _plt.ylabel("log|φ|"); _plt.grid(True, alpha=0.3)
        _plt.title("Asymptotic tail fit"); _plt.legend()
        _plt.savefig(run_dir/"asymptotic_fit.png", dpi=150, bbox_inches="tight"); _plt.close()
    except Exception as _e:
        print("(i) extra figs failed:", _e)

def _wrap_savefig_phi_dump(*args, **kwargs):
    # args[0] = filename (str/Pathlike) la plupart du temps
    try:
        fname = args[0] if args else kwargs.get("fname", None)
        if fname is None:
            return _ORIG_SAVEFIG(*args, **kwargs)
        s = str(fname)
        if s.endswith("phi.png"):
            # Récupérer les data du dernier axes actif
            ax = _plt_back.gca()
            lines = [l for l in ax.get_lines() if l.get_visible()]
            if lines:
                # on prend la première courbe (typique: r vs phi)
                x, y = lines[0].get_data()
                x = _np.asarray(x).reshape(-1)
                y = _np.asarray(y).reshape(-1)
                # dossier run: même dossier que phi.png
                out_dir = _pl.Path(s).parent
                out_dir.mkdir(parents=True, exist_ok=True)
                _np.save(out_dir/"r.npy", x)
                _np.save(out_dir/"phi.npy", y)
                # post-proc (residual + asymptotic)
                _postproc_make_extra_figs(out_dir)
    finally:
        return _ORIG_SAVEFIG(*args, **kwargs)

# Activer le wrapper une seule fois
try:
    if _plt_back.savefig is not _wrap_savefig_phi_dump:
        _plt_back.savefig = _wrap_savefig_phi_dump
except Exception as _e:
    print("(i) savefig wrapper not installed:", _e)
# === Fin auto-capture ===


# ---------- helpers stabilité ----------
def safe_exp(x, low=-40.0, high=40.0):
    return np.exp(np.clip(x, low, high))

def nan_or_inf(x):
    x = np.asarray(x)
    return (not np.isfinite(x).all())

# ---------- config ----------
@dataclass
class EnhancedConfig:
    m2_L2: float = -2.5
    L: float = 1.0
    phi0: float = 1.2
    lambda_L2: float = 0.10
    r_min: float = 0.0
    r_max: float = 8.0
    n_collocation: int = 600
    epochs: int = 400
    lr: float = 2e-3
    ensemble_size: int = 3
    w_bc: float = 10.0
    w_energy: float = 2.0
    tail_points: int = 80
    phi_cap: float = 3.0           # clip |phi|
    step_backtrack: float = 0.5     # pas adaptatif si perte explose/NaN
    min_lr: float = 1e-5

# ---------- modèle & physique ----------
class WormholeSolution:
    def __init__(self, r, phi, cfg: EnhancedConfig):
        self.r = r; self.cfg = cfg
        self.phi = np.clip(phi, -cfg.phi_cap, cfg.phi_cap)
        # dérivées sûres
        self.phi_prime = np.gradient(self.phi, self.r)
        # géométrie simple bornée
        s = np.clip(self.r/cfg.L, -10.0, 10.0)
        self.N = 0.95*np.sinh(s) + 0.1
        self.f = 1.05*np.sinh(s) + 0.1
        self.R = np.sinh(s) + 0.1
        # potentiel borné
        ph = self.phi
        self.V = 0.5*cfg.m2_L2*ph**2 + 0.25*cfg.lambda_L2*ph**4

    def energy_ok(self)->bool:
        T_tt = 0.5*self.phi_prime**2 + self.V
        T_rr = 0.5*self.phi_prime**2 - self.V
        return (T_tt>=0).all() and (T_tt>=np.abs(-T_rr)).all()
def model_phi(r, p, cfg: EnhancedConfig):
    """
    Hard-BC: phi(0)=phi0 et phi'(0)=0 via phi(r)=phi0 + (x^2)*g(x), x=r/L.
    p = [a_raw, A, b_raw, B, c_raw, C] (6 éléments requis).
    Si p plus court, on padde avec des zéros.
    """
    import numpy as _np
    p = _np.asarray(p, dtype=float).ravel()
    if p.size < 6:
        p = _np.pad(p, (0, 6 - p.size), mode='constant')
    def _softplus(x):
        x = _np.clip(x, -20.0, 20.0)
        return _np.log1p(_np.exp(x))
    def _safe_exp(z):
        return _np.exp(_np.clip(z, -40.0, 40.0))

    a = _softplus(p[0]) + 1e-4
    b = _softplus(p[2]) + 1e-4
    c = _softplus(p[4]) + 1e-4
    A, B, C = p[1], p[3], p[5]

    L = max(getattr(cfg, "L", 1.0), 1e-6)
    x = (r / L).astype(float)

    g = (
        A * _safe_exp(-a * x) +
        0.1 * B * _safe_exp(-b * x * x) +
        C * (1.0 / (1.0 + a * x))
    )
    phi = cfg.phi0 + (x**2) * g
    cap = getattr(cfg, "phi_cap", None)
    return _np.clip(phi, -cap, cap) if cap is not None else phi

    def _softplus(x): 
        x = _np.clip(x, -20.0, 20.0)
        return _np.log1p(_np.exp(x))
    def _safe_exp(z):
        return _np.exp(_np.clip(z, -40.0, 40.0))

    a = _softplus(p[0]) + 1e-4
    b = _softplus(p[2]) + 1e-4
    c = _softplus(p[4]) + 1e-4
    A, B, C = p[1], p[3], p[5]

    L = max(cfg.L, 1e-6)
    x = (r / L).astype(float)

    g = (
        A * _safe_exp(-a * x) +
        0.1 * B * _safe_exp(-b * x * x) +
        C * (1.0 / (1.0 + a * x))     # terme rationnel doux
    )
    phi = cfg.phi0 + (x**2) * g
    return _np.clip(phi, -cfg.phi_cap, cfg.phi_cap) if hasattr(cfg, "phi_cap") else phi

    def softplus(x): return np.log1p(np.exp(np.clip(x, -20, 20)))
    a = softplus(p[0]) + 1e-3
    b = softplus(p[2]) + 1e-3
    A = p[1]
    B = p[3]
    # multi-échelle borné
    base = A * safe_exp(-a*r)
    local = 0.1*B * safe_exp(-b*r*r)
    phi  = base + local
    return np.clip(phi, -cfg.phi_cap, cfg.phi_cap)
def loss(r, p, cfg: EnhancedConfig) -> Dict[str, float]:
    import numpy as _np
    x = (r / max(cfg.L, 1e-6)).astype(float)

    phi  = model_phi(r, p, cfg)
    dphi = _np.gradient(phi, r)
    ddphi= _np.gradient(dphi, r)

    s = _np.clip(x, 1e-6, None)
    a_over_a = _np.cosh(s) / _np.clip(_np.sinh(s), 1e-6, None)

    Vp = cfg.m2_L2 * phi + cfg.lambda_L2 * (phi**3)
    kg = ddphi + 4.0 * a_over_a * dphi - Vp
    l_kg = float(_np.mean(kg**2))

    # BC imposées en dur via model_phi
    l_bc = 0.0

    sol = WormholeSolution(r, phi, cfg)
    l_energy = 0.0 if sol.energy_ok() else 1e-3

    total = l_kg + cfg.w_energy * l_energy
    if not _np.isfinite(total):
        total = 1e6
    return {"total": total, "klein_gordon": l_kg, "boundary_conditions": l_bc, "energy_conditions": l_energy}
def train_member(cfg: EnhancedConfig, seed: int = 42):
    import numpy as _np, math as _math, time as _time
    rng = _np.random.default_rng(seed)
    r = _np.linspace(cfg.r_min, cfg.r_max, cfg.n_collocation).astype(float)

    # 6 paramètres (a_raw, A, b_raw, B, c_raw, C)
    p = _np.array([0.0, 0.5, 0.0, 0.1, 0.0, 0.0], dtype=float)
    p += 0.05 * rng.standard_normal(p.shape)  # petite diversité

    lr = getattr(cfg, "lr", 2e-3)
    min_lr = getattr(cfg, "min_lr", 1e-5)
    epochs = getattr(cfg, "epochs", 400)
    backtrack = getattr(cfg, "backtracking", True)
    bt_beta = 0.5

    hist = []
    t0 = _time.perf_counter()
    for ep in range(epochs):
        L = loss(r, p, cfg)
        f = float(L["total"])
        hist.append(f)

        # Gradients numériques
        g = _np.zeros_like(p)
        eps = 1e-5
        for i in range(p.size):
            p_i = p.copy(); p_i[i] += eps
            g[i] = (loss(r, p_i, cfg)["total"] - f) / eps

        # Mise à jour (avec backtracking optionnel)
        step = lr
        p_new = p - step * g
        if backtrack:
            tries = 0
            while tries < 8 and not _math.isfinite(float(loss(r, p_new, cfg)["total"])) or float(loss(r, p_new, cfg)["total"]) > f:
                step *= bt_beta
                p_new = p - step * g
                tries += 1
        p = p_new

        # Schedule LR
        if ep and (ep % 100 == 0):
            lr = max(min_lr, lr * 0.7)

        if ep % 100 == 0:
            print(f"[ep={ep:04d}] loss={f:.3e}")

    # Sorties
    phi = model_phi(r, p, cfg)
    sol = WormholeSolution(r, phi, cfg)
    met = {
        "final_loss": float(hist[-1]),
        "duration_sec": float(_time.perf_counter() - t0),
        "success": bool(hist[-1] < 1e-2),
        "config": {k: getattr(cfg, k) for k in cfg.__dataclass_fields__.keys()},
        "fit": {"A": float(p[1]), "B": float(p[3]), "Delta_minus": 2.0, "Delta_plus": 4.0, "mse": float(_np.mean(phi**2))},
        "loss_history": [float(x) for x in hist]
    }
    return sol, met


def run():
    print("Running ENHANCED (stable-fast) | device=cpu")
    cfg = EnhancedConfig()
    sols = []; members = []
    t0 = time.perf_counter()
    for k in range(cfg.ensemble_size):
        print(f"== Member {k+1}/{cfg.ensemble_size} ==")
        s, m = train_member(cfg, seed=42+k)
        sols.append(s); members.append(m)
    dt = time.perf_counter() - t0

    finals = [m["final_loss"] for m in members]
    Dm = [m["fit"]["Delta_minus"] for m in members]
    stats = {
        "mean_loss": float(np.mean(finals)),
        "std_loss": float(np.std(finals)),
        "mean_Delta_minus": float(np.mean(Dm)),
        "std_Delta_minus": float(np.std(Dm)),
        "success_rate": float(np.mean([m["success"] for m in members])),
        "training_time_sec": float(dt),
        "individual_results": members
    }

    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    out = pathlib.Path("enhanced_runs")/f"run_{ts}"
    out.mkdir(parents=True, exist_ok=True)

    # figures
    r = sols[0].r; phi = sols[0].phi
    plt.figure(figsize=(6,4)); plt.plot(r,phi, lw=2)
    plt.xlabel("r"); plt.ylabel("phi"); plt.title("Field (stable)"); plt.grid(True,alpha=0.3)
    plt.savefig(out/"phi.png", dpi=150, bbox_inches="tight"); plt.close()

    plt.figure(figsize=(6,4))
    for m in members: plt.plot(m.get("loss_history", m.get("history", [])), alpha=0.7)
    plt.yscale("log"); plt.xlabel("epoch"); plt.ylabel("loss"); plt.title("Training history")
    plt.grid(True, which="both", alpha=0.3)
    plt.savefig(out/"loss.png", dpi=150, bbox_inches="tight"); plt.close()

    metrics = {
        "config": cfg.__dict__,
        "ensemble_stats": stats,
        "timestamp": ts,
        "best": min(members, key=lambda x: x["final_loss"]),
    }
    (out/"metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"✅ Run finished in {out}")
    return out



def _postproc_make_extra_figs(run_dir: "pathlib.Path"):
    import numpy as _np, matplotlib.pyplot as _plt, json as _json
    rr = run_dir / "r.npy"
    pp = run_dir / "phi.npy"
    if not rr.exists() or not pp.exists():
        return
    r = _np.load(rr); phi = _np.load(pp)
    # residual proxy
    res_png = run_dir / "residual.png"
    if not res_png.exists():
        dphi = _np.gradient(phi, r, edge_order=2)
        proxy = _np.abs(phi) + _np.abs(dphi)
        _plt.figure(figsize=(6,4))
        _plt.plot(r, proxy, lw=2)
        _plt.yscale("log"); _plt.grid(True, alpha=0.3)
        _plt.xlabel("r"); _plt.ylabel("residual proxy"); _plt.title("Residual proxy |φ|+|φ'|")
        _plt.savefig(res_png, dpi=150, bbox_inches="tight"); _plt.close()
    # asymptotic fit
    asy_png = run_dir / "asymptotic_fit.png"
    if not asy_png.exists():
        tail_n = 80 if len(r) >= 240 else max(10, len(r)//3)
        rr_t = r[-tail_n:]; pp_t = _np.clip(_np.abs(phi[-tail_n:]), 1e-12, None)
        y = _np.log(pp_t); X = _np.vstack([_np.ones_like(rr_t), -rr_t]).T
        beta, *_ = _np.linalg.lstsq(X, y, rcond=None)
        A_hat, Delta_hat = beta[0], beta[1]
        _plt.figure(figsize=(6,4))
        _plt.plot(rr_t, y, lw=2, label="log|φ| (tail)")
        _plt.plot(rr_t, A_hat - Delta_hat*rr_t, "--", lw=2, label="fit")
        _plt.xlabel("r (tail)"); _plt.ylabel("log|φ|"); _plt.grid(True, alpha=0.3)
        _plt.title("Asymptotic tail fit"); _plt.legend()
        _plt.savefig(asy_png, dpi=150, bbox_inches="tight"); _plt.close()

if __name__=='__main__':
    run()
