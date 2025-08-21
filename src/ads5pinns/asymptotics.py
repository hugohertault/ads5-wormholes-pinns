import numpy as np
from .physics import delta_from_m2

def asymptotic_fit(r, phi, m2_L2, L: float = 1.0, n_tail: int = 24):
    r = np.asarray(r, dtype=float).reshape(-1)
    phi = np.asarray(phi, dtype=float).reshape(-1)
    n = r.size; i0 = max(0, n - n_tail)
    rr = r[i0:]; yy = phi[i0:]
    m = np.isfinite(rr) & np.isfinite(yy)
    rr = rr[m]; yy = yy[m]
    if rr.size < 3:
        dmn, dpl = delta_from_m2(m2_L2)
        A = float(yy.mean()) if yy.size else 0.0
        return dict(A=A, B=0.0, Delta_minus=float(dmn), Delta_plus=float(dpl),
                    tail_index=list(range(i0, n)))
    med = float(np.median(yy)); mad = float(np.median(np.abs(yy-med))) + 1e-12
    mm = np.abs(yy-med) <= 3.0*mad
    if not np.any(mm): mm = slice(None)
    rr = rr[mm]; yy = yy[mm]
    if rr.size < 3:
        rr = r[i0:]; yy = phi[i0:]
        rr = rr[np.isfinite(rr)]; yy = yy[np.isfinite(yy)]
    dmn, dpl = delta_from_m2(m2_L2)
    X1 = np.exp(-dmn*rr/L); X2 = np.exp(-dpl*rr/L)
    X = np.stack([X1,X2], axis=1)
    span = float(rr.max()-rr.min()) if rr.size else 0.0
    w = np.ones_like(rr) if span<=1e-12 else (rr-rr.min())/(span+1e-12)+1e-6
    W = np.diag(w)
    lam = 1e-6
    XtWX = X.T @ W @ X + lam*np.eye(2); XtWy = X.T @ W @ yy
    try: coef = np.linalg.solve(XtWX, XtWy)
    except np.linalg.LinAlgError: coef = np.linalg.lstsq(X, yy, rcond=None)[0]
    Ahat, Bhat = float(coef[0]), float(coef[1])
    return dict(A=Ahat, B=Bhat, Delta_minus=float(dmn), Delta_plus=float(dpl),
                tail_index=list(range(i0, n)))
