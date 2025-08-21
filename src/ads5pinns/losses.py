import torch
from .physics import kg_residual
from .asymptotics import asymptotic_fit

def kg_total_loss(model, r, m2_L2, lambda_L2, phi0,
                  w_pde=1.0, w_bc=10.0, w_reg=5.0, w_decay=0.1, L=1.0,
                  bc_type: str = "dirichlet", w_bc_mixed: float = 10.0, f_double_trace: float = 0.0):
    res, phi, r_t = kg_residual(model, r, m2_L2, lambda_L2, L=L)
    pde = torch.mean(res**2)
    bc_throat = (phi[0] - phi0).pow(2)
    dr0 = (r_t[1] - r_t[0])
    reg = ((phi[1] - phi[0]) / dr0).pow(2)
    decay = torch.mean(phi[-10:]**2)
    total = w_pde*pde + w_bc*bc_throat + w_reg*reg + w_decay*decay
    parts = dict(pde=float(pde.item()), bc=float(bc_throat.item()),
                 reg=float(reg.item()), decay=float(decay.item()))
    if bc_type == 'mixed':
        with torch.no_grad():
            rr = r_t.detach().cpu().squeeze().numpy()
            pp = phi.detach().cpu().squeeze().numpy()
            fit = asymptotic_fit(rr, pp, m2_L2, L=L, n_tail=20)
            A, B = fit['A'], fit['B']
            val = (B - f_double_trace*A)**2
        bc_mixed = torch.as_tensor(val, dtype=phi.dtype, device=phi.device)
        total = total + w_bc_mixed*bc_mixed
        parts.update(mixed=float(bc_mixed.item()))
    return total, parts
