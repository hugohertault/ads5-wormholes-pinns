import torch, math
from dataclasses import dataclass

def delta_from_m2(m2_L2: float):
    root = math.sqrt(max(0.0, 4.0 + m2_L2))
    return 2 - root, 2 + root

@dataclass
class KGBackground:
    L: float = 1.0
    def metric(self, r: torch.Tensor):
        a = 0.95*torch.sinh(r/self.L)+0.1
        b = 1.05*torch.sinh(r/self.L)+0.1
        return a,b
    def metric_derivatives(self, r: torch.Tensor):
        a_r = 0.95*torch.cosh(r/self.L)/self.L
        b_r = 1.05*torch.cosh(r/self.L)/self.L
        return a_r, b_r

def bf_is_valid(m2_L2: float) -> bool:
    return (m2_L2 >= -4.0)

def _central_diff_first(y: torch.Tensor, h: torch.Tensor):
    dy = torch.zeros_like(y)
    dy[1:-1] = (y[2:] - y[:-2])/(2*h)
    dy[0]  = (y[1] - y[0])/h
    dy[-1] = (y[-1] - y[-2])/h
    return dy

def _central_diff_second(y: torch.Tensor, h: torch.Tensor):
    d2y = torch.zeros_like(y)
    d2y[1:-1] = (y[2:] - 2*y[1:-1] + y[:-2])/(h*h)
    d2y[0]  = (y[2] - 2*y[1] + y[0])/(h*h)
    d2y[-1] = (y[-1] - 2*y[-2] + y[-3])/(h*h)
    return d2y

def kg_residual(model, r: torch.Tensor, m2_L2: float, lambda_L2: float, L: float = 1.0):
    phi = model(r)
    h = r[1] - r[0]
    phi_r  = _central_diff_first(phi,  h)
    phi_rr = _central_diff_second(phi, h)
    bg = KGBackground(L=L)
    a,b = bg.metric(r)
    a_r,b_r = bg.metric_derivatives(r)
    dV_dphi = (m2_L2/(L**2))*phi + (lambda_L2/(L**4))*phi**3
    residual = phi_rr + (a_r/a + 3*b_r/b)*phi_r - dV_dphi
    return residual, phi, r

def make_grid(r_min: float, r_max: float, n_points: int, device: str):
    return torch.linspace(r_min, r_max, n_points, device=device).unsqueeze(1)
