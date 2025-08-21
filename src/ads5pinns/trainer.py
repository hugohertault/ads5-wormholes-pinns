import os, time, json, math, torch, numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .config import TrainConfig, CaseConfig
from .models import MLP
from .physics import make_grid, bf_is_valid, kg_residual, delta_from_m2
from .losses import kg_total_loss
from .asymptotics import asymptotic_fit

class Trainer:
    def __init__(self, out_dir: str = 'runs'):
        self.out_dir = out_dir; os.makedirs(out_dir, exist_ok=True)
    def train_case(self, case: CaseConfig, cfg: TrainConfig):
        if not bf_is_valid(case.m2_L2): return None
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        r = make_grid(cfg.r_min, cfg.r_max, cfg.n_points, device)
        model = MLP(1,32,3,1).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
        sch = torch.optim.lr_scheduler.StepLR(opt, step_size=cfg.step_size, gamma=cfg.gamma)
        best = math.inf; bad=0; hist=[]; t0=time.time()
        for e in range(cfg.max_epochs):
            opt.zero_grad()
            loss, parts = kg_total_loss(model, r, case.m2_L2, case.lambda_L2, case.phi0,
                                        bc_type=case.bc, f_double_trace=case.f_double_trace)
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
            opt.step(); sch.step()
            L=float(loss.item()); hist.append(L)
            if L<best-1e-9: best=L; bad=0
            else: bad+=1
            if L<cfg.target_loss or (L<cfg.good_enough_loss and e>400) or bad>cfg.patience: break
        if cfg.use_lbfgs:
            def closure():
                optlb.zero_grad(); l,_=kg_total_loss(model,r,case.m2_L2,case.lambda_L2,case.phi0,
                                                  bc_type=case.bc,f_double_trace=case.f_double_trace)
                l.backward(); return l
            optlb=torch.optim.LBFGS(model.parameters(), max_iter=cfg.lbfgs_steps,
                                    tolerance_grad=1e-9, tolerance_change=1e-9)
            lb=optlb.step(closure); hist.append(float(lb.item()))
        dur=time.time()-t0; model.eval()
        with torch.no_grad():
            rr=r.detach().cpu().squeeze().numpy()
            phi=model(r).detach().cpu().squeeze().numpy()
        res,_,_=kg_residual(model,r,case.m2_L2,case.lambda_L2,L=1.0)
        res_np=res.detach().cpu().squeeze().abs().numpy()
        try: fit=asymptotic_fit(rr,phi,case.m2_L2,L=1.0,n_tail=20)
        except Exception:
            dmn,dpl=delta_from_m2(case.m2_L2); fit={'A':float(phi[-1] if len(phi)>0 else 0.0),'B':0.0,
                'Delta_minus':float(dmn),'Delta_plus':float(dpl),'tail_index':list(range(max(0,len(rr)-20),len(rr)))}
        success=(best<cfg.success_loss)
        out=os.path.join(self.out_dir,'case_'+time.strftime('%Y%m%d-%H%M%S')); os.makedirs(out,exist_ok=True)
        with open(os.path.join(out,'metrics.json'),'w') as f:
            json.dump({'final_loss':best,'history':hist,'duration_sec':dur,'success':success,'fit':fit},f,indent=2)
        fig=plt.figure(figsize=(6,4)); plt.plot(rr,phi); plt.xlabel('r/L'); plt.ylabel('phi(r)'); plt.title('Scalar field'); plt.grid(True,alpha=0.3); fig.tight_layout(); fig.savefig(os.path.join(out,'phi.png'),dpi=200); plt.close(fig)
        fig2=plt.figure(figsize=(6,4)); plt.semilogy(hist); plt.xlabel('epoch'); plt.ylabel('loss'); plt.title('Training loss'); plt.grid(True,alpha=0.3); fig2.tight_layout(); fig2.savefig(os.path.join(out,'loss.png'),dpi=200); plt.close(fig2)
        fig3=plt.figure(figsize=(6,4)); plt.semilogy(rr, np.maximum(res_np,1e-16)); plt.xlabel('r/L'); plt.ylabel('|KG residual|'); plt.title('PDE residual'); plt.grid(True,alpha=0.3); fig3.tight_layout(); fig3.savefig(os.path.join(out,'residual.png'),dpi=200); plt.close(fig3)
        dmn,dpl=delta_from_m2(case.m2_L2); A,B=fit['A'],fit['B']; phi_fit=A*np.exp(-dmn*rr)+B*np.exp(-dpl*rr)
        fig4=plt.figure(figsize=(6,4)); plt.plot(rr,phi,label='PINN'); plt.plot(rr,phi_fit,'--',label='asymptotic fit'); plt.xlabel('r/L'); plt.ylabel('phi(r)'); plt.title('Asymptotic fit'); plt.grid(True,alpha=0.3); plt.legend(); fig4.tight_layout(); fig4.savefig(os.path.join(out,'asymptotic_fit.png'),dpi=200); plt.close(fig4)
        return {'r':rr,'phi':phi,'history':hist,'final_loss':best,'success':success,'out_dir':out,'fit':fit}
