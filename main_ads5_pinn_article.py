import os, json
from ads5pinns import TrainConfig, CaseConfig, Trainer

def run_single(fast=False):
    cfg = TrainConfig(n_points=80 if fast else 200, max_epochs=300 if fast else 1500, patience=120 if fast else 300, verbose=(not fast))
    case = CaseConfig(1.0, -2.5, 0.05, 'dirichlet', 0.0)
    tr = Trainer(out_dir='runs')
    res = tr.train_case(case, cfg)
    print(json.dumps({'final_loss': float(res['final_loss']), 'success': bool(res['success']), 'fit': res['fit'], 'out_dir': res['out_dir']}, indent=2))

def run_ensemble(fast=False):
    cfg = TrainConfig(n_points=80 if fast else 200, max_epochs=300 if fast else 1500, patience=120 if fast else 300, verbose=(not fast))
    cases = [
        CaseConfig(0.8, -3.0, 0.05, 'dirichlet', 0.0),
        CaseConfig(1.2, -2.5, 0.10, 'mixed', 0.3),
        CaseConfig(1.5, -2.0, 0.15, 'dirichlet', 0.0)
    ]
    tr = Trainer(out_dir='runs'); summary = {}
    for i,c in enumerate(cases,1):
        res = tr.train_case(c, cfg)
        summary['case_'+str(i)] = {'final_loss': float(res['final_loss']), 'success': bool(res['success']), 'fit': res['fit'], 'out_dir': res['out_dir']}
        print('Case', i, 'success=', res['success'], 'loss=', '{:.3e}'.format(res['final_loss']), 'out=', res['out_dir'])
    with open('runs/reproduce_summary.json','w') as f: json.dump(summary, f, indent=2)
    print('Ensemble summary saved.')

if __name__ == '__main__':
    FAST = os.getenv('FAST_SMOKE','0') == '1'
    print('Running article main | FAST_SMOKE =', FAST)
    run_single(fast=FAST); run_ensemble(fast=FAST)
