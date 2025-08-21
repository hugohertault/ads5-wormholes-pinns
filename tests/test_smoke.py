import os
from ads5pinns import TrainConfig, CaseConfig, Trainer

def test_train_fast(tmp_path, monkeypatch):
    monkeypatch.setenv('FAST_SMOKE','1')
    tdir = tmp_path / 'runs'; tdir.mkdir()
    tr = Trainer(out_dir=str(tdir))
    cfg = TrainConfig(n_points=64, max_epochs=200, patience=80, verbose=False)
    case = CaseConfig(1.0, -2.5, 0.05, 'dirichlet', 0.0)
    res = tr.train_case(case, cfg)
    assert res['success']
    assert res['final_loss'] < 0.5
    assert len(res['r']) == len(res['phi'])
