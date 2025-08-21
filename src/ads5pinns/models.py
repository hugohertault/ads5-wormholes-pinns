import torch, torch.nn as nn

class MLP(nn.Module):
    def __init__(self, in_dim=1, hidden=32, depth=3, out_dim=1, act=nn.Tanh):
        super().__init__()
        layers=[]; d=in_dim
        for _ in range(depth):
            layers += [nn.Linear(d, hidden), act()]; d = hidden
        layers += [nn.Linear(d, out_dim)]
        self.net = nn.Sequential(*layers)
        self.apply(self._init)
    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight, gain=0.1)
            nn.init.zeros_(m.bias)
    def forward(self, x): return self.net(x)
