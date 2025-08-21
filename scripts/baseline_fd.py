import numpy as np, json, os
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

def solve_fd(rmin=0.01, rmax=8.0, n=400, m2_L2=-2.5, phi0=1.0, L=1.0):
    r = np.linspace(rmin, rmax, n); dr = r[1]-r[0]
    a = 0.95*np.sinh(r/L)+0.1
    b = 1.05*np.sinh(r/L)+0.1
    ap = (0.95*np.cosh(r/L)/L) / a
    bp = (1.05*np.cosh(r/L)/L) / b
    c1 = ap + 3*bp
    A = np.zeros((n,n)); rhs = np.zeros(n)
    A[0,0]=1.0; rhs[0]=phi0
    A[1,0]=-1.0/dr; A[1,1]=1.0/dr
    for i in range(1,n-1):
        A[i, i-1] += 1.0/dr**2 - 0.5*c1[i]/dr
        A[i, i  ] += -2.0/dr**2 - (m2_L2/(L**2))
        A[i, i+1] += 1.0/dr**2 + 0.5*c1[i]/dr
    A[n-1, n-1] = 1.0; rhs[n-1] = 0.0
    phi = np.linalg.solve(A, rhs)
    return r, phi

if __name__ == '__main__':
    os.makedirs('runs_fd', exist_ok=True)
    r, phi = solve_fd()
    np.savez('runs_fd/fd_solution.npz', r=r, phi=phi)
    fig=plt.figure(figsize=(6,4)); plt.plot(r,phi); plt.xlabel('r/L'); plt.ylabel('phi'); plt.grid(True,alpha=0.3); fig.tight_layout(); fig.savefig('runs_fd/fd_phi.png',dpi=200); plt.close(fig)
    print(json.dumps({'ok': True, 'n': int(len(r))}))
