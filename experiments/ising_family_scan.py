import torch
from common import Sys, fidelity, cdt
dev = torch.device('cuda:0')
def gs(S, J, h, gz):
    n, D = S.n, S.D; z = (1 - 2 * S.bits).double(); zz = z[:, :-1] * z[:, 1:]; idx = torch.arange(D, device=dev)
    B = J.shape[0]; H = torch.zeros(B, D, D, dtype=torch.float64, device=dev); H[:, idx, idx] = -(J @ zz.T) - (gz @ z.T)
    for q in range(n): H[:, idx, idx ^ (1 << (n - 1 - q))] -= h[:, q:q + 1]
    ev, V = torch.linalg.eigh(H); v = V[:, :, 0]; gap = ev[:, 1] - ev[:, 0]
    return v.to(cdt), gap
for n in [6, 8]:
    S = Sys(n, dev); g = torch.Generator(device=dev).manual_seed(0); B = 1000
    U = lambda shape, a, b: a + (b - a) * torch.rand(*shape, generator=g, device=dev, dtype=torch.float64)
    for name, (Jr, hr, gr) in {'A J[.5,1.5] h[.5,1.5] g[-.5,.5]': ((.5, 1.5), (.5, 1.5), (-.5, .5)),
                                'B J[.5,1.5] h[.2,1.8] g[-1,1]': ((.5, 1.5), (.2, 1.8), (-1, 1)),
                                'C J[0,2] h[0,2] g[-1,1]': ((0, 2), (0, 2), (-1, 1)),
                                'D J[.5,1.5] h[.5,1.5] g[-.3,.3]': ((.5, 1.5), (.5, 1.5), (-.3, .3))}.items():
        v, gap = gs(S, U((B, n - 1), *Jr), U((B, n), *hr), U((B, n), *gr))
        m = v.mean(0, keepdim=True); m = m / m.norm()
        A = v[:500].reshape(500, 2 ** (n // 2), -1); sv = torch.linalg.svdvals(A) ** 2
        Sent = -(sv * torch.log2(sv.clamp_min(1e-30))).sum(1); chi2w = 1 - sv[:, :2].sum(1)
        print(f"n={n} {name}: prior-only {fidelity(v, m.expand_as(v)).mean():.3f} | pair {fidelity(v[:500], v[500:]).mean():.3f} | "
              f"S_mid {Sent.mean():.3f} | mid-cut weight beyond 2 Schmidt {chi2w.mean():.4f} | min gap {gap.min():.3f} q05 {gap.quantile(.05):.3f}", flush=True)
