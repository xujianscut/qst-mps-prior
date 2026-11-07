"""Per-state compute of MLE vs MAP under the SAME batched implementation (batch of 200 states, one GPU),
min over 5 repeats; also single-state (streaming) cost. Uses the validation-chosen T of each rescue run."""
import json, time, torch
from common import Sys, CondDrop, onehot, fidelity, sync
import os
OUT = os.environ.get('QST_OUT', 'results')  # where result JSONs and checkpoints are written
dev = torch.device('cuda:0'); R = OUT + '/'; out = {}
for n in [6, 8, 10]:
    S = Sys(n, dev, shots=200); DES = S.design('local'); TAB = S.pauli_tables(DES); OHOT = onehot(DES, dev)
    sel = json.load(open(R + f'rescue_n{n}_local_S200.json'))['sel']; lam = float(sel['map'][0][3:]); Tm = sel['map'][1]; Tl = sel['mle'][1]
    net = CondDrop(4 * n, S.THETA).to(dev); net.load_state_dict(torch.load(R + f'ckpt_n{n}_local_S200_s30000.pt', map_location=dev)); net.eval()
    psi = S.shallow(200, seed=999); o = S.measure(psi, TAB, 'indep', torch.Generator(device=dev).manual_seed(4242), None)
    def fit(o, init, iters, lam=0.0, ref=None):
        th = init.clone().requires_grad_(True); opt = torch.optim.Adam([th], lr=0.05)
        for _ in range(iters):
            st = S.th_to_state(th); loss = ((S.expect(st, TAB) - o) ** 2).mean(1)
            if lam > 0: loss = loss + lam * (1 - fidelity(st, ref))
            opt.zero_grad(); loss.mean().backward(); opt.step()
        return th.detach()
    def t_mle(o): return fit(o, 0.1 * torch.randn(o.shape[0], S.THETA, device=dev), Tl)
    def t_net(o):
        with torch.no_grad(): return net(OHOT.unsqueeze(0).expand(o.shape[0], -1, -1), o)
    def t_map(o):
        th0 = t_net(o)
        with torch.no_grad(): ref = S.th_to_state(th0)
        return fit(o, th0, Tm, lam, ref)
    res = {}
    for B in [200, 1]:
        ob = o[:B]
        for name, f in [('net', t_net), ('mle', t_mle), ('map', t_map)]:
            f(ob); ts = []
            for _ in range(5):
                sync(dev); t0 = time.time(); f(ob); sync(dev); ts.append(time.time() - t0)
            res[f'{name}_B{B}_ms_per_state'] = min(ts) / B * 1e3
    res.update(T_mle=Tl, T_map=Tm, lam=lam); out[n] = res
    print(n, {k: round(v, 3) if isinstance(v, float) else v for k, v in res.items()}, flush=True)
json.dump(out, open(R + 'cost.json', 'w'), indent=1)
