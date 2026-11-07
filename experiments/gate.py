"""Residual-gated MAP: use the learned prior only when the network's estimate is consistent with the data.
Gate statistic r(o) = mean_j (<P_j>_net(o) - o_j)^2 (computable from the record alone). Threshold tau = q-quantile of r
on the in-distribution VALIDATION set (seed 555/5555). States with r > tau get prior-free MLE instead of MAP.
lam, T for MAP and T for MLE are the validation choices stored in rescue_{tag}.json. Fixed-scale per-observable conformal:
Q for the MAP branch from validation MAP residuals, Q for the MLE branch from validation MLE residuals (all val states).
Evaluated in distribution (test seed 999/4242) and under the same shifts as map_refine (seed 31337, noise 99)."""
import argparse, json, numpy as np, torch
from common import Sys, CondDrop, onehot, fidelity, conf_Q
import os
OUT = os.environ.get('QST_OUT', 'results')  # where result JSONs and checkpoints are written
ap = argparse.ArgumentParser(); ap.add_argument('--n', type=int, default=6); ap.add_argument('--shots', type=int, default=200)
ap.add_argument('--q', type=float, default=0.9); ap.add_argument('--gate', default='chi2', choices=['raw', 'chi2'])
ap.add_argument('--family', default='shallow', choices=['shallow', 'tfim']); args = ap.parse_args()
FS = '' if args.family == 'shallow' else '_' + args.family
dev = torch.device('cuda:0'); torch.manual_seed(0); n = args.n; R = OUT + '/'
S = Sys(n, dev, shots=args.shots); DES = S.design('local'); TAB = S.pauli_tables(DES); OHOT = onehot(DES, dev)
THt = S.pauli_tables(S.obs_held())
tag = f'rescue_n{n}_local_S{args.shots}{FS}'; sel = json.load(open(R + tag + '.json'))['sel']
lam = float(sel['map'][0][3:]); Tm = sel['map'][1]; Tl = sel['mle'][1]
net = CondDrop(4 * n, S.THETA).to(dev); net.load_state_dict(torch.load(R + f'ckpt_n{n}_local_S{args.shots}_s30000{FS}.pt', map_location=dev)); net.eval()
def make_set(seed, nseed, spread=None, shots=None):
    psi = S.family(args.family, 200, seed=seed, spread=spread)
    return psi, S.measure(psi, TAB, 'indep', torch.Generator(device=dev).manual_seed(nseed), None, shots=shots)
def fit(o, init, iters, lam=0.0, ref=None):
    th = init.clone().requires_grad_(True); opt = torch.optim.Adam([th], lr=0.05)
    for _ in range(iters):
        st = S.th_to_state(th); loss = ((S.expect(st, TAB) - o) ** 2).mean(1)
        if lam > 0: loss = loss + lam * (1 - fidelity(st, ref))
        opt.zero_grad(); loss.mean().backward(); opt.step()
    return th.detach()
def run(o, shots):
    with torch.no_grad(): thn = net(OHOT.unsqueeze(0).expand(o.shape[0], -1, -1), o); stn = S.th_to_state(thn)
    r = ((S.expect(stn, TAB) - o) ** 2)
    if args.gate == 'chi2': r = r / ((1 - o ** 2).clamp_min(1.0 / shots) / shots)  # chi^2 per observable, known shot count
    r = r.mean(1)
    th_map = fit(o, thn, Tm, lam, stn.detach())
    th_mle = fit(o, 0.1 * torch.randn(o.shape[0], S.THETA, device=dev, generator=torch.Generator(device=dev).manual_seed(0)), Tl)
    return dict(r=r, net=thn, map=th_map, mle=th_mle)
psv, ov = make_set(555, 5555); V = run(ov, args.shots); tau = V['r'].quantile(args.q)
oHv = S.expect(psv, THt)
P = {k: S.expect(S.th_to_state(V[k]), THt).detach() for k in ['map', 'mle']}
Q = {k: conf_Q((P[k] - oHv).abs(), 0.9) for k in P}
print(f"[cfg] {tag} lam={lam} T_map={Tm} T_mle={Tl} gate {args.gate} q={args.q} tau={tau.item():.2e}", flush=True)
rows = []
for kind, val in [('test', 0), ('spread', 0.9), ('spread', 1.2), ('shots', 50), ('shots', 20)]:
    if kind == 'test': psx, ox = make_set(999, 4242)
    else: psx, ox = make_set(31337, 99, spread=val if kind == 'spread' else None, shots=val if kind == 'shots' else None)
    X = run(ox, val if kind == 'shots' else args.shots); g = X['r'] > tau; oHx = S.expect(psx, THt); row = dict(kind=kind, val=val, gated_frac=g.float().mean().item())
    thg = torch.where(g.unsqueeze(1), X['mle'], X['map']); X['gated'] = thg
    for k in ['net', 'mle', 'map', 'gated']:
        st = S.th_to_state(X[k]); row[k + '_fid'] = fidelity(psx, st).mean().item()
    for k in ['mle', 'map', 'gated']:
        p = S.expect(S.th_to_state(X[k]), THt).detach()
        q = Q['mle'] if k == 'mle' else (Q['map'] if k == 'map' else torch.where(g.unsqueeze(1), Q['mle'], Q['map']))
        row[k + '_cov'] = ((p - oHx).abs() <= q).float().mean().item()
        row[k + '_wid'] = ((p + q).clamp(max=1) - (p - q).clamp(min=-1)).mean().item()
    fg, fm = fidelity(psx, S.th_to_state(thg)), fidelity(psx, S.th_to_state(X['mle']))
    row['gated_minus_mle'] = (fg - fm).mean().item(); row['gated_minus_mle_se'] = ((fg - fm).std() / np.sqrt(len(fg))).item()
    rows.append(row)
    print(f"[{kind}={val}] gated {row['gated_frac']*100:.0f}% | fid net {row['net_fid']:.3f} mle {row['mle_fid']:.3f} map {row['map_fid']:.3f} "
          f"gated {row['gated_fid']:.3f} (gated-mle {row['gated_minus_mle']:+.4f}+-{row['gated_minus_mle_se']:.4f}) | cov/wid mle {row['mle_cov']*100:.1f}/{row['mle_wid']:.3f} "
          f"map {row['map_cov']*100:.1f}/{row['map_wid']:.3f} gated {row['gated_cov']*100:.1f}/{row['gated_wid']:.3f}", flush=True)
json.dump(dict(tag=tag, q=args.q, tau=tau.item(), rows=rows), open(R + f'gate_{args.gate}_n{n}_S{args.shots}_q{args.q}{FS}.json', 'w'), indent=1)
