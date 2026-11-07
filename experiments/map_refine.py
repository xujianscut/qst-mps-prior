"""MAP-refinement study, all hyperparameters chosen on a VALIDATION set (seed 555), reported on the TEST set
(seed 999, noise seed 4242 at S=200, identical to amortized/mle_baselines). Compares, on the same records:
  net      : amortized estimator (point)
  mle      : prior-free MPS-MLE, 1 restart from random init (T chosen on val; also R=3 residual-selected)
  warm     : MLE warm-started from the net's output (T chosen on val)
  map      : MLE + lam*(1 - F(psi_theta, psi_net)) learned-prior regularizer, warm-started (lam,T chosen on val)
Also: held-out-observable MAE and fixed-scale conformal intervals for every estimator, coverage under shift
(Q calibrated on in-distribution val, applied to shifted test), and single-state (streaming) latency, GPU and CPU.
"""
import argparse, json, time, os, numpy as np, torch
from common import Sys, CondDrop, onehot, fidelity, conf_Q, sync
import os
OUT = os.environ.get('QST_OUT', 'results')  # where result JSONs and checkpoints are written

ap = argparse.ArgumentParser()
ap.add_argument('--dev', default='cuda:0'); ap.add_argument('--n', type=int, default=6)
ap.add_argument('--design', default='local', choices=['local1', 'local', 'local3'])
ap.add_argument('--shots', type=int, default=200, help='shots per observable, train AND test')
ap.add_argument('--steps', type=int, default=30000); ap.add_argument('--ntest', type=int, default=200)
ap.add_argument('--E', type=int, default=24); ap.add_argument('--lr', type=float, default=0.05)
ap.add_argument('--latency', type=int, default=1)
ap.add_argument('--meas', default='indep', choices=['indep', 'basis']); ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--family', default='shallow', choices=['shallow', 'tfim'])
args = ap.parse_args()
dev = torch.device(args.dev); torch.manual_seed(args.seed); np.random.seed(args.seed)
S = Sys(args.n, dev, shots=args.shots); n = args.n
DES = S.design(args.design); TAB = S.pauli_tables(DES); K = len(DES)
OL = S.obs_local(); OH_ = S.obs_held(); TL = S.pauli_tables(OL); THt = S.pauli_tables(OH_)
OHOT = onehot(DES, dev)
plan = S.setup_basis(DES) if args.meas == 'basis' else None
sfx = ('' if args.meas == 'indep' else '_basis') + ('' if args.seed == 0 else f'_seed{args.seed}') + ('' if args.family == 'shallow' else '_' + args.family)
tag = f"rescue_n{n}_{args.design}_S{args.shots}{sfx}"
R = OUT + '/'
print(f"[cfg] {tag} K={K}", flush=True)

# ---------------- network (train once, cache) ----------------
net = CondDrop(4 * n, S.THETA).to(dev)
ck = R + f"ckpt_n{n}_{args.design}_S{args.shots}_s{args.steps}{sfx}.pt"
TR = S.family(args.family, 20000, seed=1)
if os.path.exists(ck):
    net.load_state_dict(torch.load(ck, map_location=dev)); t_train = json.load(open(ck + '.json'))['t_train_s']
else:
    opt = torch.optim.AdamW(net.parameters(), lr=3e-4); sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.steps)
    gtr = torch.Generator(device=dev).manual_seed(7 + 1000 * args.seed); sync(dev); t0 = time.time()
    for it in range(1, args.steps + 1):
        bidx = torch.randint(0, TR.shape[0], (256,), device=dev, generator=gtr); psb = TR[bidx]
        with torch.no_grad(): obs = S.measure(psb, TAB, args.meas, gtr, plan)
        loss = (1 - fidelity(S.th_to_state(net(OHOT.unsqueeze(0).expand(256, -1, -1), obs)), psb)).mean()
        opt.zero_grad(); loss.backward(); opt.step(); sch.step()
    sync(dev); t_train = time.time() - t0
    torch.save(net.state_dict(), ck); json.dump({'t_train_s': t_train}, open(ck + '.json', 'w'))
print(f"[train] {t_train:.0f}s", flush=True)
net.eval()
MEAN = TR.mean(0, keepdim=True); MEAN = MEAN / MEAN.norm()


def make_set(seed, nseed, spread=None, shots=None, N=None):
    N = N or args.ntest
    psi = S.family(args.family, N, seed=seed, spread=spread)
    o = S.measure(psi, TAB, args.meas, torch.Generator(device=dev).manual_seed(nseed), plan, shots=shots)
    return psi, o


def net_pred(o):
    with torch.no_grad():
        th = net(OHOT.unsqueeze(0).expand(o.shape[0], -1, -1), o)
    return th


def fit(o, init, iters, lam=0.0, ref=None, ckp=(), psi=None):
    th = init.clone().requires_grad_(True); opt = torch.optim.Adam([th], lr=args.lr); rec = {}
    if 0 in ckp:
        with torch.no_grad(): rec[0] = th.detach().clone()
    for it in range(1, iters + 1):
        st = S.th_to_state(th); loss = ((S.expect(st, TAB) - o) ** 2).mean(1)
        if lam > 0: loss = loss + lam * (1 - fidelity(st, ref))
        loss = loss.mean(); opt.zero_grad(); loss.backward(); opt.step()
        if it in ckp: rec[it] = th.detach().clone()
    return th.detach(), rec


TS = [0, 10, 25, 50, 100, 200, 500]
LAMS = [0.001, 0.003, 0.01, 0.03, 0.1]


def evaluate(psi, o):
    """Returns dict name -> {T or (lam,T): thetas} for all estimator variants on this record."""
    out = {}
    thn = net_pred(o); out['net'] = {0: thn}; psin = S.th_to_state(thn).detach()
    init = 0.1 * torch.randn(o.shape[0], S.THETA, device=dev, generator=torch.Generator(device=dev).manual_seed(0))
    _, out['mle'] = fit(o, init, max(TS), ckp=TS[1:])
    # R=3 residual-selected
    recs = [out['mle']] + [fit(o, 0.1 * torch.randn(o.shape[0], S.THETA, device=dev, generator=torch.Generator(device=dev).manual_seed(r)), max(TS), ckp=TS[1:])[1] for r in [1, 2]]
    out['mle3'] = {}
    for T in TS[1:]:
        ths = torch.stack([rc[T] for rc in recs])
        with torch.no_grad():
            res = torch.stack([((S.expect(S.th_to_state(t), TAB) - o) ** 2).mean(1) for t in ths])
        out['mle3'][T] = ths[res.argmin(0), torch.arange(o.shape[0], device=dev)]
    _, out['warm'] = fit(o, thn, max(TS), ckp=TS)
    for lam in LAMS:
        _, out[f'map{lam}'] = fit(o, thn, max(TS), lam=lam, ref=psin, ckp=TS)
    fam = MEAN.expand(o.shape[0], -1)
    for lam in LAMS:
        _, out[f'fam{lam}'] = fit(o, init, max(TS), lam=lam, ref=fam, ckp=TS[1:])
    return out


def fids(psi, d):
    return {k: fidelity(psi, S.th_to_state(v)).detach() for k, v in d.items()}


# ---------------- validation: choose T (and lam) per method ----------------
psv, ov = make_set(555, 5555)
EV = evaluate(psv, ov)
choice = {}
for name, d in EV.items():
    f = fids(psv, d); best = max(f, key=lambda k: f[k].mean().item()); choice[name] = (best, f[best].mean().item())
mapnames = [k for k in choice if k.startswith('map')]
famnames = [k for k in choice if k.startswith('fam')]
bestfam = max(famnames, key=lambda k: choice[k][1])
bestmap = max(mapnames, key=lambda k: choice[k][1])
print("[val choice] " + " | ".join(f"{k}: T={v[0]} fid {v[1]:.4f}" for k, v in choice.items()), flush=True)
sel = {'net': ('net', 0), 'mle': ('mle', choice['mle'][0]), 'mle3': ('mle3', choice['mle3'][0]),
       'warm': ('warm', choice['warm'][0]), 'map': (bestmap, choice[bestmap][0]), 'fam': (bestfam, choice[bestfam][0])}
print(f"[val choice] map -> {bestmap} T={choice[bestmap][0]}", flush=True)

# ---------------- test ----------------
pst, ot = make_set(999, 4242)
ET = evaluate(pst, ot)
res = dict(tag=tag, n=n, design=args.design, shots=args.shots, t_train_s=t_train, sel={k: [v[0], v[1]] for k, v in sel.items()})
oLt = S.expect(pst, TL); oHt = S.expect(pst, THt)
oLv = S.expect(psv, TL); oHv = S.expect(psv, THt)
test = {}
Qs = {}
for k, (nm, T) in sel.items():
    thv = EV[nm][T]; tht = ET[nm][T]
    stv = S.th_to_state(thv).detach(); stt = S.th_to_state(tht).detach()
    f = fidelity(pst, stt)
    pHv, pHt = S.expect(stv, THt), S.expect(stt, THt)
    QH = conf_Q((pHv - oHv).abs(), 0.9); Qs[k] = QH  # fixed-scale per-observable conformal, calibrated on val
    cov = ((pHt - oHt).abs() <= QH).float().mean().item(); wid = ((pHt + QH).clamp(max=1) - (pHt - QH).clamp(min=-1)).mean().item()
    test[k] = dict(fid=f.mean().item(), fid_sd=f.std().item(), q10=f.quantile(0.1).item(), maeH=(pHt - oHt).abs().mean().item(),
                   covH_fixed=cov, widH_fixed=wid)
    print(f"[TEST] {k:5s} ({nm},T={T}): fid {f.mean().item():.4f}+-{f.std().item():.4f} q10 {f.quantile(0.1).item():.3f} | heldMAE {(pHt-oHt).abs().mean().item():.4f} "
          f"| fixed-scale covH {cov*100:.1f}% wH {wid:.3f}", flush=True)
# full test curves (every T) for transparency
res['curves'] = {nm: {str(T): fidelity(pst, S.th_to_state(th)).mean().item() for T, th in d.items()} for nm, d in ET.items()}
# paired comparison warm/map vs mle
for a in ['warm', 'map', 'net', 'fam']:
    fa = fidelity(pst, S.th_to_state(ET[sel[a][0]][sel[a][1]])); fm = fidelity(pst, S.th_to_state(ET['mle'][sel['mle'][1]]))
    d = (fa - fm).detach(); test[a]['delta_vs_mle'] = d.mean().item(); test[a]['delta_se'] = (d.std() / np.sqrt(len(d))).item()
    test[a]['win_frac'] = (d > 0).float().mean().item()
    print(f"[PAIRED] {a} - mle: {d.mean().item():+.4f} +- {(d.std()/np.sqrt(len(d))).item():.4f} (SE), wins {((d>0).float().mean().item())*100:.0f}%", flush=True)
res['test'] = test
# integrity controls on the test set: prior-only (normalized training mean) and shuffled records
with torch.no_grad():
    f_prior = fidelity(pst, MEAN.expand(pst.shape[0], -1)).mean().item()
    perm = torch.randperm(pst.shape[0], generator=torch.Generator(device=dev).manual_seed(5), device=dev)
    f_shuf = fidelity(pst, S.th_to_state(net_pred(ot[perm]))).mean().item()
res['controls'] = dict(prior_only=f_prior, shuffled=f_shuf)
print(f"[CONTROLS] prior-only {f_prior:.4f} | shuffled {f_shuf:.4f}", flush=True)

# ---------------- ensemble intervals of the net (calibrated on val) ----------------
def ens(o):
    net.train()
    with torch.no_grad():
        s = torch.stack([S.expect(S.th_to_state(net(OHOT.unsqueeze(0).expand(o.shape[0], -1, -1), o)), THt) for _ in range(args.E)])
    net.eval(); return s.mean(0), s.std(0).clamp_min(1e-4)
mv, sv = ens(ov); QE = conf_Q((mv - oHv).abs() / sv, 0.9)

# ---------------- shift: adaptivity of intervals ----------------
shifts = []
for kind, val in [('none', 0), ('spread', 0.9), ('spread', 1.2), ('shots', 50), ('shots', 20)]:
    sp = val if kind == 'spread' else None; sh = val if kind == 'shots' else None
    if kind == 'shots' and val >= args.shots: continue
    psx, ox = make_set(31337, 99, spread=sp, shots=sh)
    oHx = S.expect(psx, THt); row = dict(kind=kind, val=val)
    m, s = ens(ox); row['ens_cov'] = ((m - oHx).abs() <= QE * s).float().mean().item()
    row['ens_wid'] = ((m + QE * s).clamp(max=1) - (m - QE * s).clamp(min=-1)).mean().item()
    row['ens_fid'] = fidelity(psx, S.th_to_state(net_pred(ox))).mean().item()
    for k in ['mle', 'map']:
        nm, T = sel[k]
        if nm == 'mle':
            th, _ = fit(ox, 0.1 * torch.randn(ox.shape[0], S.THETA, device=dev, generator=torch.Generator(device=dev).manual_seed(0)), T)
        else:
            lam = float(nm[3:]); thn = net_pred(ox); th, _ = fit(ox, thn, T, lam=lam, ref=S.th_to_state(thn).detach())
        p = S.expect(S.th_to_state(th), THt).detach()
        row[f'{k}_cov'] = ((p - oHx).abs() <= Qs[k]).float().mean().item()
        row[f'{k}_wid'] = ((p + Qs[k]).clamp(max=1) - (p - Qs[k]).clamp(min=-1)).mean().item()
        row[f'{k}_fid'] = fidelity(psx, S.th_to_state(th)).mean().item()
    shifts.append(row)
    print(f"[SHIFT] {kind}={val}: ens cov {row['ens_cov']*100:.1f}% w {row['ens_wid']:.3f} fid {row['ens_fid']:.3f} | "
          f"mle+fixed cov {row['mle_cov']*100:.1f}% w {row['mle_wid']:.3f} fid {row['mle_fid']:.3f} | "
          f"map+fixed cov {row['map_cov']*100:.1f}% w {row['map_wid']:.3f} fid {row['map_fid']:.3f}", flush=True)
res['shift'] = shifts

# ---------------- streaming latency (one state at a time) ----------------
if args.latency:
    lat = {}
    for devname in ['gpu', 'cpu']:
        d2 = dev if devname == 'gpu' else torch.device('cpu')
        if devname == 'cpu': torch.set_num_threads(4)
        S2 = Sys(n, d2, shots=args.shots); TAB2 = S2.pauli_tables(DES); OH2 = onehot(DES, d2)
        net2 = CondDrop(4 * n, S2.THETA).to(d2); net2.load_state_dict({k: v.to(d2) for k, v in net.state_dict().items()}); net2.eval()
        o_all = ot[:10].to(d2)
        def t_net(o1, E=1):
            with torch.no_grad():
                x = OH2.unsqueeze(0).expand(E, -1, -1); th = net2(x, o1.expand(E, -1)); S2.cores_to_state(S2.canon(S2.split_cores(th)))
        def t_fit(o1, T, init):
            th = init.clone().requires_grad_(True); op = torch.optim.Adam([th], lr=args.lr)
            for _ in range(T):
                l = ((S2.expect(S2.th_to_state(th), TAB2) - o1) ** 2).mean(); op.zero_grad(); l.backward(); op.step()
        def timeit(fn):
            fn(o_all[0:1]); sync(d2); t0 = time.time()
            for i in range(o_all.shape[0]): fn(o_all[i:i + 1])
            sync(d2); return (time.time() - t0) / o_all.shape[0] * 1e3
        Tm = sel['mle'][1]; Tw = sel['map'][1]
        lat[devname] = dict(net_point=timeit(lambda o1: t_net(o1)), net_ens=timeit(lambda o1: t_net(o1, args.E)),
                            mle=timeit(lambda o1: t_fit(o1, Tm, 0.1 * torch.randn(1, S2.THETA, device=d2))),
                            map=timeit(lambda o1: (t_net(o1), t_fit(o1, Tw, net2(OH2.unsqueeze(0), o1).detach()))),
                            T_mle=Tm, T_map=Tw)
        print(f"[LATENCY {devname}] per state, one at a time: net {lat[devname]['net_point']:.2f}ms | net+{args.E}ens {lat[devname]['net_ens']:.2f}ms | "
              f"MLE(T={Tm}) {lat[devname]['mle']:.1f}ms | MAP(T={Tw}) {lat[devname]['map']:.1f}ms", flush=True)
    res['latency'] = lat

json.dump(res, open(R + tag + '.json', 'w'), indent=1)
print('[dumped]', flush=True)
