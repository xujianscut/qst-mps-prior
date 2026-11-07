"""Amortized conditional estimator, instrumented.
Trains the paper's DeepSets+dropout estimator (shallow family) for --steps (cosine schedule, as in the paper),
timing training separately from evaluation. Reports on a fixed held-out test set (seed 999, ntest states,
measurement-noise seed 4242, identical to mle_baselines.py):
  point fidelity, shuffled control, prior-only (mean state), point and E-pass ensemble inference time,
  per-observable conformal coverage/width for local Z/X and the paper's held-out observables, held-out MAE,
  and coverage under distribution shift (Q_O calibrated in distribution, applied to shifted states).
"""
import argparse, json, time, numpy as np, torch
from common import Sys, CondDrop, onehot, fidelity, perobs_conformal, conf_Q, sync
import os
OUT = os.environ.get('QST_OUT', 'results')  # where result JSONs and checkpoints are written

ap = argparse.ArgumentParser()
ap.add_argument('--dev', default='cuda:0'); ap.add_argument('--n', type=int, default=6)
ap.add_argument('--steps', type=int, default=30000); ap.add_argument('--design', default='local', choices=['local1', 'local', 'local3'])
ap.add_argument('--meas', default='indep', choices=['indep', 'basis']); ap.add_argument('--shots', type=int, default=200)
ap.add_argument('--ntest', type=int, default=200); ap.add_argument('--ntrain', type=int, default=20000)
ap.add_argument('--E', type=int, default=24); ap.add_argument('--drop', type=float, default=0.1)
ap.add_argument('--shift', type=int, default=1); ap.add_argument('--out', default='')
args = ap.parse_args()
dev = torch.device(args.dev); torch.manual_seed(0); np.random.seed(0)
S = Sys(args.n, dev, shots=args.shots); n = args.n
DES = S.design(args.design); TAB = S.pauli_tables(DES); K = len(DES)
OL = S.obs_local(); OH_ = S.obs_held(); TL = S.pauli_tables(OL); TH = S.pauli_tables(OH_)
if n <= 6: S.dense_check(DES[:20] + OH_[:6])
plan = S.setup_basis(DES) if args.meas == 'basis' else None
OHOT = onehot(DES, dev)
tag = args.out or f"cond_n{n}_{args.design}_{args.meas}_s{args.steps}"
print(f"[cfg] {tag}: K={K} obs, THETA={S.THETA}, m_settings={9 if args.meas=='basis' else 'n/a'}", flush=True)

net = CondDrop(4 * n, S.THETA, p=args.drop).to(dev)
opt = torch.optim.AdamW(net.parameters(), lr=3e-4); sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.steps)
TR = S.shallow(args.ntrain, seed=1)
gtr = torch.Generator(device=dev).manual_seed(7)
sync(dev); t0 = time.time()
for it in range(1, args.steps + 1):
    bidx = torch.randint(0, TR.shape[0], (256,), device=dev, generator=gtr); psb = TR[bidx]
    with torch.no_grad(): obs = S.measure(psb, TAB, args.meas, gtr, plan)
    pred = net(OHOT.unsqueeze(0).expand(256, -1, -1), obs)
    loss = (1 - fidelity(S.th_to_state(pred), psb)).mean()
    opt.zero_grad(); loss.backward(); opt.step(); sch.step()
    if it % 5000 == 0: print(f"   it{it} loss {loss.item():.4f} ({time.time()-t0:.0f}s)", flush=True)
sync(dev); t_train = time.time() - t0
print(f"[TIMING] train {args.steps} steps: {t_train:.1f}s", flush=True)

# ---------------- fixed test set (shared with mle_baselines.py) ----------------
NT = args.ntest
psi = S.shallow(NT, seed=999)
gte = torch.Generator(device=dev).manual_seed(4242)
obs = S.measure(psi, TAB, args.meas, gte, plan)
oh = OHOT.unsqueeze(0).expand(NT, -1, -1)
mean_state = TR.mean(0, keepdim=True); mean_state = mean_state / mean_state.norm()
f_prior = fidelity(psi, mean_state.expand(NT, -1)).mean().item()
net.eval()
with torch.no_grad():
    pt = S.cores_to_state(S.canon(S.split_cores(net(oh, obs))))
    perm = torch.randperm(NT, generator=torch.Generator(device=dev).manual_seed(5), device=dev)
    sh = S.cores_to_state(S.canon(S.split_cores(net(oh, obs[perm]))))
fpt = fidelity(psi, pt); f_pt = fpt.mean().item(); f_sh = fidelity(psi, sh).mean().item()
print(f"[point] fid {f_pt:.4f}+-{fpt.std().item():.4f} | shuffled {f_sh:.4f} | prior-only {f_prior:.4f}", flush=True)

# timing (point / ensemble), 20 repeats
with torch.no_grad():
    for _ in range(3): S.cores_to_state(S.canon(S.split_cores(net(oh, obs))))
    sync(dev); t0 = time.time()
    for _ in range(20): S.cores_to_state(S.canon(S.split_cores(net(oh, obs))))
    sync(dev); t_pt = (time.time() - t0) / 20 / NT
    net.train()
    for _ in range(2): [S.cores_to_state(S.canon(S.split_cores(net(oh, obs)))) for _ in range(args.E)]
    sync(dev); t0 = time.time()
    for _ in range(5):
        st = [S.cores_to_state(S.canon(S.split_cores(net(oh, obs)))) for _ in range(args.E)]
        eL = [S.expect(x, TL) for x in st]; eH = [S.expect(x, TH) for x in st]
    sync(dev); t_ens = (time.time() - t0) / 5 / NT
print(f"[TIMING] point {t_pt*1e3:.4f} ms/state | {args.E}-pass ensemble+observables {t_ens*1e3:.4f} ms/state", flush=True)


def ens_stats(psi_in, obs_in):
    net.train()
    with torch.no_grad():
        st = [S.cores_to_state(S.canon(S.split_cores(net(OHOT.unsqueeze(0).expand(obs_in.shape[0], -1, -1), obs_in)))) for _ in range(args.E)]
        sL = torch.stack([S.expect(x, TL) for x in st]); sH = torch.stack([S.expect(x, TH) for x in st])
    net.eval()
    with torch.no_grad():
        p = S.cores_to_state(S.canon(S.split_cores(net(OHOT.unsqueeze(0).expand(obs_in.shape[0], -1, -1), obs_in))))
    return (sL.mean(0), sL.std(0).clamp_min(1e-4), sH.mean(0), sH.std(0).clamp_min(1e-4), fidelity(psi_in, p))


mL, sdL, mH, sdH, _ = ens_stats(psi, obs)
oL = S.expect(psi, TL); oHt = S.expect(psi, TH)
covL = perobs_conformal(oL, mL, sdL); covH = perobs_conformal(oHt, mH, sdH)
maeH = (mH - oHt).abs().mean().item(); maeL = (mL - oL).abs().mean().item()
maeH_prior = (S.expect(mean_state, TH) - oHt).abs().mean().item()
print(f"[UQ] local Z/X: cov {covL[0]*100:.1f}+-{covL[1]*100:.1f}% width {covL[2]:.3f} MAE {maeL:.4f}", flush=True)
print(f"[UQ] held-out ({len(OH_)}): cov {covH[0]*100:.1f}+-{covH[1]*100:.1f}% width {covH[2]:.3f} MAE {maeH:.4f} (prior-only {maeH_prior:.4f})", flush=True)

res = dict(tag=tag, n=n, design=args.design, meas=args.meas, steps=args.steps, K=K, ntest=NT,
           t_train_s=t_train, t_point_ms=t_pt * 1e3, t_ens_ms=t_ens * 1e3,
           fid=f_pt, fid_sd=fpt.std().item(), fid_q10=fpt.quantile(0.1).item(), fid_shuf=f_sh, fid_prior=f_prior,
           covL=covL, covH=covH, maeL=maeL, maeH=maeH, maeH_prior=maeH_prior, nheld=len(OH_))

# ---------------- distribution shift: calibrate Q_O in distribution, apply to shifted test ----------------
if args.shift:
    N2 = NT // 2
    rL = (oL[:N2] - mL[:N2]).abs() / sdL[:N2]; rH = (oHt[:N2] - mH[:N2]).abs() / sdH[:N2]
    QL = conf_Q(rL, 0.9); QH = conf_Q(rH, 0.9)
    shifts = []
    for kind, val in [('spread', 0.6), ('spread', 0.3), ('spread', 0.9), ('spread', 1.2), ('shots', 50), ('shots', 800)]:
        sp = val if kind == 'spread' else 0.6
        shots = val if kind == 'shots' else args.shots
        ps = S.shallow(N2, seed=31337, spread=sp)
        g2 = torch.Generator(device=dev).manual_seed(99)
        ob = S.measure(ps, TAB, args.meas, g2, plan, shots=shots)
        a, b_, c, d, f = ens_stats(ps, ob)
        oLs = S.expect(ps, TL); oHs = S.expect(ps, TH)
        cl = ((oLs - a).abs() <= QL * b_).float().mean().item(); ch = ((oHs - c).abs() <= QH * d).float().mean().item()
        wl = ((a + QL * b_).clamp(max=1) - (a - QL * b_).clamp(min=-1)).clamp_min(0).mean().item()
        wh = ((c + QH * d).clamp(max=1) - (c - QH * d).clamp(min=-1)).clamp_min(0).mean().item()
        # does the ensemble spread track the error? rank correlation per state
        errs = (oHs - c).abs().mean(1); spr = d.mean(1)
        rk = lambda x: x.argsort().argsort().float()
        rho = float(np.corrcoef(rk(errs).cpu().numpy(), rk(spr).cpu().numpy())[0, 1])
        shifts.append(dict(kind=kind, val=val, fid=f.mean().item(), covL=cl, covH=ch, widL=wl, widH=wh,
                           maeH=(c - oHs).abs().mean().item(), rho_err_spread=rho))
        print(f"[SHIFT] {kind}={val}: fid {f.mean().item():.3f} covL {cl*100:.1f}% covH {ch*100:.1f}% "
              f"widL {wl:.3f} widH {wh:.3f} maeH {(c-oHs).abs().mean().item():.3f} rho(err,spread) {rho:.2f}", flush=True)
    res['shift'] = shifts

json.dump(res, open(os.path.join(OUT, f"{tag}.json"), 'w'), indent=1)
print("[dumped]", flush=True)
