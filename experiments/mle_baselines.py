"""Classical baselines on the SAME test set / measurement record as amortized.py.
(1) per-state prior-free MPS-MLE (paper's procedure: random init 0.1, Adam lr 0.05, MSE to measured values),
    swept over restarts R and iterations T; restarts selected by TRAINING RESIDUAL (practical) and, for
    reference, by true fidelity (oracle, as in the paper's Table 2). Time per state = batch time / ntest.
(2) MLE + parametric-bootstrap uncertainty (B refits warm-started from the MLE), per-observable conformal,
    so the classical baseline has an interval too.
(3) direct estimates of the held-out observables from the same bitstrings (basis mode only).
(4) classical shadows (random single-qubit Pauli bases, 1 shot each) at matched copy budget m*S and 10x.
"""
import argparse, json, time, numpy as np, torch
from common import Sys, fidelity, perobs_conformal, sync
import os
OUT = os.environ.get('QST_OUT', 'results')  # where result JSONs and checkpoints are written

ap = argparse.ArgumentParser()
ap.add_argument('--dev', default='cuda:0'); ap.add_argument('--n', type=int, default=6)
ap.add_argument('--design', default='local', choices=['local1', 'local', 'local3'])
ap.add_argument('--meas', default='indep', choices=['indep', 'basis']); ap.add_argument('--shots', type=int, default=200)
ap.add_argument('--ntest', type=int, default=200); ap.add_argument('--R', type=int, default=5)
ap.add_argument('--T', type=int, default=2000); ap.add_argument('--lr', type=float, default=0.05)
ap.add_argument('--B', type=int, default=24); ap.add_argument('--Tb', type=int, default=200)
ap.add_argument('--shadows', type=int, default=1); ap.add_argument('--out', default='')
args = ap.parse_args()
dev = torch.device(args.dev); torch.manual_seed(0); np.random.seed(0)
S = Sys(args.n, dev, shots=args.shots); n = args.n
DES = S.design(args.design); TAB = S.pauli_tables(DES); K = len(DES)
OL = S.obs_local(); OH_ = S.obs_held(); TL = S.pauli_tables(OL); TH = S.pauli_tables(OH_)
plan = S.setup_basis(DES) if args.meas == 'basis' else None
tag = args.out or f"mle_n{n}_{args.design}_{args.meas}"
NT = args.ntest
psi = S.shallow(NT, seed=999)
gte = torch.Generator(device=dev).manual_seed(4242)
obs = S.measure(psi, TAB, args.meas, gte, plan)          # identical record to amortized.py
oL = S.expect(psi, TL); oHt = S.expect(psi, TH)
CKP = [50, 100, 200, 500, 1000, 2000, 4000]; CKP = [c for c in CKP if c <= args.T]
print(f"[cfg] {tag}: K={K}, ntest={NT}, R={args.R}, T={args.T}", flush=True)


def fit(o, init, iters, lr, ckp=None, rec=None):
    th = init.clone().requires_grad_(True); opt = torch.optim.Adam([th], lr=lr)
    tt = 0.0; sync(dev); t0 = time.time()
    for it in range(1, iters + 1):
        st = S.th_to_state(th); loss = ((S.expect(st, TAB) - o) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if ckp is not None and it in ckp:
            sync(dev); tt += time.time() - t0
            with torch.no_grad():
                stt = S.th_to_state(th); res = ((S.expect(stt, TAB) - o) ** 2).mean(1)
                rec[it] = dict(t=tt, fid=fidelity(psi, stt), res=res, th=th.detach().clone())
            sync(dev); t0 = time.time()
    sync(dev); tt += time.time() - t0
    return th.detach(), tt


# warmup
fit(obs[:8], 0.1 * torch.randn(8, S.THETA, device=dev), 10, args.lr)
recs = []
for r in range(args.R):
    init = 0.1 * torch.randn(NT, S.THETA, device=dev, generator=torch.Generator(device=dev).manual_seed(r))
    rec = {}; fit(obs, init, args.T, args.lr, ckp=CKP, rec=rec); recs.append(rec)
    print(f"  restart {r}: " + " ".join(f"T{c}:{rec[c]['fid'].mean().item():.3f}({rec[c]['t']:.1f}s)" for c in CKP), flush=True)

sweep = []
for R in range(1, args.R + 1):
    for c in CKP:
        res = torch.stack([recs[r][c]['res'] for r in range(R)]); fid = torch.stack([recs[r][c]['fid'] for r in range(R)])
        sel = res.argmin(0); f_sel = fid.gather(0, sel.view(1, -1)).squeeze(0)
        f_orc = fid.max(0).values
        tps = sum(recs[r][c]['t'] for r in range(R)) / NT
        sweep.append(dict(R=R, T=c, ms_per_state=tps * 1e3, fid_res=f_sel.mean().item(), fid_res_sd=f_sel.std().item(),
                          fid_oracle=f_orc.mean().item()))
        print(f"[MLE] R={R} T={c}: {tps*1e3:.2f} ms/state, fid(residual-select) {f_sel.mean().item():.4f}+-{f_sel.std().item():.4f}, "
              f"fid(oracle-select) {f_orc.mean().item():.4f}", flush=True)

# reference configuration (paper: R=3, T=500), residual-selected
Rr, Tr = min(3, args.R), 500 if 500 in CKP else CKP[-1]
res = torch.stack([recs[r][Tr]['res'] for r in range(Rr)]); sel = res.argmin(0)
TH_hat = torch.stack([recs[r][Tr]['th'] for r in range(Rr)])[sel, torch.arange(NT, device=dev)]
st_hat = S.th_to_state(TH_hat)
pL = S.expect(st_hat, TL).detach(); pH = S.expect(st_hat, TH).detach()
mle_maeH = (pH - oHt).abs().mean().item(); mle_maeL = (pL - oL).abs().mean().item()
print(f"[MLE R{Rr} T{Tr}] held-out MAE {mle_maeH:.4f} | local MAE {mle_maeL:.4f}", flush=True)

# ---------------- parametric bootstrap UQ for MLE ----------------
with torch.no_grad():
    ohat = S.expect(st_hat, TAB); sd = S.shot_sd(ohat, args.meas)
gb = torch.Generator(device=dev).manual_seed(2024)
bL, bH, tb = [], [], 0.0
for bb in range(args.B):
    ob = ohat + sd * torch.randn(ohat.shape, device=dev, generator=gb)
    thb, t = fit(ob, TH_hat, args.Tb, args.lr); tb += t
    with torch.no_grad():
        sb = S.th_to_state(thb); bL.append(S.expect(sb, TL)); bH.append(S.expect(sb, TH))
bL = torch.stack(bL); bH = torch.stack(bH)
covL_b = perobs_conformal(oL, pL, bL.std(0).clamp_min(1e-4)); covH_b = perobs_conformal(oHt, pH, bH.std(0).clamp_min(1e-4))
covL_f = perobs_conformal(oL, pL, torch.ones_like(pL)); covH_f = perobs_conformal(oHt, pH, torch.ones_like(pH))
t_boot = tb / NT
print(f"[MLE+bootstrap B={args.B}, Tb={args.Tb}] +{t_boot*1e3:.2f} ms/state | local cov {covL_b[0]*100:.1f}% w {covL_b[2]:.3f} | "
      f"held cov {covH_b[0]*100:.1f}% w {covH_b[2]:.3f}", flush=True)
print(f"[MLE+fixed-scale conformal] local cov {covL_f[0]*100:.1f}% w {covL_f[2]:.3f} | held cov {covH_f[0]*100:.1f}% w {covH_f[2]:.3f}", flush=True)

out = dict(tag=tag, n=n, design=args.design, meas=args.meas, K=K, ntest=NT, sweep=sweep, ref=dict(R=Rr, T=Tr),
           mle_maeH=mle_maeH, mle_maeL=mle_maeL, boot=dict(B=args.B, Tb=args.Tb, ms_per_state=t_boot * 1e3, covL=covL_b, covH=covH_b),
           fixed_scale=dict(covL=covL_f, covH=covH_f), nheld=len(OH_))

# ---------------- direct estimates from the same bitstrings (basis mode) ----------------
if args.meas == 'basis':
    planH = S.setup_basis(OH_)  # which of the 9 settings cover each held-out observable
    covH_cnt = S._cover.clone()
    gd = torch.Generator(device=dev).manual_seed(4242)
    # re-simulate the identical record for the union design: same settings, same generator seed and order
    U = DES + OH_; TU = S.pauli_tables(U); planU = S.setup_basis(U)
    ou = S.sample_basis(psi, planU, len(U), gd)
    assert (ou[:, :K] - obs).abs().max() < 1e-6, "direct estimate does not reuse the identical bitstrings"
    dH = ou[:, K:]
    maeD = (dH - oHt).abs().mean().item()
    sdD = torch.sqrt(((1 - dH ** 2).clamp_min(0)) / (args.shots * covH_cnt)).clamp_min(1e-3)
    covD = perobs_conformal(oHt, dH, sdD)
    print(f"[DIRECT from bitstrings] held-out MAE {maeD:.4f} | covering settings per obs {covH_cnt.min().item():.0f}-{covH_cnt.max().item():.0f} | "
          f"conformal cov {covD[0]*100:.1f}% w {covD[2]:.3f}", flush=True)
    out['direct'] = dict(maeH=maeD, covH=covD, settings_per_obs=[covH_cnt.min().item(), covH_cnt.max().item()])

# ---------------- classical shadows ----------------
if args.shadows:
    m = 9 if args.design in ('local', 'local1') else None
    budgets = [9 * args.shots, 90 * args.shots]
    hidx = torch.tensor(OH_, device=dev); lidx = torch.tensor(OL, device=dev)
    rots = torch.stack([S._rot(1), S._rot(2), S._rot(3)])  # basis 1,2,3 -> X,Y,Z
    sh_res = []
    for N in budgets:
        estH = torch.zeros(NT, len(OH_), device=dev); estL = torch.zeros(NT, len(OL), device=dev)
        g = torch.Generator(device=dev).manual_seed(77)
        chunk = 2000
        for i in range(NT):
            accH = torch.zeros(len(OH_), device=dev); accL = torch.zeros(len(OL), device=dev)
            for c0 in range(0, N, chunk):
                Nc = min(chunk, N - c0)
                bas = torch.randint(1, 4, (Nc, n), device=dev, generator=g)
                phi = psi[i:i + 1].expand(Nc, -1).clone()
                for q in range(n): phi = S.apply_1q(phi, q, rots[bas[:, q] - 1])
                pr = phi.abs() ** 2; pr = pr / pr.sum(1, keepdim=True)
                oc = torch.multinomial(pr, 1, generator=g).squeeze(1)  # (Nc,)
                z = (1 - 2 * S.bits[oc]).float()  # (Nc,n)
                for idxs, acc in [(hidx, accH), (lidx, accL)]:
                    supp = idxs > 0  # (k,n)
                    match = ((bas.unsqueeze(1) == idxs.unsqueeze(0)) | ~supp.unsqueeze(0)).all(-1)  # (Nc,k)
                    val = torch.where(supp.unsqueeze(0), 3.0 * z.unsqueeze(1), torch.ones_like(z).unsqueeze(1)).prod(-1)
                    acc += (match.float() * val).sum(0)
            estH[i] = accH / N; estL[i] = accL / N
        maeSH = (estH.clamp(-1, 1) - oHt).abs().mean().item(); maeSL = (estL.clamp(-1, 1) - oL).abs().mean().item()
        sh_res.append(dict(N=N, maeH=maeSH, maeL=maeSL))
        print(f"[SHADOWS N={N}] held-out MAE {maeSH:.4f} | local Z/X MAE {maeSL:.4f}", flush=True)
    out['shadows'] = sh_res

json.dump(out, open(os.path.join(OUT, f"{tag}.json"), 'w'), indent=1)
print("[dumped]", flush=True)
