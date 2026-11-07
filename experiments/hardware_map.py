"""Hardware closed loop with MAP: net / tuned MLE / MAP on the 63 IBM-measured local Paulis (hw_local.npz).
Uses the n=6 S=200 checkpoint and the validation-chosen MAP config of rescue_n6_local_S200 (lam, T) and MLE T.
Fidelity is to the ideal circuit target. Convention verified on the ideal expectations first."""
import json, numpy as np, torch
from common import Sys, CondDrop, onehot, fidelity
import os
OUT = os.environ.get('QST_OUT', 'results')  # where result JSONs and checkpoints are written
dev = torch.device('cuda:0'); torch.manual_seed(0)
S = Sys(6, dev, shots=200); DES = S.design('local'); TAB = S.pauli_tables(DES); OHOT = onehot(DES, dev)
R = OUT + '/'
d = np.load(os.environ.get('QST_HW', 'data/hw_local.npz'), allow_pickle=True)
assert [list(p) for p in d['pool']] == DES, "Pauli order differs"
psi = torch.tensor(d['psi_ideal'], dtype=torch.complex64, device=dev)
e_id = torch.tensor(d['exp_ideal'], dtype=torch.float32, device=dev); e_hw = torch.tensor(d['exp_hw'], dtype=torch.float32, device=dev)
err = (S.expect(psi, TAB) - e_id).abs().max().item()
if err > 1e-4:
    rev = torch.tensor([int(f"{i:06b}"[::-1], 2) for i in range(64)], device=dev); psi = psi[:, rev]
    err = (S.expect(psi, TAB) - e_id).abs().max().item()
print(f"[convention] max|expect - ideal| = {err:.2e}", flush=True); assert err < 1e-4
cfg = json.load(open(R + 'rescue_n6_local_S200.json'))['sel']
lam = float(cfg['map'][0][3:]); Tm = cfg['map'][1]; Tl = cfg['mle'][1]
print(f"[cfg] MAP lam={lam} T={Tm}; MLE T={Tl}", flush=True)
net = CondDrop(24, S.THETA).to(dev); net.load_state_dict(torch.load(R + 'ckpt_n6_local_S200_s30000.pt', map_location=dev)); net.eval()
def fit(o, init, iters, lam=0.0, ref=None):
    th = init.clone().requires_grad_(True); opt = torch.optim.Adam([th], lr=0.05)
    for _ in range(iters):
        st = S.th_to_state(th); loss = ((S.expect(st, TAB) - o) ** 2).mean(1)
        if lam > 0: loss = loss + lam * (1 - fidelity(st, ref))
        opt.zero_grad(); loss.mean().backward(); opt.step()
    return th.detach()
out = {}
for name, o in [('ideal', e_id), ('hw', e_hw)]:
    with torch.no_grad(): th0 = net(OHOT.unsqueeze(0).expand(5, -1, -1), o); st0 = S.th_to_state(th0)
    f_net = fidelity(psi, st0)
    fm = []
    for r in range(5):  # 5 random inits, averaged (not selected), to reduce init noise on 5 states
        g = torch.Generator(device=dev).manual_seed(r)
        fm.append(fidelity(psi, S.th_to_state(fit(o, 0.1 * torch.randn(5, S.THETA, device=dev, generator=g), Tl))))
    f_mle = torch.stack(fm).mean(0)
    f_map = fidelity(psi, S.th_to_state(fit(o, th0, Tm, lam, st0.detach())))
    out[name] = dict(net=f_net.tolist(), mle=f_mle.tolist(), map=f_map.tolist())
    print(f"[{name}] net {f_net.mean():.4f}+-{f_net.std():.4f} | MLE {f_mle.mean():.4f}+-{f_mle.std():.4f} | MAP {f_map.mean():.4f}+-{f_map.std():.4f}", flush=True)
    print(f"   per-state net {np.round(f_net.cpu().numpy(),3)} mle {np.round(f_mle.cpu().numpy(),3)} map {np.round(f_map.cpu().numpy(),3)}", flush=True)
out['hw_minus_ideal'] = (e_hw - e_id).abs().mean().item()
json.dump(out, open(R + 'hw_map.json', 'w'), indent=1); print('[dumped]')
