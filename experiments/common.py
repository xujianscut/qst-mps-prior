"""Shared utilities for the experiments (shallow-circuit family, chi=2 MPS).
Pauli expectations via index-permutation + phase (no dense 2^n x 2^n matrices), checked against
dense kron at small n. Two measurement models:
  indep : o_j = <P_j> + N(0,(1-<P_j>^2)/S)  (the paper's simulation protocol)
  basis : m=9 qubit-wise basis settings (qubit q measured in basis a if q even else c, (a,c) in {X,Y,Z}^2),
          S bitstrings per setting; every design observable is estimated from the bitstrings of all
          settings that cover it (correlated shot noise, as on hardware).
"""
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F, time
import os
OUT = os.environ.get('QST_OUT', 'results')  # where result JSONs and checkpoints are written

cdt = torch.complex64


class Sys:
    def __init__(s, n, dev, chi=2, spread=0.6, shots=200):
        s.n, s.dev, s.chi, s.spread, s.shots = n, dev, chi, spread, shots
        s.D = 2 ** n
        s.bonds = [min(2 ** k, 2 ** (n - k), chi) for k in range(n + 1)]
        s.shapes = [(s.bonds[k], 2, s.bonds[k + 1]) for k in range(n)]
        sizes = [int(np.prod(x)) for x in s.shapes]
        s.off = np.cumsum([0] + sizes)
        s.THETA = 2 * int(s.off[-1])
        idx = torch.arange(s.D, device=dev)
        s.bits = torch.stack([(idx >> (n - 1 - q)) & 1 for q in range(n)], 1)  # (D,n)
        par = torch.zeros(s.D, dtype=torch.long, device=dev)
        for q in range(n):
            par ^= (idx >> q) & 1
        s.parity = par  # parity[x] = popcount(x) mod 2

    # ---------------- Pauli strings as (flip, phase) ----------------
    def pauli_tables(s, idxs):
        """idxs: (k,n) ints in {0:I,1:X,2:Y,3:Z}. Returns flipidx (k,D) long, coef (k,D) complex such that
        (P psi)(t) = coef[t] * psi[flip[t]]."""
        n, D, dev = s.n, s.D, s.dev
        idxs = torch.as_tensor(idxs, device=dev).long()
        k = idxs.shape[0]
        w = torch.tensor([1 << (n - 1 - q) for q in range(n)], device=dev)
        xm = (((idxs == 1) | (idxs == 2)).long() * w).sum(1)  # (k,)
        zym = (((idxs == 2) | (idxs == 3)).long() * w).sum(1)
        ny = (idxs == 2).sum(1)
        t = torch.arange(D, device=dev)
        src = t.unsqueeze(0) ^ xm.unsqueeze(1)  # s = t xor xmask
        sign = 1 - 2 * s.parity[src & zym.unsqueeze(1)]
        iy = torch.tensor([1, 1j, -1, -1j], dtype=cdt, device=dev)[ny % 4]
        coef = iy.unsqueeze(1) * sign.to(cdt)
        return src, coef, xm, zym

    def expect(s, psi, tab):
        src, coef = tab[0], tab[1]
        # psi (b,D) -> (b,k)
        g = psi[:, src]  # (b,k,D)
        return (psi.conj().unsqueeze(1) * coef.unsqueeze(0) * g).sum(-1).real

    def dense_check(s, idxs):
        I2 = torch.eye(2, dtype=cdt, device=s.dev)
        X = torch.tensor([[0, 1], [1, 0]], dtype=cdt, device=s.dev)
        Y = torch.tensor([[0, -1j], [1j, 0]], dtype=cdt, device=s.dev)
        Z = torch.tensor([[1, 0], [0, -1]], dtype=cdt, device=s.dev)
        P = [I2, X, Y, Z]
        psi = torch.randn(4, s.D, dtype=cdt, device=s.dev); psi = psi / psi.norm(dim=1, keepdim=True)
        tab = s.pauli_tables(idxs)
        e1 = s.expect(psi, tab)
        for j in range(len(idxs)):
            M = P[int(idxs[j][0])]
            for q in range(1, s.n): M = torch.kron(M, P[int(idxs[j][q])])
            e2 = torch.einsum('bi,ij,bj->b', psi.conj(), M, psi).real
            assert (e1[:, j] - e2).abs().max() < 1e-4, ("pauli table mismatch", j, idxs[j])
        return True

    # ---------------- designs ----------------
    def design(s, kind):
        n = s.n; seen = set(); out = []
        def add(v):
            t = tuple(v)
            if any(t) and t not in seen: seen.add(t); out.append(list(v))
        for q in range(n):
            for p in [1, 2, 3]:
                v = [0] * n; v[q] = p; add(v)
        if kind in ('local', 'local3'):
            for q in range(n - 1):
                for p1 in [1, 2, 3]:
                    for p2 in [1, 2, 3]:
                        v = [0] * n; v[q] = p1; v[q + 1] = p2; add(v)
        if kind == 'local3':
            for a in range(n - 2):
                for combo in np.ndindex(4, 4, 4):
                    if all(c == 0 for c in combo): continue
                    v = [0] * n
                    for wq, c in zip([a, a + 1, a + 2], combo): v[wq] = c
                    add(v)
        assert len(out) > 0
        return out

    def obs_local(s):  # 2n local Z/X (the paper's calibration set)
        n = s.n; out = []
        for p in [3, 1]:
            for q in range(n):
                v = [0] * n; v[q] = p; out.append(v)
        return out

    def obs_held(s):  # paper's held-out set: long-range ZZ (gap>=2) + weight-3 ZZZ / XXX
        n = s.n; out = []
        for i in range(n):
            for j in range(i + 2, n):
                v = [0] * n; v[i] = 3; v[j] = 3; out.append(v)
        for i in range(n - 2):
            for p in [3, 1]:
                v = [0] * n; v[i] = v[i + 1] = v[i + 2] = p; out.append(v)
        return out

    # ---------------- state family ----------------
    def apply_1q(s, state, q, U):
        """U: (2,2) or (b,2,2)."""
        b = state.shape[0]; n = s.n
        st = state.view(b, 2 ** q, 2, 2 ** (n - q - 1))
        if U.dim() == 2: U = U.unsqueeze(0).expand(b, 2, 2)
        out = torch.einsum('bij,bajc->baic', U, st)
        return out.reshape(b, s.D)

    def apply_ry(s, state, ang):
        for q in range(s.n):
            c = torch.cos(ang[:, q] / 2).to(cdt); sn = torch.sin(ang[:, q] / 2).to(cdt)
            U = torch.stack([torch.stack([c, -sn], -1), torch.stack([sn, c], -1)], -2)
            state = s.apply_1q(state, q, U)
        return state

    def apply_cnot(s, state, ctrl, tgt):
        n, D = s.n, s.D
        idx = torch.arange(D, device=s.dev); bc = (idx >> (n - 1 - ctrl)) & 1
        sw = idx ^ (bc << (n - 1 - tgt)); out = state.clone(); m = bc == 1
        out[:, idx[m]] = state[:, sw[m]]; return out

    def shallow(s, b, seed, spread=None):
        sp = s.spread if spread is None else spread
        g = torch.Generator(device=s.dev).manual_seed(seed)
        th1 = np.pi / 2 + sp * torch.randn(b, s.n, generator=g, device=s.dev)
        th2 = sp * torch.randn(b, s.n, generator=g, device=s.dev)
        st = torch.zeros(b, s.D, dtype=cdt, device=s.dev); st[:, 0] = 1.0
        st = s.apply_ry(st, th1)
        for q in range(s.n - 1): st = s.apply_cnot(st, q, q + 1)
        st = s.apply_ry(st, th2)
        return st / st.norm(dim=1, keepdim=True)

    # ---------------- disordered Ising ground states (second family) ----------------
    def tfim_params(s, b, seed, spread=None):
        w = 0.5 if spread is None else spread
        g = torch.Generator(device=s.dev).manual_seed(seed)
        J = 1 + w * (2 * torch.rand(b, s.n - 1, generator=g, device=s.dev, dtype=torch.float64) - 1)
        h = 1 + w * (2 * torch.rand(b, s.n, generator=g, device=s.dev, dtype=torch.float64) - 1)
        gz = w * (2 * torch.rand(b, s.n, generator=g, device=s.dev, dtype=torch.float64) - 1)
        return J, h, gz

    def tfim(s, b, seed, spread=None, chunk=None):
        """Ground states of H = -sum J_i Z_i Z_{i+1} - sum h_i X_i - sum g_i Z_i (open chain), J_i,h_i ~ U[1-w,1+w],
        g_i ~ U[-w,w], w = spread (default 0.5). Exact diagonalization in float64; sign fixed so sum(psi) > 0.
        Cached on disk (keyed by n, b, seed, w)."""
        import os
        w = 0.5 if spread is None else spread
        fn = os.environ.get('QST_OUT', 'results') + f"/cache/tfim_n{s.n}_b{b}_s{seed}_w{w}.pt"
        if os.path.exists(fn): return torch.load(fn, map_location=s.dev).to(cdt)
        J, h, gz = s.tfim_params(b, seed, spread)
        n, D = s.n, s.D; z = (1 - 2 * s.bits).double()  # (D,n)
        zz = z[:, :-1] * z[:, 1:]  # (D,n-1)
        idx = torch.arange(D, device=s.dev)
        chunk = chunk or max(1, int(2 ** 27 // (D * D)))
        out = []
        for c0 in range(0, b, chunk):
            Jc, hc, gc = J[c0:c0 + chunk], h[c0:c0 + chunk], gz[c0:c0 + chunk]; B = Jc.shape[0]
            H = torch.zeros(B, D, D, dtype=torch.float64, device=s.dev)
            diag = -(Jc @ zz.T) - (gc @ z.T)  # (B,D)
            H[:, idx, idx] = diag
            for q in range(n):
                fl = idx ^ (1 << (n - 1 - q))
                H[:, idx, fl] -= hc[:, q:q + 1]
            ev, V = torch.linalg.eigh(H); v = V[:, :, 0]
            v = v * torch.sign(v.sum(1, keepdim=True) + 1e-12)
            out.append(v)
        st = torch.cat(out).to(cdt); st = st / st.norm(dim=1, keepdim=True)
        os.makedirs(os.path.dirname(fn), exist_ok=True); torch.save(st.cpu(), fn)
        return st

    def family(s, name, b, seed, spread=None):
        return s.shallow(b, seed, spread) if name == 'shallow' else s.tfim(b, seed, spread)

    # ---------------- MPS ----------------
    def split_cores(s, th):
        b = th.shape[0]; c = torch.complex(th[:, :s.THETA // 2], th[:, s.THETA // 2:])
        return [c[:, s.off[k]:s.off[k + 1]].reshape(b, *s.shapes[k]) for k in range(s.n)]

    def cores_to_state(s, cs):
        b = cs[0].shape[0]; psi = cs[0].reshape(b, 2, s.bonds[1])
        for k in range(1, s.n):
            psi = torch.einsum('bar,brjs->bajs', psi, cs[k].reshape(b, s.bonds[k], 2, s.bonds[k + 1])).reshape(b, 2 ** (k + 1), s.bonds[k + 1])
        psi = psi.reshape(b, s.D); return psi / psi.norm(dim=1, keepdim=True).clamp_min(1e-9)

    def canon(s, cs):
        b = cs[0].shape[0]; cs = [g.clone() for g in cs]; bonds, shapes = s.bonds, s.shapes
        for k in range(s.n - 1):
            M = cs[k].reshape(b, bonds[k] * 2, bonds[k + 1]); Q, Rm = torch.linalg.qr(M)
            d = torch.diagonal(Rm, dim1=-2, dim2=-1); ph = (d / d.abs().clamp_min(1e-9)).conj()
            Q = Q * ph.unsqueeze(1); Rm = ph.unsqueeze(2) * Rm
            cs[k] = Q.reshape(b, bonds[k], 2, bonds[k + 1])
            cs[k + 1] = torch.einsum('brs,bsjt->brjt', Rm, cs[k + 1].reshape(b, bonds[k + 1], 2, bonds[k + 2])).reshape(b, *shapes[k + 1])
        nrm = cs[-1].reshape(b, -1).norm(dim=1).clamp_min(1e-9); cs[-1] = cs[-1] / nrm.view(-1, 1, 1, 1)
        return cs

    def th_to_state(s, th):
        return s.cores_to_state(s.split_cores(th))

    # ---------------- measurement models ----------------
    def basis_settings(s):
        n = s.n; out = []
        for a in [1, 2, 3]:
            for c in [1, 2, 3]:
                out.append([a if q % 2 == 0 else c for q in range(n)])
        return out

    def setup_basis(s, design_idx):
        """For each setting, which design observables it covers. Returns list of (setting, obs_ids, masks)."""
        n = s.n; settings = s.basis_settings(); plan = []
        cover = np.zeros(len(design_idx))
        for st in settings:
            ids = [j for j, v in enumerate(design_idx) if all(v[q] == 0 or v[q] == st[q] for q in range(n))]
            masks = [sum(1 << (n - 1 - q) for q in range(n) if design_idx[j][q] != 0) for j in ids]
            plan.append((st, torch.tensor(ids, device=s.dev), torch.tensor(masks, device=s.dev)))
            cover[ids] += 1
        assert (cover > 0).all(), "design not covered by basis settings"
        s._cover = torch.tensor(cover, device=s.dev, dtype=torch.float32)
        return plan

    def _rot(s, b):
        H = torch.tensor([[1, 1], [1, -1]], dtype=cdt, device=s.dev) / np.sqrt(2)
        Sd = torch.tensor([[1, 0], [0, -1j]], dtype=cdt, device=s.dev)
        I2 = torch.eye(2, dtype=cdt, device=s.dev)
        return {1: H, 2: H @ Sd, 3: I2}[b]

    def sample_basis(s, psi, plan, k, gen, shots=None):
        """Bitstring simulation. Returns estimates (b,k) of design observables, pooling all covering settings."""
        S = s.shots if shots is None else shots
        b = psi.shape[0]; acc = torch.zeros(b, k, device=s.dev)
        for st, ids, masks in plan:
            phi = psi
            for q in range(s.n):
                if st[q] != 3: phi = s.apply_1q(phi, q, s._rot(st[q]))
            pr = (phi.abs() ** 2); pr = pr / pr.sum(1, keepdim=True)
            outc = torch.multinomial(pr, S, replacement=True, generator=gen)  # (b,S)
            val = 1 - 2 * s.parity[outc.unsqueeze(-1) & masks.view(1, 1, -1)]  # (b,S,ks)
            acc[:, ids] += val.float().mean(1)
        return acc / s._cover

    def measure(s, psi, tab, mode, gen, plan=None, shots=None):
        S = s.shots if shots is None else shots
        if mode == 'indep':
            te = s.expect(psi, tab)
            sd = torch.sqrt(((1 - te ** 2).clamp_min(0)) / S)
            return te + sd * torch.randn(te.shape, device=s.dev, generator=gen)
        return s.sample_basis(psi, plan, tab[0].shape[0], gen, shots=S)

    def shot_sd(s, val, mode, shots=None):
        S = s.shots if shots is None else shots
        eff = S if mode == 'indep' else S * s._cover
        return torch.sqrt(((1 - val ** 2).clamp_min(0)) / eff)


def fidelity(a, b):
    return (torch.einsum('bi,bi->b', a.conj(), b).abs()) ** 2


class CondDrop(nn.Module):  # identical to the paper's DeepSets+dropout estimator
    def __init__(s, dp, dout, h=512, z=512, p=0.1):
        super().__init__()
        s.enc = nn.Sequential(nn.Linear(dp + 1, h), nn.SiLU(), nn.Dropout(p), nn.Linear(h, h), nn.SiLU(), nn.Linear(h, z))
        s.post = nn.Sequential(nn.Linear(z, h), nn.SiLU(), nn.Dropout(p), nn.Linear(h, h), nn.SiLU(), nn.Dropout(p), nn.Linear(h, h), nn.SiLU(), nn.Linear(h, dout))

    def forward(s, oh, ov):
        x = torch.cat([oh, ov.unsqueeze(-1)], -1); return s.post(s.enc(x).mean(1))


def onehot(idxs, dev):
    t = torch.as_tensor(idxs, device=dev).long()
    return F.one_hot(t, 4).float().reshape(t.shape[0], -1)


def conf_Q(rc, lv=0.9):
    N = rc.shape[0]; kk = int(np.ceil(lv * (N + 1)))
    if kk > N: return torch.full((rc.shape[1],), float('inf'), device=rc.device)
    return rc.sort(dim=0).values[kk - 1]


def perobs_conformal(otrue, mean, std, nsplit=10, lv=0.9, seed0=0):
    """Per-observable split conformal (own Q_O), averaged over random splits. Returns cov, cov_sd, width."""
    N = otrue.shape[0]; covs, wids = [], []
    for sp in range(nsplit):
        g = torch.Generator(device=otrue.device).manual_seed(seed0 + sp)
        idx = torch.randperm(N, generator=g, device=otrue.device); ci, ti = idx[:N // 2], idx[N // 2:]
        r = (otrue - mean).abs() / std; Q = conf_Q(r[ci], lv)
        lo = mean[ti] - Q * std[ti]; hi = mean[ti] + Q * std[ti]
        covs.append(((otrue[ti] >= lo) & (otrue[ti] <= hi)).float().mean().item())
        wids.append((hi.clamp(max=1) - lo.clamp(min=-1)).clamp_min(0).mean().item())
    return float(np.mean(covs)), float(np.std(covs)), float(np.mean(wids))


def sync(dev):
    if dev.type == 'cuda': torch.cuda.synchronize(dev)
