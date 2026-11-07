"""#1 GAUGE-STABILITY experiment (the key theory point). Does the left-canonical QR sweep + sign/norm
fixing actually collapse gauge-equivalent MPS to the same coordinates? Apply random INVERTIBLE bond
gauge transforms G_l (state-preserving) to a physical state's cores, re-canonicalize, and measure:
  (a) canonical core-space distance ||theta - theta'||   (should be ~0 if gauge is fixed)
  (b) physical-state fidelity (must be ~1: gauge preserves the state)
  (c) flow prior log-density stability across gauge copies
  (d) reconstruction stability: reconstruct from the SAME measurements starting from different gauge
      copies -> same state?
Calibrated against the typical inter-state distance (what 'far' means)."""
import time, argparse, json, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
ap=argparse.ArgumentParser(); ap.add_argument('--dev',default='cuda:0'); ap.add_argument('--n',type=int,default=6)
ap.add_argument('--chi',type=int,default=2); ap.add_argument('--sigma',type=float,default=0.35); ap.add_argument('--shots',type=int,default=200)
ap.add_argument('--steps',type=int,default=40000); ap.add_argument('--family',default='tfim',choices=['random','tfim'])
ap.add_argument('--gauge_scale',type=float,default=0.5); ap.add_argument('--ncopies',type=int,default=8)
args=ap.parse_args(); dev=torch.device(args.dev); torch.manual_seed(0); np.random.seed(0); cdt=torch.complex64
n,chi,D=args.n,args.chi,2**args.n
bonds=[min(2**k,2**(n-k),chi) for k in range(n+1)]; shapes=[(bonds[k],2,bonds[k+1]) for k in range(n)]
sizes=[np.prod(s) for s in shapes]; off=np.cumsum([0]+sizes); THETA=2*int(off[-1])
print(f"[cfg] n={n} chi={chi} D={D} theta={THETA} family={args.family} gauge_scale={args.gauge_scale}")
I2=torch.tensor([[1,0],[0,1]],dtype=cdt,device=dev);Xp=torch.tensor([[0,1],[1,0]],dtype=cdt,device=dev)
Yp=torch.tensor([[0,-1j],[1j,0]],dtype=cdt,device=dev);Zp=torch.tensor([[1,0],[0,-1]],dtype=cdt,device=dev)
PAULI=[I2,Xp,Yp,Zp]
def kron_list(ms):
    o=ms[0]
    for m in ms[1:]: o=torch.kron(o,m)
    return o
def rand_paulis(k,seed):
    rng=np.random.default_rng(seed); mats=[]; seen=set()
    while len(mats)<k:
        s=tuple(rng.integers(0,4,size=n).tolist())
        if s==(0,)*n or s in seen: continue
        seen.add(s); mats.append(kron_list([PAULI[i] for i in s]))
    return torch.stack(mats,0)
POOL=rand_paulis(min(4**n-1,220),1)
def split_cores(th):
    b=th.shape[0]; c=torch.complex(th[:,:THETA//2],th[:,THETA//2:]); return [c[:,off[k]:off[k+1]].reshape(b,*shapes[k]) for k in range(n)]
def cat_cores(cores):
    b=cores[0].shape[0]; c=torch.cat([g.reshape(b,-1) for g in cores],1); return torch.cat([c.real,c.imag],1)
def cores_to_state(cores):
    b=cores[0].shape[0]; psi=cores[0].reshape(b,2,bonds[1])
    for k in range(1,n): psi=torch.einsum('bar,brjs->bajs',psi,cores[k].reshape(b,bonds[k],2,bonds[k+1])).reshape(b,2**(k+1),bonds[k+1])
    psi=psi.reshape(b,D); return psi/psi.norm(dim=1,keepdim=True).clamp_min(1e-9)
def canon(cores):
    b=cores[0].shape[0]; cs=[g.clone() for g in cores]
    for k in range(n-1):
        M=cs[k].reshape(b,bonds[k]*2,bonds[k+1]); Q,Rm=torch.linalg.qr(M)
        d=torch.diagonal(Rm,dim1=-2,dim2=-1); ph=(d/d.abs().clamp_min(1e-9)).conj()
        Q=Q*ph.unsqueeze(1); Rm=ph.unsqueeze(2)*Rm; cs[k]=Q.reshape(b,bonds[k],2,bonds[k+1])
        cs[k+1]=torch.einsum('brs,bsjt->brjt',Rm,cs[k+1].reshape(b,bonds[k+1],2,bonds[k+2])).reshape(b,*shapes[k+1])
    nrm=cs[-1].reshape(b,-1).norm(dim=1).clamp_min(1e-9); cs[-1]=cs[-1]/nrm.view(-1,1,1,1)
    st=cores_to_state(cs); ph=(st[:,0]/st[:,0].abs().clamp_min(1e-9)).conj(); cs[-1]=cs[-1]*ph.view(-1,1,1,1)
    return cs
def apply_gauge(cores, scale):
    # random invertible bond gauge G_l on each internal bond: A^(l) -> A^(l) G_l, A^(l+1) -> G_l^{-1} A^(l+1). State-preserving.
    cs=[g.clone() for g in cores]; b=cores[0].shape[0]
    for l in range(n-1):
        r=bonds[l+1]
        G=torch.eye(r,dtype=cdt,device=dev).unsqueeze(0).repeat(b,1,1)+scale*(torch.randn(b,r,r,device=dev)+1j*torch.randn(b,r,r,device=dev))
        Ginv=torch.linalg.inv(G)
        cs[l]=torch.einsum('bair,brc->baic', cs[l].reshape(b,bonds[l],2,r), G).reshape(b,*shapes[l])
        cs[l+1]=torch.einsum('brc,bcjd->brjd', Ginv, cs[l+1].reshape(b,r,2,bonds[l+2])).reshape(b,*shapes[l+1])
    return cs
def make_family(b):
    g=torch.Generator(device=dev).manual_seed(12345)
    base=[torch.randn(1,*s,generator=g,device=dev)+1j*torch.randn(1,*s,generator=g,device=dev) for s in shapes]
    return [(base[k]+args.sigma*(torch.randn(b,*shapes[k],device=dev)+1j*torch.randn(b,*shapes[k],device=dev))) for k in range(n)]
def _single(op,q): return kron_list([op if i==q else I2 for i in range(n)])
ZZ=torch.stack([_single(Zp,q)@_single(Zp,q+1) for q in range(n-1)],0) if n>1 else None; XO=torch.stack([_single(Xp,q) for q in range(n)],0)
def tfim_ground(b):
    J=0.2+0.4*torch.rand(b,n-1,device=dev); h=1.0+1.0*torch.rand(b,n,device=dev)
    ch=24; psis=[]
    for s in range(0,b,ch):
        H=-torch.einsum('bi,ijk->bjk',J[s:s+ch].to(cdt),ZZ)-torch.einsum('bi,ijk->bjk',h[s:s+ch].to(cdt),XO)
        w,V=torch.linalg.eigh(0.5*(H+H.conj().transpose(-1,-2))); psis.append(V[:,:,0].clone()); del H,w,V
    return torch.cat(psis,0)
def fit_mps(psi_t,iters=400,lr=0.05,restarts=2):
    b=psi_t.shape[0]; best=None;beste=None
    for _ in range(restarts):
        x=(0.1*torch.randn(b,THETA,device=dev)).requires_grad_(True); op=torch.optim.Adam([x],lr=lr)
        for _ in range(iters):
            psi=cores_to_state(split_cores(x)); loss=(1-(torch.einsum('bi,bi->b',psi.conj(),psi_t).abs())**2).mean()
            op.zero_grad(); loss.backward(); op.step()
        with torch.no_grad():
            e=1-(torch.einsum('bi,bi->b',cores_to_state(split_cores(x)).conj(),psi_t).abs())**2
            if best is None: best,beste=x.detach().clone(),e
            else: sw=e<beste; best[sw]=x.detach()[sw]; beste=torch.where(sw,e,beste)
    return split_cores(best)
def make_cores(b):
    if args.family=='tfim': return fit_mps(tfim_ground(b))
    return make_family(b)
def expect(psi,ops):
    Opsi=torch.einsum('kij,bj->bki',ops,psi); return torch.einsum('bi,bki->bk',psi.conj(),Opsi).real
def fidelity(a,b): return (torch.einsum('bi,bi->b',a.conj(),b).abs())**2

# ---- flow prior (for prior-density stability) ----
print("[prep] bank + flow ...")
NB=6000 if args.family=='tfim' else 30000
BANK=cat_cores(canon(make_cores(NB))); BM,BS=BANK.mean(0,keepdim=True),BANK.std(0,keepdim=True).clamp_min(1e-4); BANKn=(BANK-BM)/BS
sf=lambda t:(t-BM)/BS; uf=lambda t:t*BS+BM
class Vel(nn.Module):
    def __init__(s,d,h=1024):
        super().__init__(); s.t=nn.Sequential(nn.Linear(1,128),nn.SiLU(),nn.Linear(128,128))
        s.n=nn.Sequential(nn.Linear(d+128,h),nn.SiLU(),nn.Linear(h,h),nn.SiLU(),nn.Linear(h,h),nn.SiLU(),nn.Linear(h,h),nn.SiLU(),nn.Linear(h,d))
    def forward(s,x,t): return s.n(torch.cat([x,s.t(t.view(-1,1))],1))
vel=Vel(THETA).to(dev); opt=torch.optim.AdamW(vel.parameters(),lr=2e-4); sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,args.steps)
t0=time.time()
for it in range(1,args.steps+1):
    th=BANKn[torch.randint(0,BANKn.shape[0],(256,),device=dev)]; eps=torch.randn_like(th); t=torch.rand(th.shape[0],device=dev)
    loss=F.mse_loss(vel((1-t).view(-1,1)*eps+t.view(-1,1)*th,t),th-eps); opt.zero_grad(); loss.backward(); opt.step(); sch.step()
print(f"[flow trained {time.time()-t0:.0f}s]")
def flow_energy(theta_std,ns=20):
    # cheap prior-density proxy: reverse-flow to t=0 and report ||z||^2/D (Gaussian NLL up to const); lower=more typical
    th=theta_std.clone()
    for i in range(ns):
        t=1.0-i/ns
        with torch.no_grad(): v=vel(th,torch.full((th.shape[0],),t,device=dev))
        th=th-(1.0/ns)*v
    return (th**2).mean(1)   # per-sample mean-squared latent

# =========================== GAUGE STABILITY ===========================
NT=60
cores0=canon(make_cores(NT))               # canonical reference cores
th0=sf(cat_cores(cores0))                   # standardized canonical coords
psi0=cores_to_state(cores0)
# inter-state distance calibration (how far apart are DIFFERENT states in canonical coords)
perm=torch.randperm(NT,device=dev); inter=(th0-th0[perm]).norm(dim=1)
print(f"\n[calibration] typical inter-state canonical-coord distance: mean {inter.mean():.3f} (median {inter.median():.3f})")

# For each state, make K gauge copies, re-canonicalize, measure spread
Kc=args.ncopies
core_dists=[]; state_fids=[]; energy_spread=[]; recon_fids_across=[]
ops=POOL[:16]
# reconstruction op (proximal, best-of-1) starting from a given canonical anchor's measurements
def meas_loss(psi,ops,obs): return (expect(psi/psi.norm(dim=1,keepdim=True),ops)-obs).pow(2).mean(1)
def prox_from(anchor_th_std,ops,obs,n_inner=25,lam=0.3,lr=0.1):
    x=uf(anchor_th_std).clone().requires_grad_(True); op=torch.optim.Adam([x],lr=lr)
    psi_a=cores_to_state(split_cores(uf(anchor_th_std))).detach()
    for _ in range(n_inner):
        psi=cores_to_state(split_cores(x)); loss=meas_loss(psi,ops,obs).mean()+lam*(1-(torch.einsum('bi,bi->b',psi.conj(),psi_a).abs())**2).mean()
        op.zero_grad(); loss.backward(); op.step()
    with torch.no_grad(): return cores_to_state(canon(split_cores(x)))
th_copies=[]  # (Kc, NT, THETA) standardized canonical coords of gauge copies
for c in range(Kc):
    g=apply_gauge(cores0, args.gauge_scale)     # gauge-transform the SAME states
    gc=canon(g)                                  # re-canonicalize
    thc=sf(cat_cores(gc)); th_copies.append(thc)
    state_fids.append(fidelity(psi0, cores_to_state(gc)))   # must be ~1
th_copies=torch.stack(th_copies,0)              # (Kc,NT,THETA)
# per-state spread across gauge copies (max pairwise / std)
mean_c=th_copies.mean(0)                         # (NT,THETA)
spread=(th_copies-mean_c.unsqueeze(0)).norm(dim=2)   # (Kc,NT)
gauge_core_dist=spread.mean(0)                   # per-state mean distance to centroid
state_fid=torch.stack(state_fids,0).mean(0)
# prior-density stability: flow energy across gauge copies (std within state)
en=torch.stack([flow_energy(th_copies[c]) for c in range(Kc)],0)  # (Kc,NT)
en_std=en.std(0); en_mean=en.mean(0)
# reconstruction stability: measurements are gauge-invariant; reconstruct starting from 2 different gauge copies
obs=expect(psi0,ops)  # clean gauge-invariant measurements
r1=prox_from(th_copies[0],ops,obs); r2=prox_from(th_copies[min(1,Kc-1)],ops,obs)
recon_cross_fid=fidelity(r1,r2)                  # do 2 gauge-starting reconstructions agree?
recon_to_true_1=fidelity(psi0,r1)

print("\n================= GAUGE STABILITY =================")
print(f"state fidelity across gauge copies (should be ~1): mean {state_fid.mean():.4f} min {state_fid.min():.4f}")
print(f"canonical CORE distance across gauge copies:  mean {gauge_core_dist.mean():.4f}  (vs inter-state {inter.mean():.3f})")
print(f"   ratio gauge-spread / inter-state = {(gauge_core_dist.mean()/inter.mean()).item():.4f}  (small => gauge well fixed)")
print(f"flow prior-energy across gauge copies: within-state std {en_std.mean():.4f}  (mean energy {en_mean.mean():.3f})")
print(f"reconstruction cross-gauge agreement fidelity: mean {recon_cross_fid.mean():.4f} min {recon_cross_fid.min():.4f}")
print(f"reconstruction-to-true fidelity: {recon_to_true_1.mean():.4f}")
OUT={'n':n,'family':args.family,'gauge_scale':args.gauge_scale,'ncopies':Kc,
     'inter_state_dist':inter.mean().item(),'gauge_core_dist':gauge_core_dist.mean().item(),
     'ratio':(gauge_core_dist.mean()/inter.mean()).item(),'state_fid':state_fid.mean().item(),
     'energy_within_std':en_std.mean().item(),'energy_mean':en_mean.mean().item(),
     'recon_cross_fid':recon_cross_fid.mean().item(),'recon_to_true':recon_to_true_1.mean().item()}
# also scan gauge scale to see if larger gauges break it more
scan={}
for gs in [0.1,0.3,0.5,1.0,2.0]:
    cps=[canon(apply_gauge(cores0,gs)) for _ in range(4)]
    ths=torch.stack([sf(cat_cores(c)) for c in cps],0); m=ths.mean(0); sp=(ths-m).norm(dim=2).mean()
    fid=torch.stack([fidelity(psi0,cores_to_state(c)) for c in cps],0).mean()
    scan[gs]={'core_dist':sp.item(),'ratio':(sp/inter.mean()).item(),'state_fid':fid.item()}
    print(f"  gauge_scale={gs}: core-dist {sp.item():.4f} ratio {(sp/inter.mean()).item():.4f} state-fid {fid.item():.4f}")
OUT['scan']=scan
json.dump(OUT,open(f'./qst_gauge_{args.family}_n{n}.json','w'),indent=1)
print("[dumped]")
