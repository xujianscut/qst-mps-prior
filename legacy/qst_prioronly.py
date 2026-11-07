"""#1 THE critical control: is this real measurement-efficient tomography, or nearest-neighbor prediction
from a concentrated family? Reports:
  (a) k=0 PRIOR-ONLY baselines (no measurements): random prior sample, bank medoid, bank-mean state.
  (b) reconstruction with TRUE measurements vs SHUFFLED measurements (from other states) vs prior-only.
  (c) information gain = fid(k) - fid(prior-only), and fid(true k) - fid(shuffled k).
  (d) distribution of each test state's fidelity to its NEAREST training-bank neighbor.
If shuffled ~ prior-only << true, measurements carry real information. If prior-only ~ true, it is
nearest-neighbor prediction. Headline family TFIM n=6."""
import time, argparse, json, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
ap=argparse.ArgumentParser(); ap.add_argument('--dev',default='cuda:0'); ap.add_argument('--n',type=int,default=6); ap.add_argument('--chi',type=int,default=2)
ap.add_argument('--shots',type=int,default=200); ap.add_argument('--steps',type=int,default=40000); ap.add_argument('--ntest',type=int,default=120)
ap.add_argument('--family',default='tfim',choices=['tfim','random']); ap.add_argument('--sigma',type=float,default=0.35)
ap.add_argument('--bestof',type=int,default=8); ap.add_argument('--ns',type=int,default=40); ap.add_argument('--ni',type=int,default=25); ap.add_argument('--bank',type=int,default=6000)
args=ap.parse_args(); dev=torch.device(args.dev); torch.manual_seed(0); np.random.seed(0); cdt=torch.complex64
n,chi,D=args.n,args.chi,2**args.n
bonds=[min(2**k,2**(n-k),chi) for k in range(n+1)]; shapes=[(bonds[k],2,bonds[k+1]) for k in range(n)]
sizes=[np.prod(s) for s in shapes]; off=np.cumsum([0]+sizes); THETA=2*int(off[-1])
print(f"[cfg] n={n} family={args.family} theta={THETA}",flush=True)
I2=torch.tensor([[1,0],[0,1]],dtype=cdt,device=dev);Xp=torch.tensor([[0,1],[1,0]],dtype=cdt,device=dev)
Yp=torch.tensor([[0,-1j],[1j,0]],dtype=cdt,device=dev);Zp=torch.tensor([[1,0],[0,-1]],dtype=cdt,device=dev); PAULI=[I2,Xp,Yp,Zp]
def kron_list(ms):
    o=ms[0]
    for m in ms[1:]: o=torch.kron(o,m)
    return o
def rand_paulis(k,seed):
    rng=np.random.default_rng(seed); mats=[];seen=set()
    while len(mats)<k:
        s=tuple(rng.integers(0,4,size=n).tolist())
        if s==(0,)*n or s in seen: continue
        seen.add(s); mats.append(kron_list([PAULI[i] for i in s]))
    return torch.stack(mats,0)
POOL=rand_paulis(min(4**n-1,220),1)
def split_cores(th):
    b=th.shape[0]; c=torch.complex(th[:,:THETA//2],th[:,THETA//2:]); return [c[:,off[k]:off[k+1]].reshape(b,*shapes[k]) for k in range(n)]
def cat_cores(cs):
    b=cs[0].shape[0]; c=torch.cat([g.reshape(b,-1) for g in cs],1); return torch.cat([c.real,c.imag],1)
def cores_to_state(cs):
    b=cs[0].shape[0]; psi=cs[0].reshape(b,2,bonds[1])
    for k in range(1,n): psi=torch.einsum('bar,brjs->bajs',psi,cs[k].reshape(b,bonds[k],2,bonds[k+1])).reshape(b,2**(k+1),bonds[k+1])
    psi=psi.reshape(b,D); return psi/psi.norm(dim=1,keepdim=True).clamp_min(1e-9)
def canon(cs):
    b=cs[0].shape[0]; cs=[g.clone() for g in cs]
    for k in range(n-1):
        M=cs[k].reshape(b,bonds[k]*2,bonds[k+1]); Q,Rm=torch.linalg.qr(M)
        d=torch.diagonal(Rm,dim1=-2,dim2=-1); ph=(d/d.abs().clamp_min(1e-9)).conj(); Q=Q*ph.unsqueeze(1); Rm=ph.unsqueeze(2)*Rm
        cs[k]=Q.reshape(b,bonds[k],2,bonds[k+1]); cs[k+1]=torch.einsum('brs,bsjt->brjt',Rm,cs[k+1].reshape(b,bonds[k+1],2,bonds[k+2])).reshape(b,*shapes[k+1])
    nrm=cs[-1].reshape(b,-1).norm(dim=1).clamp_min(1e-9); cs[-1]=cs[-1]/nrm.view(-1,1,1,1)
    st=cores_to_state(cs); ph=(st[:,0]/st[:,0].abs().clamp_min(1e-9)).conj(); cs[-1]=cs[-1]*ph.view(-1,1,1,1); return cs
def make_family(b):
    g=torch.Generator(device=dev).manual_seed(12345)
    base=[torch.randn(1,*s,generator=g,device=dev)+1j*torch.randn(1,*s,generator=g,device=dev) for s in shapes]
    return canon([(base[k]+args.sigma*(torch.randn(b,*shapes[k],device=dev)+1j*torch.randn(b,*shapes[k],device=dev))) for k in range(n)])
def _single(op,q): return kron_list([op if i==q else I2 for i in range(n)])
ZZ=torch.stack([_single(Zp,q)@_single(Zp,q+1) for q in range(n-1)],0); XO=torch.stack([_single(Xp,q) for q in range(n)],0)
def tfim_ground(b,seed=0):
    g=torch.Generator(device=dev).manual_seed(seed); J=0.2+0.4*torch.rand(b,n-1,generator=g,device=dev); h=1.0+1.0*torch.rand(b,n,generator=g,device=dev)
    ch=24; psis=[]
    for s in range(0,b,ch):
        H=-torch.einsum('bi,ijk->bjk',J[s:s+ch].to(cdt),ZZ)-torch.einsum('bi,ijk->bjk',h[s:s+ch].to(cdt),XO)
        w,V=torch.linalg.eigh(0.5*(H+H.conj().transpose(-1,-2))); psis.append(V[:,:,0].clone()); del H,w,V
    return torch.cat(psis,0)
def fit_mps(psi_t,iters=400,lr=0.05,restarts=2,seed=0):
    b=psi_t.shape[0]; g=torch.Generator(device=dev).manual_seed(seed); best=None;beste=None
    for _ in range(restarts):
        x=(0.1*torch.randn(b,THETA,generator=g,device=dev)).requires_grad_(True); op=torch.optim.Adam([x],lr=lr)
        for _ in range(iters):
            psi=cores_to_state(split_cores(x)); loss=(1-(torch.einsum('bi,bi->b',psi.conj(),psi_t).abs())**2).mean(); op.zero_grad(); loss.backward(); op.step()
        with torch.no_grad():
            e=1-(torch.einsum('bi,bi->b',cores_to_state(split_cores(x)).conj(),psi_t).abs())**2
            if best is None: best,beste=x.detach().clone(),e
            else: sw=e<beste; best[sw]=x.detach()[sw]; beste=torch.where(sw,e,beste)
    return canon(split_cores(best))
def make_cores(b,seed=0): return fit_mps(tfim_ground(b,seed),seed=seed) if args.family=='tfim' else make_family(b)
def expect(psi,ops):
    Opsi=torch.einsum('kij,bj->bki',ops,psi); return torch.einsum('bi,bki->bk',psi.conj(),Opsi).real
def fidelity(a,b): return (torch.einsum('bi,bi->b',a.conj(),b).abs())**2
def meas_loss(psi,ops,obs): return (expect(psi/psi.norm(dim=1,keepdim=True),ops)-obs).pow(2).mean(1)

print("[prep] bank + flow ...",flush=True)
BANK=cat_cores(make_cores(args.bank,seed=0)); BM,BS=BANK.mean(0,keepdim=True),BANK.std(0,keepdim=True).clamp_min(1e-4); BANKn=(BANK-BM)/BS
sf=lambda t:(t-BM)/BS; uf=lambda t:t*BS+BM
BANK_PSI=cores_to_state(split_cores(uf(BANKn)))                # bank states (for medoid / NN)
class Vel(nn.Module):
    def __init__(s,d,h=1024):
        super().__init__(); s.t=nn.Sequential(nn.Linear(1,128),nn.SiLU(),nn.Linear(128,128)); s.n=nn.Sequential(nn.Linear(d+128,h),nn.SiLU(),nn.Linear(h,h),nn.SiLU(),nn.Linear(h,h),nn.SiLU(),nn.Linear(h,h),nn.SiLU(),nn.Linear(h,d))
    def forward(s,x,t): return s.n(torch.cat([x,s.t(t.view(-1,1))],1))
vel=Vel(THETA).to(dev); opt=torch.optim.AdamW(vel.parameters(),lr=2e-4); sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,args.steps)
for it in range(1,args.steps+1):
    th=BANKn[torch.randint(0,BANKn.shape[0],(256,),device=dev)]; eps=torch.randn_like(th); t=torch.rand(th.shape[0],device=dev)
    loss=F.mse_loss(vel((1-t).view(-1,1)*eps+t.view(-1,1)*th,t),th-eps); opt.zero_grad(); loss.backward(); opt.step(); sch.step()
print(f"[train] flow done {loss.item():.3f}",flush=True)
@torch.no_grad()
def prior_samples(b,ns=120):
    th=torch.randn(b,THETA,device=dev)
    for i in range(ns): th=th+(1.0/ns)*vel(th,torch.full((b,),i/ns,device=dev))
    return cores_to_state(canon(split_cores(uf(th))))
def prox(a,ops,obs,ni=None,lam=0.3,lr=0.1):
    ni=args.ni if ni is None else ni
    x=uf(a).clone().requires_grad_(True); op=torch.optim.Adam([x],lr=lr); pa=cores_to_state(split_cores(uf(a))).detach()
    for _ in range(ni):
        psi=cores_to_state(split_cores(x)); loss=meas_loss(psi,ops,obs).mean()+lam*(1-(torch.einsum('bi,bi->b',psi.conj(),pa).abs())**2).mean()
        op.zero_grad(); loss.backward(); op.step()
    with torch.no_grad(): return sf(cat_cores(canon(split_cores(x))))
def flow_recon(ops,obs,ns=None,bo=None):
    ns=args.ns if ns is None else ns; bo=args.bestof if bo is None else bo
    b=obs.shape[0]
    def one():
        th=torch.randn(b,THETA,device=dev)
        for i in range(ns):
            t=i/ns; t1=(i+1)/ns
            with torch.no_grad(): v=vel(th,torch.full((b,),t,device=dev))
            x0=th+(1-t)*v; x0=prox(x0,ops,obs); eps=(th-t*x0)/max(1-t,1e-3); th=((1-t1)*eps+t1*x0).clamp(-6,6)
        return cores_to_state(canon(split_cores(uf(th))))
    cand=torch.stack([one() for _ in range(bo)],0); mls=torch.stack([meas_loss(cand[j],ops,obs) for j in range(bo)],0)
    return cand[mls.argmin(0),torch.arange(b,device=dev)]

psi_true=cores_to_state(make_cores(args.ntest,seed=999))
OUT={'n':n,'family':args.family}
# ---- (d) nearest-neighbor fidelity distribution of test states to the bank ----
nn_fid=(psi_true@BANK_PSI.conj().T).abs()**2                       # (ntest, 6000)
nn_max=nn_fid.max(1).values
q=torch.quantile(nn_max,torch.tensor([0.1,0.5,0.9],device=dev))
OUT['nn_neighbor']={'mean':nn_max.mean().item(),'median':q[1].item(),'q10':q[0].item(),'q90':q[2].item()}
print(f"[d] test->nearest-bank-neighbor fidelity: mean {nn_max.mean():.3f} median {q[1]:.3f} (q10 {q[0]:.3f}, q90 {q[2]:.3f})",flush=True)
# ---- (a) k=0 prior-only baselines ----
rs=prior_samples(args.ntest)                                       # random prior sample per test (independent draw)
f_rand=fidelity(psi_true,rs).mean().item()
# medoid: bank state maximizing mean fidelity to all bank (family centroid-ish)
sub=BANK_PSI[:1500]; mfid=((sub@sub.conj().T).abs()**2).mean(1); medoid=sub[mfid.argmax():mfid.argmax()+1]
f_med=fidelity(psi_true,medoid.expand(args.ntest,-1)).mean().item()
# prior "mean": normalized mean of bank states (not a family member; a crude central estimate)
mean_state=BANK_PSI.mean(0,keepdim=True); mean_state=mean_state/mean_state.norm(); f_mean=fidelity(psi_true,mean_state.expand(args.ntest,-1)).mean().item()
OUT['prior_only']={'random_sample':f_rand,'medoid':f_med,'mean_state':f_mean}
print(f"[a] PRIOR-ONLY (k=0): random-sample {f_rand:.3f} | medoid {f_med:.3f} | mean-state {f_mean:.3f}",flush=True)
# ---- (b,c) true vs shuffled measurements, info gain vs k ----
perm=torch.randperm(args.ntest,device=dev)
KS=[8,16,32,64]; res={'k':KS,'true':[],'shuffled':[],'gain_over_prioronly':[],'true_minus_shuffled':[]}
best_prior=max(f_rand,f_med,f_mean)
for k in KS:
    ops=POOL[:k]; te=expect(psi_true,ops); sh=torch.sqrt(((1-te**2).clamp_min(0))/args.shots); obs=te+sh*torch.randn_like(te)
    f_true=fidelity(psi_true,flow_recon(ops,obs)).mean().item()
    # shuffled: feed measurements from a DIFFERENT state (permuted), reconstruct, compare to the true state
    obs_shuf=obs[perm]
    f_shuf=fidelity(psi_true,flow_recon(ops,obs_shuf)).mean().item()
    res['true'].append(f_true); res['shuffled'].append(f_shuf)
    res['gain_over_prioronly'].append(f_true-best_prior); res['true_minus_shuffled'].append(f_true-f_shuf)
    print(f"[b] k={k:>3}: TRUE-meas {f_true:.3f} | SHUFFLED-meas {f_shuf:.3f} | gain-over-prior-only {f_true-best_prior:+.3f} | true-shuffled {f_true-f_shuf:+.3f}",flush=True)
OUT['measurements']=res
json.dump(OUT,open(f'./qst_prioronly_{args.family}_n{n}.json','w'),indent=1); print("[dumped]",flush=True)
