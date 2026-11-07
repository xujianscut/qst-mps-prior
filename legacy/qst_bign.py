"""Larger-n scaling: random-MPS family with EFFICIENT Pauli-string expectation (no dense 2^n x 2^n
matrices), demonstrating the core parameterization scales where dense methods are infeasible."""
import argparse, math, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
ap=argparse.ArgumentParser(); ap.add_argument('--dev',default='cuda:5'); ap.add_argument('--n',type=int,default=12)
ap.add_argument('--chi',type=int,default=2); ap.add_argument('--sigma',type=float,default=0.35)
ap.add_argument('--shots',type=int,default=200); ap.add_argument('--steps',type=int,default=40000); ap.add_argument('--ntest',type=int,default=80)
args=ap.parse_args(); dev=torch.device(args.dev); torch.manual_seed(0); np.random.seed(0); cdt=torch.complex64
n,chi,D=args.n,args.chi,2**args.n
bonds=[min(2**k,2**(n-k),chi) for k in range(n+1)]; shapes=[(bonds[k],2,bonds[k+1]) for k in range(n)]
sizes=[int(np.prod(s)) for s in shapes]; off=np.cumsum([0]+sizes); THETA=2*int(off[-1])
print(f"[cfg] n={n} chi={chi} core-params(complex)={int(off[-1])} theta={THETA} vs dense-Pauli-matrix={D}^2={D*D}")
PA=[torch.tensor(m,dtype=cdt,device=dev) for m in ([[1,0],[0,1]],[[0,1],[1,0]],[[0,-1j],[1j,0]],[[1,0],[0,-1]])]
def split_cores(th):
    b=th.shape[0]; c=torch.complex(th[:,:THETA//2],th[:,THETA//2:]); return [c[:,off[k]:off[k+1]].reshape(b,*shapes[k]) for k in range(n)]
def cat_cores(cs):
    b=cs[0].shape[0]; c=torch.cat([g.reshape(b,-1) for g in cs],1); return torch.cat([c.real,c.imag],1)
def cores_to_state(cs):
    b=cs[0].shape[0]; psi=cs[0].reshape(b,2,bonds[1])
    for k in range(1,n): psi=torch.einsum('bar,brjs->bajs',psi,cs[k].reshape(b,bonds[k],2,bonds[k+1])).reshape(b,2**(k+1),bonds[k+1])
    return (psi.reshape(b,D)/psi.reshape(b,D).norm(dim=1,keepdim=True).clamp_min(1e-9))
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
    cs=[base[k]+args.sigma*(torch.randn(b,*shapes[k],device=dev)+1j*torch.randn(b,*shapes[k],device=dev)) for k in range(n)]
    return canon(cs)
def fid(a,b): return (torch.einsum('bi,bi->b',a.conj(),b).abs())**2
# ---- efficient Pauli-string expectation: apply single-qubit Paulis, no dense 2^n matrices ----
def apply_ps(psi,s):
    b=psi.shape[0]; out=psi
    for q in range(n):
        if s[q]==0: continue
        st=out.view(b,2**q,2,2**(n-q-1)); out=torch.einsum('ij,bajc->baic',PA[s[q]],st).reshape(b,D)
    return out
def pexp(psi,pool): return torch.stack([(psi.conj()*apply_ps(psi,pool[j])).sum(1).real for j in range(len(pool))],1)
rng=np.random.default_rng(1); POOL=[]; seen=set()
while len(POOL)<64:
    s=tuple(int(x) for x in rng.integers(0,4,size=n))
    if s==(0,)*n or s in seen: continue
    seen.add(s); POOL.append(s)
# ---- flow prior ----
BANK=cat_cores(make_family(20000)); BM,BS=BANK.mean(0,keepdim=True),BANK.std(0,keepdim=True).clamp_min(1e-4); BANKn=(BANK-BM)/BS
sf=lambda t:(t-BM)/BS; uf=lambda t:t*BS+BM
class Vel(nn.Module):
    def __init__(s,d,h=1024):
        super().__init__(); s.t=nn.Sequential(nn.Linear(1,128),nn.SiLU(),nn.Linear(128,128)); s.n=nn.Sequential(nn.Linear(d+128,h),nn.SiLU(),nn.Linear(h,h),nn.SiLU(),nn.Linear(h,h),nn.SiLU(),nn.Linear(h,h),nn.SiLU(),nn.Linear(h,d))
    def forward(s,x,t): return s.n(torch.cat([x,s.t(t.view(-1,1))],1))
vel=Vel(THETA).to(dev); opt=torch.optim.AdamW(vel.parameters(),lr=2e-4); sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,args.steps)
import time; t0=time.time()
for it in range(1,args.steps+1):
    th=BANKn[torch.randint(0,BANKn.shape[0],(256,),device=dev)]; eps=torch.randn_like(th); t=torch.rand(th.shape[0],device=dev)
    loss=F.mse_loss(vel((1-t).view(-1,1)*eps+t.view(-1,1)*th,t),th-eps); opt.zero_grad(); loss.backward(); opt.step(); sch.step()
print(f"[train] done {time.time()-t0:.0f}s loss {loss.item():.3f}")
def prox(anchor,obs,pool,ni,lam,lr=0.1):
    x=uf(anchor).clone().requires_grad_(True); op=torch.optim.Adam([x],lr=lr); pa=cores_to_state(split_cores(uf(anchor))).detach()
    for _ in range(ni):
        psi=cores_to_state(split_cores(x)); loss=((pexp(psi,pool)-obs)**2).sum(1).mean()+lam*(1-(torch.einsum('bi,bi->b',psi.conj(),pa).abs())**2).mean()
        op.zero_grad(); loss.backward(); op.step()
    with torch.no_grad(): return sf(cat_cores(canon(split_cores(x))))
def flow_recon(obs,pool,ns=40,lam=0.3,ni=25):
    b=obs.shape[0]; th=torch.randn(b,THETA,device=dev)
    for i in range(ns):
        t=i/ns; t1=(i+1)/ns
        with torch.no_grad(): v=vel(th,torch.full((b,),t,device=dev))
        x0=th+(1-t)*v; x0=prox(x0,obs,pool,ni,lam); eps=(th-t*x0)/max(1-t,1e-3); th=((1-t1)*eps+t1*x0).clamp(-6,6)
    return cores_to_state(canon(split_cores(uf(th))))
def mle(obs,pool,restarts=8,iters=500,lr=0.1):
    b=obs.shape[0]; best=None; be=None
    for _ in range(restarts):
        x=(0.3*torch.randn(b,THETA,device=dev)).requires_grad_(True); op=torch.optim.Adam([x],lr=lr)
        for _ in range(iters):
            loss=((pexp(cores_to_state(split_cores(x)),pool)-obs)**2).sum(1).mean(); op.zero_grad(); loss.backward(); op.step()
        with torch.no_grad():
            e=((pexp(cores_to_state(split_cores(x)),pool)-obs)**2).sum(1)
            if best is None: best,be=x.detach().clone(),e
            else: sw=e<be; best[sw]=x.detach()[sw]; be=torch.where(sw,e,be)
    return cores_to_state(split_cores(best))
psi_true=cores_to_state(make_family(args.ntest))
print(f"\n============ LARGER-n (n={n}, random family, efficient Pauli) ============")
for k in [16,32,64]:
    pool=POOL[:k]; te=pexp(psi_true,pool); sh=torch.sqrt(((1-te**2).clamp_min(0))/args.shots); obs=te+sh*torch.randn_like(te)
    ff=fid(flow_recon(obs,pool),psi_true).mean().item(); fm=fid(mle(obs,pool),psi_true).mean().item()
    print(f"  k={k:3d} | MPS-flow(ours) {ff:.4f} | MPS-MLE {fm:.4f}")
print(f"[note] core params/state = {int(off[-1])} complex; dense Pauli matrix would be {D}x{D} (infeasible to store {2*D*D*8/1e9:.1f}GB x k)")
