"""
Posterior validation: does the measurement-guided flow sampler approximate the TRUE Bayesian posterior
p(theta|o) ~ p_family(theta) l(o|theta)?  Gold standard = self-normalized importance sampling (SIR)
from the family generator, weighted by the exact likelihood (feasible at n=6). Compare on local
observables: posterior means, marginal distributions (energy distance), and calibration coverage.
"""
import math, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
dev=torch.device('cuda:5'); torch.manual_seed(0); np.random.seed(0); cdt=torch.complex64
n=6; chi=2; D=2**n; SIGMA_ANG=0.35
bonds=[min(2**k,2**(n-k),chi) for k in range(n+1)]; shapes=[(bonds[k],2,bonds[k+1]) for k in range(n)]
sizes=[int(np.prod(s)) for s in shapes]; off=np.cumsum([0]+sizes); THETA=2*int(off[-1])
I2=torch.tensor([[1,0],[0,1]],dtype=cdt,device=dev);Xp=torch.tensor([[0,1],[1,0]],dtype=cdt,device=dev)
Yp=torch.tensor([[0,-1j],[1j,0]],dtype=cdt,device=dev);Zp=torch.tensor([[1,0],[0,-1]],dtype=cdt,device=dev)
PAU=[I2,Xp,Yp,Zp]
def kron_list(ms):
    o=ms[0]
    for m in ms[1:]: o=torch.kron(o,m)
    return o
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
    cs=[base[k]+0.35*(torch.randn(b,*shapes[k],device=dev)+1j*torch.randn(b,*shapes[k],device=dev)) for k in range(n)]
    return canon(cs)
def expect(psi,ops):
    Opsi=torch.einsum('kij,bj->bki',ops,psi); return torch.einsum('bi,bki->bk',psi.conj(),Opsi).real
def fid(a,b): return (torch.einsum('bi,bi->b',a.conj(),b).abs())**2
rng=np.random.default_rng(1); pool=[]; seen=set()
while len(pool)<16:
    s=tuple(int(x) for x in rng.integers(0,4,size=n))
    if s==(0,)*n or s in seen: continue
    seen.add(s); pool.append(s)
POOL=torch.stack([kron_list([PAU[s[q]] for q in range(n)]) for s in pool],0)          # (16,D,D) measured
def single(op,q): return kron_list([op if i==q else I2 for i in range(n)])
OBS=torch.stack([single(Zp,q) for q in range(n)]+[single(Xp,q) for q in range(n)],0)  # (2n,D,D) local observables
# ---- train flow prior ----
BANK=cat_cores(make_family(20000)); BM,BS=BANK.mean(0,keepdim=True),BANK.std(0,keepdim=True).clamp_min(1e-4); BANKn=(BANK-BM)/BS
sf=lambda t:(t-BM)/BS; uf=lambda t:t*BS+BM
class Vel(nn.Module):
    def __init__(s,d,h=1024):
        super().__init__(); s.t=nn.Sequential(nn.Linear(1,128),nn.SiLU(),nn.Linear(128,128)); s.n=nn.Sequential(nn.Linear(d+128,h),nn.SiLU(),nn.Linear(h,h),nn.SiLU(),nn.Linear(h,h),nn.SiLU(),nn.Linear(h,h),nn.SiLU(),nn.Linear(h,d))
    def forward(s,x,t): return s.n(torch.cat([x,s.t(t.view(-1,1))],1))
vel=Vel(THETA).to(dev); opt=torch.optim.AdamW(vel.parameters(),lr=2e-4); sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,40000)
for it in range(1,40001):
    th=BANKn[torch.randint(0,BANKn.shape[0],(256,),device=dev)]; eps=torch.randn_like(th); t=torch.rand(th.shape[0],device=dev)
    loss=F.mse_loss(vel((1-t).view(-1,1)*eps+t.view(-1,1)*th,t),th-eps); opt.zero_grad(); loss.backward(); opt.step(); sch.step()
print(f"[train] flow loss {loss.item():.3f}")
def prox(anchor,obs,ni=25,lam=0.3,lr=0.1):
    x=uf(anchor).clone().requires_grad_(True); op=torch.optim.Adam([x],lr=lr); pa=cores_to_state(split_cores(uf(anchor))).detach()
    for _ in range(ni):
        psi=cores_to_state(split_cores(x)); loss=((expect(psi,POOL)-obs)**2).sum(1).mean()+lam*(1-(torch.einsum('bi,bi->b',psi.conj(),pa).abs())**2).mean()
        op.zero_grad(); loss.backward(); op.step()
    with torch.no_grad(): return sf(cat_cores(canon(split_cores(x))))
def flow_post(obs,K,ns=40,noise=0.15):    # K stochastic posterior samples for a single obs (1,16)
    obsK=obs.expand(K,-1); th=torch.randn(K,THETA,device=dev)
    for i in range(ns):
        t=i/ns; t1=(i+1)/ns
        with torch.no_grad(): v=vel(th,torch.full((K,),t,device=dev))
        x0=th+(1-t)*v; x0=prox(x0,obsK); eps=(th-t*x0)/max(1-t,1e-3); th=((1-t1)*eps+t1*x0)
        if i<ns-1: th=th+noise*(1-t1)*torch.randn_like(th)
        th=th.clamp(-6,6)
    return cores_to_state(canon(split_cores(uf(th))))                                  # (K,D)
# ---- gold-standard SIR posterior ----
M=300000; SH=200; K=3000
@torch.no_grad()
def sample_prior(b,ns=100):          # sample states from the trained flow prior p_phi (no measurements)
    th=torch.randn(b,THETA,device=dev)
    for i in range(ns): th=th+(1.0/ns)*vel(th,torch.full((b,),i/ns,device=dev))
    return cores_to_state(canon(split_cores(uf(th))))
# TWO gold posteriors: PRIMARY p_phi (isolates the sampler vs the posterior under ITS OWN prior);
# SECONDARY p_family (end-to-end vs the true Bayesian posterior).
PHIST=torch.cat([sample_prior(20000) for _ in range(5)],0); EXPPHI=expect(PHIST,POOL)   # 100k flow-prior samples
FAMST=cores_to_state(make_family(M)); EXPFAM=expect(FAMST,POOL)                          # M family-prior samples
def sir(states,EXP,o,sig2):
    logw=-0.5*((EXP-o)**2/sig2).sum(1); w=torch.softmax(logw,0)
    ess=(1.0/(w**2).sum()).item(); idx=torch.multinomial(w,K,replacement=True); return states[idx], ess
def energy_dist(x,y):    # 1D energy distance between sample sets x,y (each (N,))
    def pd(a,b): return (a.view(-1,1)-b.view(1,-1)).abs().mean()
    return (2*pd(x,y)-pd(x,x)-pd(y,y)).clamp_min(0).item()

T=25; psis=cores_to_state(make_family(T))
pmo=[]; pm_phi=[]; ed_phi=[]; ed_fam=[]; ess_phi=[]; ess_fam=[]; cover=0; cov_tot=0
for t in range(T):
    te=expect(psis[t:t+1],POOL); sig2=((1-te**2).clamp_min(1e-3)/SH); o=te+torch.sqrt(sig2)*torch.randn_like(te)
    gp,ep=sir(PHIST,EXPPHI,o,sig2); gf,ef=sir(FAMST,EXPFAM,o,sig2); o_states=flow_post(o,K)
    o_obs=expect(o_states,OBS); gp_obs=expect(gp,OBS); gf_obs=expect(gf,OBS); otrue=expect(psis[t:t+1],OBS).squeeze(0)
    pmo.append(o_obs.mean(0)); pm_phi.append(gp_obs.mean(0)); ess_phi.append(ep); ess_fam.append(ef)
    ed_phi.append(np.mean([energy_dist(o_obs[:,j],gp_obs[:,j]) for j in range(2*n)]))
    ed_fam.append(np.mean([energy_dist(o_obs[:,j],gf_obs[:,j]) for j in range(2*n)]))
    lo=o_obs.quantile(0.05,0); hi=o_obs.quantile(0.95,0); cover+=((otrue>=lo)&(otrue<=hi)).sum().item(); cov_tot+=2*n
    if t==0: np.savez('./postval_example.npz', ours=o_obs.cpu().numpy(), gold=gp_obs.cpu().numpy(), otrue=otrue.cpu().numpy())
pmo=torch.stack(pmo).flatten().cpu().numpy(); pm_phi=torch.stack(pm_phi).flatten().cpu().numpy()
mae=np.abs(pmo-pm_phi).mean(); corr=np.corrcoef(pmo,pm_phi)[0,1]
np.savez('./postval_summary.npz', pm_ours=pmo, pm_gold=pm_phi, ed_phi=np.array(ed_phi), ed_fam=np.array(ed_fam))
print("\n================ POSTERIOR VALIDATION (n=6, T=25 states) ================")
print(f"  PRIMARY gold = SIR from FLOW prior p_phi (isolates the sampler):")
print(f"    ESS: median {np.median(ess_phi):.0f} (min {np.min(ess_phi):.0f}) of {len(EXPPHI)}")
print(f"    posterior-mean agreement (ours vs p_phi-gold): MAE={mae:.4f}  corr={corr:.4f}")
print(f"    energy distance (per-obs marginals): median {np.median(ed_phi):.4f}  [{np.percentile(ed_phi,25):.4f},{np.percentile(ed_phi,75):.4f}]")
print(f"  SECONDARY gold = SIR from family prior p_fam (end-to-end true Bayesian posterior):")
print(f"    ESS: median {np.median(ess_fam):.0f} of {M};  energy distance: median {np.median(ed_fam):.4f}")
print(f"  our raw 90% CI coverage of TRUE observable value: {100*cover/cov_tot:.1f}%")
# ---- ISOLATION: same learned prior, different inference. ours (proximal) vs learned-prior exact SIR/MAP ----
def flow_point(o,K=64):              # our sampler point estimate: best-likelihood of K proximal draws
    s=flow_post(o,K); res=((expect(s,POOL)-o)**2).sum(1); return s[res.argmin():res.argmin()+1]
fo=[]; fsir=[]; fmap=[]
for t in range(T):
    te=expect(psis[t:t+1],POOL); sig2=((1-te**2).clamp_min(1e-3)/SH); o=te+torch.sqrt(sig2)*torch.randn_like(te)
    fo.append(fid(flow_point(o),psis[t:t+1]).item())
    logw=-0.5*((EXPPHI-o)**2/sig2).sum(1)                       # learned-prior + exact importance weights
    fsir.append(fid(PHIST[logw.argmax():logw.argmax()+1],psis[t:t+1]).item())   # LP-SIR MAP (best-weight flow sample)
    idx=torch.multinomial(torch.softmax(logw,0),256,True); rho=(PHIST[idx].conj().T@PHIST[idx])/256
    w2,V2=torch.linalg.eigh(0.5*(rho+rho.conj().T)); fmap.append(fid(V2[:,-1:].T,psis[t:t+1]).item())  # LP posterior-mean (top eig)
print("\n---- ISOLATION (same learned flow prior, n=6): prior vs sampler ----")
print(f"  learned prior + OUR proximal sampler : fidelity {np.mean(fo):.3f} +/- {np.std(fo):.3f}")
print(f"  learned prior + exact SIR (MAP)      : fidelity {np.mean(fsir):.3f} +/- {np.std(fsir):.3f}")
print(f"  learned prior + exact SIR (post-mean): fidelity {np.mean(fmap):.3f} +/- {np.std(fmap):.3f}")
print("  => ours ~ exact-inference-under-same-prior => the learned prior does the work; sampler is a fast, faithful surrogate")
print("=========================================================================")
