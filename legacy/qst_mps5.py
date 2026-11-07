"""
#29 MPS-core-flow QST -- integrated build (items 1+2+3).
  1) gauge-canonical MPS cores + flow prior over cores (scalable: O(n*chi^2) params vs 2^n).
  2) baselines: MPS-chi-MLE (the 'TT-CS'-like structural prior), pure-state MLE, oracle, NN-prior.
  3) calibration: coverage of posterior credible intervals on observables (SBI/NPE-style).
Key test: does the MPS-core flow beat MPS-chi-MLE at FEW measurements at n where FULL-STATE flow underfit (n=6)?
Family = narrow structured low-bond-dim states (fixed base cores + perturbation), canonicalized.
"""
import time, argparse, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
ap=argparse.ArgumentParser()
ap.add_argument('--dev',default='cuda:8'); ap.add_argument('--n',type=int,default=6); ap.add_argument('--chi',type=int,default=2)
ap.add_argument('--sigma',type=float,default=0.35); ap.add_argument('--shots',type=int,default=200)
ap.add_argument('--steps',type=int,default=40000); ap.add_argument('--bs',type=int,default=256); ap.add_argument('--ntest',type=int,default=120)
ap.add_argument('--nsamp',type=int,default=24); ap.add_argument('--family',default='random',choices=['random','tfim'])
ap.add_argument('--bestof',type=int,default=8); ap.add_argument('--bank',type=int,default=30000)
ap.add_argument('--dump_proj',action='store_true'); ap.add_argument('--critical',action='store_true')
args=ap.parse_args()
dev=torch.device(args.dev); torch.manual_seed(0); np.random.seed(0); cdt=torch.complex64
n,chi,D=args.n,args.chi,2**args.n
# bond dims: proper staircase capped at chi (boundary bonds <= 2^k), consistent with left-canonical form
bonds=[min(2**k, 2**(n-k), chi) for k in range(n+1)]
shapes=[(bonds[k],2,bonds[k+1]) for k in range(n)]
sizes=[np.prod(s) for s in shapes]; off=np.cumsum([0]+sizes); THETA=2*int(off[-1])   # real dim
print(f"[cfg-mps] n={n} chi={chi} dim={D} core-params(complex)={int(off[-1])} theta(real)={THETA} vs full-state real={2*D}")

# ---- Pauli pool ----
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

# ---- MPS cores <-> theta, contraction, canonicalization ----
def split_cores(th):                 # th (b,THETA) real -> list of complex cores
    b=th.shape[0]; c=torch.complex(th[:,:THETA//2], th[:,THETA//2:])
    return [c[:,off[k]:off[k+1]].reshape(b,*shapes[k]) for k in range(n)]
def cat_cores(cores):
    b=cores[0].shape[0]; c=torch.cat([g.reshape(b,-1) for g in cores],1)
    return torch.cat([c.real,c.imag],1)
def cores_to_state(cores):           # contract MPS -> (b,D) normalized
    b=cores[0].shape[0]; psi=cores[0].reshape(b,2,bonds[1])          # (b,2,chi)
    for k in range(1,n):
        psi=torch.einsum('bar,brjs->bajs', psi, cores[k].reshape(b,bonds[k],2,bonds[k+1])).reshape(b,2**(k+1),bonds[k+1])
    psi=psi.reshape(b,D)
    return psi/psi.norm(dim=1,keepdim=True).clamp_min(1e-9)
def canon(cores):                    # left-canonical sweep + global phase fix (gauge fixing)
    b=cores[0].shape[0]; cs=[g.clone() for g in cores]
    for k in range(n-1):
        M=cs[k].reshape(b,bonds[k]*2,bonds[k+1])
        Q,Rm=torch.linalg.qr(M)                                     # Q:(b,bk*2,r) R:(b,r,bk+1)
        # sign-fix: make R diagonal real positive (removes residual gauge)
        d=torch.diagonal(Rm,dim1=-2,dim2=-1); ph=(d/d.abs().clamp_min(1e-9)).conj()
        Q=Q*ph.unsqueeze(1); Rm=ph.unsqueeze(2)*Rm
        cs[k]=Q.reshape(b,bonds[k],2,bonds[k+1])
        cs[k+1]=torch.einsum('brs,bsjt->brjt', Rm, cs[k+1].reshape(b,bonds[k+1],2,bonds[k+2])).reshape(b,*shapes[k+1])
    # normalize last core (left cores are isometries => ||state||=||last core||_F): removes gauge scale pathology
    nrm=cs[-1].reshape(b,-1).norm(dim=1).clamp_min(1e-9); cs[-1]=cs[-1]/nrm.view(-1,1,1,1)
    st=cores_to_state(cs); ph=(st[:,0]/st[:,0].abs().clamp_min(1e-9)).conj()   # global phase
    cs[-1]=cs[-1]*ph.view(-1,1,1,1)
    return cs
def make_family(b):                  # narrow structured family: fixed base cores + perturbation
    g=torch.Generator(device=dev).manual_seed(12345)
    base=[torch.randn(1,*s,generator=g,device=dev)+1j*torch.randn(1,*s,generator=g,device=dev) for s in shapes]
    cores=[(base[k]+args.sigma*(torch.randn(b,*shapes[k],device=dev)+1j*torch.randn(b,*shapes[k],device=dev))) for k in range(n)]
    return canon(cores)

# ---- physical family: disordered TFIM ground states, compressed to chi-MPS ----
def _single(op,q): return kron_list([op if i==q else I2 for i in range(n)])
ZZ_OPS=torch.stack([_single(Zp,q)@_single(Zp,q+1) for q in range(n-1)],0) if n>1 else None  # (n-1,D,D)
X_OPS=torch.stack([_single(Xp,q) for q in range(n)],0)                                         # (n,D,D)
def tfim_ground(b):                  # disordered TFIM ground states; --critical => near-critical (higher entanglement)
    if args.critical: J=0.8+0.4*torch.rand(b,n-1,device=dev); h=0.8+0.4*torch.rand(b,n,device=dev)   # J~h~1 critical
    else: J=0.2+0.4*torch.rand(b,n-1,device=dev); h=1.0+1.0*torch.rand(b,n,device=dev)                # paramagnetic
    ch=8 if D>=1024 else 24; psis=[]
    for s in range(0,b,ch):
        with torch.no_grad():
            jj=J[s:s+ch].to(cdt); hh=h[s:s+ch].to(cdt)
            H=-torch.einsum('bi,ijk->bjk',jj,ZZ_OPS)-torch.einsum('bi,ijk->bjk',hh,X_OPS)
            w,V=torch.linalg.eigh(0.5*(H+H.conj().transpose(-1,-2))); psis.append(V[:,:,0].clone())   # ground state
        del H,w,V; torch.cuda.empty_cache()          # avoid fragmentation leak across many chunks (n=10)
    return torch.cat(psis,0)
def fit_mps_to_state(psi_t,iters=300,lr=0.05,restarts=2):   # optimize chi-MPS cores to match a target state
    b=psi_t.shape[0]; best=None; beste=None
    for _ in range(restarts):
        x=(0.1*torch.randn(b,THETA,device=dev)).requires_grad_(True); op=torch.optim.Adam([x],lr=lr)
        for _ in range(iters):
            psi=cores_to_state(split_cores(x))
            loss=(1-(torch.einsum('bi,bi->b',psi.conj(),psi_t).abs())**2).mean()
            op.zero_grad(); loss.backward(); op.step()
        with torch.no_grad():
            e=1-(torch.einsum('bi,bi->b',cores_to_state(split_cores(x)).conj(),psi_t).abs())**2
            if best is None: best,beste=x.detach().clone(),e
            else: sw=e<beste; best[sw]=x.detach()[sw]; beste=torch.where(sw,e,beste)
    return canon(split_cores(best))
def make_states(b):                  # dispatch: random-MPS family or physical TFIM family
    if args.family=='tfim': return fit_mps_to_state(tfim_ground(b))
    return make_family(b)

def expect(psi,ops):                 # <psi|P_k|psi> without materializing (b,k,D,D): two-step contraction
    Opsi=torch.einsum('kij,bj->bki', ops, psi)      # (b,k,D)
    return torch.einsum('bi,bki->bk', psi.conj(), Opsi).real
def fidelity(a,b): return (torch.einsum('bi,bi->b',a.conj(),b).abs())**2

# ---- flow prior over canonical cores ----
print(f"[prep] state bank (family={args.family}) ...")
NB=6000 if args.family=='tfim' else args.bank
BANK=cat_cores(make_states(NB)); BM,BS=BANK.mean(0,keepdim=True),BANK.std(0,keepdim=True).clamp_min(1e-4); BANKn=(BANK-BM)/BS
sf=lambda t:(t-BM)/BS; uf=lambda t:t*BS+BM
# sanity: is family genuinely low-chi & narrow? oracle chi-MLE gap and family spread
fam_state=cores_to_state(split_cores(uf(BANKn[:500])))
print(f"[bank] {BANK.shape} theta-std {BS.min():.3f}..{BS.max():.3f}")

class Vel(nn.Module):
    def __init__(s,d,h=1024):
        super().__init__(); s.t=nn.Sequential(nn.Linear(1,128),nn.SiLU(),nn.Linear(128,128))
        s.n=nn.Sequential(nn.Linear(d+128,h),nn.SiLU(),nn.Linear(h,h),nn.SiLU(),nn.Linear(h,h),nn.SiLU(),nn.Linear(h,h),nn.SiLU(),nn.Linear(h,d))
    def forward(s,x,t): return s.n(torch.cat([x,s.t(t.view(-1,1))],1))
vel=Vel(THETA).to(dev); opt=torch.optim.AdamW(vel.parameters(),lr=2e-4); sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,args.steps)
print("[train] MPS-core flow prior ..."); t0=time.time()
for it in range(1,args.steps+1):
    th=BANKn[torch.randint(0,BANKn.shape[0],(args.bs,),device=dev)]; eps=torch.randn_like(th); t=torch.rand(th.shape[0],device=dev)
    loss=F.mse_loss(vel((1-t).view(-1,1)*eps+t.view(-1,1)*th,t),th-eps); opt.zero_grad(); loss.backward(); opt.step(); sch.step()
    if it%10000==0: print(f"   it{it} loss {loss.item():.4f} ({time.time()-t0:.0f}s)")

@torch.no_grad()
def sample_prior(b,ns=120):
    th=torch.randn(b,THETA,device=dev)
    for i in range(ns): th=th+(1.0/ns)*vel(th,torch.full((b,),i/ns,device=dev))
    return cores_to_state(canon(split_cores(uf(th)))), uf(th)
ps,_=sample_prior(500); fam=cores_to_state(make_states(2000))
prior_fid=((ps@fam.conj().T).abs()**2).max(1).values.mean().item()
print(f"[prior sanity] MPS-core flow nearest-family fidelity {prior_fid:.4f} (full-state flow got 0.44 at n=6)")
BANK_PSI=cores_to_state(make_states(4000))

# ---- reconstruction ----
def meas_loss(psi,ops,obs): return (expect(psi/psi.norm(dim=1,keepdim=True),ops)-obs).pow(2).mean(1)
def prox_cores(anchor_std,ops,obs,n_inner,lam,lr=0.1):   # optimize raw cores: fit + state-space prox to anchor
    x=uf(anchor_std).clone().requires_grad_(True); op=torch.optim.Adam([x],lr=lr)
    psi_a=cores_to_state(split_cores(uf(anchor_std))).detach()
    for _ in range(n_inner):
        psi=cores_to_state(split_cores(x))
        loss=meas_loss(psi,ops,obs).mean()+lam*(1-(torch.einsum('bi,bi->b',psi.conj(),psi_a).abs())**2).mean()
        op.zero_grad(); loss.backward(); op.step()
    with torch.no_grad(): return sf(cat_cores(canon(split_cores(x))))
def flow_recon(ops,obs,ns=40,lam=0.3,n_inner=25,nsamp=1,noise=0.0):
    b=obs.shape[0]
    def one():
        th=torch.randn(b,THETA,device=dev)
        for i in range(ns):
            t=i/ns; t1=(i+1)/ns
            with torch.no_grad(): v=vel(th,torch.full((b,),t,device=dev))
            x0=th+(1-t)*v; x0=prox_cores(x0,ops,obs,n_inner,lam); eps=(th-t*x0)/max(1-t,1e-3)
            th=((1-t1)*eps+t1*x0)
            if noise>0 and i<ns-1: th=th+noise*(1-t1)*torch.randn_like(th)   # stochastic posterior spread
            th=th.clamp(-6,6)
        return cores_to_state(canon(split_cores(uf(th))))
    if nsamp>1: return torch.stack([one() for _ in range(nsamp)],0)              # (nsamp,b,D) posterior samples
    bo=max(1,args.bestof)
    if bo==1: return one()
    cand=torch.stack([one() for _ in range(bo)],0)                              # (bo,b,D): best-of-N by measurement fit
    ml=torch.stack([meas_loss(cand[j],ops,obs) for j in range(bo)],0)           # (bo,b)
    sel=ml.argmin(0)                                                            # (b,)
    return cand[sel,torch.arange(b,device=dev)]
def mps_mle(ops,obs,restarts=12,iters=700,lr=0.1,return_ens=False):  # STRENGTHENED chi-MPS fit (fair TT-CS-like baseline)
    b=obs.shape[0]; best=None; beste=None; ens=[]
    for _ in range(restarts):
        x=(0.3*torch.randn(b,THETA,device=dev)).requires_grad_(True); op=torch.optim.Adam([x],lr=lr)
        for _ in range(iters):
            loss=meas_loss(cores_to_state(split_cores(x)),ops,obs).mean(); op.zero_grad(); loss.backward(); op.step()
        with torch.no_grad():
            st=cores_to_state(split_cores(x)); e=meas_loss(st,ops,obs); ens.append(st)
            if best is None: best,beste=x.detach().clone(),e
            else: sw=e<beste; best[sw]=x.detach()[sw]; beste=torch.where(sw,e,beste)
    stbest=cores_to_state(split_cores(best))
    return (stbest, torch.stack(ens,0)) if return_ens else stbest
def shadow_recon(ops,obs):        # linear inversion (classical-shadow-style) -> nearest physical pure
    b=obs.shape[0]
    rho=(torch.eye(D,dtype=cdt,device=dev).unsqueeze(0).expand(b,-1,-1)/D
         + torch.einsum('bk,kij->bij', obs.to(cdt), ops)/D)
    w,V=torch.linalg.eigh(0.5*(rho+rho.conj().transpose(-1,-2))); return V[:,:,-1]
def pure_mle(ops,obs,restarts=6,iters=400,lr=0.1):       # full rank-1 (unstructured pure) baseline
    b=obs.shape[0]; best=None; beste=None
    for _ in range(restarts):
        x=(0.3*torch.randn(b,2*D,device=dev)).requires_grad_(True); op=torch.optim.Adam([x],lr=lr)
        for _ in range(iters):
            psi=torch.complex(x[:,:D],x[:,D:]); loss=meas_loss(psi,ops,obs).mean(); op.zero_grad(); loss.backward(); op.step()
        with torch.no_grad():
            psi=torch.complex(x[:,:D],x[:,D:]); e=meas_loss(psi,ops,obs)
            if best is None: best,beste=x.detach().clone(),e
            else: sw=e<beste; best[sw]=x.detach()[sw]; beste=torch.where(sw,e,beste)
    psi=torch.complex(best[:,:D],best[:,D:]); return psi/psi.norm(dim=1,keepdim=True)
def _nn(ops,obs,idx):                                    # NN-retrieval prior: nearest bank state's cores as anchor
    anchor=sf(BANK[idx]); return cores_to_state(split_cores(uf(prox_cores(anchor,ops,obs,120,0.3))))

import math
def pcn_bayes(ops,obs,n_iter=2000,burn=800,beta=0.12,keep=24):
    # SMC/pCN Bayesian QST: generic N(0,I) prior over chi-MPS cores, pCN-MCMC posterior (Lukens 2020-style).
    b=obs.shape[0]; te=obs; sig2=((1-te**2).clamp_min(1e-3)/args.shots).clamp_min(1e-4)   # per-Pauli shot variance
    def loglik(t):
        r=expect(cores_to_state(split_cores(t)),ops)-obs; return -0.5*(r**2/sig2).sum(1)
    th=torch.randn(b,THETA,device=dev); ll=loglik(th); S=[]; LLs=[]; a=1.0-beta**2
    step=max(1,(n_iter-burn)//keep)
    for it in range(n_iter):
        prop=math.sqrt(a)*th+beta*torch.randn_like(th); llp=loglik(prop)   # pCN proposal preserves prior
        acc=torch.rand(b,device=dev).log()<(llp-ll)
        th=torch.where(acc.unsqueeze(1),prop,th); ll=torch.where(acc,llp,ll)
        if it>=burn and (it-burn)%step==0: S.append(cores_to_state(split_cores(th))); LLs.append(ll.clone())
    S=torch.stack(S,0); LLs=torch.stack(LLs,0)                             # (keep,b,D),(keep,b)
    est=S[LLs.argmax(0),torch.arange(b,device=dev)]                       # MAP-ish point estimate
    return est,S
def shadows_proper(psi_t,n_shadows):
    # standard randomized classical shadows: single-qubit random Pauli-basis snapshots (Huang 2020). Small n only.
    b=psi_t.shape[0]
    Hd=torch.tensor([[1,1],[1,-1]],dtype=cdt,device=dev)/math.sqrt(2)
    Sd=torch.tensor([[1,0],[0,-1j]],dtype=cdt,device=dev); I1=torch.eye(2,dtype=cdt,device=dev)
    Uall=torch.stack([I1,Hd,Hd@Sd],0)                                    # (3,2,2) for Z,X,Y measurement
    rho=torch.zeros(b,D,D,dtype=cdt,device=dev)
    for _ in range(n_shadows):
        bas=torch.randint(0,3,(b,n),device=dev)                         # (b,n)
        psi=psi_t.clone()
        for q in range(n):
            Uq=Uall[bas[:,q]]                                           # (b,2,2) vectorized
            st=psi.view(b,2**q,2,2**(n-q-1)); psi=torch.einsum('bij,bajc->baic',Uq,st).reshape(b,D)
        s=torch.multinomial((psi.abs()**2),1).squeeze(1)                # sampled comp-basis outcome
        snap=torch.ones(b,1,1,dtype=cdt,device=dev)
        for q in range(n):
            sq=(s>>(n-1-q))&1; Uq=Uall[bas[:,q]]
            e=torch.zeros(b,2,dtype=cdt,device=dev); e[torch.arange(b,device=dev),sq]=1.0
            ket=torch.einsum('bji,bj->bi',Uq.conj(),e).unsqueeze(2)     # U_q^dag |s_q>
            proj=3*ket@ket.conj().transpose(1,2)-I1
            snap=torch.einsum('bij,bkl->bikjl',snap,proj).reshape(b,snap.shape[1]*2,snap.shape[2]*2)
        rho=rho+snap
    rho=rho/n_shadows; w,V=torch.linalg.eigh(0.5*(rho+rho.conj().transpose(-1,-2))); return V[:,:,-1]

# ---- experiment: sweep measurement budget (with strengthened baselines) ----
psi_true=cores_to_state(make_states(args.ntest))
print(f"\n================ MPS-core QST (n={n},chi={chi}) fidelity, higher=better ================")
print(f"{'k':>5} | {'MPS-flow':>9} | {'pCN-Bayes':>9} | {'MPS-MLE*':>9} | {'shadow':>8} | {'NN(ceil)':>9}")
KS=[8,16,32,64]; RES={'n':n,'chi':chi,'family':args.family,'bestof':args.bestof,'core_params':int(off[-1]),'full_real':2*D,'prior_fid':prior_fid,'k':KS,
     'flow':[],'nn':[],'mps_mle':[],'pure_mle':[],'shadow':[],'pcn':[]}
for k in KS:
    ops=POOL[:k]; te=expect(psi_true,ops); sh=torch.sqrt(((1-te**2).clamp_min(0))/args.shots); obs=te+sh*torch.randn_like(te)
    f_flow=fidelity(psi_true,flow_recon(ops,obs)).mean().item()
    f_nn=fidelity(psi_true,_nn(ops,obs,torch.cdist(obs,expect(BANK_PSI,ops)).argmin(1))).mean().item()
    f_mps=fidelity(psi_true,mps_mle(ops,obs)).mean().item()
    f_pure=fidelity(psi_true,pure_mle(ops,obs)).mean().item()
    f_sh=fidelity(psi_true,shadow_recon(ops,obs)).mean().item()
    f_pcn=fidelity(psi_true,pcn_bayes(ops,obs)[0]).mean().item()
    RES['flow'].append(f_flow); RES['nn'].append(f_nn); RES['mps_mle'].append(f_mps); RES['pure_mle'].append(f_pure); RES['shadow'].append(f_sh); RES['pcn'].append(f_pcn)
    print(f"{k:>5} | {f_flow:>9.4f} | {f_pcn:>9.4f} | {f_mps:>9.4f} | {f_sh:>8.4f} | {f_nn:>9.4f}")
# best-of-N ablation (Prop 5) at k=16
ops=POOL[:16]; te=expect(psi_true,ops); sh=torch.sqrt(((1-te**2).clamp_min(0))/args.shots); obs16=te+sh*torch.randn_like(te)
Nsweep=[1,2,4,8,16,32]; bo_fid=[]
_orig_bo=args.bestof
for Nb in Nsweep:
    args.bestof=Nb; bo_fid.append(fidelity(psi_true,flow_recon(ops,obs16)).mean().item())
args.bestof=_orig_bo
RES['bestof_sweep']={'N':Nsweep,'fid':bo_fid}
print(f"[best-of-N sweep @k=16] N={Nsweep} -> fid={[round(x,3) for x in bo_fid]}")
# proper classical shadows (small n only; matched shot budget)
if n<=6:
    budget=1600; sub=psi_true[:40]
    f_shp=fidelity(sub,shadows_proper(sub,budget)).mean().item()
    RES['shadows_proper']={'budget':budget,'fid':f_shp}
    print(f"[classical shadows (proper randomized, {budget} shots, 40 states)] fid={f_shp:.4f}")

# ---- CALIBRATION: conformal recalibration + sharpness vs MLE-ensemble uncertainty ----
# observables: all single-qubit Z and X (local, informative)
def single(op,q): return kron_list([op if i==q else I2 for i in range(n)])
OBS=torch.stack([single(Zp,q) for q in range(n)]+[single(Xp,q) for q in range(n)],0)   # (2n,D,D)
cal_k=32; ops=POOL[:cal_k]; te=expect(psi_true,ops); sh=torch.sqrt(((1-te**2).clamp_min(0))/args.shots); obs=te+sh*torch.randn_like(te)
K=args.nsamp
flow_s=flow_recon(ops,obs,ns=30,nsamp=K,noise=0.15)                 # (K,ntest,D) stochastic posterior
_,mle_ens=mps_mle(ops,obs,restarts=K,iters=500,return_ens=True)     # (K,ntest,D) frequentist ensemble
_,pcn_s=pcn_bayes(ops,obs,keep=K)                                   # (K,ntest,D) pCN Bayesian posterior samples
otrue=expect(psi_true,OBS)                                          # (ntest,2n)
def obs_stats(samps):                                               # -> mean,std per (state,observable)
    e=torch.stack([expect(samps[j],OBS) for j in range(samps.shape[0])],0)  # (K,ntest,2n)
    return e.mean(0), e.std(0).clamp_min(1e-4)
fm,fs=obs_stats(flow_s); mm,ms=obs_stats(mle_ens); pm,ps=obs_stats(pcn_s)
nt=args.ntest; half=nt//2; idx=torch.randperm(nt,device=dev); ci,ti=idx[:half],idx[half:]   # calib/test split
def conformal_eval(mean,std,alpha=0.1):
    r=( (otrue-mean).abs()/std )                                    # nonconformity (ntest,2n)
    Q=torch.quantile(r[ci].flatten(), 1-alpha)                      # calibrated multiplier on CALIB
    lo=mean[ti]-Q*std[ti]; hi=mean[ti]+Q*std[ti]
    covered=((otrue[ti]>=lo)&(otrue[ti]<=hi)).float().mean().item()
    sharp=(2*Q*std[ti]).mean().item()                              # RAW calibrated interval width
    # local Pauli observables live in [-1,1]; clip intervals (never reduces coverage) -> physical width
    sharp_clip=(hi.clamp(max=1.0)-lo.clamp(min=-1.0)).clamp_min(0).mean().item()
    return covered,sharp,sharp_clip
fc,fsh,fshc=conformal_eval(fm,fs); mc,msh,mshc=conformal_eval(mm,ms); pc,psh,pshc=conformal_eval(pm,ps)
print("\n---------------- CALIBRATION (local Z/X observables, k=32, nominal 90%) ----------------")
print(f"  {'method':<16} {'conf-cover':>11} {'raw-width':>10} {'clipped[-1,1]-width':>20}")
print(f"  {'MPS-flow(ours)':<16} {fc*100:>10.1f}% {fsh:>10.4f} {fshc:>20.4f}")
print(f"  {'pCN-Bayes':<16} {pc*100:>10.1f}% {psh:>10.4f} {pshc:>20.4f}")
print(f"  {'MPS-MLE ensemble':<16} {mc*100:>10.1f}% {msh:>10.4f} {mshc:>20.4f}")
print(f"  => clipped-width ratio (physical): ours {fshc:.4f} vs pCN {pshc:.4f} vs MLE {mshc:.4f}  (MLE/ours={mshc/fshc:.1f}x)")
print("================================================================================")
print(f"VERDICT n={n}: prior-fid {prior_fid:.3f} | ours-sharper-clipped-than-MLE:{fshc<mshc}")
RES['calib']={'flow':{'conf':fc,'sharp':fsh,'sharp_clip':fshc},'mle':{'conf':mc,'sharp':msh,'sharp_clip':mshc},'pcn':{'conf':pc,'sharp':psh,'sharp_clip':pshc}}
RES['theta_std_max']=float(BS.max().item())
# (c) reliability curve: sweep nominal level, record conformal coverage + sharpness (flow vs mle)
levels=[0.5,0.6,0.7,0.8,0.9,0.95]
rel={'level':levels,'flow_cov':[],'flow_sharp':[],'mle_cov':[],'mle_sharp':[]}
for lv in levels:
    fcv,_,fs2=conformal_eval(fm,fs,alpha=1-lv); mcv,_,ms2=conformal_eval(mm,ms,alpha=1-lv)   # clipped width
    rel['flow_cov'].append(fcv); rel['flow_sharp'].append(fs2); rel['mle_cov'].append(mcv); rel['mle_sharp'].append(ms2)
RES['reliability']=rel
import json
with open('./qst_results_v5.jsonl','a') as fh: fh.write(json.dumps(RES)+"\n")
# (c) prior 2D-projection dump: family cores vs prior-sampled cores (standardized), for PCA scatter
if args.dump_proj:
    import numpy as _np
    _,pri_theta=sample_prior(1500,ns=120)                            # unstandardized prior cores
    pri_std=sf(cat_cores(canon(split_cores(pri_theta)))).cpu().numpy()
    _np.savez(f'./proj_{args.family}_n{n}.npz', fam=BANKn[:1500].cpu().numpy(), pri=pri_std)
    print("[dumped projection]")
print("[dumped results]")
