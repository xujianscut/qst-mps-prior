"""
Review-response experiments for #29 QST paper. Reuses qst_mps5 machinery.
  A) MPS-MLE baseline is NOT broken: noiseless/high-k recovery + convergence sweep + residual-vs-fidelity
     => the 31-247x ratio is genuine underdetermination, not a weak/unconverged baseline.
  B) Supervised amortized measurement->core regressor (the 'obvious alternative'): does supervised-alone match
     learned-prior+proximal? Per-budget MLP conditioned on the Pauli measurement vector.
  C) Prior-mismatch / OOD sweep: push test states off the training manifold (delta) + critical-regime OOD;
     fidelity vs mismatch for flow vs MLE vs supervised, plus manifold-distance to detect hallucination.
  D) Per-observable conformal (separate Q_O): the rigorous variant, coverage per observable.
"""
import time, argparse, json, math, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
ap=argparse.ArgumentParser()
ap.add_argument('--dev',default='cuda:0'); ap.add_argument('--n',type=int,default=6); ap.add_argument('--chi',type=int,default=2)
ap.add_argument('--sigma',type=float,default=0.35); ap.add_argument('--shots',type=int,default=200)
ap.add_argument('--steps',type=int,default=40000); ap.add_argument('--bs',type=int,default=256); ap.add_argument('--ntest',type=int,default=120)
ap.add_argument('--family',default='tfim',choices=['random','tfim']); ap.add_argument('--bank',type=int,default=30000)
args=ap.parse_args()
dev=torch.device(args.dev); torch.manual_seed(0); np.random.seed(0); cdt=torch.complex64
n,chi,D=args.n,args.chi,2**args.n
bonds=[min(2**k,2**(n-k),chi) for k in range(n+1)]; shapes=[(bonds[k],2,bonds[k+1]) for k in range(n)]
sizes=[np.prod(s) for s in shapes]; off=np.cumsum([0]+sizes); THETA=2*int(off[-1])
print(f"[cfg] n={n} chi={chi} D={D} core(complex)={int(off[-1])} theta={THETA} family={args.family}")

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
    b=th.shape[0]; c=torch.complex(th[:,:THETA//2],th[:,THETA//2:])
    return [c[:,off[k]:off[k+1]].reshape(b,*shapes[k]) for k in range(n)]
def cat_cores(cores):
    b=cores[0].shape[0]; c=torch.cat([g.reshape(b,-1) for g in cores],1)
    return torch.cat([c.real,c.imag],1)
def cores_to_state(cores):
    b=cores[0].shape[0]; psi=cores[0].reshape(b,2,bonds[1])
    for k in range(1,n):
        psi=torch.einsum('bar,brjs->bajs',psi,cores[k].reshape(b,bonds[k],2,bonds[k+1])).reshape(b,2**(k+1),bonds[k+1])
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
def make_family(b):
    g=torch.Generator(device=dev).manual_seed(12345)
    base=[torch.randn(1,*s,generator=g,device=dev)+1j*torch.randn(1,*s,generator=g,device=dev) for s in shapes]
    cores=[(base[k]+args.sigma*(torch.randn(b,*shapes[k],device=dev)+1j*torch.randn(b,*shapes[k],device=dev))) for k in range(n)]
    return canon(cores)
def _single(op,q): return kron_list([op if i==q else I2 for i in range(n)])
ZZ_OPS=torch.stack([_single(Zp,q)@_single(Zp,q+1) for q in range(n-1)],0) if n>1 else None
X_OPS=torch.stack([_single(Xp,q) for q in range(n)],0)
def tfim_ground(b,critical=False):
    if critical: J=0.8+0.4*torch.rand(b,n-1,device=dev); h=0.8+0.4*torch.rand(b,n,device=dev)
    else: J=0.2+0.4*torch.rand(b,n-1,device=dev); h=1.0+1.0*torch.rand(b,n,device=dev)
    ch=8 if D>=1024 else 24; psis=[]
    for s in range(0,b,ch):
        with torch.no_grad():
            jj=J[s:s+ch].to(cdt); hh=h[s:s+ch].to(cdt)
            H=-torch.einsum('bi,ijk->bjk',jj,ZZ_OPS)-torch.einsum('bi,ijk->bjk',hh,X_OPS)
            w,V=torch.linalg.eigh(0.5*(H+H.conj().transpose(-1,-2))); psis.append(V[:,:,0].clone())
        del H,w,V; torch.cuda.empty_cache()
    return torch.cat(psis,0)
def fit_mps_to_state(psi_t,iters=300,lr=0.05,restarts=2):
    b=psi_t.shape[0]; best=None; beste=None
    for _ in range(restarts):
        x=(0.1*torch.randn(b,THETA,device=dev)).requires_grad_(True); op=torch.optim.Adam([x],lr=lr)
        for _ in range(iters):
            psi=cores_to_state(split_cores(x)); loss=(1-(torch.einsum('bi,bi->b',psi.conj(),psi_t).abs())**2).mean()
            op.zero_grad(); loss.backward(); op.step()
        with torch.no_grad():
            e=1-(torch.einsum('bi,bi->b',cores_to_state(split_cores(x)).conj(),psi_t).abs())**2
            if best is None: best,beste=x.detach().clone(),e
            else: sw=e<beste; best[sw]=x.detach()[sw]; beste=torch.where(sw,e,beste)
    return canon(split_cores(best))
def make_states(b,critical=False):
    if args.family=='tfim': return fit_mps_to_state(tfim_ground(b,critical))
    return make_family(b)
def expect(psi,ops):
    Opsi=torch.einsum('kij,bj->bki',ops,psi); return torch.einsum('bi,bki->bk',psi.conj(),Opsi).real
def fidelity(a,b): return (torch.einsum('bi,bi->b',a.conj(),b).abs())**2

# ---- flow prior ----
print(f"[prep] bank ...")
NB=6000 if args.family=='tfim' else args.bank
BANK=cat_cores(make_states(NB)); BM,BS=BANK.mean(0,keepdim=True),BANK.std(0,keepdim=True).clamp_min(1e-4); BANKn=(BANK-BM)/BS
sf=lambda t:(t-BM)/BS; uf=lambda t:t*BS+BM
class Vel(nn.Module):
    def __init__(s,d,h=1024):
        super().__init__(); s.t=nn.Sequential(nn.Linear(1,128),nn.SiLU(),nn.Linear(128,128))
        s.n=nn.Sequential(nn.Linear(d+128,h),nn.SiLU(),nn.Linear(h,h),nn.SiLU(),nn.Linear(h,h),nn.SiLU(),nn.Linear(h,h),nn.SiLU(),nn.Linear(h,d))
    def forward(s,x,t): return s.n(torch.cat([x,s.t(t.view(-1,1))],1))
vel=Vel(THETA).to(dev); opt=torch.optim.AdamW(vel.parameters(),lr=2e-4); sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,args.steps)
print("[train] flow prior ..."); t0=time.time()
for it in range(1,args.steps+1):
    th=BANKn[torch.randint(0,BANKn.shape[0],(args.bs,),device=dev)]; eps=torch.randn_like(th); t=torch.rand(th.shape[0],device=dev)
    loss=F.mse_loss(vel((1-t).view(-1,1)*eps+t.view(-1,1)*th,t),th-eps); opt.zero_grad(); loss.backward(); opt.step(); sch.step()
    if it%20000==0: print(f"   it{it} loss {loss.item():.4f} ({time.time()-t0:.0f}s)")
BANK_PSI=cores_to_state(make_states(4000))
@torch.no_grad()
def prior_fid_of(psi):    # max fidelity of psi to the training family bank => 'is it on the manifold'
    return ((psi@BANK_PSI.conj().T).abs()**2).max(1).values

def meas_loss(psi,ops,obs): return (expect(psi/psi.norm(dim=1,keepdim=True),ops)-obs).pow(2).mean(1)
def prox_cores(anchor_std,ops,obs,n_inner,lam,lr=0.1):
    x=uf(anchor_std).clone().requires_grad_(True); op=torch.optim.Adam([x],lr=lr)
    psi_a=cores_to_state(split_cores(uf(anchor_std))).detach()
    for _ in range(n_inner):
        psi=cores_to_state(split_cores(x))
        loss=meas_loss(psi,ops,obs).mean()+lam*(1-(torch.einsum('bi,bi->b',psi.conj(),psi_a).abs())**2).mean()
        op.zero_grad(); loss.backward(); op.step()
    with torch.no_grad(): return sf(cat_cores(canon(split_cores(x))))
def flow_recon(ops,obs,ns=40,lam=0.3,n_inner=25,bestof=8):
    b=obs.shape[0]
    def one():
        th=torch.randn(b,THETA,device=dev)
        for i in range(ns):
            t=i/ns; t1=(i+1)/ns
            with torch.no_grad(): v=vel(th,torch.full((b,),t,device=dev))
            x0=th+(1-t)*v; x0=prox_cores(x0,ops,obs,n_inner,lam); eps=(th-t*x0)/max(1-t,1e-3)
            th=((1-t1)*eps+t1*x0).clamp(-6,6)
        return cores_to_state(canon(split_cores(uf(th))))
    if bestof==1: return one()
    cand=torch.stack([one() for _ in range(bestof)],0); ml=torch.stack([meas_loss(cand[j],ops,obs) for j in range(bestof)],0)
    sel=ml.argmin(0); return cand[sel,torch.arange(b,device=dev)]
def mps_mle(ops,obs,restarts=12,iters=700,lr=0.1,return_resid=False):
    b=obs.shape[0]; best=None; beste=None
    for _ in range(restarts):
        x=(0.3*torch.randn(b,THETA,device=dev)).requires_grad_(True); op=torch.optim.Adam([x],lr=lr)
        for _ in range(iters):
            loss=meas_loss(cores_to_state(split_cores(x)),ops,obs).mean(); op.zero_grad(); loss.backward(); op.step()
        with torch.no_grad():
            e=meas_loss(cores_to_state(split_cores(x)),ops,obs)
            if best is None: best,beste=x.detach().clone(),e
            else: sw=e<beste; best[sw]=x.detach()[sw]; beste=torch.where(sw,e,beste)
    st=cores_to_state(split_cores(best))
    return (st,beste) if return_resid else st

OUT={'n':n,'chi':chi,'family':args.family}
# =========================================================================================
# A) MPS-MLE baseline is not broken: noiseless recovery + high-k + convergence + residual-vs-fid
# =========================================================================================
print("\n===== A) MPS-MLE sanity (is the near-zero baseline genuine underdetermination or a broken solver?) =====")
psi_true=cores_to_state(make_states(args.ntest))
A={'noiseless':{}, 'noisy':{}, 'conv':{}}
for k in [8,16,32,64,128,min(4**n-1,220)]:
    if k>POOL.shape[0]: continue
    ops=POOL[:k]; te=expect(psi_true,ops)
    # NOISELESS
    st,res=mps_mle(ops,te,return_resid=True)
    A['noiseless'][k]={'fid':fidelity(psi_true,st).mean().item(),'resid':res.mean().item()}
    # NOISY (shots=200)
    sh=torch.sqrt(((1-te**2).clamp_min(0))/args.shots); obs=te+sh*torch.randn_like(te)
    st2,res2=mps_mle(ops,obs,return_resid=True)
    A['noisy'][k]={'fid':fidelity(psi_true,st2).mean().item(),'resid':res2.mean().item()}
    print(f"  k={k:>4}  NOISELESS fid={A['noiseless'][k]['fid']:.4f} resid={A['noiseless'][k]['resid']:.2e} | NOISY fid={A['noisy'][k]['fid']:.4f} resid={A['noisy'][k]['resid']:.2e}")
# convergence sweep at k=16 noiseless: more compute should NOT rescue it if underdetermined
ops=POOL[:16]; te=expect(psi_true,ops)
print("  convergence @k=16 (noiseless): fid should plateau (underdetermined), not climb with compute")
for R,IT in [(12,700),(24,1500),(48,3000),(96,3000)]:
    st,res=mps_mle(ops,te,restarts=R,iters=IT,return_resid=True); f=fidelity(psi_true,st).mean().item()
    A['conv'][f"R{R}_I{IT}"]={'fid':f,'resid':res.mean().item()}
    print(f"    restarts={R:>3} iters={IT:>4}: fid={f:.4f} resid={res.mean().item():.2e}")
OUT['A_mle_sanity']=A

# =========================================================================================
# B) Supervised amortized measurement->core regressor (the obvious alternative)
# =========================================================================================
print("\n===== B) Supervised amortized measurement->core baseline (does supervised-alone match prior+proximal?) =====")
class Amort(nn.Module):
    def __init__(s,kin,dout,h=1024):
        super().__init__(); s.net=nn.Sequential(nn.Linear(kin,h),nn.SiLU(),nn.Linear(h,h),nn.SiLU(),
            nn.Linear(h,h),nn.SiLU(),nn.Linear(h,h),nn.SiLU(),nn.Linear(h,dout))
    def forward(s,m): return s.net(m)
B={}
SUP={}
for k in [8,16,32]:
    ops=POOL[:k]
    # supervised training set: bank states' (noisy) measurements -> standardized canonical cores
    net=Amort(k,THETA).to(dev); o2=torch.optim.AdamW(net.parameters(),lr=3e-4)
    NB2=6000 if args.family=='tfim' else 20000
    # supervised training pairs: fresh family states -> (noisy measurements, standardized canonical cores)
    fresh=make_states(NB2); th_tr=sf(cat_cores(fresh)); st_tr=cores_to_state(fresh)
    for ep in range(4000):
        idx=torch.randint(0,NB2,(512,),device=dev); psb=st_tr[idx]; te=expect(psb,ops)
        shn=torch.sqrt(((1-te**2).clamp_min(0))/args.shots); mb=te+shn*torch.randn_like(te)
        pred=net(mb); loss=F.mse_loss(pred,th_tr[idx]); o2.zero_grad(); loss.backward(); o2.step()
    with torch.no_grad():
        te=expect(psi_true,ops); sh=torch.sqrt(((1-te**2).clamp_min(0))/args.shots); obs=te+sh*torch.randn_like(te)
        pred=net(obs); st=cores_to_state(canon(split_cores(uf(pred))))
        fsup=fidelity(psi_true,st).mean().item()
    fflow=fidelity(psi_true,flow_recon(ops,obs)).mean().item()
    B[k]={'supervised':fsup,'flow':fflow}; SUP[k]=net
    print(f"  k={k:>3}: supervised-amortized fid={fsup:.4f} | flow(prior+proximal) fid={fflow:.4f}")
OUT['B_supervised']=B

# =========================================================================================
# C) Prior-mismatch / OOD sweep: push states off training manifold + critical-regime OOD
# =========================================================================================
print("\n===== C) Prior-mismatch / OOD (graceful degradation vs hallucination?) =====")
def haar(b):
    z=torch.randn(b,D,device=dev)+1j*torch.randn(b,D,device=dev); return z/z.norm(dim=1,keepdim=True)
psi_in=cores_to_state(make_states(80))
C={'delta':[], 'ood_offmanifold':[], 'critical':{}}
k=16; ops=POOL[:k]
DELTAS=[0.0,0.05,0.10,0.20,0.35,0.5]
for d in DELTAS:
    rp=haar(80); mix=(1-d)*psi_in+d*rp; mix=mix/mix.norm(dim=1,keepdim=True)
    te=expect(mix,ops); sh=torch.sqrt(((1-te**2).clamp_min(0))/args.shots); obs=te+sh*torch.randn_like(te)
    rf=flow_recon(ops,obs); rm=mps_mle(ops,obs)
    sup=SUP[16];
    with torch.no_grad(): rs=cores_to_state(canon(split_cores(uf(sup(obs)))))
    row={'delta':d,
         'flow_fid':fidelity(mix,rf).mean().item(),'mle_fid':fidelity(mix,rm).mean().item(),'sup_fid':fidelity(mix,rs).mean().item(),
         'flow_reconPriorFid':prior_fid_of(rf).mean().item(),   # hallucination probe: is recon glued to manifold?
         'true_priorFid':prior_fid_of(mix).mean().item()}       # how far the TRUE state itself left the manifold
    C['delta'].append(row)
    print(f"  delta={d:.2f}: flow={row['flow_fid']:.3f} mle={row['mle_fid']:.3f} sup={row['sup_fid']:.3f} | recon-onManifold={row['flow_reconPriorFid']:.3f} true-onManifold={row['true_priorFid']:.3f}")
# critical-regime OOD (TFIM only): prior trained paramagnetic, test near-critical (higher entanglement)
if args.family=='tfim':
    psi_c=cores_to_state(make_states(80,critical=True))
    te=expect(psi_c,ops); sh=torch.sqrt(((1-te**2).clamp_min(0))/args.shots); obs=te+sh*torch.randn_like(te)
    C['critical']={'flow_fid':fidelity(psi_c,flow_recon(ops,obs)).mean().item(),
                   'mle_fid':fidelity(psi_c,mps_mle(ops,obs)).mean().item(),
                   'true_priorFid':prior_fid_of(psi_c).mean().item()}
    print(f"  CRITICAL-regime OOD: flow={C['critical']['flow_fid']:.3f} mle={C['critical']['mle_fid']:.3f} true-onManifold={C['critical']['true_priorFid']:.3f}")
OUT['C_ood']=C

# =========================================================================================
# D) Per-observable conformal (rigorous variant: separate Q_O per observable)
# =========================================================================================
print("\n===== D) Per-observable conformal (separate Q_O; rigorous fixed-O guarantee) =====")
def single(op,q): return kron_list([op if i==q else I2 for i in range(n)])
OBS=torch.stack([single(Zp,q) for q in range(n)]+[single(Xp,q) for q in range(n)],0)
cal_k=32; ops=POOL[:cal_k]; te=expect(psi_true,ops); sh=torch.sqrt(((1-te**2).clamp_min(0))/args.shots); obs=te+sh*torch.randn_like(te)
K=24
def flow_samples(ops,obs,ns=30,K=24,noise=0.15):
    b=obs.shape[0]; outs=[]
    for _ in range(K):
        th=torch.randn(b,THETA,device=dev)
        for i in range(ns):
            t=i/ns; t1=(i+1)/ns
            with torch.no_grad(): v=vel(th,torch.full((b,),t,device=dev))
            x0=th+(1-t)*v; x0=prox_cores(x0,ops,obs,25,0.3); eps=(th-t*x0)/max(1-t,1e-3)
            th=((1-t1)*eps+t1*x0)
            if noise>0 and i<ns-1: th=th+noise*(1-t1)*torch.randn_like(th)
            th=th.clamp(-6,6)
        outs.append(cores_to_state(canon(split_cores(uf(th)))))
    return torch.stack(outs,0)
flow_s=flow_samples(ops,obs,K=K)
e=torch.stack([expect(flow_s[j],OBS) for j in range(K)],0); mean=e.mean(0); std=e.std(0).clamp_min(1e-4)
otrue=expect(psi_true,OBS); nt=args.ntest; idx=torch.randperm(nt,device=dev); ci,ti=idx[:nt//2],idx[nt//2:]
# pooled (as in paper) vs per-observable Q_O
r=((otrue-mean).abs()/std)
Qp=torch.quantile(r[ci].flatten(),0.9)
cov_pool=((otrue[ti]>=mean[ti]-Qp*std[ti])&(otrue[ti]<=mean[ti]+Qp*std[ti])).float().mean().item()
percov=[]; perw=[]
for o in range(OBS.shape[0]):
    Qo=torch.quantile(r[ci,o],0.9); lo=mean[ti,o]-Qo*std[ti,o]; hi=mean[ti,o]+Qo*std[ti,o]
    percov.append(((otrue[ti,o]>=lo)&(otrue[ti,o]<=hi)).float().mean().item())
    perw.append((hi.clamp(max=1)-lo.clamp(min=-1)).clamp_min(0).mean().item())
D_={'pooled_cover':cov_pool,'perobs_cover_mean':float(np.mean(percov)),'perobs_cover_min':float(np.min(percov)),
    'perobs_cover_max':float(np.max(percov)),'perobs_width_mean':float(np.mean(perw))}
print(f"  pooled cover={cov_pool*100:.1f}% | per-observable Q_O: mean cover={np.mean(percov)*100:.1f}% (min {np.min(percov)*100:.0f}%, max {np.max(percov)*100:.0f}%) width={np.mean(perw):.3f}")
OUT['D_perobs_conformal']=D_

with open(f'./qst_review_{args.family}_n{n}.json','w') as fh: json.dump(OUT,fh,indent=1)
print(f"\n[dumped] qst_review_{args.family}_n{n}.json")
