"""DE-RISK the pivot: amortized CONDITIONAL posterior for QST. A measurement-agnostic set-encoder over
{(P_j, o_j)} -> MPS cores (trained with a gauge-invariant STATE-space fidelity loss). Crux checks:
  (a) does conditioning on measurements clearly BEAT the k=0 prior-only baseline (medoid/mean)?
  (b) does it USE the measurements -- SHUFFLED measurements (from other states) should degrade badly?
  (c) is it measurement-AGNOSTIC -- one model trained on mixed (k, Pauli set), tested at each k?
If yes on all three, the conditional-posterior redirection is validated."""
import time, argparse, json, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
ap=argparse.ArgumentParser(); ap.add_argument('--dev',default='cuda:1'); ap.add_argument('--n',type=int,default=6); ap.add_argument('--chi',type=int,default=2)
ap.add_argument('--shots',type=int,default=200); ap.add_argument('--steps',type=int,default=30000); ap.add_argument('--ntest',type=int,default=120)
ap.add_argument('--family',default='tfim',choices=['tfim','random','shallow']); ap.add_argument('--sigma',type=float,default=0.35); ap.add_argument('--bank',type=int,default=20000); ap.add_argument('--fixedk',type=int,default=0); ap.add_argument('--arch',default='deepsets',choices=['deepsets','transformer','mlp']); ap.add_argument('--spread',type=float,default=1.0); ap.add_argument('--design',default='random',choices=['random','local','local1','local3']); ap.add_argument('--seed',type=int,default=0)
ap.add_argument('--hw_eval',default=''); ap.add_argument('--depth',type=int,default=1); ap.add_argument('--strengthen',action='store_true'); args=ap.parse_args(); dev=torch.device(args.dev); torch.manual_seed(args.seed); np.random.seed(args.seed); cdt=torch.complex64
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
def paulis_from_idx(idxs):    # idxs: (K,n) ints -> (K,D,D)
    return torch.stack([kron_list([PAULI[int(idxs[j,q])] for q in range(n)]) for j in range(idxs.shape[0])],0)
POOLIDX=None
def rand_pauli_idx(k,seed):
    rng=np.random.default_rng(seed); out=[];seen=set()
    while len(out)<k:
        s=tuple(rng.integers(0,4,size=n).tolist())
        if s==(0,)*n or s in seen: continue
        seen.add(s); out.append(list(s))
    return torch.tensor(out,device=dev)
POOLIDX=rand_pauli_idx(min(4**n-1,1024),1); POOL=paulis_from_idx(POOLIDX)   # large pool, precomputed ONCE
def local_design_idx(kind='local'):
    seen=set(); idxs=[]
    def add(v):
        t=tuple(v)
        if t not in seen and any(t): seen.add(t); idxs.append(list(v))
    if kind in ('local','local1','local3'):
        for q in range(n):
            for p in [1,2,3]: v=[0]*n; v[q]=p; add(v)                      # weight-1
    if kind=='local':
        for q in range(n-1):
            for p1 in [1,2,3]:
                for p2 in [1,2,3]: v=[0]*n; v[q]=p1; v[q+1]=p2; add(v)      # NN weight-2
    if kind=='local3':   # theory-complete: ALL non-identity Paulis on each contiguous 3-site window
        for q in range(n-2):
            for a in range(4):
                for b in range(4):
                    for c in range(4):
                        if a==0 and b==0 and c==0: continue
                        v=[0]*n; v[q]=a; v[q+1]=b; v[q+2]=c; add(v)
    return torch.tensor(idxs,device=dev)
if args.design in ('local','local1','local3'):
    POOLIDX=local_design_idx(args.design); POOL=paulis_from_idx(POOLIDX); print(f'[design] {args.design}: {POOL.shape[0]} observables',flush=True)
POOLOH=None  # set after pauli_onehot defined
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
def apply_ry(state,ang):
    b=state.shape[0]
    for q in range(n):
        c=torch.cos(ang[:,q]/2).to(cdt); s=torch.sin(ang[:,q]/2).to(cdt)
        st=state.view(b,2**q,2,2**(n-q-1)); a0=st[:,:,0,:]; a1=st[:,:,1,:]
        state=torch.stack([c.view(-1,1,1)*a0-s.view(-1,1,1)*a1, s.view(-1,1,1)*a0+c.view(-1,1,1)*a1],dim=2).reshape(b,D)
    return state
def apply_cnot(state,ctrl,tgt):
    b=state.shape[0]; idx=torch.arange(D,device=dev); bit_c=(idx>>(n-1-ctrl))&1
    swap=idx^((bit_c<<(n-1-tgt))); out=state.clone(); mask=bit_c==1; out[:,idx[mask]]=state[:,swap[mask]]; return out
def shallow_states(b,seed=0):   # physical family: depth D_blocks of (RY - CNOT ladder) + final RY. depth=1 => chi=2 (RY-ladder-RY); depth>=2 => deeper brickwork, chi up to 2^depth
    g=torch.Generator(device=dev).manual_seed(seed)
    sp=args.spread
    st=torch.zeros(b,D,dtype=cdt,device=dev); st[:,0]=1.0
    for d in range(args.depth):
        th=(np.pi/2 if d==0 else 0.0)+sp*torch.randn(b,n,generator=g,device=dev)
        st=apply_ry(st,th)
        for q in range(n-1): st=apply_cnot(st,q,q+1)
    th_last=sp*torch.randn(b,n,generator=g,device=dev)
    st=apply_ry(st,th_last); return st/st.norm(dim=1,keepdim=True)
def make_states(b,seed=0):
    if args.family=='tfim': return cores_to_state(fit_mps(tfim_ground(b,seed),seed=seed))
    if args.family=='shallow': return shallow_states(b,seed)
    return cores_to_state(make_family(b))
def expect(psi,ops):
    Opsi=torch.einsum('kij,bj->bki',ops,psi); return torch.einsum('bi,bki->bk',psi.conj(),Opsi).real
def fidelity(a,b): return (torch.einsum('bi,bi->b',a.conj(),b).abs())**2

print("[prep] state bank ...",flush=True)
BANK_PSI=make_states(args.bank,seed=0)
mean_state=BANK_PSI.mean(0,keepdim=True); mean_state=mean_state/mean_state.norm()
sub=BANK_PSI[:1500]; mfid=((sub@sub.conj().T).abs()**2).mean(1); medoid=sub[mfid.argmax():mfid.argmax()+1]

# ---- CONDITIONAL amortized models over {(P_j, o_j)} -> cores ----
class CondNet(nn.Module):   # DeepSets (mean-pool)
    def __init__(s,dp,dout,h=512,z=512):
        super().__init__()
        s.enc=nn.Sequential(nn.Linear(dp+1,h),nn.SiLU(),nn.Linear(h,h),nn.SiLU(),nn.Linear(h,z))
        s.post=nn.Sequential(nn.Linear(z,h),nn.SiLU(),nn.Linear(h,h),nn.SiLU(),nn.Linear(h,h),nn.SiLU(),nn.Linear(h,dout))
    def forward(s,pfeat,ovals,mask=None):
        x=torch.cat([pfeat,ovals.unsqueeze(-1)],-1); e=s.enc(x); z=e.mean(1); return s.post(z)
class SetTransformer(nn.Module):   # self-attention over measurements + attention pooling (permutation-invariant, any K)
    def __init__(s,dp,dout,d=256,nhead=4,nlayers=4):
        super().__init__()
        s.embed=nn.Linear(dp+1,d)
        layer=nn.TransformerEncoderLayer(d,nhead,dim_feedforward=4*d,batch_first=True,activation='gelu',dropout=0.0)
        s.enc=nn.TransformerEncoder(layer,nlayers)
        s.query=nn.Parameter(0.02*torch.randn(1,1,d)); s.pool=nn.MultiheadAttention(d,nhead,batch_first=True)
        s.post=nn.Sequential(nn.Linear(d,512),nn.SiLU(),nn.Linear(512,512),nn.SiLU(),nn.Linear(512,dout))
    def forward(s,pfeat,ovals,mask=None):
        x=torch.cat([pfeat,ovals.unsqueeze(-1)],-1); h=s.enc(s.embed(x))
        q=s.query.expand(h.shape[0],-1,-1); pooled,_=s.pool(q,h,h); return s.post(pooled.squeeze(1))
class PlainMLP(nn.Module):   # ignores set structure: fixed-order value vector -> cores (baseline)
    def __init__(s,kin,dout,h=1024):
        super().__init__(); s.net=nn.Sequential(nn.Linear(kin,h),nn.SiLU(),nn.Linear(h,h),nn.SiLU(),nn.Linear(h,h),nn.SiLU(),nn.Linear(h,h),nn.SiLU(),nn.Linear(h,dout))
    def forward(s,pfeat,ovals,mask=None): return s.net(ovals)
def pauli_onehot(idxs):   # (K,n) -> (K, 4n)
    oh=F.one_hot(idxs.long(),4).float().reshape(idxs.shape[0],-1); return oh
POOLOH=pauli_onehot(POOLIDX)                                           # (npool, 4n) precomputed
EX_ALL=None                                                           # cache of <P> for a fixed state batch is not reusable across states
Kfix=(POOL.shape[0] if args.design in ('local','local1','local3') else (args.fixedk if args.fixedk>0 else 32))
net=(PlainMLP(Kfix,THETA) if args.arch=='mlp' else (SetTransformer(4*n,THETA) if args.arch=='transformer' else CondNet(4*n,THETA))).to(dev); opt=torch.optim.AdamW(net.parameters(),lr=3e-4); sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,args.steps)
print("[train] conditional model (variable k, variable Pauli design, state-loss) ...",flush=True); t0=time.time()
TR=make_states(6000 if args.family=='tfim' else args.bank, seed=1)     # fresh training states
NPOOL=POOL.shape[0]; rng=np.random.default_rng(0)
for it in range(1,args.steps+1):
    bidx=torch.randint(0,TR.shape[0],(256,),device=dev); psb=TR[bidx]
    if args.design in ('local','local1','local3'):
        k=NPOOL; sel=torch.arange(NPOOL,device=dev)                    # FIXED full local design
    elif args.fixedk>0:
        k=args.fixedk; sel=torch.arange(k,device=dev)
    else:
        k=int(rng.integers(6,65)); sel=torch.randperm(NPOOL,device=dev)[:k]  # variable design
    ops=POOL[sel]; te=expect(psb,ops); shn=torch.sqrt(((1-te**2).clamp_min(0))/args.shots); obs=te+shn*torch.randn_like(te)
    oh=POOLOH[sel].unsqueeze(0).expand(256,-1,-1)                      # (256,k,4n)
    pred=net(oh,obs); st=cores_to_state(split_cores(pred))   # canon is state-preserving -> unneeded for the fidelity loss
    loss=(1-fidelity(st,psb)).mean(); opt.zero_grad(); loss.backward(); opt.step(); sch.step()
    if it%10000==0: print(f"   it{it} loss {loss.item():.4f} ({time.time()-t0:.0f}s)",flush=True)

psi_true=make_states(args.ntest,seed=999)
# prior-only baselines (context)
f_med=fidelity(psi_true,medoid.expand(args.ntest,-1)).mean().item(); f_mean=fidelity(psi_true,mean_state.expand(args.ntest,-1)).mean().item()
print(f"\n[prior-only k=0] medoid {f_med:.3f} | mean-state {f_mean:.3f}",flush=True)
@torch.no_grad()
def cond_recon(pidx,obs):
    oh=pauli_onehot(pidx).unsqueeze(0).expand(obs.shape[0],-1,-1); return cores_to_state(canon(split_cores(net(oh,obs))))
perm=torch.randperm(args.ntest,device=dev); KS=[8,16,32,64]; res={'k':KS,'true':[],'shuffled':[]}
print(f"{'k':>4} | {'COND true':>10} | {'COND shuffled':>13} | {'gain-over-prior':>15} | {'true-shuffled':>13}",flush=True)
KS_eval=([POOL.shape[0]] if args.design in ('local','local1','local3') else KS)
for k in KS_eval:
    pidx=POOLIDX[:k]; ops=POOL[:k]; te=expect(psi_true,ops); shn=torch.sqrt(((1-te**2).clamp_min(0))/args.shots); obs=te+shn*torch.randn_like(te)
    _ft=fidelity(psi_true,cond_recon(pidx,obs)); f_true=_ft.mean().item(); print(f'   [per-state] mean {f_true:.3f} std {_ft.std().item():.3f} q10 {_ft.quantile(0.1).item():.3f} q90 {_ft.quantile(0.9).item():.3f}',flush=True)
    f_shuf=fidelity(psi_true,cond_recon(pidx,obs[perm])).mean().item()
    res['true'].append(f_true); res['shuffled'].append(f_shuf)
    print(f"{k:>4} | {f_true:>10.3f} | {f_shuf:>13.3f} | {f_true-max(f_med,f_mean):>+15.3f} | {f_true-f_shuf:>+13.3f}",flush=True)
# measurement-agnostic check: test on a DIFFERENT random Pauli design (unseen at that exact set)
pidx2=rand_pauli_idx(24, 55555); ops2=paulis_from_idx(pidx2); te=expect(psi_true,ops2); shn=torch.sqrt(((1-te**2).clamp_min(0))/args.shots); obs=te+shn*torch.randn_like(te)
f_alt=fidelity(psi_true,cond_recon(pidx2,obs)).mean().item()
print(f"[meas-agnostic] fresh unseen 24-Pauli design: cond fid {f_alt:.3f}",flush=True)
res['prior_medoid']=f_med; res['prior_mean']=f_mean; res['alt_design_k24']=f_alt
json.dump({'n':n,'family':args.family,**res},open(f'./qst_cond_{args.family}_n{n}.json','w'),indent=1)
print("[dumped]",flush=True)


if args.hw_eval:
    import numpy as _np
    d=_np.load(args.hw_eval)
    psi_hw=torch.tensor(d["psi_ideal"],dtype=cdt,device=dev)
    pidx_hw=torch.tensor(d["pool"],dtype=torch.long,device=dev)
    obs_hw=torch.tensor(d["exp_hw"],dtype=torch.float32,device=dev)
    obs_id=torch.tensor(d["exp_ideal"],dtype=torch.float32,device=dev)
    # convention cross-check: torch expect on stored ideal state vs stored ideal expectations
    te=expect(psi_hw,paulis_from_idx(pidx_hw)); conv=(te-obs_id).abs().max().item()
    print(f"[hw_eval] convention cross-check max|torch-qiskit| = {conv:.2e}",flush=True)
    Mh=psi_hw.shape[0]
    f_hw=fidelity(psi_hw,cond_recon(pidx_hw,obs_hw))
    f_id=fidelity(psi_hw,cond_recon(pidx_hw,obs_id))
    f_pr=fidelity(psi_hw,medoid.expand(Mh,-1))
    dev_err=(obs_hw-obs_id).abs().mean().item()
    print(f"[hw_eval] {Mh} states | device |hw-ideal|={dev_err:.4f} | cond-from-HW fid={f_hw.mean().item():.4f}+-{f_hw.std().item():.4f} | cond-from-ideal fid={f_id.mean().item():.4f} | prior-only medoid={f_pr.mean().item():.4f}",flush=True)
    print("[hw_eval] per-state HW fid:", [round(x,3) for x in f_hw.tolist()],flush=True)


if args.strengthen:
    print("[strengthen] family-aware simple baselines",flush=True)
    # recompute LOCAL-design test measurements (obs was overwritten by the agnostic check above)
    _te=expect(psi_true,POOL); _sh=torch.sqrt(((1-_te**2).clamp_min(0))/args.shots); obsL=_te+_sh*torch.randn_like(_te)
    with torch.no_grad():
        bank_obs=torch.cat([expect(BANK_PSI[i:i+512],POOL) for i in range(0,BANK_PSI.shape[0],512)],0)  # (bank,K) noiseless db
        # (a) measurement-space 1-NN: nearest training measurement vector -> its state
        dmat=torch.cdist(obsL,bank_obs); nnidx=dmat.argmin(1)
        f_knn=fidelity(psi_true,BANK_PSI[nnidx]).mean().item()
        # (a2) 5-NN average state (renormalized)
        nn5=dmat.topk(5,largest=False).indices  # (ntest,5)
        avg=BANK_PSI[nn5].mean(1); avg=avg/avg.norm(dim=1,keepdim=True); f_knn5=fidelity(psi_true,avg).mean().item()
    print(f"[strengthen] measurement-space 1-NN fid {f_knn:.4f} | 5-NN {f_knn5:.4f}",flush=True)
    # (b) direct DENSE-STATE regressor (MLP obs -> 2^n complex amplitudes), same fidelity loss, NO MPS core
    dnet=nn.Sequential(nn.Linear(POOL.shape[0],512),nn.ReLU(),nn.Linear(512,512),nn.ReLU(),nn.Linear(512,512),nn.ReLU(),nn.Linear(512,2*D)).to(dev)
    dopt=torch.optim.AdamW(dnet.parameters(),lr=3e-4)
    def dstate(o):
        out=dnet(o); st=torch.complex(out[:,:D],out[:,D:]); return st/st.norm(dim=1,keepdim=True)
    for it in range(12000):
        idx=torch.randint(0,BANK_PSI.shape[0],(256,),device=dev); psb2=BANK_PSI[idx]
        te=expect(psb2,POOL); shn=torch.sqrt(((1-te**2).clamp_min(0))/args.shots); o2=te+shn*torch.randn_like(te)
        st=dstate(o2); loss=(1-fidelity(psb2,st)).mean()
        dopt.zero_grad(); loss.backward(); dopt.step()
    with torch.no_grad(): f_dir=fidelity(psi_true,dstate(obsL)).mean().item()
    print(f"[strengthen] direct dense-state MLP (no MPS core) fid {f_dir:.4f}",flush=True)
