"""SHORE-UP #2: calibrated UQ for the (validated) fixed-design conditional model. Train ONE conditional
model with dropout; at test do E stochastic (MC-dropout) forward passes -> posterior over the 2n local
Z/X observables -> conformal recalibration -> coverage + width. Prior-only gives no calibrated intervals;
this shows the conditional posterior does. Fixed design (POOL[:k]), random+tfim families."""
import time, argparse, json, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
ap=argparse.ArgumentParser(); ap.add_argument('--dev',default='cuda:2'); ap.add_argument('--n',type=int,default=6); ap.add_argument('--chi',type=int,default=2)
ap.add_argument('--shots',type=int,default=200); ap.add_argument('--steps',type=int,default=20000); ap.add_argument('--ntest',type=int,default=200)
ap.add_argument('--family',default='random',choices=['tfim','random','shallow']); ap.add_argument('--spread',type=float,default=0.6); ap.add_argument('--sigma',type=float,default=0.35)
ap.add_argument('--k',type=int,default=32); ap.add_argument('--drop',type=float,default=0.1); ap.add_argument('--E',type=int,default=24); ap.add_argument('--design',default='random',choices=['random','local'])
args=ap.parse_args(); dev=torch.device(args.dev); torch.manual_seed(0); np.random.seed(0); cdt=torch.complex64
n,chi,D=args.n,args.chi,2**args.n
bonds=[min(2**k,2**(n-k),chi) for k in range(n+1)]; shapes=[(bonds[k],2,bonds[k+1]) for k in range(n)]
sizes=[np.prod(s) for s in shapes]; off=np.cumsum([0]+sizes); THETA=2*int(off[-1])
print(f"[cfg] n={n} family={args.family} k={args.k} drop={args.drop} E={args.E}",flush=True)
I2=torch.tensor([[1,0],[0,1]],dtype=cdt,device=dev);Xp=torch.tensor([[0,1],[1,0]],dtype=cdt,device=dev)
Yp=torch.tensor([[0,-1j],[1j,0]],dtype=cdt,device=dev);Zp=torch.tensor([[1,0],[0,-1]],dtype=cdt,device=dev); PAULI=[I2,Xp,Yp,Zp]
def kron_list(ms):
    o=ms[0]
    for m in ms[1:]: o=torch.kron(o,m)
    return o
def rand_pauli_idx(k,seed):
    rng=np.random.default_rng(seed); out=[];seen=set()
    while len(out)<k:
        s=tuple(rng.integers(0,4,size=n).tolist())
        if s==(0,)*n or s in seen: continue
        seen.add(s); out.append(list(s))
    return torch.tensor(out,device=dev)
def paulis_from_idx(idxs): return torch.stack([kron_list([PAULI[int(idxs[j,q])] for q in range(n)]) for j in range(idxs.shape[0])],0)
def local_design_idx():
    idxs=[]
    for q in range(n):
        for pp in [1,2,3]:
            v=[0]*n; v[q]=pp; idxs.append(v)
    for q in range(n-1):
        for p1 in [1,2,3]:
            for p2 in [1,2,3]:
                v=[0]*n; v[q]=p1; v[q+1]=p2; idxs.append(v)
    return torch.tensor(idxs,device=dev)
POOLIDX=(local_design_idx() if args.design=='local' else rand_pauli_idx(args.k,1)); POOL=paulis_from_idx(POOLIDX)  # FIXED design
def split_cores(th):
    b=th.shape[0]; c=torch.complex(th[:,:THETA//2],th[:,THETA//2:]); return [c[:,off[k]:off[k+1]].reshape(b,*shapes[k]) for k in range(n)]
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
    nrm=cs[-1].reshape(b,-1).norm(dim=1).clamp_min(1e-9); cs[-1]=cs[-1]/nrm.view(-1,1,1,1); return cs
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
def shallow_states(b,seed=0):
    g=torch.Generator(device=dev).manual_seed(seed); sp=args.spread
    th1=np.pi/2+sp*torch.randn(b,n,generator=g,device=dev); th2=sp*torch.randn(b,n,generator=g,device=dev)
    st=torch.zeros(b,D,dtype=cdt,device=dev); st[:,0]=1.0; st=apply_ry(st,th1)
    for q in range(n-1): st=apply_cnot(st,q,q+1)
    st=apply_ry(st,th2); return st/st.norm(dim=1,keepdim=True)
def make_states(b,seed=0):
    if args.family=='tfim': return cores_to_state(fit_mps(tfim_ground(b,seed),seed=seed))
    if args.family=='shallow': return shallow_states(b,seed)
    return cores_to_state(make_family(b))
def expect(psi,ops):
    Opsi=torch.einsum('kij,bj->bki',ops,psi); return torch.einsum('bi,bki->bk',psi.conj(),Opsi).real
def fidelity(a,b): return (torch.einsum('bi,bi->b',a.conj(),b).abs())**2
def pauli_onehot(idxs): return F.one_hot(idxs.long(),4).float().reshape(idxs.shape[0],-1)
OBS=torch.stack([_single(Zp,q) for q in range(n)]+[_single(Xp,q) for q in range(n)],0)   # 2n local observables
def _multi(m): return kron_list([m.get(i,I2) for i in range(n)])
_H=[]
for i in range(n):
    for j in range(i+2,n): _H.append(_multi({i:Zp,j:Zp}))          # long-range ZZ (gap>=2), NOT in local design
for i in range(n-2):
    _H.append(_multi({i:Zp,i+1:Zp,i+2:Zp})); _H.append(_multi({i:Xp,i+1:Xp,i+2:Xp}))  # weight-3 ZZZ/XXX
OBS_HELD=torch.stack(_H,0)  # unmeasured observables (not in the local protocol)
OH=pauli_onehot(POOLIDX)

class CondDrop(nn.Module):   # DeepSets with dropout for MC-dropout posterior
    def __init__(s,dp,dout,h=512,z=512,p=0.1):
        super().__init__()
        s.enc=nn.Sequential(nn.Linear(dp+1,h),nn.SiLU(),nn.Dropout(p),nn.Linear(h,h),nn.SiLU(),nn.Linear(h,z))
        s.post=nn.Sequential(nn.Linear(z,h),nn.SiLU(),nn.Dropout(p),nn.Linear(h,h),nn.SiLU(),nn.Dropout(p),nn.Linear(h,h),nn.SiLU(),nn.Linear(h,dout))
    def forward(s,oh,ov):
        x=torch.cat([oh,ov.unsqueeze(-1)],-1); return s.post(s.enc(x).mean(1))
net=CondDrop(4*n,THETA,p=args.drop).to(dev); opt=torch.optim.AdamW(net.parameters(),lr=3e-4); sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,args.steps)
TR=make_states(6000 if args.family=='tfim' else 20000, seed=1); print(f"[train] fixed-design cond+dropout ...",flush=True); t0=time.time()
for it in range(1,args.steps+1):
    bidx=torch.randint(0,TR.shape[0],(256,),device=dev); psb=TR[bidx]
    te=expect(psb,POOL); shn=torch.sqrt(((1-te**2).clamp_min(0))/args.shots); obs=te+shn*torch.randn_like(te)
    oh=OH.unsqueeze(0).expand(256,-1,-1); pred=net(oh,obs); st=cores_to_state(split_cores(pred))
    loss=(1-fidelity(st,psb)).mean(); opt.zero_grad(); loss.backward(); opt.step(); sch.step()
    if it%10000==0: print(f"   it{it} loss {loss.item():.4f} ({time.time()-t0:.0f}s)",flush=True)

psi_true=make_states(args.ntest,seed=999); te=expect(psi_true,POOL); shn=torch.sqrt(((1-te**2).clamp_min(0))/args.shots); obs=te+shn*torch.randn_like(te)
oh=OH.unsqueeze(0).expand(args.ntest,-1,-1)
# point estimate (dropout off)
net.eval()
with torch.no_grad(): pt=cores_to_state(canon(split_cores(net(oh,obs))))
f_pt=fidelity(psi_true,pt).mean().item()
mean_state=TR.mean(0,keepdim=True); mean_state=mean_state/mean_state.norm(); f_prior=fidelity(psi_true,mean_state.expand(args.ntest,-1)).mean().item()
print(f"\n[point] cond fid {f_pt:.3f} | prior-only mean-state {f_prior:.3f}",flush=True)
# MC-dropout posterior over observables
net.train()   # dropout ON for sampling
with torch.no_grad():
    samps=torch.stack([expect(cores_to_state(canon(split_cores(net(oh,obs)))),OBS) for _ in range(args.E)],0)  # (E,ntest,2n)
mean=samps.mean(0); std=samps.std(0).clamp_min(1e-4); otrue=expect(psi_true,OBS)
# conformal recalibration + coverage/width, averaged over 10 splits
covs=[];wids=[];rawc=[]
for sp in range(10):
    g=torch.Generator(device=dev).manual_seed(sp); idx=torch.randperm(args.ntest,generator=g,device=dev); ci,ti=idx[:args.ntest//2],idx[args.ntest//2:]
    r=((otrue-mean).abs()/std); Q=torch.quantile(r[ci].flatten(),0.9)
    lo=mean[ti]-Q*std[ti]; hi=mean[ti]+Q*std[ti]
    covs.append(((otrue[ti]>=lo)&(otrue[ti]<=hi)).float().mean().item()); wids.append((hi.clamp(max=1)-lo.clamp(min=-1)).clamp_min(0).mean().item())
    rawc.append(((otrue[ti]>=mean[ti]-1.645*std[ti])&(otrue[ti]<=mean[ti]+1.645*std[ti])).float().mean().item())
print(f"[calibration] MC-dropout posterior over {2*n} local Z/X observables, {args.E} samples:",flush=True)
print(f"  RAW 90%-interval coverage {np.mean(rawc)*100:.1f}% (uncalibrated)",flush=True)
print(f"  CONFORMAL coverage {np.mean(covs)*100:.1f}±{np.std(covs)*100:.1f}% | clipped width {np.mean(wids):.3f}±{np.std(wids):.3f}  (10 splits)",flush=True)
# reliability curve: coverage + width vs nominal level
rel={'level':[],'cov':[],'wid':[]}
for lv in [0.5,0.6,0.7,0.8,0.9,0.95]:
    cs=[];ws=[]
    for sp in range(10):
        g=torch.Generator(device=dev).manual_seed(sp); idx=torch.randperm(args.ntest,generator=g,device=dev); ci,ti=idx[:args.ntest//2],idx[args.ntest//2:]
        r=((otrue-mean).abs()/std); Q=torch.quantile(r[ci].flatten(),lv)
        lo=mean[ti]-Q*std[ti]; hi=mean[ti]+Q*std[ti]
        cs.append(((otrue[ti]>=lo)&(otrue[ti]<=hi)).float().mean().item()); ws.append((hi.clamp(max=1)-lo.clamp(min=-1)).clamp_min(0).mean().item())
    rel['level'].append(lv); rel['cov'].append(float(np.mean(cs))); rel['wid'].append(float(np.mean(ws)))
print("[reliability]",rel,flush=True)
# ---- UNMEASURED-observable UQ: calibrated intervals on observables NOT in the input protocol ----
net.train()
with torch.no_grad():
    sH=torch.stack([expect(cores_to_state(canon(split_cores(net(oh,obs)))),OBS_HELD) for _ in range(args.E)],0)
meanH=sH.mean(0); stdH=sH.std(0).clamp_min(1e-4); otH=expect(psi_true,OBS_HELD)
maeH=(meanH-otH).abs().mean().item()
mae_prior=(expect(mean_state.expand(args.ntest,-1),OBS_HELD)-otH).abs().mean().item()
covsH=[];widsH=[]
for sp in range(10):
    g=torch.Generator(device=dev).manual_seed(sp); idx=torch.randperm(args.ntest,generator=g,device=dev); ci,ti=idx[:args.ntest//2],idx[args.ntest//2:]
    r=((otH-meanH).abs()/stdH); Q=torch.quantile(r[ci].flatten(),0.9)
    lo=meanH[ti]-Q*stdH[ti]; hi=meanH[ti]+Q*stdH[ti]
    covsH.append(((otH[ti]>=lo)&(otH[ti]<=hi)).float().mean().item()); widsH.append((hi.clamp(max=1)-lo.clamp(min=-1)).clamp_min(0).mean().item())
print(f"[UNMEASURED-UQ] {OBS_HELD.shape[0]} held-out observables (long-range ZZ + weight-3), NONE in the input protocol:",flush=True)
print(f"  prediction MAE {maeH:.4f} vs prior-only-mean MAE {mae_prior:.4f}  (reconstruction predicts unmeasured observables)",flush=True)
print(f"  CONFORMAL coverage {np.mean(covsH)*100:.1f}±{np.std(covsH)*100:.1f}% | clipped width {np.mean(widsH):.3f}  (a shot-based interval does not exist: these were never measured)",flush=True)

# ================= PER-OBSERVABLE CONFORMAL (one Q_O per observable, finite-sample rank) =================
def conf_Q(rc, lv):
    N = rc.shape[0]; kk = int(np.ceil(lv*(N+1)))
    if kk > N: return torch.full((rc.shape[1],), float("inf"), device=rc.device)
    return rc.sort(dim=0).values[kk-1]
covsPO=[];widsPO=[];perobs=[]
for sp in range(10):
    g=torch.Generator(device=dev).manual_seed(sp); idx=torch.randperm(args.ntest,generator=g,device=dev); ci,ti=idx[:args.ntest//2],idx[args.ntest//2:]
    r=((otrue-mean).abs()/std); Q=conf_Q(r[ci],0.9)
    lo=mean[ti]-Q*std[ti]; hi=mean[ti]+Q*std[ti]; hit=((otrue[ti]>=lo)&(otrue[ti]<=hi)).float()
    covsPO.append(hit.mean().item()); perobs.append(hit.mean(0).cpu().numpy())
    widsPO.append((hi.clamp(max=1)-lo.clamp(min=-1)).clamp_min(0).mean().item())
po=np.stack(perobs,0).mean(0)
print("[PER-OBS calibration] %d local Z/X observables, own Q_O each, ceil(0.9*(Ncal+1))-th score:"%(2*n),flush=True)
print("  CONFORMAL coverage %.1f+-%.1f%% | clipped width %.3f+-%.3f  (10 splits)"%(np.mean(covsPO)*100,np.std(covsPO)*100,np.mean(widsPO),np.std(widsPO)),flush=True)
print("  worst/best single-observable coverage %.1f%% / %.1f%%"%(po.min()*100,po.max()*100),flush=True)
relPO={"level":[],"cov":[],"wid":[]}
for lv in [0.5,0.6,0.7,0.8,0.9,0.95]:
    cs=[];ws=[]
    for sp in range(10):
        g=torch.Generator(device=dev).manual_seed(sp); idx=torch.randperm(args.ntest,generator=g,device=dev); ci,ti=idx[:args.ntest//2],idx[args.ntest//2:]
        r=((otrue-mean).abs()/std); Q=conf_Q(r[ci],lv)
        lo=mean[ti]-Q*std[ti]; hi=mean[ti]+Q*std[ti]
        cs.append(((otrue[ti]>=lo)&(otrue[ti]<=hi)).float().mean().item()); ws.append((hi.clamp(max=1)-lo.clamp(min=-1)).clamp_min(0).mean().item())
    relPO["level"].append(lv); relPO["cov"].append(float(np.mean(cs))); relPO["wid"].append(float(np.mean(ws)))
print("[PER-OBS reliability]",relPO,flush=True)
covsHPO=[];widsHPO=[]
for sp in range(10):
    g=torch.Generator(device=dev).manual_seed(sp); idx=torch.randperm(args.ntest,generator=g,device=dev); ci,ti=idx[:args.ntest//2],idx[args.ntest//2:]
    r=((otH-meanH).abs()/stdH); Q=conf_Q(r[ci],0.9)
    lo=meanH[ti]-Q*stdH[ti]; hi=meanH[ti]+Q*stdH[ti]
    covsHPO.append(((otH[ti]>=lo)&(otH[ti]<=hi)).float().mean().item()); widsHPO.append((hi.clamp(max=1)-lo.clamp(min=-1)).clamp_min(0).mean().item())
print("[PER-OBS UNMEASURED-UQ] %d held-out observables, own Q_O each:"%OBS_HELD.shape[0],flush=True)
print("  CONFORMAL coverage %.1f+-%.1f%% | clipped width %.3f"%(np.mean(covsHPO)*100,np.std(covsHPO)*100,np.mean(widsHPO)),flush=True)
json.dump({"perobs_cover":float(np.mean(covsPO)),"perobs_cover_std":float(np.std(covsPO)),"perobs_width":float(np.mean(widsPO)),
           "perobs_width_std":float(np.std(widsPO)),"perobs_worst":float(po.min()),"perobs_best":float(po.max()),
           "perobs_reliability":relPO,"perobs_held_cover":float(np.mean(covsHPO)),"perobs_held_cover_std":float(np.std(covsHPO)),
           "perobs_held_width":float(np.mean(widsHPO))}, open("./qst_perobs_%s.json"%args.family,"w"), indent=1)

json.dump({'reliability':rel,'n':n,'family':args.family,'k':args.k,'point_fid':f_pt,'prior_fid':f_prior,'raw_cover':float(np.mean(rawc)),
           'conf_cover':float(np.mean(covs)),'conf_cover_std':float(np.std(covs)),'width':float(np.mean(wids)),'width_std':float(np.std(wids))},
          open(f'./qst_cond_cal_{args.family}.json','w'),indent=1)
print("[dumped]",flush=True)
