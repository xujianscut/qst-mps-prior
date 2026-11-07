"""#2 NATIVE MPS-contraction pipeline: Pauli expectations + overlaps via transfer matrices at O(n*chi^3),
NEVER forming the 2^n state vector. Runs the full prior+proximal pipeline at n=16,20 on a random-MPS
family (generatable natively). Refutes 'only an O(2^n) implementation'. Includes a small-n cross-check
against the dense contraction to prove the native primitives are correct."""
import time, argparse, json, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
ap=argparse.ArgumentParser(); ap.add_argument('--dev',default='cuda:2'); ap.add_argument('--n',type=int,default=16)
ap.add_argument('--chi',type=int,default=2); ap.add_argument('--sigma',type=float,default=0.3); ap.add_argument('--shots',type=int,default=200)
ap.add_argument('--steps',type=int,default=30000); ap.add_argument('--ntest',type=int,default=40); ap.add_argument('--k',type=int,default=32)
ap.add_argument('--bestof',type=int,default=8); ap.add_argument('--crosscheck',action='store_true')
args=ap.parse_args(); dev=torch.device(args.dev); torch.manual_seed(0); np.random.seed(0); cdt=torch.complex64
n,chi=args.n,args.chi
bonds=[min(2**k,2**(n-k),chi) for k in range(n+1)]; shapes=[(bonds[k],2,bonds[k+1]) for k in range(n)]
sizes=[np.prod(s) for s in shapes]; off=np.cumsum([0]+sizes); THETA=2*int(off[-1])
print(f"[cfg] n={n} chi={chi} core(complex)={int(off[-1])} theta={THETA}  (dense 2^n={2**n if n<=24 else 'HUGE'})",flush=True)
I2=torch.tensor([[1,0],[0,1]],dtype=cdt,device=dev);Xp=torch.tensor([[0,1],[1,0]],dtype=cdt,device=dev)
Yp=torch.tensor([[0,-1j],[1j,0]],dtype=cdt,device=dev);Zp=torch.tensor([[1,0],[0,-1]],dtype=cdt,device=dev)
PAULI=[I2,Xp,Yp,Zp]
def split_cores(th):
    b=th.shape[0]; c=torch.complex(th[:,:THETA//2],th[:,THETA//2:]); return [c[:,off[k]:off[k+1]].reshape(b,*shapes[k]) for k in range(n)]
def cat_cores(cores):
    b=cores[0].shape[0]; c=torch.cat([g.reshape(b,-1) for g in cores],1); return torch.cat([c.real,c.imag],1)

# ---------- NATIVE transfer-matrix contractions: O(n chi^3), no 2^n ----------
def overlap_mps(A,B):
    # <psi_A | psi_B> via transfer matrices. A,B: lists of (b, l, 2, r). Returns (b,) complex.
    b=A[0].shape[0]; L=torch.ones(b,1,1,dtype=cdt,device=dev)   # (b, lA, lB)
    for i in range(n):
        Ai=A[i]; Bi=B[i]                                        # (b,l,2,r)
        # tmp[b, lA, s, rB] = sum_lB L[b,lA,lB] B[b,lB,s,rB]
        tmp=torch.einsum('bxy,bysr->bxsr', L, Bi)
        # L'[b, rA, rB] = sum_{lA,s} conj(A[b,lA,s,rA]) tmp[b,lA,s,rB]
        L=torch.einsum('bxsr,bxsq->brq', Ai.conj(), tmp)
    return L.reshape(b)
def norm2_mps(A): return overlap_mps(A,A).real.clamp_min(1e-12)
def pauli_expect_mps(A, pstr):
    # <psi|(x)P_i|psi> / <psi|psi> for a Pauli string pstr (list of n ints in 0..3). Native, O(n chi^3).
    b=A[0].shape[0]; L=torch.ones(b,1,1,dtype=cdt,device=dev)
    for i in range(n):
        Ai=A[i]; P=PAULI[pstr[i]]                               # (2,2)
        PB=torch.einsum('st,bxtr->bxsr', P, Ai)                 # apply P to ket physical index
        tmp=torch.einsum('bxy,bysr->bxsr', L, PB)
        L=torch.einsum('bxsr,bxsq->brq', Ai.conj(), tmp)
    num=L.reshape(b).real
    return num/norm2_mps(A)
def expect_batch_mps(A, pstrs):
    return torch.stack([pauli_expect_mps(A,p) for p in pstrs],1)  # (b,k)
def fidelity_mps(A,B):
    ov=overlap_mps(A,B); return (ov.abs()**2)/(norm2_mps(A)*norm2_mps(B))
def canon(cores):
    b=cores[0].shape[0]; cs=[g.clone() for g in cores]
    for k in range(n-1):
        M=cs[k].reshape(b,bonds[k]*2,bonds[k+1]); Q,Rm=torch.linalg.qr(M)
        d=torch.diagonal(Rm,dim1=-2,dim2=-1); ph=(d/d.abs().clamp_min(1e-9)).conj()
        Q=Q*ph.unsqueeze(1); Rm=ph.unsqueeze(2)*Rm; cs[k]=Q.reshape(b,bonds[k],2,bonds[k+1])
        cs[k+1]=torch.einsum('brs,bsjt->brjt',Rm,cs[k+1].reshape(b,bonds[k+1],2,bonds[k+2])).reshape(b,*shapes[k+1])
    nrm=torch.sqrt(norm2_mps(cs)).clamp_min(1e-9); cs[-1]=cs[-1]/nrm.view(-1,1,1,1)
    return cs
def make_random_mps(b):
    g=torch.Generator(device=dev).manual_seed(777)
    base=[torch.randn(1,*s,generator=g,device=dev)+1j*torch.randn(1,*s,generator=g,device=dev) for s in shapes]
    cores=[(base[k]+args.sigma*(torch.randn(b,*shapes[k],device=dev)+1j*torch.randn(b,*shapes[k],device=dev))) for k in range(n)]
    return canon(cores)
def rand_pstrs(k,seed):
    rng=np.random.default_rng(seed); out=[]; seen=set()
    while len(out)<k:
        s=tuple(rng.integers(0,4,size=n).tolist())
        if s==(0,)*n or s in seen: continue
        seen.add(s); out.append(list(s))
    return out
PSTRS=rand_pstrs(args.k,1)

# ---------- optional cross-check native vs dense at small n ----------
if args.crosscheck and n<=8:
    def kron_list(ms):
        o=ms[0]
        for m in ms[1:]: o=torch.kron(o,m)
        return o
    def dense_state(cores):
        b=cores[0].shape[0]; psi=cores[0].reshape(b,2,bonds[1])
        for k in range(1,n): psi=torch.einsum('bar,brjs->bajs',psi,cores[k].reshape(b,bonds[k],2,bonds[k+1])).reshape(b,2**(k+1),bonds[k+1])
        psi=psi.reshape(b,2**n); return psi/psi.norm(dim=1,keepdim=True).clamp_min(1e-9)
    A=make_random_mps(5)
    psid=dense_state(A)
    for p in PSTRS[:3]:
        Pd=kron_list([PAULI[i] for i in p]); ed=torch.einsum('bi,ij,bj->b',psid.conj(),Pd,psid).real
        en=pauli_expect_mps(A,p)
        print(f"[crosscheck] Pauli {p[:4]}...: dense {ed[0].item():.6f} native {en[0].item():.6f} diff {(ed-en).abs().max().item():.2e}",flush=True)

# ---------- flow prior over cores ----------
print("[prep] bank + flow ...",flush=True)
BANK=cat_cores(make_random_mps(20000)); BM,BS=BANK.mean(0,keepdim=True),BANK.std(0,keepdim=True).clamp_min(1e-4); BANKn=(BANK-BM)/BS
sf=lambda t:(t-BM)/BS; uf=lambda t:t*BS+BM
class Vel(nn.Module):
    def __init__(s,d,h=1024):
        super().__init__(); s.t=nn.Sequential(nn.Linear(1,128),nn.SiLU(),nn.Linear(128,128))
        s.n=nn.Sequential(nn.Linear(d+128,h),nn.SiLU(),nn.Linear(h,h),nn.SiLU(),nn.Linear(h,h),nn.SiLU(),nn.Linear(h,h),nn.SiLU(),nn.Linear(h,d))
    def forward(s,x,t): return s.n(torch.cat([x,s.t(t.view(-1,1))],1))
vel=Vel(THETA).to(dev); opt=torch.optim.AdamW(vel.parameters(),lr=2e-4); sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,args.steps)
torch.cuda.reset_peak_memory_stats(dev); t0=time.time()
for it in range(1,args.steps+1):
    th=BANKn[torch.randint(0,BANKn.shape[0],(256,),device=dev)]; eps=torch.randn_like(th); t=torch.rand(th.shape[0],device=dev)
    loss=F.mse_loss(vel((1-t).view(-1,1)*eps+t.view(-1,1)*th,t),th-eps); opt.zero_grad(); loss.backward(); opt.step(); sch.step()
    if it%10000==0: print(f"   it{it} loss {loss.item():.4f} ({time.time()-t0:.0f}s)",flush=True)
train_s=time.time()-t0

# ---------- native proximal reconstruction ----------
def meas_loss(cores,obs):
    e=expect_batch_mps(cores,PSTRS); return (e-obs).pow(2).mean(1)
def prox(anchor_std,obs,n_inner=12,lam=0.3,lr=0.1):
    x=uf(anchor_std).clone().requires_grad_(True); op=torch.optim.Adam([x],lr=lr)
    A0=[c.detach() for c in split_cores(uf(anchor_std))]
    for _ in range(n_inner):
        cs=split_cores(x)
        f=(overlap_mps(cs,A0).abs()**2)/(norm2_mps(cs)*norm2_mps(A0)).clamp_min(1e-9)
        loss=meas_loss(cs,obs).mean()+lam*(1-f).mean()
        op.zero_grad(); loss.backward(); op.step()
    with torch.no_grad(): return sf(cat_cores(canon(split_cores(x))))
def flow_recon(obs,ns=16):
    b=obs.shape[0]
    def one():
        th=torch.randn(b,THETA,device=dev)
        for i in range(ns):
            t=i/ns; t1=(i+1)/ns
            with torch.no_grad(): v=vel(th,torch.full((b,),t,device=dev))
            x0=th+(1-t)*v; x0=prox(x0,obs); eps=(th-t*x0)/max(1-t,1e-3)
            th=((1-t1)*eps+t1*x0).clamp(-6,6)
        return split_cores(uf(th))
    cands=[one() for _ in range(args.bestof)]
    ml=torch.stack([meas_loss(c,obs) for c in cands],0); sel=ml.argmin(0)
    best=[torch.stack([cands[sel[j]][i][j] for j in range(b)],0) for i in range(n)]
    return canon(best)
def mps_mle(obs,restarts=3,iters=200,lr=0.1):
    b=obs.shape[0]; best=None;beste=None
    for _ in range(restarts):
        x=(0.3*torch.randn(b,THETA,device=dev)).requires_grad_(True); op=torch.optim.Adam([x],lr=lr)
        for _ in range(iters):
            loss=meas_loss(split_cores(x),obs).mean(); op.zero_grad(); loss.backward(); op.step()
        with torch.no_grad():
            e=meas_loss(split_cores(x),obs)
            if best is None: best,beste=x.detach().clone(),e
            else: sw=e<beste; best[sw]=x.detach()[sw]; beste=torch.where(sw,e,beste)
    return canon(split_cores(best))

A_true=make_random_mps(args.ntest)
te=expect_batch_mps(A_true,PSTRS); sh=torch.sqrt(((1-te**2).clamp_min(0))/args.shots); obs=te+sh*torch.randn_like(te)
t1=time.time(); rec=flow_recon(obs); infer_s=time.time()-t1
f_flow=fidelity_mps(A_true,rec).mean().item()
f_mle=fidelity_mps(A_true,mps_mle(obs)).mean().item()
peak_gb=torch.cuda.max_memory_allocated(dev)/1e9
print(f"\n================ NATIVE MPS pipeline n={n} (NO 2^n object) ================",flush=True)
print(f"  k={args.k}: flow fid {f_flow:.4f}  |  MPS-MLE fid {f_mle:.4f}  |  ratio {f_flow/max(f_mle,1e-6):.1f}x",flush=True)
print(f"  peak GPU mem {peak_gb:.3f} GB   train {train_s:.0f}s   inference(batch {args.ntest}) {infer_s:.1f}s",flush=True)
print(f"  core params (complex) {int(off[-1])} vs dense-state dim 2^{n}={2**n}",flush=True)
json.dump({'n':n,'chi':chi,'k':args.k,'flow':f_flow,'mle':f_mle,'core_params':int(off[-1]),
           'peak_gb':peak_gb,'train_s':train_s,'infer_s':infer_s,'ntest':args.ntest},
          open(f'./qst_native_n{n}.json','w'),indent=1)
print("[dumped]",flush=True)
