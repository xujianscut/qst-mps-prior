"""Reviewer-requested amortized state-reconstruction baseline that EXPLOITS the known circuit ansatz:
regress the 2n shallow-circuit angles directly from the 63 local observables, rebuild the state, measure
fidelity. If this hits ~0.99, it shows that GIVEN the exact ansatz one can beat the ansatz-agnostic MPS
estimator---which is the honest point (our method does not assume the circuit structure)."""
import numpy as np, torch, torch.nn as nn
dev=torch.device("cuda:0"); cdt=torch.complex64
torch.manual_seed(0); np.random.seed(0)
n=6; D=2**n; SPREAD=0.6; SHOTS=200
def apply_ry(state,ang):
    b=state.shape[0]
    for q in range(n):
        c=torch.cos(ang[:,q]/2).to(cdt); s=torch.sin(ang[:,q]/2).to(cdt)
        st=state.view(b,2**q,2,2**(n-q-1)); a0=st[:,:,0,:]; a1=st[:,:,1,:]
        state=torch.stack([c.view(-1,1,1)*a0-s.view(-1,1,1)*a1, s.view(-1,1,1)*a0+c.view(-1,1,1)*a1],dim=2).reshape(b,D)
    return state
def apply_cnot(state,ctrl,tgt):
    idx=torch.arange(D,device=dev); bit_c=(idx>>(n-1-ctrl))&1
    swap=idx^((bit_c<<(n-1-tgt))); out=state.clone(); mask=bit_c==1; out[:,idx[mask]]=state[:,swap[mask]]; return out
def build(th1,th2):
    b=th1.shape[0]; st=torch.zeros(b,D,dtype=cdt,device=dev); st[:,0]=1.0; st=apply_ry(st,th1)
    for q in range(n-1): st=apply_cnot(st,q,q+1)
    st=apply_ry(st,th2); return st/st.norm(dim=1,keepdim=True)
def gen_angles(b,seed):
    g=torch.Generator(device=dev).manual_seed(seed)
    th1=np.pi/2+SPREAD*torch.randn(b,n,generator=g,device=dev); th2=SPREAD*torch.randn(b,n,generator=g,device=dev)
    return th1,th2
# local design (same order as qst_cond)
seen=set(); pool=[]
def add(v):
    t=tuple(v)
    if t not in seen and any(t): seen.add(t); pool.append(list(v))
for q in range(n):
    for p in [1,2,3]: v=[0]*n; v[q]=p; add(v)
for q in range(n-1):
    for p1 in [1,2,3]:
        for p2 in [1,2,3]: v=[0]*n; v[q]=p1; v[q+1]=p2; add(v)
POOLIDX=torch.tensor(pool,device=dev); K=len(pool)
I2=torch.eye(2,dtype=cdt,device=dev);X=torch.tensor([[0,1],[1,0]],dtype=cdt,device=dev);Y=torch.tensor([[0,-1j],[1j,0]],dtype=cdt,device=dev);Z=torch.tensor([[1,0],[0,-1]],dtype=cdt,device=dev);P=[I2,X,Y,Z]
def kron_list(ms):
    o=ms[0]
    for m in ms[1:]: o=torch.kron(o,m)
    return o
OPS=torch.stack([kron_list([P[int(POOLIDX[k,q])] for q in range(n)]) for k in range(K)])
def expect(psi):  # (b,D) -> (b,K)
    return torch.einsum("bi,kij,bj->bk", psi.conj(), OPS, psi).real
def fidelity(a,b): return (torch.einsum("bi,bi->b",a.conj(),b).abs())**2
# training data (compute expectations in chunks to avoid a huge einsum)
print("[angle-baseline] building data ...",flush=True)
B=6000; th1,th2=gen_angles(B,111); psb=build(th1,th2)
def expect_chunked(psi,cs=512):
    outs=[]
    for i in range(0,psi.shape[0],cs): outs.append(expect(psi[i:i+cs]))
    return torch.cat(outs,0)
te=expect_chunked(psb); shn=torch.sqrt(((1-te**2).clamp_min(0))/SHOTS); obs=te+shn*torch.randn_like(te)
targ=torch.cat([th1-np.pi/2,th2],1)
net=nn.Sequential(nn.Linear(K,512),nn.ReLU(),nn.Linear(512,512),nn.ReLU(),nn.Linear(512,512),nn.ReLU(),nn.Linear(512,2*n)).to(dev)
opt=torch.optim.AdamW(net.parameters(),lr=3e-4)
print("[angle-baseline] training ...",flush=True)
for it in range(15000):
    idx=torch.randint(0,B,(256,),device=dev); pred=net(obs[idx]); loss=((pred-targ[idx])**2).mean()
    opt.zero_grad(); loss.backward(); opt.step()
    if it%5000==0: print(f"  it{it} angle-mse {loss.item():.4f}",flush=True)
# test
th1t,th2t=gen_angles(200,999); pst=build(th1t,th2t)
tet=expect_chunked(pst); shnt=torch.sqrt(((1-tet**2).clamp_min(0))/SHOTS); obst=tet+shnt*torch.randn_like(tet)
with torch.no_grad():
    pr=net(obst); ph1=pr[:,:n]+np.pi/2; ph2=pr[:,n:]; rec=build(ph1,ph2)
    f=fidelity(pst,rec)
print(f"[angle-baseline] known-ansatz angle regression: fid {f.mean().item():.4f}+-{f.std().item():.4f} (per-state q10 {f.quantile(0.1).item():.3f} q90 {f.quantile(0.9).item():.3f})",flush=True)
