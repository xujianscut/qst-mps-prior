"""#3: classical MPS-MLE (no learned prior) fidelity vs measurement design local1/local/local3.
Fits chi-MPS cores per test state to the measured Pauli values (random restarts), reports mean fidelity.
Shows the classical local reconstruction IMPROVES with design completeness -> MLE~0 at few obs is an
information gap, and local3 (theory-complete, 207) is the fair classical bar."""
import numpy as np, torch, torch.nn as nn
dev=torch.device("cuda:0"); cdt=torch.complex64; torch.manual_seed(0); np.random.seed(0)
n=6; D=2**n; chi=2; SPREAD=0.6; SHOTS=200
shapes=[(1,2,min(2,2**(n-1),chi))]+[ (min(2**l,2**(n-l),chi),2,min(2**(l+1),2**(n-l-1),chi)) for l in range(1,n-1)]+[(min(2**(n-1),2,chi),2,1)]
sizes=[int(np.prod(s)) for s in shapes]; off=np.cumsum([0]+sizes); TH=2*int(off[-1])
def split_cores(th):
    b=th.shape[0]; c=torch.complex(th[:,:TH//2],th[:,TH//2:]); return [c[:,off[k]:off[k+1]].reshape(b,*shapes[k]) for k in range(n)]
def cores_to_state(cs):
    b=cs[0].shape[0]; psi=cs[0].reshape(b,-1,cs[0].shape[-1])
    for k in range(1,n): psi=torch.einsum("bar,brdc->badc",psi,cs[k]).reshape(b,-1,cs[k].shape[-1])
    psi=psi.reshape(b,D); return psi/psi.norm(dim=1,keepdim=True)
I2=torch.eye(2,dtype=cdt,device=dev);X=torch.tensor([[0,1],[1,0]],dtype=cdt,device=dev);Y=torch.tensor([[0,-1j],[1j,0]],dtype=cdt,device=dev);Z=torch.tensor([[1,0],[0,-1]],dtype=cdt,device=dev);P=[I2,X,Y,Z]
def kron_list(ms):
    o=ms[0]
    for m in ms[1:]: o=torch.kron(o,m)
    return o
def apply_ry(state,ang):
    b=state.shape[0]
    for q in range(n):
        c=torch.cos(ang[:,q]/2).to(cdt); s=torch.sin(ang[:,q]/2).to(cdt)
        st=state.view(b,2**q,2,2**(n-q-1)); a0=st[:,:,0,:]; a1=st[:,:,1,:]
        state=torch.stack([c.view(-1,1,1)*a0-s.view(-1,1,1)*a1, s.view(-1,1,1)*a0+c.view(-1,1,1)*a1],dim=2).reshape(b,D)
    return state
def apply_cnot(state,ctrl,tgt):
    idx=torch.arange(D,device=dev); bc=(idx>>(n-1-ctrl))&1; sw=idx^((bc<<(n-1-tgt))); out=state.clone(); m=bc==1; out[:,idx[m]]=state[:,sw[m]]; return out
def shallow(b,seed):
    g=torch.Generator(device=dev).manual_seed(seed); th1=np.pi/2+SPREAD*torch.randn(b,n,generator=g,device=dev); th2=SPREAD*torch.randn(b,n,generator=g,device=dev)
    st=torch.zeros(b,D,dtype=cdt,device=dev); st[:,0]=1.0; st=apply_ry(st,th1)
    for q in range(n-1): st=apply_cnot(st,q,q+1)
    st=apply_ry(st,th2); return st/st.norm(dim=1,keepdim=True)
def design(kind):
    seen=set(); idx=[]
    def add(v):
        t=tuple(v)
        if t not in seen and any(t): seen.add(t); idx.append(list(v))
    for q in range(n):
        for p in [1,2,3]: v=[0]*n; v[q]=p; add(v)
    if kind=="local":
        for q in range(n-1):
            for p1 in [1,2,3]:
                for p2 in [1,2,3]: v=[0]*n; v[q]=p1;v[q+1]=p2; add(v)
    if kind=="local3":
        for a in range(n-2):
            W=[a,a+1,a+2]
            for combo in np.ndindex(4,4,4):
                if all(c==0 for c in combo): continue
                v=[0]*n
                for w,c in zip(W,combo): v[w]=c
                add(v)
    return torch.tensor(idx,device=dev)
def paulis(idx): return torch.stack([kron_list([P[int(idx[j,q])] for q in range(n)]) for j in range(idx.shape[0])],0)
def expect(psi,ops): return torch.einsum("bi,kij,bj->bk",psi.conj(),ops,psi).real
def fidelity(a,b): return (torch.einsum("bi,bi->b",a.conj(),b).abs())**2
NT=20; psi_t=shallow(NT,999)
for kind in ["local1","local","local3"]:
    idx=design("local" if kind=="local" else ("local3" if kind=="local3" else "l1"))
    if kind=="local1": idx=design("l1")
    OPS=paulis(idx); K=idx.shape[0]
    te=expect(psi_t,OPS); shn=torch.sqrt(((1-te**2).clamp_min(0))/SHOTS); o=te+shn*torch.randn_like(te)
    # per-state MLE: fit cores to minimize ||<P>_theta - o||^2, best of restarts
    best=torch.full((NT,),-1.0,device=dev)
    for rs in range(3):
        th=(0.1*torch.randn(NT,TH,device=dev,generator=torch.Generator(device=dev).manual_seed(rs))).requires_grad_(True)
        opt=torch.optim.Adam([th],lr=0.05)
        for it in range(500):
            st=cores_to_state(split_cores(th)); pred=expect(st,OPS); loss=((pred-o)**2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            f=fidelity(psi_t,cores_to_state(split_cores(th))); best=torch.maximum(best,f)
    print(f"[MLE-design] {kind:7s} ({K:3d} obs): MPS-MLE fidelity {best.mean().item():.4f} +- {best.std().item():.4f}",flush=True)
