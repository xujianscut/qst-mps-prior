"""Closed-loop IBM hardware run with the INFORMATIVE LOCAL design (matches qst_cond.py --design local).
Prepare M held-out shallow states (spread 0.6, exactly matching qst_cond shallow family) on ibm_aachen,
measure the 63 local Paulis (single-qubit XYZ + nearest-neighbor weight-2), one batched EstimatorV2 job.
Token read from env IBM_TOKEN (never hardcoded). Conventions match hw_multi.py (verified <1e-6)."""
import os, numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit_ibm_runtime import QiskitRuntimeService, EstimatorV2
n=6; SHOTS=4096; SPREAD=0.6; M=5
seen=set(); pool=[]
def add(v):
    t=tuple(v)
    if t not in seen and any(t): seen.add(t); pool.append(t)
for q in range(n):
    for p in [1,2,3]: v=[0]*n; v[q]=p; add(v)
for q in range(n-1):
    for p1 in [1,2,3]:
        for p2 in [1,2,3]: v=[0]*n; v[q]=p1; v[q+1]=p2; add(v)
K=len(pool); print(f"[design] local: {K} observables",flush=True)
rng=np.random.default_rng(2027); TH1=[]; TH2=[]
for m in range(M):
    TH1.append(np.pi/2+SPREAD*rng.standard_normal(n)); TH2.append(SPREAD*rng.standard_normal(n))
def build_circuit(a1,a2):
    qc=QuantumCircuit(n)
    for q in range(n): qc.ry(float(a1[q]),q)
    for q in range(n-1): qc.cx(q,q+1)
    for q in range(n): qc.ry(float(a2[q]),q)
    return qc
def to_label(s): return "".join("IXYZ"[s[q]] for q in reversed(range(n)))
labels=[to_label(s) for s in pool]; obs=[SparsePauliOp(l) for l in labels]
I2=np.eye(2,dtype=complex);X=np.array([[0,1],[1,0]],dtype=complex);Y=np.array([[0,-1j],[1j,0]]);Z=np.diag([1,-1]).astype(complex); P=[I2,X,Y,Z]
def kron_all(ms):
    o=ms[0]
    for m in ms[1:]: o=np.kron(o,m)
    return o
idx=np.arange(2**n); rev=np.array([int(f"{i:0{n}b}"[::-1],2) for i in idx])
PSI=[]; EXP_IDEAL=[]
for m in range(M):
    qc=build_circuit(TH1[m],TH2[m]); psi_t=Statevector(qc).data[rev]; PSI.append(psi_t)
    EXP_IDEAL.append(np.array([(psi_t.conj()@kron_all([P[s[q]] for q in range(n)])@psi_t).real for s in pool]))
EXP_IDEAL=np.array(EXP_IDEAL); PSI=np.array(PSI)
tok=os.environ["IBM_TOKEN"]
svc=QiskitRuntimeService(channel="ibm_quantum_platform", token=tok); bk=svc.backend("ibm_aachen")
print("[hw] backend", bk.name, "pending", bk.status().pending_jobs, flush=True)
pubs=[]
for m in range(M):
    qct=transpile(build_circuit(TH1[m],TH2[m]), bk, optimization_level=2)
    obs_isa=[o.apply_layout(qct.layout) for o in obs]; pubs.append((qct, obs_isa))
est=EstimatorV2(mode=bk); est.options.default_shots=SHOTS
job=est.run(pubs); print("[hw] job", job.job_id(), "submitted, waiting...", flush=True)
res=job.result(); EXP_HW=np.array([np.array(res[m].data.evs).real for m in range(M)])
print("[hw] done. per-state mean|hw-ideal|:", np.round(np.abs(EXP_HW-EXP_IDEAL).mean(1),4), flush=True)
np.savez("./hw_local.npz", th1=np.array(TH1), th2=np.array(TH2), spread=SPREAD,
         pool=np.array(pool), exp_hw=EXP_HW, exp_ideal=EXP_IDEAL, psi_ideal=PSI, backend=bk.name, shots=SHOTS, M=M)
print("[hw] saved hw_local.npz", flush=True)
