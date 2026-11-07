import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"font.size":11,"axes.spines.top":False,"axes.spines.right":False,"figure.dpi":150,"font.family":"serif","mathtext.fontset":"cm"})
D="./"
C={"prior":"#7f7f7f","cond":"#1f77b4","shuf":"#d62728","gain":"#2ca02c"}
ns=[6,8,10]; fid=[0.967,0.939,0.900]; fstd=[0.046,0.077,0.093]
prior=[0.375,0.258,0.187]; shuf=[0.154,0.095,0.064]; gain=[0.592,0.681,0.713]
fig,ax=plt.subplots(1,2,figsize=(9,3.4))
ax[0].errorbar(ns,fid,yerr=fstd,fmt="-o",color=C["cond"],ms=6,capsize=3,label="conditional (ours)")
ax[0].plot(ns,prior,"--^",color=C["prior"],ms=5,label="prior-only ($k{=}0$)")
ax[0].plot(ns,shuf,":s",color=C["shuf"],ms=5,label="shuffled control")
ax[0].set_xticks(ns); ax[0].set_xlabel("qubits $n$"); ax[0].set_ylabel("reconstruction fidelity"); ax[0].set_ylim(0,1.02)
ax[0].set_title("(a) Fidelity holds; family de-concentrates"); ax[0].legend(frameon=False,fontsize=9,loc="center left")
ax[1].bar(range(len(ns)),gain,0.5,color=C["gain"]); ax[1].set_xticks(range(len(ns))); ax[1].set_xticklabels(ns)
for i,g in enumerate(gain): ax[1].text(i,g+0.01,f"+{g:.2f}",ha="center",fontsize=9)
ax[1].set_xlabel("qubits $n$"); ax[1].set_ylabel("gain over prior-only"); ax[1].set_ylim(0,0.85)
ax[1].set_title("(b) Measurement gain grows with $n$")
fig.tight_layout(); fig.savefig(D+"fig_scaling_cond.pdf",bbox_inches="tight"); print("fig_scaling_cond")
