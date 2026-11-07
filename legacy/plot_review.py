import json, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':150,'font.family':'serif','mathtext.fontset':'cm'})
import sys
FAM=sys.argv[1] if len(sys.argv)>1 else 'random'
D=json.load(open(f'./qst_review_{FAM}_n6.json'))
C=D['C_ood']['delta']
d=[r['delta'] for r in C]
flow=[r['flow_fid'] for r in C]; mle=[r['mle_fid'] for r in C]; sup=[r['sup_fid'] for r in C]
recon_on=[r['flow_reconPriorFid'] for r in C]; true_on=[r['true_priorFid'] for r in C]
fig,ax=plt.subplots(1,2,figsize=(9,3.4))
# (a) fidelity-to-true vs mismatch
ax[0].plot(d,flow,'-o',color='#1f77b4',ms=5,label='MPS-flow (ours)')
ax[0].plot(d,sup,'-^',color='#ff7f0e',ms=5,label='supervised amortized')
ax[0].plot(d,mle,'--s',color='#d62728',ms=4,label='MPS-MLE')
ax[0].set_xlabel('off-manifold mismatch $\\delta$'); ax[0].set_ylabel('fidelity to true state')
ax[0].set_title('(a) Graceful degradation under prior mismatch'); ax[0].legend(frameon=False,fontsize=8.5); ax[0].set_ylim(0,1.02)
# (b) hallucination probe: recon stays on manifold while truth leaves it
ax[1].plot(d,true_on,'-o',color='#2ca02c',ms=5,label='true state $\\to$ manifold')
ax[1].plot(d,recon_on,'-s',color='#1f77b4',ms=5,label='our recon $\\to$ manifold')
ax[1].plot(d,flow,':',color='#1f77b4',lw=1.3,label='our recon $\\to$ true (fidelity)')
ax[1].set_xlabel('off-manifold mismatch $\\delta$'); ax[1].set_ylabel('max fidelity to training family')
ax[1].set_title('(b) Hallucination probe'); ax[1].legend(frameon=False,fontsize=8); ax[1].set_ylim(0,1.02)
fig.tight_layout(); fig.savefig('./fig_ood.pdf',bbox_inches='tight')
print("saved fig_ood.pdf")
# print a compact summary for the tex
print("delta:",d); print("flow:",[round(x,3) for x in flow]); print("mle:",[round(x,3) for x in mle]); print("sup:",[round(x,3) for x in sup])
print("recon_on:",[round(x,3) for x in recon_on]); print("true_on:",[round(x,3) for x in true_on])
print("critical:",D['C_ood'].get('critical'))
print("A_noiseless:",{k:round(v['fid'],3) for k,v in D['A_mle_sanity']['noiseless'].items()})
print("A_resid:",{k:('%.1e'%v['resid']) for k,v in D['A_mle_sanity']['noiseless'].items()})
print("A_conv:",{k:round(v['fid'],3) for k,v in D['A_mle_sanity']['conv'].items()})
print("B_supervised:",D['B_supervised'])
print("D_perobs:",D['D_perobs_conformal'])
