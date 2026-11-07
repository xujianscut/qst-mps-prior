import numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':150,'font.family':'serif','mathtext.fontset':'cm'})
S=np.load('./postval_summary.npz'); E=np.load('./postval_example.npz')
po,pg=S['pm_ours'],S['pm_gold']; corr=np.corrcoef(po,pg)[0,1]
fig,ax=plt.subplots(1,2,figsize=(9,3.5))
ax[0].plot([-1,1],[-1,1],'k:',lw=1); ax[0].scatter(pg,po,s=14,alpha=.5,color='#1f77b4')
ax[0].set_xlabel('gold (SIR) posterior mean'); ax[0].set_ylabel('our posterior mean')
ax[0].set_title(f'(a) Posterior means agree ($r{{=}}{corr:.3f}$)'); ax[0].set_xlim(-1,1); ax[0].set_ylim(-1,1)
# example marginals for a few observables
ours=E['ours']; gold=E['gold']; otrue=E['otrue']  # (K,2n),(K,2n),(2n,)
sel=[0,2,7]; cols=['#1f77b4','#ff7f0e','#2ca02c']
for i,j in enumerate(sel):
    ax[1].hist(gold[:,j],bins=30,density=True,histtype='step',color=cols[i],lw=1.6,ls='--')
    ax[1].hist(ours[:,j],bins=30,density=True,histtype='stepfilled',color=cols[i],alpha=.25)
    ax[1].axvline(otrue[j],color=cols[i],lw=0.8)
ax[1].plot([],[],'-',color='gray',label='ours (filled)'); ax[1].plot([],[],'--',color='gray',label='gold SIR')
ax[1].set_xlabel('observable value $\\langle O\\rangle$'); ax[1].set_ylabel('posterior density')
ax[1].set_title('(b) Posterior marginals (3 observables)'); ax[1].legend(frameon=False,fontsize=9)
fig.tight_layout(); fig.savefig('./fig_postval.pdf',bbox_inches='tight'); print('saved fig_postval.pdf')
