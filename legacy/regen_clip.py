import json, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':150,'font.family':'serif','mathtext.fontset':'cm'})
D={}
for l in open('./qst_results_v5.jsonl'):
    d=json.loads(l); D[(d['family'],d['n'])]=d
C={'flow':'#1f77b4','pcn':'#9467bd','mle':'#d62728','nn':'#2ca02c'}; mk={6:'o',8:'s',10:'^'}
def cs(fam,n,who): return D[(fam,n)]['calib'][who]['sharp_clip']

# fig_baselines: clipped sharpness ours/pcn/mle (log-y)
groups=[('tfim',6,'TFIM $n{=}6$'),('tfim',8,'TFIM $n{=}8$'),('random',8,'random $n{=}8$')]
fig,ax=plt.subplots(1,1,figsize=(5.2,3.4)); x=np.arange(len(groups)); w=0.26
o=[cs(f,n,'flow') for f,n,_ in groups]; p=[cs(f,n,'pcn') for f,n,_ in groups]; m=[cs(f,n,'mle') for f,n,_ in groups]
ax.bar(x-w,o,w,color=C['flow'],label='MPS-flow (ours)'); ax.bar(x,p,w,color=C['pcn'],label='pCN Bayesian'); ax.bar(x+w,m,w,color=C['mle'],label='MLE ensemble')
ax.axhline(2.0,ls=':',color='gray',lw=1); ax.text(len(groups)-0.5,2.05,'max physical width $=2$',fontsize=7.5,ha='right')
for i in range(len(groups)): ax.text(i-w,o[i]+0.03,f'{o[i]:.2f}',ha='center',fontsize=8)
ax.set_xticks(x); ax.set_xticklabels([g[2] for g in groups]); ax.set_ylabel('clipped interval width $\\downarrow$'); ax.set_ylim(0,2.3)
ax.set_title('Calibrated width (clipped to $[-1,1]$) at $\\sim\\!90\\%$ cover'); ax.legend(frameon=False,fontsize=9)
fig.tight_layout(); fig.savefig('./fig_baselines.pdf',bbox_inches='tight')

# fig_tfim: (a) fidelity vs k, (b) clipped sharpness bars
TF=[6,8,10]; fig,ax=plt.subplots(1,2,figsize=(9,3.4))
for n in TF:
    d=D[('tfim',n)]; k=d['k']
    ax[0].plot(k,d['flow'],'-',marker=mk[n],color=C['flow'],ms=5,label=f'$n{{=}}{n}$')
    ax[0].plot(k,d['nn'],':',color=C['nn'],lw=1.3,alpha=.85); ax[0].plot(k,d['mps_mle'],'--',marker=mk[n],color=C['mle'],ms=3,alpha=.6)
ax[0].plot([],[],'-',color=C['flow'],label='MPS-flow (ours)');ax[0].plot([],[],':',color=C['nn'],label='NN (ceiling)');ax[0].plot([],[],'--',color=C['mle'],label='MPS-MLE')
ax[0].set_xscale('log',base=2); ax[0].set_xticks(k); ax[0].set_xticklabels(k); ax[0].set_ylim(0,1.03)
ax[0].set_xlabel('measurements $k$'); ax[0].set_ylabel('reconstruction fidelity'); ax[0].set_title('(a) TFIM ground states')
h,l=ax[0].get_legend_handles_labels(); keep=['MPS-flow (ours)','NN (ceiling)','MPS-MLE']
ax[0].legend([h[l.index(x)] for x in keep],keep,frameon=False,fontsize=8.5,loc='center right')
x=np.arange(len(TF)); w=0.35; fsh=[cs('tfim',n,'flow') for n in TF]; msh=[cs('tfim',n,'mle') for n in TF]
ax[1].bar(x-w/2,fsh,w,color=C['flow'],label='ours'); ax[1].bar(x+w/2,msh,w,color=C['mle'],label='MLE'); ax[1].axhline(2,ls=':',color='gray',lw=1)
for i,(f,mm2) in enumerate(zip(fsh,msh)): ax[1].text(i,max(f,mm2)+0.03,f'{mm2/f:.0f}$\\times$',ha='center',fontsize=9)
ax[1].set_xticks(x); ax[1].set_xticklabels([f'$n{{=}}{n}$' for n in TF]); ax[1].set_ylabel('clipped width $\\downarrow$'); ax[1].set_ylim(0,2.3)
ax[1].set_title('(b) Sharper calibrated intervals'); ax[1].legend(frameon=False,fontsize=9)
fig.tight_layout(); fig.savefig('./fig_tfim.pdf',bbox_inches='tight')

# fig_calib: (a) conformal coverage per n (random), (b) clipped sharpness ours vs mle
RN=[6,8,10]; fig,ax=plt.subplots(1,2,figsize=(9,3.4)); x=np.arange(len(RN)); w=0.35
cov=[D[('random',n)]['calib']['flow']['conf']*100 for n in RN]
ax[0].bar(x,cov,0.5,color=C['flow']); ax[0].axhline(90,ls=':',color='k',lw=1); ax[0].text(len(RN)-0.5,91,'nominal 90\\%',fontsize=8,ha='right')
ax[0].set_xticks(x); ax[0].set_xticklabels([f'$n{{=}}{n}$' for n in RN]); ax[0].set_ylabel('conformal coverage (\\%)'); ax[0].set_ylim(0,100)
ax[0].set_title('(a) Conformal coverage $\\approx$ nominal')
fsh=[cs('random',n,'flow') for n in RN]; msh=[cs('random',n,'mle') for n in RN]
ax[1].bar(x-w/2,fsh,w,color=C['flow'],label='MPS-flow (ours)'); ax[1].bar(x+w/2,msh,w,color=C['mle'],label='MLE ensemble'); ax[1].axhline(2,ls=':',color='gray',lw=1)
ax[1].set_xticks(x); ax[1].set_xticklabels([f'$n{{=}}{n}$' for n in RN]); ax[1].set_ylabel('clipped width $\\downarrow$'); ax[1].set_ylim(0,2.3)
ax[1].set_title('(b) Sharper intervals (random family)'); ax[1].legend(frameon=False,fontsize=9)
fig.tight_layout(); fig.savefig('./fig_calib.pdf',bbox_inches='tight')

# fig_reliability (random): coverage vs level + clipped width vs level
fig,ax=plt.subplots(1,2,figsize=(9,3.4)); cols={6:'#9ecae1',8:'#4292c6',10:'#08519c'}
for n in RN:
    r=D[('random',n)]['reliability']; lv=r['level']
    ax[0].plot(lv,r['flow_cov'],'-o',color=cols[n],ms=4,label=f'ours $n{{=}}{n}$')
    ax[1].plot(lv,r['flow_sharp'],'-o',color=cols[n],ms=4,label=f'ours $n{{=}}{n}$'); ax[1].plot(lv,r['mle_sharp'],'--',color=C['mle'],alpha=.5,lw=1)
ax[0].plot([0.5,0.95],[0.5,0.95],'k:',lw=1,label='ideal'); ax[0].set_xlabel('nominal level $1-\\alpha$'); ax[0].set_ylabel('empirical coverage')
ax[0].set_title('(a) Reliability: conformal coverage'); ax[0].legend(frameon=False,fontsize=8.5)
ax[1].plot([],[],'--',color=C['mle'],label='MLE ensemble'); ax[1].set_xlabel('nominal level $1-\\alpha$'); ax[1].set_ylabel('clipped width $\\downarrow$')
ax[1].set_title('(b) Sharpness vs.\\ level'); ax[1].legend(frameon=False,fontsize=8.5)
fig.tight_layout(); fig.savefig('./fig_reliability.pdf',bbox_inches='tight')
print("regenerated fig_baselines, fig_tfim, fig_calib, fig_reliability with CLIPPED widths")
