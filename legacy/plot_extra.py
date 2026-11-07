import json, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':150,
                     'font.family':'serif','mathtext.fontset':'cm'})
D={}
for l in open('./qst_results_v4.jsonl'):
    d=json.loads(l); D[(d['family'],d['n'],d.get('bestof'))]=d
C={'flow':'#1f77b4','nn':'#2ca02c','mps_mle':'#d62728'}; mk={6:'o',8:'s',10:'^'}

# ---- fig_tfim: n=6,8,10 (bestof=8) ----
TF=[6,8,10]; fig,ax=plt.subplots(1,2,figsize=(9,3.4))
for n in TF:
    d=D[('tfim',n,8)]; k=d['k']
    ax[0].plot(k,d['flow'],'-',marker=mk[n],color=C['flow'],ms=5,label=f'$n{{=}}{n}$')
    ax[0].plot(k,d['nn'],':',color=C['nn'],lw=1.3,alpha=.85)
    ax[0].plot(k,d['mps_mle'],'--',marker=mk[n],color=C['mps_mle'],ms=3,alpha=.6)
ax[0].plot([],[],'-',color=C['flow'],label='MPS-flow (ours)'); ax[0].plot([],[],':',color=C['nn'],label='NN (ceiling)'); ax[0].plot([],[],'--',color=C['mps_mle'],label='MPS-MLE')
ax[0].set_xscale('log',base=2); ax[0].set_xticks(k); ax[0].set_xticklabels(k); ax[0].set_ylim(0,1.03)
ax[0].set_xlabel('measurements $k$'); ax[0].set_ylabel('reconstruction fidelity'); ax[0].set_title('(a) TFIM ground states')
h,l=ax[0].get_legend_handles_labels(); keep=['MPS-flow (ours)','NN (ceiling)','MPS-MLE']
ax[0].legend([h[l.index(x)] for x in keep],keep,frameon=False,fontsize=8.5,loc='center right')
x=np.arange(len(TF)); w=0.35
fsh=[D[('tfim',n,8)]['calib']['flow']['sharp'] for n in TF]; msh=[D[('tfim',n,8)]['calib']['mle']['sharp'] for n in TF]
ax[1].bar(x-w/2,fsh,w,color=C['flow'],label='ours'); ax[1].bar(x+w/2,msh,w,color=C['mps_mle'],label='MLE')
for i,(f,m) in enumerate(zip(fsh,msh)): ax[1].text(i,max(f,m)*1.02,f'{m/f:.0f}$\\times$',ha='center',va='bottom',fontsize=9)
ax[1].set_xticks(x); ax[1].set_xticklabels([f'$n{{=}}{n}$' for n in TF]); ax[1].set_ylabel('calibrated width $\\downarrow$')
ax[1].set_title('(b) Sharper calibrated intervals'); ax[1].legend(frameon=False,fontsize=9)
fig.tight_layout(); fig.savefig('./fig_tfim.pdf',bbox_inches='tight')

# ---- fig_reliability: coverage & sharpness vs nominal level (random family) ----
RN=[6,8,10]; fig,ax=plt.subplots(1,2,figsize=(9,3.4))
cols={6:'#9ecae1',8:'#4292c6',10:'#08519c'}
for n in RN:
    r=D[('random',n,8)]['reliability']; lv=r['level']
    ax[0].plot(lv,r['flow_cov'],'-o',color=cols[n],ms=4,label=f'ours $n{{=}}{n}$')
    ax[1].plot(lv,r['flow_sharp'],'-o',color=cols[n],ms=4,label=f'ours $n{{=}}{n}$')
    ax[1].plot(lv,r['mle_sharp'],'--',color=C['mps_mle'],alpha=.5,lw=1)
ax[0].plot([0.5,0.95],[0.5,0.95],'k:',lw=1,label='ideal'); ax[0].set_xlabel('nominal level $1-\\alpha$')
ax[0].set_ylabel('empirical coverage'); ax[0].set_title('(a) Reliability: conformal coverage'); ax[0].legend(frameon=False,fontsize=8.5)
ax[1].plot([],[],'--',color=C['mps_mle'],label='MLE ensemble'); ax[1].set_xlabel('nominal level $1-\\alpha$')
ax[1].set_ylabel('calibrated width $\\downarrow$'); ax[1].set_title('(b) Sharpness vs.\\ level'); ax[1].legend(frameon=False,fontsize=8.5)
fig.tight_layout(); fig.savefig('./fig_reliability.pdf',bbox_inches='tight')

# ---- fig_projection: PCA of prior samples vs family (tfim n=10 and random n=8) ----
fig,ax=plt.subplots(1,2,figsize=(9,3.4))
for j,(tag,fname,ttl) in enumerate([('tfim','proj_tfim_n10.npz','(a) TFIM $n{=}10$'),('random','proj_random_n8.npz','(b) random $n{=}8$')]):
    z=np.load('./'+fname); fam=z['fam']; pri=z['pri']
    X=np.concatenate([fam,pri],0); X=X-X.mean(0)
    U,S,Vt=np.linalg.svd(X,full_matrices=False); P=X@Vt[:2].T
    nf=len(fam)
    ax[j].scatter(P[:nf,0],P[:nf,1],s=6,alpha=.35,color='#ff7f0e',label='family',rasterized=True)
    ax[j].scatter(P[nf:,0],P[nf:,1],s=6,alpha=.35,color=C['flow'],label='prior samples',rasterized=True)
    ax[j].set_title(ttl); ax[j].set_xlabel('PC 1'); ax[j].set_ylabel('PC 2'); ax[j].legend(frameon=False,fontsize=9,markerscale=2)
fig.tight_layout(); fig.savefig('./fig_projection.pdf',bbox_inches='tight')
print("saved fig_tfim, fig_reliability, fig_projection")
