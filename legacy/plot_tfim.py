import json, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':150,
                     'font.family':'serif','mathtext.fontset':'cm'})
R={}
for line in open('./qst_results_v4.jsonl'):
    d=json.loads(line)
    if d.get('family')=='tfim' and d.get('bestof')==8: R[d['n']]=d
NS=sorted(R); C={'flow':'#1f77b4','nn':'#2ca02c','mps_mle':'#d62728'}
mk={6:'o',8:'s',10:'^'}
fig,ax=plt.subplots(1,2,figsize=(9,3.4))
for n in NS:
    d=R[n]; k=d['k']
    ax[0].plot(k,d['flow'],'-',marker=mk.get(n,'o'),color=C['flow'],ms=5,label=f'$n{{=}}{n}$')
    ax[0].plot(k,d['nn'],':',color=C['nn'],lw=1.3,alpha=.8)
    ax[0].plot(k,d['mps_mle'],'--',marker=mk.get(n,'o'),color=C['mps_mle'],ms=3,alpha=.7)
ax[0].plot([],[],'-',color=C['flow'],label='MPS-flow (ours)')
ax[0].plot([],[],':',color=C['nn'],label='NN prior (ceiling)')
ax[0].plot([],[],'--',color=C['mps_mle'],label='MPS-MLE')
ax[0].set_xscale('log',base=2); ax[0].set_xticks(d['k']); ax[0].set_xticklabels(d['k'])
ax[0].set_xlabel('measurements $k$'); ax[0].set_ylabel('reconstruction fidelity'); ax[0].set_ylim(0,1.03)
ax[0].set_title('(a) TFIM ground states: fidelity vs.\\ budget')
h,l=ax[0].get_legend_handles_labels()
keep=['MPS-flow (ours)','NN prior (ceiling)','MPS-MLE']
ax[0].legend([h[l.index(x)] for x in keep],keep,frameon=False,fontsize=8.5,loc='center right')
# calibration sharpness bars
x=np.arange(len(NS)); w=0.35
fsh=[R[n]['calib']['flow']['sharp'] for n in NS]; msh=[R[n]['calib']['mle']['sharp'] for n in NS]
ax[1].bar(x-w/2,fsh,w,color=C['flow'],label='MPS-flow (ours)')
ax[1].bar(x+w/2,msh,w,color=C['mps_mle'],label='MLE ensemble')
for i,(f,m) in enumerate(zip(fsh,msh)): ax[1].text(i,max(f,m)*1.02,f'{m/f:.0f}$\\times$',ha='center',va='bottom',fontsize=9)
ax[1].set_xticks(x); ax[1].set_xticklabels([f'$n{{=}}{n}$' for n in NS])
ax[1].set_ylabel('calibrated interval width $\\downarrow$'); ax[1].set_title('(b) Calibrated intervals: much sharper')
ax[1].legend(frameon=False,fontsize=9)
fig.tight_layout(); fig.savefig('./fig_tfim.pdf',bbox_inches='tight')
print("saved fig_tfim.pdf for n=",NS)
