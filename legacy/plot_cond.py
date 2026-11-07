import json, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':150,'font.family':'serif','mathtext.fontset':'cm'})
D='./'
C={'prior':'#7f7f7f','cond':'#1f77b4','shuf':'#d62728','local':'#2ca02c','rand':'#ff7f0e'}

# ---------- fig_control: prior-only + shuffled control (concentrated vs diverse) ----------
KS=[8,16,32,64]
tf_true=[0.849,0.878,0.891,0.913]; tf_shuf=[0.859,0.873,0.895,0.915]; tf_med=0.997
rd_true=[0.539,0.560,0.604,0.647]; rd_shuf=[0.509,0.500,0.512,0.514]; rd_med=0.646
fig,ax=plt.subplots(1,2,figsize=(9,3.4))
for a,(tr,sh,md,ttl) in zip(ax,[(tf_true,tf_shuf,tf_med,'(a) concentrated family (paramagnetic TFIM)'),
                                (rd_true,rd_shuf,rd_med,'(b) diverse family (random MPS)')]):
    a.plot(KS,tr,'-o',color=C['cond'],ms=5,label='true measurements')
    a.plot(KS,sh,'--s',color=C['shuf'],ms=4,label='shuffled measurements')
    a.axhline(md,ls=':',color=C['prior'],lw=1.4); a.text(KS[-1],md+0.006 if md<0.9 else md-0.03,f'prior-only (medoid) {md:.2f}',ha='right',fontsize=8,color=C['prior'])
    a.set_xscale('log',base=2); a.set_xticks(KS); a.set_xticklabels(KS); a.set_xlabel('measurements $k$'); a.set_ylabel('reconstruction fidelity')
    a.set_title(ttl,fontsize=10); a.legend(frameon=False,fontsize=9,loc='lower right')
ax[0].set_ylim(0.8,1.01); ax[1].set_ylim(0.45,0.75)
fig.tight_layout(); fig.savefig(D+'fig_control.pdf',bbox_inches='tight'); print("fig_control")

# ---------- fig_main: headline bars (prior-only / conditional / shuffled), local design ----------
fams=['random','shallow (physical)']; prior=[0.680,0.375]; cond=[0.954,0.967]; shuf=[0.505,0.154]
fig,ax=plt.subplots(figsize=(5.0,3.4)); x=np.arange(2); w=0.26
ax.bar(x-w,prior,w,color=C['prior'],label='prior-only ($k{=}0$)')
ax.bar(x,cond,w,color=C['cond'],label='conditional (ours)')
ax.bar(x+w,shuf,w,color=C['shuf'],label='shuffled control')
for i in range(2):
    ax.text(i,cond[i]+0.02,f'{cond[i]:.2f}',ha='center',fontsize=9,color=C['cond'])
    ax.annotate('',xy=(i-w,prior[i]),xytext=(i-w,cond[i]),arrowprops=dict(arrowstyle='<->',color='k',lw=0.8))
    ax.text(i-w-0.02,(prior[i]+cond[i])/2,f'+{cond[i]-prior[i]:.2f}',ha='right',va='center',fontsize=8)
ax.set_xticks(x); ax.set_xticklabels(fams); ax.set_ylabel('reconstruction fidelity'); ax.set_ylim(0,1.05)
ax.set_title('Conditional estimator, informative local design'); ax.legend(frameon=False,fontsize=9,loc='upper center',ncol=1)
fig.tight_layout(); fig.savefig(D+'fig_main.pdf',bbox_inches='tight'); print("fig_main")

# ---------- fig_design: the lever (random Paulis vs local), random family ----------
fig,ax=plt.subplots(figsize=(4.6,3.4)); labels=['random\nPaulis','informative\nlocal']; x=np.arange(2); w=0.32
cd=[0.79,0.954]; sh=[0.60,0.505]; pr=0.680
ax.bar(x-w/2,cd,w,color=C['cond'],label='conditional (true)')
ax.bar(x+w/2,sh,w,color=C['shuf'],label='shuffled')
ax.axhline(pr,ls=':',color=C['prior'],lw=1.4); ax.text(1.4,pr+0.01,'prior-only 0.68',ha='right',fontsize=8,color=C['prior'])
for i in range(2): ax.text(i-w/2,cd[i]+0.015,f'{cd[i]:.2f}',ha='center',fontsize=9,color=C['cond'])
ax.set_xticks(x); ax.set_xticklabels(labels); ax.set_ylabel('fidelity'); ax.set_ylim(0,1.05)
ax.set_title('Measurement design is the lever'); ax.legend(frameon=False,fontsize=9,loc='upper left')
fig.tight_layout(); fig.savefig(D+'fig_design.pdf',bbox_inches='tight'); print("fig_design")

# ---------- fig_agnostic: fixed vs measurement-agnostic collapse (random family) ----------
fig,ax=plt.subplots(figsize=(5.0,3.4)); grp=['fixed design\n(local)','agnostic\nDeepSets','agnostic\nset-transf.']; x=np.arange(3); w=0.34
tr=[0.954,0.688,0.686]; sh=[0.505,0.679,0.686]
ax.bar(x-w/2,tr,w,color=C['cond'],label='true measurements')
ax.bar(x+w/2,sh,w,color=C['shuf'],label='shuffled')
ax.axhline(0.680,ls=':',color=C['prior'],lw=1.4); ax.text(2.4,0.69,'prior-only',ha='right',fontsize=8,color=C['prior'])
ax.set_xticks(x); ax.set_xticklabels(grp,fontsize=9); ax.set_ylabel('fidelity'); ax.set_ylim(0,1.05)
ax.set_title('Design-agnostic conditioning collapses'); ax.legend(frameon=False,fontsize=9)
fig.tight_layout(); fig.savefig(D+'fig_agnostic.pdf',bbox_inches='tight'); print("fig_agnostic")

# ---------- fig_scale: native contraction memory flat vs 2^n ----------
ns=[16,20]; mem=[0.134,0.147]; dense=[2**16,2**20]; params=[120,152]
fig,ax=plt.subplots(figsize=(5.0,3.4)); ax2=ax.twinx()
ax.plot(ns,mem,'-o',color=C['cond'],ms=6,label='peak GPU memory (GB)')
ax2.plot(ns,dense,'--s',color=C['shuf'],ms=5,label='dense dim $2^n$'); ax2.set_yscale('log')
ax.set_xlabel('qubits $n$'); ax.set_ylabel('peak memory (GB)',color=C['cond']); ax.set_ylim(0,0.3); ax.set_xticks(ns)
ax2.set_ylabel('dense state dim $2^n$ (log)',color=C['shuf'])
ax.set_ylim(0,0.25); ax.annotate('0.134 GB (120 params)',xy=(16,0.134),xytext=(16.1,0.055),fontsize=8,color=C['cond'],arrowprops=dict(arrowstyle='->',color=C['cond'],lw=0.7)); ax.annotate('0.147 GB (152 params)',xy=(20,0.147),xytext=(17.4,0.205),fontsize=8,color=C['cond'],arrowprops=dict(arrowstyle='->',color=C['cond'],lw=0.7))
ax.set_title('Native cost $\\mathcal{O}(n\\chi^{3})$: flat memory in $n$')
fig.tight_layout(); fig.savefig(D+'fig_scale.pdf',bbox_inches='tight'); print("fig_scale")

# ---------- fig_calib: reliability (coverage + width vs nominal), from cal_rel JSONs ----------
try:
    rr=json.load(open(D+'qst_cond_cal_random.json'))['reliability']; ss=json.load(open(D+'qst_cond_cal_shallow.json'))['reliability']
    fig,ax=plt.subplots(1,2,figsize=(9,3.4))
    ax[0].plot(rr['level'],rr['cov'],'-o',color=C['cond'],ms=5,label='random')
    ax[0].plot(ss['level'],ss['cov'],'-s',color=C['local'],ms=5,label='shallow')
    ax[0].plot([0.5,0.95],[0.5,0.95],'k:',lw=1,label='ideal'); ax[0].set_xlabel('nominal level $1-\\alpha$'); ax[0].set_ylabel('empirical coverage')
    ax[0].set_title('(a) Reliability (conformal)'); ax[0].legend(frameon=False,fontsize=9)
    ax[1].plot(rr['level'],rr['wid'],'-o',color=C['cond'],ms=5,label='random'); ax[1].plot(ss['level'],ss['wid'],'-s',color=C['local'],ms=5,label='shallow')
    ax[1].set_xlabel('nominal level $1-\\alpha$'); ax[1].set_ylabel('clipped interval width'); ax[1].set_title('(b) Sharpness'); ax[1].legend(frameon=False,fontsize=9)
    fig.tight_layout(); fig.savefig(D+'fig_calib.pdf',bbox_inches='tight'); print("fig_calib")
except Exception as e:
    print("fig_calib SKIPPED (reliability json not ready):",e)
print("done")
