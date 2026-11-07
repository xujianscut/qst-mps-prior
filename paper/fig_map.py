"""Main figure: fidelity vs shots per setting for n=6,8,10 (network, MLE, MAP). Reads rescue JSONs; asserts inputs."""
import json, os, sys, math
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
D, out = sys.argv[1], sys.argv[2]
NS, SS = [6, 8, 10], [20, 50, 100, 200]
C = {'map': '#2a78d6', 'mle': '#eb6834', 'net': '#1baf7a'}   # categorical slots 1-3 (validated all-pairs)
LAB = {'map': 'MAP refinement', 'mle': 'tuned MLE', 'net': 'network alone'}
MK = {'map': dict(marker='o', mfc=C['map']), 'mle': dict(marker='s', mfc='white'), 'net': dict(marker='^', mfc='white')}
plt.rcParams.update({'font.size': 9, 'font.family': 'serif', 'axes.spines.top': False, 'axes.spines.right': False})
fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.5), sharey=True)
for ax, n in zip(axs, NS):
    for k in ['net', 'mle', 'map']:
        m, e = [], []
        for S in SS:
            p = os.path.join(D, f'rescue_n{n}_local_S{S}.json'); assert os.path.exists(p), p
            t = json.load(open(p))['test'][k]; m.append(t['fid']); e.append(2 * t['fid_sd'] / math.sqrt(200))
        ax.errorbar(SS, m, yerr=e, color=C[k], lw=1.5, ms=5, mec=C[k], mew=1.2, capsize=2, label=LAB[k], **MK[k])
    ax.set_xscale('log'); ax.set_xticks(SS); ax.set_xticklabels([str(s) for s in SS]); ax.minorticks_off()
    ax.set_title(f'$n={n}$', fontsize=9); ax.set_xlabel('shots per setting $S$')
    ax.grid(axis='y', color='#e4e3df', lw=0.6); ax.set_axisbelow(True)
axs[0].set_ylabel('mean fidelity'); axs[0].set_ylim(0.68, 1.0)
axs[0].legend(frameon=False, loc='lower right', fontsize=8)
fig.tight_layout(); fig.savefig(out); fig.savefig(out.replace('.pdf', '.png'), dpi=150); print('saved', out)
