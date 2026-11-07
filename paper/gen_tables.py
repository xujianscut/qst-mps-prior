"""Generate LaTeX tables from the result JSONs. Usage: python gen_tables.py <results_dir> <out_dir>.
Every number in the tables is read from JSON; missing inputs raise."""
import json, os, sys, glob
D, O = sys.argv[1], sys.argv[2]
def J(name):
    p = os.path.join(D, name); assert os.path.exists(p), f"missing {p}"; return json.load(open(p))
f3 = lambda x: f"{x:.3f}"
NS = [6, 8, 10]; SS = [20, 50, 100, 200]

# ---------------- main table ----------------
rows = []
for n in NS:
    for i, S in enumerate(SS):
        r = J(f"rescue_n{n}_local_S{S}.json"); t = r['test']
        for k in ['net', 'mle', 'warm', 'map', 'fam']: assert k in t, (n, S, k)
        g = J(f"gate_chi2_n{n}_S{S}_q0.9.json"); gt = [x for x in g['rows'] if x['kind'] == 'test']; assert len(gt) == 1
        gt = gt[0]; assert abs(gt['map_fid'] - t['map']['fid']) < 2e-3, (n, S, gt['map_fid'], t['map']['fid'])
        best = max([t[k]['fid'] for k in ['net', 'mle', 'warm', 'fam', 'map']] + [gt['gated_fid']])
        bf = lambda v: (r"\textbf{%s}" % f3(v)) if f3(v) == f3(best) else f3(v)
        cell = lambda k: bf(t[k]['fid'])
        lead = (r"\multirow{4}{*}{%d}" % n) if i == 0 else ""
        rows.append(f"{lead} & {S} & {cell('net')} & {cell('mle')} & {cell('warm')} & {cell('fam')} & {cell('map')} & "
                    f"{bf(gt['gated_fid'])} & ${t['map']['delta_vs_mle']:+.3f}\\,({t['map']['delta_se']*1e3:.0f})$ & {t['map']['win_frac']*100:.0f}\\% \\\\")
    if n != NS[-1]: rows.append(r"\midrule")
tab = r"""\begin{table}[tbp]
\caption{Mean reconstruction fidelity on $200$ test states (shallow family, \texttt{local} design, independent
measurement model, $S$ shots per setting; best in bold). All hyperparameters are chosen on a separate validation
set. ``Warm start'' is MLE initialized at the network output ($\lambda=0$); ``family prior'' replaces the
network's estimate in the penalty by the family's mean state; ``gated'' applies the $\chi^2$ gate of
section~\ref{sec:gate}. The last two columns give the paired difference MAP$-$MLE with its standard error in
units of $10^{-3}$, and the fraction of test states on which MAP is better.}
\label{tab:main}
\centering
\small
\begin{tabular}{rrcccccccc}
\toprule
$n$ & $S$ & network & MLE & warm start & family prior & MAP & gated & MAP$-$MLE (SE) & wins \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
open(os.path.join(O, 'tab_main.tex'), 'w').write(tab)

# ---------------- controls + design ladder ----------------
crow = []
for n in NS:
    c = J(f"cond_n{n}_local_indep_s30000.json")
    crow.append(f"{n} & {c['K']} & {f3(c['fid'])} & {f3(c['fid_prior'])} & {f3(c['fid_shuf'])} & ${c['fid']-c['fid_prior']:+.2f}$ \\\\")
drow = []
for des, m in [('local1', 3), ('local', 9), ('local3', 27)]:
    r = J(f"rescue_n6_{des}_S200.json"); t = r['test']; c = J(f"cond_n6_{des}_indep_s30000.json")
    drow.append(f"\\texttt{{{des}}} & {c['K']} & {m} & {f3(c['fid_prior'])} & {f3(c['fid_shuf'])} & {f3(t['mle']['fid'])} & {f3(t['net']['fid'])} & {f3(t['map']['fid'])} \\\\")
tab = r"""\begin{table}[tbp]
\caption{Integrity controls and the role of the design ($S=200$, $200$ test states). Top: the network against
a prior-only guess (normalized mean of the training states, no measurements) and a shuffled control (the
network applied to another test state's record), \texttt{local} design. Bottom, $n=6$: designs with $k$
observables covered by $m$ settings; MLE, network and MAP as in table~\ref{tab:main}. The network columns in
the two blocks come from separately trained networks and agree to $0.01$.}
\label{tab:controls}
\centering
\small
\begin{tabular}{rccccc}
\toprule
$n$ & $k$ & network & prior-only & shuffled & network $-$ prior-only \\
\midrule
""" + "\n".join(crow) + r"""
\bottomrule
\end{tabular}

\vspace{6pt}
\begin{tabular}{lccccccc}
\toprule
design ($n=6$) & $k$ & $m$ & prior-only & shuffled & MLE & network & MAP \\
\midrule
""" + "\n".join(drow) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
open(os.path.join(O, 'tab_controls.tex'), 'w').write(tab)

# ---------------- shift table ----------------
srow = []
for n in [6, 10]:
    g = J(f"gate_chi2_n{n}_S200_q0.9.json"); r = J(f"rescue_n{n}_local_S200.json")
    ens = {(s['kind'], s['val']): s for s in r['shift']}
    lab = {('test', 0): 'in distribution', ('spread', 0.9): 'spread $0.9$', ('spread', 1.2): 'spread $1.2$',
           ('shots', 50): '$S=50$ at test', ('shots', 20): '$S=20$ at test'}
    for i, x in enumerate(g['rows']):
        key = (x['kind'], x['val']); e = ens[('none', 0)] if x['kind'] == 'test' else ens[key]
        lead = (r"\multirow{5}{*}{%d}" % n) if i == 0 else ""
        srow.append(f"{lead} & {lab[key]} & {f3(e['ens_fid'])} & {f3(x['mle_fid'])} & {f3(x['map_fid'])} & {f3(x['gated_fid'])} & "
                    f"{x['gated_frac']*100:.0f}\\% & {e['ens_cov']*100:.0f} & {x['mle_cov']*100:.0f} & {x['map_cov']*100:.0f} & {x['gated_cov']*100:.0f} \\\\")
    if n == 6: srow.append(r"\midrule")
tab = r"""\begin{table}[tbp]
\caption{Distribution shift (networks trained at $S=200$ on spread $0.6$; $200$ states per row). Fidelity of
the network, MLE, MAP and the $\chi^2$-gated estimator; fraction of states the gate routes to MLE; and
coverage ($\%$, nominal $90$) of intervals for the held-out observables, with multipliers calibrated in
distribution: the network's conformalized dropout ensemble, and fixed-scale intervals around MLE, MAP and
the gated estimate. The ensemble's in-distribution row is measured on a second in-distribution set.}
\label{tab:shift}
\centering
\small
\begin{tabular}{rlccccc|cccc}
\toprule
 & & \multicolumn{4}{c}{fidelity} & & \multicolumn{4}{c}{coverage (\%)} \\
$n$ & test condition & network & MLE & MAP & gated & to MLE & ens. & MLE & MAP & gated \\
\midrule
""" + "\n".join(srow) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
open(os.path.join(O, 'tab_shift.tex'), 'w').write(tab)

# ---------------- observables table ----------------
orow_mae, orow_w = [], []
for n in NS:
    r = J(f"rescue_n{n}_local_S200.json"); t = r['test']; c = J(f"cond_n{n}_local_indep_s30000.json"); m = J(f"mle_n{n}_local_indep.json")
    sh = {s['N']: s['maeH'] for s in m['shadows']}; assert set(sh) == {1800, 18000}
    best = min(t[k]['maeH'] for k in ['net', 'mle', 'map'])
    b = lambda v: (r"\textbf{%s}" % f3(v)) if v == best else f3(v)
    orow_mae.append(f"{n} & {c['nheld']} & {f3(c['maeH_prior'])} & {b(t['net']['maeH'])} & {b(t['mle']['maeH'])} & {b(t['map']['maeH'])} & {f3(sh[1800])} & {f3(sh[18000])} \\\\")
    covs = [c['covH'][0], t['net']['covH_fixed'], t['mle']['covH_fixed'], t['map']['covH_fixed']]
    orow_w.append(f"{n} & {f3(c['covH'][2])} & {f3(t['net']['widH_fixed'])} & {f3(t['mle']['widH_fixed'])} & \\textbf{{{f3(t['map']['widH_fixed'])}}} & {min(covs)*100:.1f}--{max(covs)*100:.1f} \\\\")
bm = J("rescue_n6_local_S200_basis.json")['test']; mb6 = J("mle_n6_local_basis.json"); mb10 = J("mle_n10_local_basis.json")
tab = r"""\begin{table}[tbp]
\caption{Prediction of held-out observables outside the design ($S=200$, $200$ test states). Top: mean absolute
error. Classical shadows use $9S=1800$ (the same copy budget) or $90S$ single-shot random-basis snapshots.
Bottom: mean width of $90\%$ conformal intervals (heteroscedastic dropout ensemble for the network,
fixed-scale for the others) and the range of empirical coverage over the four interval types.}
\label{tab:obs}
\centering
\small
\begin{tabular}{rccccccc}
\toprule
$n$ & observables & prior-only & network & MLE & MAP & shadows ($9S$) & shadows ($90S$) \\
\midrule
""" + "\n".join(orow_mae) + r"""
\bottomrule
\end{tabular}

\vspace{6pt}
\begin{tabular}{rccccc}
\toprule
$n$ & ensemble & network (fixed) & MLE (fixed) & MAP (fixed) & coverage (\%) \\
\midrule
""" + "\n".join(orow_w) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
open(os.path.join(O, 'tab_obs.tex'), 'w').write(tab)
print("basis n6: MAP maeH %.4f, MLE %.4f, net %.4f | direct n6 %.4f n10 %.4f" % (bm['map']['maeH'], bm['mle']['maeH'], bm['net']['maeH'], mb6['direct']['maeH'], mb10['direct']['maeH']))
print("written:", [f for f in os.listdir(O) if f.startswith('tab_')])

# ---------------- second family (disordered Ising ground states) ----------------
irow = []
for n in NS:
    for i, S in enumerate([50, 200]):
        r = J(f"rescue_n{n}_local_S{S}_tfim.json"); t = r['test']; c = r['controls']
        g = J(f"gate_chi2_n{n}_S{S}_q0.9_tfim.json"); gt = [x for x in g['rows'] if x['kind'] == 'test'][0]
        assert abs(gt['map_fid'] - t['map']['fid']) < 2e-3
        best = max([t[k]['fid'] for k in ['net', 'mle', 'warm', 'fam', 'map']] + [gt['gated_fid']])
        bf = lambda v: (r"\textbf{%s}" % f3(v)) if f3(v) == f3(best) else f3(v)
        lead = (r"\multirow{2}{*}{%d}" % n) if i == 0 else ""
        irow.append(f"{lead} & {S} & {f3(c['prior_only'])} & {f3(c['shuffled'])} & {bf(t['net']['fid'])} & {bf(t['mle']['fid'])} & "
                    f"{bf(t['warm']['fid'])} & {bf(t['fam']['fid'])} & {bf(t['map']['fid'])} & {bf(gt['gated_fid'])} & "
                    f"${t['map']['delta_vs_mle']:+.3f}\\,({t['map']['delta_se']*1e3:.0f})$ & {t['map']['win_frac']*100:.0f}\\% \\\\")
    if n != NS[-1]: irow.append(r"\midrule")
tab = r"""\begin{table}[tbp]
\caption{Second family: ground states of disordered Ising chains with random longitudinal fields
(section~\ref{sec:ising}), \texttt{local} design, independent measurement model, $200$ test states, all
hyperparameters chosen on a validation set. Columns as in tables~\ref{tab:controls} and~\ref{tab:main}.}
\label{tab:ising}
\centering
\footnotesize
\setlength{\tabcolsep}{4pt}
\begin{tabular}{rrcccccccccc}
\toprule
$n$ & $S$ & prior-only & shuffled & network & MLE & warm & family & MAP & gated & MAP$-$MLE (SE) & wins \\
\midrule
""" + "\n".join(irow) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
open(os.path.join(O, 'tab_ising.tex'), 'w').write(tab)
for n in NS:
    g = J(f"gate_chi2_n{n}_S200_q0.9_tfim.json")
    print(f"ising gate n={n}: " + " | ".join(f"{x['kind']}={x['val']}: to-MLE {x['gated_frac']*100:.0f}% mle {x['mle_fid']:.3f} map {x['map_fid']:.3f} gated {x['gated_fid']:.3f} cov mle/map/gated {x['mle_cov']*100:.0f}/{x['map_cov']*100:.0f}/{x['gated_cov']*100:.0f}" for x in g['rows']))
print("written tab_ising.tex")
