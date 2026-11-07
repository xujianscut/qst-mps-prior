import json, glob, sys
D = sys.argv[1] if len(sys.argv) > 1 else '.'
fs = sorted(glob.glob(D + '/rescue_*.json')); assert fs
for f in fs:
    d = json.load(open(f)); t = d['test']
    print(f"== {d['tag']}  train {d['t_train_s']:.0f}s  sel {d['sel']}")
    for m in [k for k in ['net', 'mle', 'mle3', 'warm', 'map', 'fam'] if k in t]:
        x = t[m]
        dl = f" d={x['delta_vs_mle']:+.4f}({x['delta_se']:.4f}) win {x['win_frac']*100:.0f}%" if 'delta_vs_mle' in x else ''
        print(f"   {m:5s} fid {x['fid']:.4f}+-{x['fid_sd']:.4f} q10 {x['q10']:.3f} maeH {x['maeH']:.4f} cov {x['covH_fixed']*100:.1f}% w {x['widH_fixed']:.3f}{dl}")
    for s in d.get('shift', []):
        print(f"   shift {s['kind']}={s['val']}: " + " | ".join(f"{k} cov {s[k+'_cov']*100:.1f} w {s[k+'_wid']:.3f} fid {s[k+'_fid']:.3f}" for k in ['ens', 'mle', 'map']))
