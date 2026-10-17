#!/usr/bin/env python3
"""Paired cluster-percentile bootstrap for fixed trained models; NumPy only."""
import argparse
import csv
import hashlib
import json
import warnings
from pathlib import Path
import numpy as np
from common import json_write, sha256


def keyed(doc):
    rows = doc['rows']; out = {r['pair_id']: r for r in rows}
    if len(out) != len(rows): raise ValueError('Duplicate pair_id')
    if not out: raise ValueError('No pairs')
    return out


def paired_arrays(a, b, metric):
    x, y = keyed(a), keyed(b)
    if a['dataset'] != b['dataset']: raise ValueError('Dataset mismatch')
    if a.get('evaluation') != b.get('evaluation'): raise ValueError('Evaluation-policy mismatch')
    if set(x) != set(y):
        raise ValueError(f'Pair sets differ ({len(set(x)-set(y))} baseline-only; {len(set(y)-set(x))} comparator-only)')
    clusters, av, bv = [], [], []
    for pid in sorted(x):
        r, s = x[pid], y[pid]
        if r['cluster_id'] != s['cluster_id']: raise ValueError(f'Cluster mismatch: {pid}')
        if not r['cluster_id']: raise ValueError('Empty cluster ID')
        if r.get('support_sha256') != s.get('support_sha256'):
            raise ValueError(f'GT/mask fingerprint mismatch: {pid}')
        aa, bb = r['metrics'].get(metric), s['metrics'].get(metric)
        aa = float(aa) if aa is not None else np.nan
        bb = float(bb) if bb is not None else np.nan
        if np.isfinite(aa) != np.isfinite(bb):
            raise ValueError(f'Unequal metric availability for {metric}: {pid}')
        if np.isinf(aa) or np.isinf(bb): raise ValueError('Infinite metric')
        clusters.append(r['cluster_id']); av.append(aa); bv.append(bb)
    return np.asarray(av), np.asarray(bv), np.asarray(clusters)


def bootstrap(a, b, clusters, replicates, seed, confidence=.95, family_size=1):
    """Cluster sums/counts retain the pair-weighted estimand under unequal sizes.

    Every replicate draws G clusters uniformly with replacement. Identical draws
    weight both models. Empty metric-support replicates are counted and reported.
    """
    names, idx = np.unique(clusters, return_inverse=True)
    g = len(names); valid = np.isfinite(a) & np.isfinite(b)
    if not valid.any(): return None
    counts = np.bincount(idx[valid], minlength=g).astype(float)
    sx = np.bincount(idx[valid], weights=a[valid], minlength=g)
    sy = np.bincount(idx[valid], weights=b[valid], minlength=g)
    ge = int((counts > 0).sum())
    if ge < 2: raise ValueError('Fewer than two independent units with metric support; CI not estimable.')
    rng = np.random.default_rng(seed)
    draws = np.full((replicates, 3), np.nan)
    for start in range(0, replicates, 128):
        n = min(128, replicates-start)
        weights = rng.multinomial(g, np.full(g, 1/g), size=n)
        denominator = weights @ counts
        good = denominator > 0
        z = np.full((n, 3), np.nan)
        z[good,0] = (weights[good] @ sx) / denominator[good]
        z[good,1] = (weights[good] @ sy) / denominator[good]
        z[good,2] = z[good,0] - z[good,1]
        draws[start:start+n] = z
    good = np.isfinite(draws[:,2]); empty = int((~good).sum())
    if empty / replicates > .01:
        raise ValueError('Over 1% of resamples lack metric support; do not report a conditional CI as a standard CI.')
    if ge < 10: warnings.warn(f'Only {ge} contributing clusters: percentile coverage may be poor; inspect leave-one-cluster-out.')
    alpha = 1-confidence
    lo, hi = np.quantile(draws[good], [alpha/2,1-alpha/2], axis=0)
    adj = np.quantile(draws[good,2], [alpha/(2*family_size),1-alpha/(2*family_size)])
    total = counts.sum(); diff = sx.sum()-sy.sum()
    loo = [{'omitted_cluster':str(name), 'delta':float((diff-(sx[k]-sy[k]))/(total-counts[k]))}
           for k,name in enumerate(names) if counts[k] > 0 and total > counts[k]]
    result = {'n_pairs':int(valid.sum()), 'n_pairs_total':len(a), 'n_clusters':g,
              'n_clusters_with_metric':ge,'A_mean':float(a[valid].mean()), 'B_mean':float(b[valid].mean()),
              'delta':float((a[valid]-b[valid]).mean()), 'A_ci_low':float(lo[0]),'A_ci_high':float(hi[0]),
              'B_ci_low':float(lo[1]),'B_ci_high':float(hi[1]),'ci_low':float(lo[2]),'ci_high':float(hi[2]),
              'adjusted_low':float(adj[0]),'adjusted_high':float(adj[1]),
              'replicates':replicates,'empty_replicates':empty,'confidence':confidence,'family_size':family_size,
              'few_clusters':ge<10,'loo_min':min(v['delta'] for v in loo),'loo_max':max(v['delta'] for v in loo)}
    return result, loo


def tex_escape(s):
    return str(s).replace('_', r'\_').replace('%',r'\%').replace('&',r'\&')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',required=True, help='Export directory containing DATASET/MODEL.json')
    p.add_argument('--out',required=True)
    p.add_argument('--baseline',default='A0')
    p.add_argument('--comparators',nargs='+',default=['A1','B1','B3','E7'])
    p.add_argument('--datasets',nargs='+')
    p.add_argument('--metrics',nargs='+',default=['epe_all','f1'])
    p.add_argument('--replicates',type=int,default=20000)
    p.add_argument('--seed',type=int,default=941004)
    p.add_argument('--confidence',type=float,default=.95)
    p.add_argument('--family-size',type=int,default=1, help='Bonferroni interval family size; 1 means pointwise only')
    args=p.parse_args()
    if not 0 < args.confidence < 1 or args.family_size<1 or args.replicates<1000:
        p.error('Require 0<confidence<1, family-size>=1, replicates>=1000')
    if args.replicates*(1-args.confidence)/(2*args.family_size)<20:
        p.error('Too few expected tail draws; increase --replicates for the requested family-size.')
    root=Path(args.input); out=Path(args.out)
    datasets=args.datasets or sorted(p.parent.name for p in root.glob('*/'+args.baseline+'.json'))
    if not datasets: raise ValueError('No datasets found')
    results=[]; sensitivity=[]; skipped=[]; inputs=[]
    for dataset in datasets:
        a_path=root/dataset/(args.baseline+'.json'); a=json.loads(a_path.read_text())
        inputs.append({'path':str(a_path),'sha256':sha256(a_path)})
        # Dataset-specific reproducible stream reused for ALL contrasts and metrics.
        seed=int.from_bytes(hashlib.sha256(f'{args.seed}:{dataset}'.encode()).digest()[:8],'little')
        for comparator in args.comparators:
            b_path=root/dataset/(comparator+'.json'); b=json.loads(b_path.read_text())
            inputs.append({'path':str(b_path),'sha256':sha256(b_path)})
            if a['model'] != args.baseline or b['model'] != comparator: raise ValueError('Model ID/file mismatch')
            for metric in args.metrics:
                av,bv,clusters=paired_arrays(a,b,metric)
                r=bootstrap(av,bv,clusters,args.replicates,seed,args.confidence,args.family_size)
                identity={'dataset':dataset,'baseline':args.baseline,'comparator':comparator,'metric':metric}
                if r is None:
                    skipped.append({**identity,'reason':'metric unavailable for all pairs'}); continue
                stats,loo=r
                results.append({**identity,'cluster_policy':a.get('cluster_policy','unspecified'),**stats})
                sensitivity.extend({**identity,**v} for v in loo)
    if args.family_size>1 and len(results)>args.family_size:
        raise ValueError('Requested family-size smaller than the number of reported comparisons.')
    out.mkdir(parents=True,exist_ok=True)
    for filename,rows in [('paired_intervals.csv',results),('leave_one_cluster_out.csv',sensitivity)]:
        if rows:
            with open(out/filename,'w',newline='') as f:
                w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    json_write(out/'analysis.json',{'arguments':vars(args),'inputs':inputs,'results':results,'skipped':skipped,
        'interpretation':'Delta = baseline minus comparator. Pointwise percentile cluster bootstrap for fixed checkpoints. No training-seed or selection uncertainty.'})
    lines=[r'% Requires booktabs. Use inside tabular or longtable; split floats if needed.',
           r'\begin{tabular}{llrrrr}',r'\toprule',
           r'Dataset / metric & Contrast & Pairs & Units & $\Delta$ & 95\% CI \\',r'\midrule']
    # Header confidence is generated, not hard-coded when CLI differs.
    lines[3]=lines[3].replace('95',f'{100*args.confidence:g}')
    for r in results:
        label=tex_escape(r['dataset']+' / '+r['metric'])
        contrast=tex_escape(r['baseline']+' -- '+r['comparator'])
        lines.append(f"{label} & {contrast} & {r['n_pairs']} & {r['n_clusters_with_metric']} & {r['delta']:.4f} & [{r['ci_low']:.4f}, {r['ci_high']:.4f}] " + r'\\')
    lines += [r'\bottomrule',r'\end{tabular}']
    (out/'paired_intervals.tex').write_text('\n'.join(lines)+'\n')
    print(f'Saved {len(results)} intervals to {out}; {len(skipped)} unavailable metrics omitted.')

if __name__=='__main__': main()
