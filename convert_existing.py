#!/usr/bin/env python3
"""Import existing metrics_detailed.json using an explicit ID/cluster CSV.

No order-based or scene-name guessing: map each unique legacy sample_name to
pair_id and cluster_id. All models must use the same evaluation policy label.
"""
import argparse,csv,json,math
from pathlib import Path
from common import json_write,sha256

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--metrics',required=True)
    p.add_argument('--mapping',required=True,help='CSV: sample_name,pair_id,cluster_id')
    p.add_argument('--dataset',required=True); p.add_argument('--model',required=True)
    p.add_argument('--policy',required=True,help='Shared audited evaluation-policy identifier')
    p.add_argument('--out',required=True)
    a=p.parse_args(); raw=json.loads(Path(a.metrics).read_text())
    with open(a.mapping,newline='') as f: maps=list(csv.DictReader(f))
    mapping={r['sample_name']:r for r in maps}
    if len(mapping)!=len(maps): raise ValueError('Duplicate sample_name in mapping')
    if len({r['sample_name'] for r in raw})!=len(raw): raise ValueError('Duplicate legacy sample_name')
    if set(mapping)!={r['sample_name'] for r in raw}: raise ValueError('Mapping and metrics must have exactly matching keys')
    rows=[]
    for r in raw:
        m=mapping[r['sample_name']]
        metrics={k:float(v) if math.isfinite(float(v)) else None for k,v in r.items()
                 if isinstance(v,(int,float)) and not isinstance(v,bool)}
        rows.append({'pair_id':m['pair_id'],'cluster_id':m['cluster_id'],'metrics':metrics})
    if len({r['pair_id'] for r in rows})!=len(rows): raise ValueError('Duplicate stable pair IDs')
    json_write(a.out,{'schema':1,'model':a.model,'dataset':a.dataset,
         'evaluation':{'legacy_audited_policy':a.policy},'cluster_policy':'explicit audited CSV mapping',
         'provenance':{'metrics_sha256':sha256(a.metrics),'mapping_sha256':sha256(a.mapping),
                       'warning':'Legacy export has no GT/mask fingerprints; equivalence must be checked by the user.'},'rows':rows})

if __name__=='__main__': main()
