#!/usr/bin/env python3
"""Export deterministic native-resolution per-pair metrics, or just pair inventories."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from common import json_write, load_model, path, predict, repo_setup, sha256


def inventory(ds, cfg, entry, repo):
    root = Path(cfg.root).resolve()
    kind = str(cfg.name).lower()
    records = getattr(ds, '_samples', None)
    if records is None or len(records) != len(ds):
        raise ValueError('Use one unmixed native dataset per entry; no sampled/repeated wrappers.')
    rows = []
    for r in records:
        a, b = Path(r['image1']).resolve(), Path(r['image2']).resolve()
        rel1, rel2 = a.relative_to(root).as_posix(), b.relative_to(root).as_posix()
        pid = rel1 + '->' + rel2
        if kind == 'sintel':
            cluster = a.parent.name
        elif kind == 'spring':
            cluster = a.parent.parent.name
        elif kind in ('hd1k', 'h1dk'):
            cluster = a.stem.split('_')[0]
        elif kind in ('chairs', 'kitti', 'kitti2012', 'kitti2015'):
            cluster = pid
        elif kind == 'things':
            cluster = None  # resolved below for the flattened subset
        else:
            raise ValueError(f'Unsupported cluster policy for {kind}')
        rows.append({'pair_id': pid, 'cluster_id': cluster, 'image1': rel1,
                     'image2': rel2, 'sample_id': a.stem + '__' + b.stem})
    policy = {'sintel': 'scene', 'spring': 'scene', 'hd1k': 'sequence', 'h1dk': 'sequence',
              'chairs': 'pair', 'kitti': 'pair', 'kitti2012': 'pair', 'kitti2015': 'pair'}.get(kind)
    if kind == 'things':
        # The subset flattens scene names. Connected adjacent-frame components
        # keep temporal neighbours together; equal frame indices join both cameras.
        if all(Path(r['image1']).parent.name in ('left', 'right') and
               Path(r['image1']).parent.parent.name == 'image_clean' for r in rows):
            parent = {}
            def find(x):
                parent.setdefault(x, x)
                if parent[x] != x: parent[x] = find(parent[x])
                return parent[x]
            def union(a, b):
                a, b = find(a), find(b)
                if a != b: parent[max(a,b)] = min(a,b)
            for r in rows:
                union(Path(r['image1']).stem, Path(r['image2']).stem)
            for r in rows:
                r['cluster_id'] = 'component_' + find(Path(r['image1']).stem)
            policy = 'connected-frame-component; stereo coupled; verify against source scenes'
        else:
            for r in rows:
                r['cluster_id'] = str(Path(r['image1']).parent.parent)
            policy = 'source-sequence-directory; stereo coupled'
    if entry.get('cluster_map'):
        with open(path(entry['cluster_map'], repo), newline='') as f:
            mapping_rows = list(csv.DictReader(f))
        mapping = {r['pair_id']: r['cluster_id'] for r in mapping_rows}
        if len(mapping) != len(mapping_rows) or set(mapping) != {r['pair_id'] for r in rows}:
            raise ValueError('cluster_map must contain exactly one row per exported pair_id.')
        if not all(mapping.values()): raise ValueError('Empty cluster ID')
        for r in rows: r['cluster_id'] = mapping[r['pair_id']]
        policy = 'user-supplied source scene/sequence map'
    if len({r['pair_id'] for r in rows}) != len(rows):
        raise ValueError('Duplicate pair IDs')
    return rows, policy


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo', default='.')
    p.add_argument('--manifest', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--device', default='cuda')
    p.add_argument('--models', nargs='+')
    p.add_argument('--datasets', nargs='+')
    p.add_argument('--inventory-only', action='store_true')
    args = p.parse_args()
    repo = Path(args.repo).resolve(); mf = Path(args.manifest).resolve(); out = Path(args.out).resolve()
    provenance = repo_setup(repo)
    from omegaconf import OmegaConf
    from data import build_dataset
    import torch
    from utils.metrics import compute_metrics
    torch.manual_seed(941004)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    spec = json.loads(mf.read_text())
    selected = []
    for entry in spec['datasets']:
        label = entry['id']
        if args.datasets and label not in args.datasets: continue
        doc = OmegaConf.load(path(entry['data_config'], repo))
        cfg = doc.get('val_data', doc.get('data', doc))
        cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
        if entry.get('root'): cfg.root = str(path(entry['root'], repo))
        cfg.root = str(path(cfg.root, repo))
        if not isinstance(cfg.name, str) or cfg.get('mode', 'single') != 'single':
            raise ValueError('Use one single dataset in each data config.')
        ev = doc.get('evaluation', {})
        if ev.get('crop', False) or ev.get('resize', False):
            raise ValueError('This adapter evaluates native resolution only.')
        for k in ('crop_size', 'image_size', 'resize', 'height', 'width'):
            if k in cfg: del cfg[k]
        if cfg.name == 'chairs' and not (Path(cfg.root)/'train_val.txt').is_file():
            raise ValueError('Missing Chairs train_val.txt; refusing the loader 90/10 fallback.')
        ds = build_dataset(cfg, split='val')
        rows, policy = inventory(ds, cfg, entry, repo)
        selected.append((entry, cfg, ds, rows, policy, int(ev.get('pad_to_stride', 8)), ev.get('pad_mode','replicate')))
        json_write(out / 'inventories' / (label + '.json'), {'dataset':label, 'cluster_policy':policy,'rows':rows})
        print(label, len(rows), 'pairs;', len({r['cluster_id'] for r in rows}), 'clusters;', policy, flush=True)
    if not selected: raise ValueError('No datasets selected')
    if args.inventory_only: return
    selected_models = [m for m in spec['models'] if not args.models or m['id'] in args.models]
    if not selected_models: raise ValueError('No models selected')
    for m in selected_models:
        checkpoint = path(m['checkpoint'], repo)
        config = path(m['config'], repo) if m.get('config') else None
        model, meta = load_model(checkpoint, config, args.device)
        for entry, cfg, ds, index, policy, stride, pad_mode in selected:
            target = out / entry['id'] / (m['id'] + '.json')
            if target.exists(): raise FileExistsError(f'{target}; use a new output directory to avoid stale mixtures.')
            rows = []
            for i, identity in enumerate(index):
                s = ds[i]
                f = predict(model, s['image1'], s['image2'], args.device, stride, pad_mode)
                if s['flow'].shape != f.shape:
                    raise ValueError(f'GT/pred shape mismatch for {identity["pair_id"]}')
                metrics = compute_metrics(f, s['flow'], s['valid'], occ_mask=s.get('occlusion'), invalid_mask=s.get('invalid'))
                # Fingerprint the exact GT and masks, so comparisons cannot silently
                # combine different reliability definitions or modified annotations.
                h = hashlib.sha256()
                for key in ('flow', 'valid', 'occlusion', 'invalid'):
                    v = s.get(key); h.update(key.encode())
                    if v is None: h.update(b'NONE')
                    else:
                        a = v.detach().cpu().numpy(); h.update(str((a.shape,str(a.dtype))).encode()); h.update(a.tobytes())
                rows.append({**identity, 'support_sha256':h.hexdigest(),
                             'metrics':{k:float(v) if math.isfinite(float(v)) else None for k,v in metrics.items()}})
                if i % 25 == 0: print(f'{m["id"]}/{entry["id"]}: {i+1}/{len(ds)}', flush=True)
            json_write(target, {'schema':1, 'model':m['id'], 'dataset':entry['id'],
                'provenance':{**provenance, **meta, 'manifest_sha256':sha256(mf), 'metric_code_sha256':sha256(repo / 'utils/metrics.py')},
                'data_config':OmegaConf.to_container(cfg,resolve=True), 'cluster_policy':policy,
                'evaluation':{'native_resolution':True,'pad_to_stride':stride,'pad_mode':pad_mode,
                              'estimator':'equal-weight mean over defined image-pair metrics'}, 'rows':rows})
            print('Saved',target,flush=True)
        del model
        if torch.cuda.is_available(): torch.cuda.empty_cache()

if __name__ == '__main__': main()
