"""Shared, deliberately small adapter for bkr11a/torch_flow ablation_pipeline."""
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

PINNED_COMMIT = '1d3d82b5c01b46c83b21cae3648dcfa8bee6472f'

def path(value, base):
    p = Path(os.path.expandvars(os.path.expanduser(str(value))))
    return (p if p.is_absolute() else Path(base) / p).resolve()

def sha256(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def json_write(p, obj):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')
    tmp.replace(p)

def repo_setup(repo):
    repo = Path(repo).resolve()
    if not (repo / 'models/hqs_flow.py').exists():
        raise ValueError(f'Not the torch_flow repository: {repo}')
    sys.path.insert(0, str(repo))
    # Existing configs may use paths relative to the repository.
    os.chdir(repo)
    def git(*args):
        p = subprocess.run(['git', *args], cwd=repo, text=True, capture_output=True)
        return p.stdout.strip() if p.returncode == 0 else 'unknown'
    return {'repo': str(repo), 'commit': git('rev-parse', 'HEAD'),
            'dirty': bool(git('status', '--porcelain', '--untracked-files=no')),
            'adapter_tested_commit': PINNED_COMMIT}

def load_model(checkpoint_path, config_path, device):
    import torch
    from omegaconf import OmegaConf
    from models import build_model
    # Only load your own/trusted PyTorch checkpoints (cfg contains Python objects).
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    saved = ckpt.get('cfg') if isinstance(ckpt, dict) else None
    cfg = OmegaConf.load(config_path) if config_path else saved
    if cfg is None:
        raise ValueError('Checkpoint has no cfg; supply its resolved_config.yaml in config.')
    if not OmegaConf.is_config(cfg):
        cfg = OmegaConf.create(cfg)
    if saved is not None and config_path:
        if not OmegaConf.is_config(saved):
            saved = OmegaConf.create(saved)
        for k in ('model', 'model_type'):
            def plain(v):
                return OmegaConf.to_container(v, resolve=True) if OmegaConf.is_config(v) else v
            if plain(cfg.get(k)) != plain(saved.get(k)):
                raise ValueError(f'External config differs from checkpoint {k}; use checkpoint cfg.')
    original = OmegaConf.to_container(cfg, resolve=True)
    build_cfg = OmegaConf.create(original)
    # Full checkpoints contain these weights. Suppress only redundant pretrained
    # file reads; preserve the architecture flag whose default depends on the path.
    def suppress_pretrain(node):
        if not OmegaConf.is_dict(node):
            return
        if node.get('gmflow_pretrained_checkpoint'):
            if 'gmflow_released_weights_compatible' not in node:
                node.gmflow_released_weights_compatible = True
            node.gmflow_pretrained_checkpoint = None
            node.gmflow_freeze_pretrained = False  # inference only; all params loaded below
        for v in node.values():
            suppress_pretrain(v)
    suppress_pretrain(build_cfg)
    state = ckpt.get('model', ckpt.get('state_dict', ckpt))
    if not isinstance(state, dict) or not state:
        raise ValueError('Expected a model state_dict or checkpoint[model].')
    if all(k.startswith('module.') for k in state):
        state = {k[7:]: v for k, v in state.items()}
    model = build_model(build_cfg)
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    return model, {'checkpoint': str(checkpoint_path), 'checkpoint_sha256': sha256(checkpoint_path),
                   'checkpoint_step': ckpt.get('step'), 'resolved_config': original,
                   'strict_state_loading': True, 'precision': 'float32',
                   'torch_version': torch.__version__, 'device': str(device)}

def predict(model, a, b, device, stride=8, pad_mode='replicate'):
    import torch
    import torch.nn.functional as F
    if a.shape != b.shape or a.ndim != 3 or a.shape[0] != 3:
        raise ValueError('Expected equal RGB tensors [3,H,W].')
    if stride < 1:
        raise ValueError('stride must be positive')
    h, w = a.shape[-2:]
    dh, dw = (-h) % stride, (-w) % stride
    top, left = dh // 2, dw // 2
    pad = (left, dw-left, top, dh-top)
    a, b = a[None].to(device), b[None].to(device)
    if dh or dw:
        a, b = F.pad(a, pad, mode=pad_mode), F.pad(b, pad, mode=pad_mode)
    with torch.inference_mode():
        out = model(a, b)
        f = out['flow_preds'][-1]
    if f.shape != (1, 2, h+dh, w+dw):
        raise ValueError(f'Unexpected final flow shape {tuple(f.shape)}')
    f = f[0, :, top:top+h, left:left+w].float().cpu()
    if not torch.isfinite(f).all():
        raise ValueError('Nonfinite prediction; refusing to omit it or encode it as valid.')
    return f

def read_rgb(p):
    import cv2
    import numpy as np
    import torch
    a = cv2.imread(str(p), cv2.IMREAD_COLOR)
    if a is None:
        raise FileNotFoundError(p)
    return torch.from_numpy(np.ascontiguousarray(a[:, :, ::-1].transpose(2, 0, 1))).float() / 255.
