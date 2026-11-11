import os
import re
import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from omegaconf import OmegaConf

sys.path.insert(0, str(Path.cwd()))  # Repository root, even if this file is in Downloads.

from data import build_dataset, FlowDataset
from models import build_model, resize_flow
from utils import flow_to_hsv
from evaluate_comprehensive import _draw_flow_arrows
from evaluate_selected_dataset import StridePaddedModel

# Run from the repository root (ablation_pipeline branch).
manifest = json.loads(Path('evaluation_manifest.json').read_text())
model_id = os.getenv('MODEL_ID', 'A0')
chosen = next((m for m in manifest['models'] if m['id'] == model_id), None)
if chosen is None and not os.getenv('CHECKPOINT'):
    raise SystemExit(f'MODEL_ID={model_id} is not in evaluation_manifest.json')
checkpoint_path = Path(os.getenv('CHECKPOINT', (chosen or {}).get('checkpoint', '')))
out_root = Path(os.getenv('VIDEO_DIR', 'results/knowledge_share_videos'))
splits = [x.strip() for x in os.getenv('SPLITS', 'val').split(',') if x.strip()]
max_width = int(os.getenv('PANEL_WIDTH', '960'))
fps = int(os.getenv('FPS', '12'))
arrow_step = int(os.getenv('ARROW_STEP', '24'))
arrow_scale = float(os.getenv('ARROW_SCALE', '1.0'))
flow_range = float(os.getenv('FLOW_RANGE', '40'))
if not checkpoint_path.is_file():
    raise SystemExit(f'Missing checkpoint: {checkpoint_path}. Set CHECKPOINT=/path/to/best.pth')
if min(max_width, fps, arrow_step) <= 0 or flow_range <= 0:
    raise SystemExit('PANEL_WIDTH, FPS, ARROW_STEP and FLOW_RANGE must be positive')

device = torch.device('cuda' if torch.cuda.is_available() else
                      'mps' if torch.backends.mps.is_available() else 'cpu')
checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
config = checkpoint.get('cfg') if isinstance(checkpoint, dict) else None
if config is None:
    config_file = os.getenv('MODEL_CONFIG')
    if not config_file:
        raise SystemExit('Checkpoint has no embedded cfg; specify MODEL_CONFIG=/path/to/resolved_config.yaml')
    config = OmegaConf.load(config_file)
else:
    config = OmegaConf.create(config)
model = build_model(config)
state = next((checkpoint[k] for k in ('model', 'state_dict', 'model_state_dict')
              if isinstance(checkpoint, dict) and isinstance(checkpoint.get(k), dict)), checkpoint)
if all(str(k).startswith('module.') for k in state):
    state = {str(k)[7:]: v for k, v in state.items()}
model.load_state_dict(state, strict=True)
del checkpoint, state
model = StridePaddedModel(model.to(device).eval(), stride=8, pad_mode='replicate')
print(f'Model: {checkpoint_path} | device: {device} | splits: {splits}', flush=True)


def safe_name(value):
    return re.sub(r'[^A-Za-z0-9_.-]+', '_', str(value)).strip('._')[:130]


def scene_group(dataset_id, record):
    p = Path(record['image1'])
    q = Path(record['image2'])
    if p.parent != q.parent:
        return None
    if dataset_id.startswith('sintel_'):
        return p.parent.name
    if dataset_id.startswith('spring_') and p.parent.name.startswith('frame_'):
        return p.parent.parent.name + '_' + p.parent.name
    if dataset_id == 'hd1k':
        match = re.match(r'^(\d{6})_\d{6}$', p.stem)
        return 'hd1k_' + match.group(1) if match else None
    if dataset_id == 'things_subset':
        if p.parent.name in ('left', 'right') and p.parent.parent.name.startswith('image_'):
            return 'inferred_' + p.parent.name
        if p.parent.name in ('left', 'right'):
            return 'things_' + p.parent.parent.parent.name + '_' + p.parent.parent.name + '_' + p.parent.name
    return None  # FlyingChairs and KITTI have unrelated isolated pairs.


def temporal_chains(indexed_records):
    # Only join pairs where the next pair's first image IS the current second image.
    by_source = {}
    for index, record in indexed_records:
        key = str(Path(record['image1']))
        if key in by_source:
            return []  # Ambiguous branching source => do not guess.
        by_source[key] = (index, record)
    targets = {str(Path(record['image2'])) for _, record in indexed_records}
    starts = sorted(set(by_source) - targets) + sorted(set(by_source) & targets)
    visited, chains = set(), []
    for start in starts:
        current, chain = start, []
        while current in by_source and current not in visited:
            visited.add(current)
            index, record = by_source[current]
            chain.append((index, record))
            current = str(Path(record['image2']))
        if chain:
            chains.append(chain)
    return chains


def read_pair(record):
    a, b = (cv2.imread(str(record[key]), cv2.IMREAD_COLOR)
            for key in ('image1', 'image2'))
    if a is None or b is None or a.shape != b.shape:
        raise ValueError('missing images or mismatched dimensions')
    h, w = a.shape[:2]
    width = min(max_width, w)
    width = max(2, width - width % 2)
    height = max(2, int(round(h * width / w)))
    height += height % 2
    if (width, height) != (w, h):
        a = cv2.resize(a, (width, height), interpolation=cv2.INTER_AREA)
        b = cv2.resize(b, (width, height), interpolation=cv2.INTER_AREA)
    return a, b


def panel(img, title):
    img = img.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = max(0.45, img.shape[1] / 1500)
    thick = max(1, int(round(scale * 2)))
    box_h = max(28, int(round(36 * scale)))
    cv2.rectangle(img, (0, 0), (img.shape[1], box_h), (14, 14, 14), -1)
    cv2.putText(img, title, (10, max(19, int(25 * scale))), font, scale,
                (255, 255, 255), thick, cv2.LINE_AA)
    return img


def run_video(chain, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    writer = None
    frames = 0
    try:
        with torch.inference_mode():
            for _, record in chain:
                a, b = read_pair(record)
                image1 = FlowDataset._to_tensor(a).unsqueeze(0).to(device)
                image2 = FlowDataset._to_tensor(b).unsqueeze(0).to(device)
                with torch.autocast(device_type=device.type, enabled=(device.type == 'cuda')):
                    outputs = model(image1, image2)
                flow = outputs['flow_preds'][-1].float()
                flow = resize_flow(flow, a.shape[:2])
                flow_xy = flow[0].permute(1, 2, 0).cpu().numpy()
                flow_xy = np.nan_to_num(flow_xy, nan=0.0, posinf=0.0, neginf=0.0)
                image_rgb = cv2.cvtColor(a, cv2.COLOR_BGR2RGB)
                hsv = flow_to_hsv(flow_xy, max_magnitude=flow_range)
                top_right = _draw_flow_arrows(flow_xy, image_rgb.astype(np.float32) / 255,
                                              step=arrow_step, scale=arrow_scale)
                bottom_right = _draw_flow_arrows(flow_xy, hsv.astype(np.float32) / 255,
                                                 step=arrow_step, scale=arrow_scale)
                top = np.concatenate([panel(image_rgb, 'SOURCE VIDEO'),
                                      panel(top_right, 'ESTIMATED FLOW: ARROWS')], axis=1)
                bottom = np.concatenate([panel(hsv, 'ESTIMATED FLOW: HSV'),
                                         panel(bottom_right, 'HSV + ARROWS')], axis=1)
                frame = cv2.cvtColor(np.concatenate([top, bottom], axis=0),
                                     cv2.COLOR_RGB2BGR)
                if writer is None:
                    size = (frame.shape[1], frame.shape[0])
                    for codec in ('mp4v', 'avc1'):
                        test = cv2.VideoWriter(str(destination), cv2.VideoWriter_fourcc(*codec), fps, size)
                        if test.isOpened():
                            writer = test
                            break
                        test.release()
                    if writer is None:
                        raise RuntimeError('OpenCV cannot open H.264 or MPEG4 video writer')
                writer.write(frame)
                frames += 1
    except Exception:
        if writer is not None:
            writer.release()
        destination.unlink(missing_ok=True)
        raise
    else:
        if writer is not None:
            writer.release()
        if frames == 0 or not destination.is_file() or destination.stat().st_size == 0:
            destination.unlink(missing_ok=True)
            raise RuntimeError('video file was empty')
    return frames


for entry in manifest['datasets']:
    name = entry['id']
    doc = OmegaConf.load(entry['data_config'])
    cfg = doc.get('val_data') or doc.get('data') or doc
    for split in splits:
        try:
            dataset = build_dataset(cfg, split=split)
        except Exception as error:
            print(f'[SKIP] {name}/{split}: dataset unavailable: {error}', flush=True)
            continue
        groups = defaultdict(list)
        for index, record in enumerate(dataset._samples):
            group = scene_group(name, record)
            if group is not None:
                groups[group].append((index, record))
        if not groups:
            print(f'[SKIP] {name}/{split}: no defensible multi-frame scene identity', flush=True)
            continue
        made = 0
        for group, records in sorted(groups.items()):
            chains = temporal_chains(records)
            for part, chain in enumerate(chains, 1):
                if len(chain) < 2:
                    continue
                base = safe_name(group)
                if len(chains) > 1 or base.startswith('inferred_'):
                    base += f'__{safe_name(Path(chain[0][1]["image1"]).stem)}_to_{safe_name(Path(chain[-1][1]["image2"]).stem)}'
                destination = out_root / name / split / f'{base}.mp4'
                try:
                    n = run_video(chain, destination)
                    made += 1
                    print(f'[OK] {destination} ({n} frames)', flush=True)
                except Exception as error:
                    print(f'[SKIP] {name}/{split}/{base}: {error}', flush=True)
                    if device.type == 'cuda':
                        torch.cuda.empty_cache()
        if made == 0:
            print(f'[SKIP] {name}/{split}: no usable temporal chains of 2+ pairs', flush=True)
print(f'Done. MP4-only output under {out_root}/', flush=True)
