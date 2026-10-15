#!/usr/bin/env python3
"""HQSFlow publication figures. Run init, check, run, or render; see README.md."""
from __future__ import annotations
import argparse
import copy
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import textwrap

import numpy as np
import yaml

VERSION = '1.0.0'
IDS = ['A0','A1','A2','A3','A4','B1','B2','B3','B4','E1','E2','E3','E4','E5','E7','E8']
LABELS = dict(zip(IDS, ['Reference','Generic recurrence','Loss-only physics','Augmented loss','No split state',
    'Analytic data','Learned data','Analytic proximal','Learned proximal','Single depth','Half depth',
    'Coarse allocation','Fine allocation','1/8 allocation','Moderate updates','Large updates']))
DATASETS = {'chairs':'FlyingChairs','things_subset_clean':'FlyingThings3D','sintel_clean':'MPI-Sintel Clean',
    'sintel_final':'MPI-Sintel Final','spring_left':'Spring','kitti2012':'KITTI 2012','kitti2015':'KITTI 2015','hd1k':'HD1K'}

def dump(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str)+'\n')

def digest_file(path):
    h = hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024), b''): h.update(chunk)
    return h.hexdigest()

def digest_arrays(arrays):
    h = hashlib.sha256()
    for a in arrays:
        a = np.ascontiguousarray(a); h.update(str((a.shape,str(a.dtype))).encode()); h.update(a.tobytes())
    return h.hexdigest()

def safe_id(s):
    if not re.fullmatch(r'[A-Za-z0-9_-]+',str(s)): raise ValueError(f'Unsafe ID: {s}')
    return str(s)

def resolve(p, root):
    p = Path(os.path.expandvars(str(p))).expanduser()
    return p.resolve() if p.is_absolute() else (root/p).resolve()

def initialize(args):
    root = Path(args.repo_root).resolve()
    matrix = yaml.safe_load((root/'configs/ablation/ablation_matrix.yaml').read_text())
    suite_path = resolve(args.suite,root)
    suite = yaml.safe_load(suite_path.read_text())
    entries = {d['id']:d for d in suite['datasets']}
    cfg = {'version':1, 'repo_root':str(root), 'output_dir':'outputs/publication_plots',
           'device':'auto', 'amp':False, 'pad_divisor':8, 'resolution':'suite',
           'style':{'width_inches':7.2,'dpi':300,'font_size':8,'datasets_per_plate':4,
                    'models_per_plate':5,'atlas_columns':4,'atlas_models_per_page':16,'show_epe':True},
           'datasets':[], 'models':[]}
    for key,title in DATASETS.items():
        if key not in entries: raise ValueError(f'Suite missing {key}')
        cfg['datasets'].append({'id':key,'title':title,'data':entries[key]['data'],
                                 'selection':{'index':0}})
    for mid in IDS:
        job=matrix['jobs'][mid]
        cfg['models'].append({'id':mid,'label':LABELS[mid],'group':job['group'],
            'checkpoint':str(Path(args.checkpoint_root)/job['base_run_name']/args.stage/args.checkpoint_name),
            'checkpoints':{}, 'config':None, 'overrides':[]})
    dest=Path(args.config)
    if dest.exists(): raise FileExistsError(f'Refusing to overwrite {dest}')
    dest.parent.mkdir(parents=True,exist_ok=True)
    dest.write_text('# Paths are relative to repo_root. Verify checkpoint STAGES against your results table.\n'
                    '# E6 intentionally excluded. Dataset selection is within split=val.\n'+yaml.safe_dump(cfg,sort_keys=False))
    print(f'Created {dest}. Edit dataset roots and checkpoint paths/stages before running.')

def read_config(path):
    cfg=yaml.safe_load(Path(path).read_text()); root=Path(cfg['repo_root']).expanduser().resolve()
    for key in ('models','datasets'):
        ids=[safe_id(x['id']) for x in cfg[key]]
        if not ids or len(ids)!=len(set(ids)): raise ValueError(f'Empty/duplicate {key} IDs')
    if 'A0' not in [m['id'] for m in cfg['models']]: raise ValueError('A0 reference is required')
    cfg['_root']=root;cfg['_out']=resolve(cfg['output_dir'],root)
    if cfg.get('resolution','suite') not in ('suite','full'): raise ValueError('resolution must be suite or full')
    return cfg

def checkpoint_for(m,d,root):
    return resolve(m.get('checkpoints',{}).get(d['id'],m['checkpoint']),root)

def validate_paths(cfg):
    failures=[]
    for d in cfg['datasets']:
        p=resolve(d['data']['root'],cfg['_root'])
        if not p.is_dir(): failures.append(f"{d['id']} data root: {p}")
    for m in cfg['models']:
        for d in cfg['datasets']:
            p=checkpoint_for(m,d,cfg['_root'])
            if not p.is_file(): failures.append(f"{m['id']}/{d['id']} checkpoint: {p}")
        for p in ([m['config']] if m.get('config') else [])+m.get('overrides',[]):
            if not resolve(p,cfg['_root']).is_file(): failures.append(f'{m["id"]} config: {p}')
    if failures: raise FileNotFoundError('Missing files (no substitutions made):\n'+'\n'.join(failures))

def collect_samples(cfg):
    from omegaconf import OmegaConf
    from data import build_dataset
    samples=[]
    for d in cfg['datasets']:
        dc=copy.deepcopy(d['data']);dc['root']=str(resolve(dc['root'],cfg['_root']))
        ds=build_dataset(OmegaConf.create(dc),split='val')
        records=getattr(ds,'_samples',None)
        if records is None: raise TypeError('Use individual datasets, not mixed datasets')
        selection=d.get('selection',{'index':0})
        if 'image1_suffix' in selection:
            suffix=selection['image1_suffix'].replace('\\','/')
            matches=[i for i,r in enumerate(records) if str(r['image1']).replace('\\','/').endswith(suffix)]
            if len(matches)!=1: raise ValueError(f"{d['id']}: suffix matched {len(matches)} validation samples")
            idx=matches[0]
        else: idx=int(selection.get('index',0))
        if idx<0 or idx>=len(ds): raise IndexError(f"{d['id']}: validation index {idx} outside 0..{len(ds)-1}")
        record=records[idx]
        if not record.get('flow'): raise ValueError('Selected sample has no ground truth')
        item=ds[idx]
        a=item['image1'].numpy().transpose(1,2,0);b=item['image2'].numpy().transpose(1,2,0)
        gt=item['flow'].numpy().transpose(1,2,0);valid=item['valid'].numpy().astype(bool)
        if gt.shape!=(*a.shape[:2],2) or a.shape!=b.shape or valid.shape!=a.shape[:2]:
            raise ValueError(f"{d['id']}: incompatible image/flow/mask shapes")
        h,w=a.shape[:2]; crop=[0,0,h,w]
        if cfg.get('resolution','suite')=='suite' and dc.get('eval_center_crop',False):
            size=dc.get('eval_crop_size',dc.get('crop_size'))
            if not size: raise ValueError('eval_center_crop needs eval_crop_size')
            ch,cw=min(int(size[0]),h),min(int(size[1]),w)
            if min(ch,cw)<=0: raise ValueError('Invalid crop size')
            y,x=(h-ch)//2,(w-cw)//2;crop=[y,x,ch,cw]
            a,b,gt,valid=[v[y:y+ch,x:x+cw].copy() for v in (a,b,gt,valid)]
        valid &= np.isfinite(gt).all(-1) & (np.linalg.norm(gt,axis=-1)<1e6)
        if not valid.any(): raise ValueError(f"{d['id']}: no valid GT pixels")
        if not np.isfinite(a).all() or not np.isfinite(b).all(): raise ValueError('Nonfinite input image')
        scale=max(float(np.percentile(np.linalg.norm(gt[valid],axis=-1),95)),1e-6)
        sid='/'.join(Path(record['image1']).parts[-2:])
        meta={'id':d['id'],'title':d.get('title',d['id']),'index':idx,'split':'val','dataset_size':len(ds),
              'sample_id':sid,'record':{k:str(v) if v is not None else None for k,v in record.items()},
              'native_hw':[h,w],'crop_yxhw':crop,'flow_scale_px':scale,'data':dc}
        meta['sample_hash']=digest_arrays([a,b,gt,valid])
        samples.append((meta,{'image1':a,'image2':b,'gt':gt,'valid':valid}))
    sintel=[s for s in samples if s[0]['data'].get('name')=='sintel']
    if len(sintel)==2 and sintel[0][0]['sample_id']!=sintel[1][0]['sample_id']:
        raise ValueError('Select the same scene/frame for Sintel Clean and Final')
    return samples

def load_model(m,cp,root,device):
    import torch
    from omegaconf import OmegaConf
    from models import build_model
    ckpt=torch.load(cp,map_location='cpu',weights_only=False) # trusted research checkpoints only
    cfg=OmegaConf.load(resolve(m['config'],root)) if m.get('config') else ckpt.get('cfg')
    if cfg is None: raise ValueError(f'{cp}: no embedded cfg; set model.config to its resolved training YAML')
    cfg=OmegaConf.create(OmegaConf.to_container(cfg,resolve=True) if OmegaConf.is_config(cfg) else cfg)
    for overlay in m.get('overrides',[]): cfg=OmegaConf.merge(cfg,OmegaConf.load(resolve(overlay,root)))
    # A full trained state replaces pretrained initialisation; preserve constructor compatibility mode.
    core=OmegaConf.select(cfg,'model.hqs_core')
    if core is None: core=OmegaConf.select(cfg,'model')
    if core is not None and core.get('gmflow_pretrained_checkpoint'):
        core['gmflow_released_weights_compatible']=core.get('gmflow_released_weights_compatible',True)
        core['gmflow_pretrained_checkpoint']=None;core['gmflow_freeze_pretrained']=False
    model=build_model(cfg)
    state=next((ckpt[k] for k in ('model','state_dict','model_state_dict') if k in ckpt),ckpt)
    if state and all(k.startswith('module.') for k in state): state={k[7:]:v for k,v in state.items()}
    model.load_state_dict(state,strict=True)
    return model.to(device).eval(),OmegaConf.to_container(cfg,resolve=True)

def infer(model,arrays,device,cfg):
    import torch
    from utils.flow_utils import InputPadder
    a,b=[torch.from_numpy(np.ascontiguousarray(arrays[k].transpose(2,0,1))).unsqueeze(0).to(device) for k in ('image1','image2')]
    pad=InputPadder(a.shape,divisor=int(cfg.get('pad_divisor',8)));a,b=pad.pad(a,b)
    amp=bool(cfg.get('amp',False))
    if amp and device.type!='cuda': raise ValueError('AMP requires CUDA; otherwise use amp: false')
    with torch.inference_mode(),torch.autocast(device_type=device.type,enabled=amp): out=model(a,b)
    pred=out['flow_preds'][-1]
    if tuple(pred.shape)!=(1,2,*a.shape[-2:]): raise ValueError(f'Expected full-resolution flow_preds, got {pred.shape}')
    pred=pad.unpad(pred.float())[0].permute(1,2,0).cpu().numpy()
    if not np.isfinite(pred).all(): raise ValueError('Prediction contains NaN/Inf; not publishing this panel')
    return pred

def run(cfg,check=False):
    validate_paths(cfg)
    sys.path.insert(0,str(cfg['_root']))
    # Repository configs sometimes use relative asset paths.
    os.chdir(cfg['_root'])
    import torch
    torch.manual_seed(0);np.random.seed(0)
    samples=collect_samples(cfg);out=cfg['_out'];out.mkdir(parents=True,exist_ok=True)
    dump(out/'selected_samples.json',[m for m,a in samples])
    for meta,a in samples: print(f"{meta['id']}: {meta['sample_id']} | crop {meta['crop_yxhw']} | p95 {meta['flow_scale_px']:.2f}px",flush=True)
    if check:
        print('Paths and selected validation samples verified. Checkpoint weights are checked during run.');return
    requested=cfg.get('device','auto')
    device=torch.device(('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu') if requested=='auto' else requested)
    cache=out/'cache';cache.mkdir(exist_ok=True)
    for meta,a in samples: np.savez_compressed(cache/f"sample_{meta['id']}.npz",**a)
    try: git=subprocess.check_output(['git','rev-parse','HEAD'],cwd=cfg['_root'],text=True).strip()
    except (OSError,subprocess.CalledProcessError): git='unknown'
    script_hash=digest_file(__file__);hashes={};reports=[]
    for m in cfg['models']:
        by_cp={}
        for d in cfg['datasets']: by_cp.setdefault(checkpoint_for(m,d,cfg['_root']),[]).append(d['id'])
        for cp,dids in by_cp.items():
            hashes.setdefault(str(cp),digest_file(cp)) if str(cp) not in hashes else None
            config_hashes=[digest_file(resolve(p,cfg['_root'])) for p in ([m['config']] if m.get('config') else [])+m.get('overrides',[])]
            model=None;resolved_cfg=None
            for meta,a in samples:
                if meta['id'] not in dids: continue
                token={'version':VERSION,'script':script_hash,'git':git,'checkpoint_sha256':hashes[str(cp)],
                       'config_hashes':config_hashes,'sample_hash':meta['sample_hash'],'amp':cfg.get('amp',False),
                       'pad_divisor':cfg.get('pad_divisor',8),'device':str(device)}
                name=f"{m['id']}__{meta['id']}";npz=cache/f'{name}.npz';jp=cache/f'{name}.json'
                if npz.exists() and jp.exists() and json.loads(jp.read_text()).get('token')==token:
                    report=json.loads(jp.read_text());print('Cached '+name,flush=True)
                else:
                    if model is None: model,resolved_cfg=load_model(m,cp,cfg['_root'],device)
                    print('Predicting '+name,flush=True);pred=infer(model,a,device,cfg)
                    epe=float(np.linalg.norm(pred[a['valid']]-a['gt'][a['valid']],axis=-1).mean())
                    report={'model':m['id'],'dataset':meta['id'],'checkpoint':str(cp),'token':token,
                            'sample_epe_px':epe,'resolved_model_config':resolved_cfg}
                    np.savez_compressed(npz,flow=pred);dump(jp,report)
                reports.append(report)
            del model
            if device.type=='cuda': torch.cuda.empty_cache()
    manifest={'version':VERSION,'git':git,'models':cfg['models'],'datasets':[m for m,a in samples],
              'predictions':reports,'resolution':cfg.get('resolution','suite'),'demo':False}
    dump(out/'manifest.json',manifest)
    with open(out/'sample_metrics.csv','w',newline='') as f:
        w=csv.writer(f);w.writerow(['model','dataset','sample_epe_px','checkpoint'])
        for r in reports:w.writerow([r[k] for k in ('model','dataset','sample_epe_px','checkpoint')])
    render(out,cfg.get('style',{}))

def flow_rgb(flow,scale,valid=None):
    from matplotlib.colors import hsv_to_rgb
    mag=np.linalg.norm(flow,axis=-1);finite=np.isfinite(flow).all(-1)
    clean=np.nan_to_num(flow);angle=np.arctan2(clean[...,1],clean[...,0])
    hsv=np.stack([((angle+np.pi)/(2*np.pi))%1,np.clip(np.nan_to_num(mag)/scale,0,1),np.ones_like(mag)],axis=-1)
    rgb=hsv_to_rgb(hsv);rgb[~finite if valid is None else ~(finite&valid)]=.72
    return rgb

def render(out,style):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    out=Path(out);manifest=json.loads((out/'manifest.json').read_text());models=manifest['models'];ds=manifest['datasets']
    style={'width_inches':7.2,'dpi':300,'font_size':8,'datasets_per_plate':4,'models_per_plate':5,
           'atlas_columns':4,'atlas_models_per_page':16,'show_epe':True,**style}
    for k in ('datasets_per_plate','models_per_plate','atlas_columns','atlas_models_per_page'):
        if int(style[k])<1:raise ValueError(f'{k} must be positive')
    if style['models_per_plate']<2:raise ValueError('models_per_plate must allow reference + variant')
    plt.rcParams.update({'font.family':'serif','font.serif':['Times New Roman','Liberation Serif','DejaVu Serif'],
        'font.size':style['font_size'],'pdf.fonttype':42,'ps.fonttype':42,'axes.titlesize':style['font_size'], 'savefig.facecolor':'white'})
    arrays={d['id']:dict(np.load(out/'cache'/f"sample_{d['id']}.npz",allow_pickle=False)) for d in ds}
    preds={(m['id'],d['id']):np.load(out/'cache'/f"{m['id']}__{d['id']}.npz",allow_pickle=False)['flow'] for m in models for d in ds}
    if not manifest.get('demo'):
        for d in ds:
            a=arrays[d['id']]
            if digest_arrays([a[k] for k in ('image1','image2','gt','valid')])!=d['sample_hash']:
                raise ValueError('Cache/manifest mismatch: finish run before rendering')
            for m in models:
                report=json.loads((out/'cache'/f"{m['id']}__{d['id']}.json").read_text())
                saved=next(r for r in manifest['predictions'] if r['model']==m['id'] and r['dataset']==d['id'])
                if report['token']!=saved['token']:raise ValueError('Prediction cache changed: finish run before rendering')
                pred=preds[m['id'],d['id']]
                if pred.shape!=a['gt'].shape or not np.isfinite(pred).all():raise ValueError('Invalid prediction cache')
    metrics={(r['model'],r['dataset']):r['sample_epe_px'] for r in manifest['predictions']}
    titles={m['id']:m.get('label',m['id']) for m in models};files=[]
    def panel(ax,img,border='#999999',badge=None):
        ax.imshow(img,interpolation='nearest');ax.set_xticks([]);ax.set_yticks([])
        for sp in ax.spines.values():sp.set_color(border);sp.set_linewidth(.6)
        if badge:ax.text(.98,.04,badge,ha='right',va='bottom',transform=ax.transAxes,fontsize=6,
                        bbox={'facecolor':'white','edgecolor':'none','alpha':.88,'pad':1})
    def image(kind,d,mid=None):
        a=arrays[d['id']]
        if kind=='input':return np.concatenate([a['image1'],a['image2']],axis=1)
        return flow_rgb(a['gt'] if kind=='gt' else preds[mid,d['id']],d['flow_scale_px'],a['valid'] if kind=='gt' else None)
    def finish(fig,path,title):
        fig.text(.5,.982,('SYNTHETIC DEMO — ' if manifest.get('demo') else '')+title,ha='center',va='top',weight='bold',fontsize=10)
        # A matching direction wheel (image y increases downwards).
        yy,xx=np.mgrid[-1:1:65j,-1:1:65j];wheel=flow_rgb(np.stack([xx,yy],-1),1);wheel[xx*xx+yy*yy>1]=1
        ax=fig.add_axes([.035,.012,.045,.045]);ax.imshow(wheel);ax.axis('off')
        fig.text(.095,.037,'Hue: direction; saturation: magnitude. Scale = valid-GT p95 per pair; larger magnitudes saturate.',fontsize=6.5)
        fig.text(.095,.021,'Grey GT: unannotated/invalid. Predictions shown densely. EPE: selected pair only (px).',fontsize=6.5)
        path.parent.mkdir(parents=True,exist_ok=True)
        for ext in ('pdf','png'):fig.savefig(path.with_suffix('.'+ext),dpi=int(style['dpi']))
        plt.close(fig);files.append(path.relative_to(out).as_posix()+'.pdf')
    groups={}
    for m in models:
        if m['id']!='A0':groups.setdefault(m.get('group','Other'),[]).append(m['id'])
    nmax=int(style['models_per_plate'])-1
    for group,ids in groups.items():
        for start in range(0,len(ids),nmax):
            cohort=['A0']+ids[start:start+nmax]
            for first in range(0,len(ds),int(style['datasets_per_plate'])):
                block=ds[first:first+int(style['datasets_per_plate'])];nr=2+len(cohort);nc=len(block)
                width=float(style['width_inches']);height=.8+nr*.76
                fig=plt.figure(figsize=(width,height));gs=fig.add_gridspec(nr,nc,left=.12,right=.985,top=.90,bottom=.09,hspace=.19,wspace=.07,height_ratios=[.55]+[1]*(nr-1))
                for j,d in enumerate(block):
                    for r in range(nr):
                        ax=fig.add_subplot(gs[r,j]);kind='input' if r==0 else 'gt' if r==1 else 'pred';mid=cohort[r-2] if r>=2 else None
                        badge=f"p95 {d['flow_scale_px']:.1f}px" if r==1 else f"EPE {metrics[mid,d['id']]:.3f}" if r>=2 and style['show_epe'] else None
                        panel(ax,image(kind,d,mid),'#df8c26' if r==0 else '#269a50' if r==1 else '#999999',badge)
                        if r==0:ax.set_title(d['title']+'\n'+textwrap.shorten(d['sample_id'],width=39,placeholder='…'),fontsize=7,pad=3)
                        if j==0:
                            label='Input pair' if r==0 else 'Ground truth' if r==1 else mid
                            ax.text(-.06,.5,label,transform=ax.transAxes,ha='right',va='center',fontsize=8)
                finish(fig,out/'plates'/f"{safe_id(group)}_{start//nmax+1}_{first//int(style['datasets_per_plate'])+1}",f'{group}: qualitative comparison')
    for d in ds:
        count=int(style['atlas_models_per_page']);nc=int(style['atlas_columns'])
        for first in range(0,len(models),count):
            cohort=models[first:first+count];nr=math.ceil(len(cohort)/nc)
            fig=plt.figure(figsize=(float(style['width_inches']),2.0+nr*1.02))
            gs=fig.add_gridspec(nr+1,nc,left=.025,right=.985,top=.88,bottom=.10,hspace=.32,wspace=.07)
            # Reference row uses two halves; normal atlas requires >=2 columns.
            if nc<2:raise ValueError('atlas_columns must be >=2')
            cut=nc//2
            for sl,kind in [(slice(0,cut),'input'),(slice(cut,nc),'gt')]:
                ax=fig.add_subplot(gs[0,sl]);panel(ax,image(kind,d),'#df8c26' if kind=='input' else '#269a50')
                ax.set_title('Input pair' if kind=='input' else f"Ground truth | p95 {d['flow_scale_px']:.1f}px",weight='bold')
            for k,m in enumerate(cohort):
                ax=fig.add_subplot(gs[1+k//nc,k%nc]);mid=m['id']
                panel(ax,image('pred',d,mid),badge=f"EPE {metrics[mid,d['id']]:.3f}" if style['show_epe'] else None)
                ax.set_title(mid+' '+titles[mid],fontsize=7)
            finish(fig,out/'atlases'/f"{d['id']}_{first//count+1}",d['title']+' | '+d['sample_id'])
    caption=('Qualitative optical-flow estimates on fixed internal validation pairs. '
             'Input frames have orange borders and ground truth has green borders. '
             'All predictions for a pair share the flow colour scale determined by the 95th percentile '
             'of valid ground-truth magnitude; larger magnitudes saturate. Grey ground-truth pixels '
             'lack reliable annotation. Prediction fields remain dense. Panel EPE values refer only '
             'to valid annotated pixels of the displayed pair, not dataset averages.')
    latex=['% Requires \\usepackage{graphicx}. Paths relative to this output directory.']
    for i,f in enumerate(files):
        latex += ['\\begin{figure*}[t]','\\centering',f'\\includegraphics[width=\\textwidth]{{{f}}}',
                  '\\caption{'+caption+'}',f'\\label{{fig:hqs-qual-{i+1}}}','\\end{figure*}','']
    (out/'figures.tex').write_text('\n'.join(latex))
    dump(out/'figure_index.json',{'files':files,'style':style,'caption':caption})
    print(f'Wrote {len(files)} figures, PDF + PNG, and figures.tex to {out}')

def demo(out):
    out=Path(out);cache=out/'cache';cache.mkdir(parents=True,exist_ok=True)
    yy,xx=np.mgrid[0:96,0:160].astype('float32');gt=np.stack([xx/16,yy/24],-1)
    img=np.stack([xx/160,yy/96,np.ones_like(xx)*.35],-1);valid=np.ones(xx.shape,bool)
    models=[{'id':mid,'label':LABELS[mid],'group':'OF-'+mid[0]} for mid in IDS];ds=[];reports=[]
    for i,(did,title) in enumerate(DATASETS.items()):
        v=valid.copy()
        if did in ('hd1k','kitti2012','kitti2015'):v[::2]=False
        a={'image1':img,'image2':np.roll(img,5,axis=1),'gt':gt,'valid':v}
        np.savez_compressed(cache/f'sample_{did}.npz',**a)
        ds.append({'id':did,'title':title,'sample_id':'synthetic fixture','flow_scale_px':10})
        for k,m in enumerate(models):
            pred=gt+np.float32(k*.03);np.savez_compressed(cache/f"{m['id']}__{did}.npz",flow=pred)
            reports.append({'model':m['id'],'dataset':did,'sample_epe_px':float(np.linalg.norm(pred[v]-gt[v],axis=-1).mean())})
    dump(out/'manifest.json',{'models':models,'datasets':ds,'predictions':reports,'demo':True})
    render(out,{'dpi':100})

def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    q=sub.add_parser('init');q.add_argument('--repo-root',default='.');q.add_argument('--suite',default='configs/eval/universal_eval_suite.yaml')
    q.add_argument('--config',default='publication_plots.yaml');q.add_argument('--checkpoint-root',default='checkpoints')
    q.add_argument('--stage',default='curr_u04_universal_balanced_final');q.add_argument('--checkpoint-name',default='best.pth')
    for name in ('check','run'):
        q=sub.add_parser(name);q.add_argument('--config',default='publication_plots.yaml')
    q=sub.add_parser('render');q.add_argument('--output-dir',required=True);q.add_argument('--config',help='Optional configuration for style settings only')
    q=sub.add_parser('demo');q.add_argument('--output-dir',default='outputs/publication_demo')
    args=p.parse_args()
    if args.command=='init':initialize(args)
    elif args.command in ('check','run'):run(read_config(args.config),check=args.command=='check')
    elif args.command=='render':render(args.output_dir,yaml.safe_load(Path(args.config).read_text()).get('style',{}) if args.config else {})
    else:demo(args.output_dir)

if __name__=='__main__':main()
