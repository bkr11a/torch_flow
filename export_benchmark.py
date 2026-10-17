#!/usr/bin/env python3
"""A0 native-resolution test predictions and official benchmark packaging.

Sintel: official Bundler -> LZMA. Spring: official flow_subsampling -> HDF5.
KITTI2012: root-level PNG ZIP. KITTI2015: flow/ PNG ZIP.
No login, submission, training, parameter search, or test-ground-truth access.
"""
import argparse
import json
import re
import struct
import subprocess
import zipfile
from pathlib import Path
import numpy as np
from common import json_write, load_model, predict, read_rgb, repo_setup, sha256


def adjacent(directory):
    frames=sorted(Path(directory).glob('*.png'))
    if len(frames)<2: raise ValueError(f'Fewer than two frames: {directory}')
    def number(p): return int(re.search(r'(\d+)$',p.stem).group(1))
    for a,b in zip(frames,frames[1:]):
        if number(b)!=number(a)+1: raise ValueError(f'Missing adjacent frame between {a} and {b}')
        yield a,b


def records(benchmark, root):
    root=Path(root); result=[]
    if root.name.lower() in ('train','training'):
        raise ValueError('Public submissions require test images, not the training split.')
    def add(a,b,rel):
        if not a.is_file() or not b.is_file(): raise FileNotFoundError(f'{a}, {b}')
        result.append({'image1':str(a.resolve()),'image2':str(b.resolve()),'output':str(rel)})
    if benchmark=='sintel':
        base=root/'test' if (root/'test').is_dir() else root
        scenes_a=sorted(p.name for p in (base/'clean').iterdir() if p.is_dir())
        scenes_b=sorted(p.name for p in (base/'final').iterdir() if p.is_dir())
        if len(scenes_a)!=12 or scenes_a!=scenes_b:
            raise ValueError('Expected the same 12 official Sintel test scenes in clean/ and final/.')
        for scene in scenes_a:
            clean=list(adjacent(base/'clean'/scene)); final=list(adjacent(base/'final'/scene))
            if [(a.name,b.name) for a,b in clean] != [(a.name,b.name) for a,b in final]:
                raise ValueError('Sintel rendering passes have different frame pairs')
            for render,pairs in [('clean',clean),('final',final)]:
                for a,b in pairs:
                    # Bundler README uses frame0001.flo (no underscore), although
                    # the image archive uses frame_0001.png.
                    n=re.search(r'(\d+)$',a.stem).group(1)
                    add(a,b,Path(render)/scene/f'frame{n}.flo')
    elif benchmark in ('kitti2012','kitti2015'):
        base=root/'testing' if (root/'testing').is_dir() else root
        camera='image_0' if benchmark=='kitti2012' else 'image_2'
        count=195 if benchmark=='kitti2012' else 200
        expected={f'{i:06d}_10.png' for i in range(count)}
        found={p.name for p in (base/camera).glob('*_10.png')}
        if found!=expected: raise ValueError(f'Expected exactly {count} {benchmark} test pairs; got {len(found)}')
        for name in sorted(expected):
            rel=Path(name) if benchmark=='kitti2012' else Path('flow')/name
            add(base/camera/name,base/camera/name.replace('_10.png','_11.png'),rel)
    elif benchmark=='spring':
        base=root/'test' if (root/'test').is_dir() else root
        scenes=sorted(p for p in base.iterdir() if p.is_dir() and (p/'frame_left').is_dir())
        if not scenes: raise ValueError('No Spring test scenes found')
        for scene in scenes:
            pairs_left=list(adjacent(scene/'frame_left')); pairs_right=list(adjacent(scene/'frame_right'))
            if len(pairs_left)!=len(pairs_right): raise ValueError('Spring stereo frame counts differ')
            for side,pairs in [('left',pairs_left),('right',pairs_right)]:
                for a,b in pairs:
                    for direction,src,dst in [('FW',a,b),('BW',b,a)]:
                        n=re.search(r'(\d+)$',src.stem).group(1)
                        add(src,dst,Path(scene.name)/f'flow_{direction}_{side}'/f'flow_{direction}_{side}_{n}.flo5')
    else: raise ValueError(benchmark)
    if len({r['output'] for r in result})!=len(result): raise ValueError('Duplicate output name')
    return result


def write_flow(p, f):
    """f: float32 H,W,2; fully dense, finite xy flow in input-image pixels."""
    import cv2
    if f.ndim!=3 or f.shape[-1]!=2 or not np.isfinite(f).all(): raise ValueError('Invalid flow')
    p=Path(p); p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_name(p.stem+'.tmp'+p.suffix)
    if p.suffix=='.flo':
        with open(tmp,'wb') as h:
            h.write(struct.pack('<fii',202021.25,f.shape[1],f.shape[0])); h.write(f.astype('<f4').tobytes())
    elif p.suffix=='.flo5':
        import h5py
        with h5py.File(tmp,'w') as h:
            h.create_dataset('flow',data=f.astype(np.float32),compression='gzip',compression_opts=5)
    elif p.suffix=='.png':
        encoded=np.rint(f.astype(np.float64)*64+32768)
        if encoded.min()<0 or encoded.max()>65535:
            raise ValueError('Flow exceeds KITTI PNG range [-512,511.984375]; no silent clipping is permitted.')
        rgb=np.concatenate([encoded.astype(np.uint16),np.ones((*f.shape[:2],1),np.uint16)],axis=2)
        if not cv2.imwrite(str(tmp),rgb[:,:,::-1]): raise IOError(f'Failed writing {p}')
    else: raise ValueError(p.suffix)
    tmp.replace(p)


def verify_flow(p, shape):
    import cv2
    p=Path(p)
    if not p.is_file(): raise FileNotFoundError(p)
    if p.suffix=='.flo':
        with open(p,'rb') as h:
            tag,w,hh=struct.unpack('<fii',h.read(12)); data=np.frombuffer(h.read(),dtype='<f4')
        if tag!=202021.25 or (hh,w)!=tuple(shape) or data.size!=hh*w*2 or not np.isfinite(data).all():
            raise ValueError(f'Invalid FLO {p}')
    elif p.suffix=='.flo5':
        import h5py
        with h5py.File(p) as h: data=h['flow'][()]
        if data.shape!=(*shape,2) or not np.isfinite(data).all(): raise ValueError(f'Invalid FLO5 {p}')
    else:
        data=cv2.imread(str(p),cv2.IMREAD_UNCHANGED)
        if data is None or data.dtype!=np.uint16 or data.shape!=(*shape,3) or not np.all(data[:,:,0]==1):
            raise ValueError(f'Invalid dense KITTI PNG {p}')


def pack(benchmark, predictions, out, recs, sintel_bundler=None, spring_subsampler=None):
    out=Path(out); predictions=Path(predictions)
    if benchmark.startswith('kitti'):
        dest=out/(benchmark+'_A0.zip')
        with zipfile.ZipFile(dest,'w',compression=zipfile.ZIP_DEFLATED,allowZip64=True) as z:
            for r in recs: z.write(predictions/r['output'],r['output'])
        return [dest]
    if benchmark=='sintel':
        if not sintel_bundler: return []
        dest=out/'sintel_A0.lzma'
        subprocess.run([str(Path(sintel_bundler).resolve()),str(predictions/'clean'),str(predictions/'final'),str(dest)],check=True)
        if not dest.is_file() or dest.stat().st_size==0: raise RuntimeError('Bundler produced no submission')
        return [dest]
    if not spring_subsampler: return []
    work=out/'submission'; work.mkdir(exist_ok=True)
    if list(work.glob('*.hdf5')): raise FileExistsError('Move the previous Spring HDF5 before repackaging.')
    subprocess.run([str(Path(spring_subsampler).resolve()),str(predictions)],cwd=work,check=True)
    files=list(work.glob('*.hdf5'))
    if not files or any(p.stat().st_size==0 for p in files): raise RuntimeError('Spring subsampler produced no HDF5')
    return files


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo',default='.')
    p.add_argument('--benchmark',required=True,choices=['sintel','kitti2012','kitti2015','spring'])
    p.add_argument('--root',required=True)
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--config',help='Only needed for a checkpoint without cfg')
    p.add_argument('--out',required=True)
    p.add_argument('--device',default='cuda')
    p.add_argument('--stride',type=int,default=8)
    p.add_argument('--sintel-bundler')
    p.add_argument('--spring-subsampler')
    p.add_argument('--resume',action='store_true')
    p.add_argument('--pack-only',action='store_true')
    args=p.parse_args()
    # Resolve before changing cwd to the repository.
    for k in ('repo','root','checkpoint','config','out','sintel_bundler','spring_subsampler'):
        if getattr(args,k,None): setattr(args,k,str(Path(getattr(args,k)).expanduser().resolve()))
    provenance=repo_setup(args.repo)
    out=Path(args.out); pred=out/'predictions'; recs=records(args.benchmark,args.root)
    if args.stride<1: raise ValueError('stride must be positive')
    plan={'benchmark':args.benchmark,'checkpoint_sha256':sha256(args.checkpoint),
          'config_sha256':sha256(args.config) if args.config else None,
          'stride':args.stride,'pad_mode':'replicate','precision':'float32','records':recs,
          'repo_commit':provenance['commit']}
    old=out/'plan.json'
    if old.exists():
        if not(args.resume or args.pack_only): raise FileExistsError('Use --resume or a fresh output directory')
        if json.loads(old.read_text())!=plan: raise ValueError('Existing export uses a different model/config/source plan')
    else:
        if args.pack_only: raise FileNotFoundError('No completed export plan to package')
        if out.exists() and any(out.iterdir()): raise FileExistsError('Output directory is not empty')
        json_write(old,plan)
    if not args.pack_only:
        import torch
        torch.manual_seed(941004); torch.backends.cudnn.benchmark=False
        torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
        model,meta=load_model(args.checkpoint,args.config,args.device)
        json_write(out/'provenance.json',{**provenance,**meta,'test_time_augmentation':False,
                   'resize':False,'backward_flow':'separate inference with reversed frame order'})
        for i,r in enumerate(recs):
            dest=pred/r['output']
            a,b=read_rgb(r['image1']),read_rgb(r['image2'])
            if dest.exists(): verify_flow(dest,tuple(a.shape[-2:]))
            else:
                f=predict(model,a,b,args.device,args.stride).permute(1,2,0).numpy()
                write_flow(dest,f); verify_flow(dest,tuple(a.shape[-2:]))
            if i%25==0: print(f'{args.benchmark}: {i+1}/{len(recs)}',flush=True)
        del model
    import cv2
    for r in recs:
        im=cv2.imread(r['image1'],cv2.IMREAD_GRAYSCALE)
        if im is None: raise FileNotFoundError(r['image1'])
        verify_flow(pred/r['output'],im.shape)
    files=pack(args.benchmark,pred,out,recs,args.sintel_bundler,args.spring_subsampler)
    json_write(out/'export_status.json',{'predictions_verified':len(recs),
               'upload_files':[{'path':str(f),'sha256':sha256(f)} for f in files],
               'status':'packaged' if files else 'predictions complete; official bundler/subsampler still required'})
    print('Upload files:',*[str(f) for f in files],sep='\n')
    if not files: print('Supply the official --sintel-bundler or --spring-subsampler and rerun with --pack-only.')

if __name__=='__main__': main()
