"""Step 4.1: explainable brute-force top-5 retrieval from step-3 SQLite.

Run from fish_step3 directory:
  python step4_baseline.py --query-id 0000 --zip ../images.zip --method equal
  python step4_baseline.py --query-image ../query_images/fish1.png --zip ../images.zip --method hsv

This is a functionality baseline. It is NOT evaluated/optimized for human similarity.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sqlite3
import zipfile
from pathlib import Path

import cv2
import numpy as np

from fishcbir.database import config_key, load_feature_matrix
from fishcbir.features import FeatureParameters, extract
from fishcbir.preprocessing import preprocess

FAMILIES = ('glcm', 'lbp', 'geometry', 'contour', 'fourier', 'hsv', 'moments', 'spatial')
METHODS = {
    'equal': FAMILIES,
    'color': ('hsv', 'moments', 'spatial'),
    'shape': ('geometry', 'contour', 'fourier'),
    'texture': ('glcm', 'lbp'),
    **{name: (name,) for name in FAMILIES},
}


def load_gallery(db_file: Path):
    """Load exactly the step3-v1 descriptors and metadata (not later configurations)."""
    if not db_file.is_file():
        raise FileNotFoundError(f'Database not found: {db_file}')
    params = FeatureParameters()
    with sqlite3.connect(str(db_file)) as con:
        rows = con.execute('SELECT image_id,zip_member,sha256 FROM images ORDER BY image_id').fetchall()
        if not rows:
            raise RuntimeError('No images in database')
        ids = [r[0] for r in rows]
        members = dict((r[0],r[1]) for r in rows)
        sha_to_ids = {}
        for iid, _, digest in rows:
            sha_to_ids.setdefault(digest, set()).add(iid)
        matrices = {}
        for name in FAMILIES:
            key = config_key(name, params.config_for(name))
            family_ids, X = load_feature_matrix(con, key)
            if family_ids != ids or X.shape != (len(ids), params.dimensions()[name]):
                raise RuntimeError(f'Invalid/incomplete matrix for {name}: {X.shape}; expected IDs {len(ids)}')
            if not np.isfinite(X).all():
                raise RuntimeError(f'Nonfinite descriptor in {name}')
            matrices[name] = np.asarray(X, dtype=np.float64)
    return ids, members, sha_to_ids, matrices


def prepare_query_image(filename: Path):
    """Resize with white padding, consistent with the 640x320 image canvas."""
    raw = filename.read_bytes()
    bgr = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f'Unable to decode image: {filename}')
    h,w = bgr.shape[:2]
    target_w,target_h=640,320
    if (w,h)!=(target_w,target_h):
        scale = min(target_w/w,target_h/h)
        nw,nh = max(1,round(w*scale)),max(1,round(h*scale))
        inter = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
        bgr_small = cv2.resize(bgr,(nw,nh),interpolation=inter)
        canvas = np.full((target_h,target_w,3),255,np.uint8)
        left,top=(target_w-nw)//2,(target_h-nh)//2
        canvas[top:top+nh,left:left+nw]=bgr_small
        bgr=canvas
        print(f'NOTE: Query resized with white padding: {w}x{h} -> {target_w}x{target_h}')
    rgb = cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
    prep = preprocess(rgb)
    if prep.status != 'ok':
        print(f'WARNING: Query segmentation flag: {prep.status}; inspect mask before interpreting results')
    return extract(prep, FeatureParameters()), bgr, hashlib.sha256(raw).hexdigest()


def distances(query: np.ndarray, gallery: np.ndarray, name: str, std: np.ndarray):
    """One distance per gallery image. Gallery-fitted std is used for scalar descriptors."""
    eps = 1e-12
    q = np.asarray(query, dtype=np.float64)
    x = np.asarray(gallery, dtype=np.float64)
    if name == 'lbp':
        return 0.5*np.sum((x-q)**2/(x+q+eps),axis=1)
    if name == 'hsv':
        return np.sqrt(np.sum((np.sqrt(np.maximum(x,0))-np.sqrt(np.maximum(q,0)))**2,axis=1)/2)
    if name == 'spatial':
        n_regions = FeatureParameters().spatial_grid[0]*FeatureParameters().spatial_grid[1]
        xx = x.reshape(len(x), n_regions,-1)
        qq = q.reshape(1,n_regions,-1)
        per_region = np.sqrt(np.sum((np.sqrt(np.maximum(xx,0))-np.sqrt(np.maximum(qq,0)))**2,axis=2)/2)
        return per_region.mean(axis=1)
    z = (x-q) / std
    return np.sqrt(np.mean(z*z,axis=1))


def calibration(matrices:dict[str,np.ndarray], seed:int=2026, n_pairs:int=1200):
    """Median pairwise raw distance per descriptor, gallery-only and reproducible."""
    rng = np.random.default_rng(seed)
    n=len(next(iter(matrices.values())))
    i=rng.integers(0,n,size=n_pairs)
    j=rng.integers(0,n,size=n_pairs)
    use=i!=j
    i,j=i[use],j[use]
    scales={}
    stds={}
    for name,X in matrices.items():
        sd=np.maximum(np.std(X,axis=0),1e-6)
        stds[name]=sd
        # Reuse the same distance implementation to avoid train/query differences.
        # Limit pairs to maintain low memory even with the spatial descriptor.
        d=np.array([distances(X[k], X[t:t+1], name, sd)[0]
                    for k,t in zip(i,j)],dtype=np.float64)
        good=d[np.isfinite(d)&(d>1e-9)]
        if good.size==0:
            raise RuntimeError(f'Cannot calibrate descriptor {name}: all zero/invalid distances')
        scales[name]=float(np.median(good))
    return stds,scales


def make_collage(query_bgr, results, zip_file: Path, members: dict, output: Path, title: str):
    panel_w,panel_h=640,350
    canvas = np.full((panel_h*2,panel_w*3,3),245,np.uint8)
    with zipfile.ZipFile(zip_file) as z:
        tiles=[('QUERY',query_bgr)]
        for r in results:
            raw=z.read(members[r['image_id']])
            img=cv2.imdecode(np.frombuffer(raw,np.uint8),cv2.IMREAD_COLOR)
            if img is None: raise ValueError(f'Cannot decode {members[r["image_id"]]}')
            label=f"#{r['rank']}  {r['image_id']}  D={r['distance_total']:.4f}"
            tiles.append((label,img))
    for idx,(label,img) in enumerate(tiles):
        img=cv2.resize(img,(640,320))
        r,c=divmod(idx,3)
        y,x=r*panel_h,c*panel_w
        canvas[y:y+320,x:x+640]=img
        cv2.putText(canvas,label,(x+14,y+341),cv2.FONT_HERSHEY_SIMPLEX,0.65,(15,15,15),2,cv2.LINE_AA)
    output.parent.mkdir(parents=True,exist_ok=True)
    if not cv2.imwrite(str(output),canvas):
        raise OSError(f'Cannot save {output}')


def main():
    parser=argparse.ArgumentParser(description='Step 4.1: top-5 fish image retrieval (baseline, no training)')
    q=parser.add_mutually_exclusive_group(required=True)
    q.add_argument('--query-id',help='Existing 4-digit image ID for a smoke test; self match excluded')
    q.add_argument('--query-image',type=Path,help='New external image; will be preprocessed and extracted')
    parser.add_argument('--db',type=Path,default=Path('data/fish.db'))
    parser.add_argument('--zip',type=Path,default=Path('../images.zip'))
    parser.add_argument('--method',choices=tuple(METHODS),default='equal')
    parser.add_argument('--out',type=Path,default=Path('outputs/step4_baseline'))
    a=parser.parse_args()
    if not a.zip.is_file():
        parser.error(f'Original image ZIP missing: {a.zip} (required for output visualization)')
    ids,members,sha_to_ids,matrices=load_gallery(a.db)
    id_to_ix={name:i for i,name in enumerate(ids)}
    if a.query_id:
        if a.query_id not in id_to_ix:
            parser.error(f'Invalid query ID {a.query_id}; use 0000...0999')
        ix=id_to_ix[a.query_id]
        query={name:X[ix] for name,X in matrices.items()}
        with zipfile.ZipFile(a.zip) as z:
            raw=z.read(members[a.query_id])
        query_bgr=cv2.imdecode(np.frombuffer(raw,np.uint8),cv2.IMREAD_COLOR)
        excluded={a.query_id}
        source=f'gallery:{a.query_id} (self excluded)'
        print('SMOKE TEST ONLY: using an indexed image; not a valid external-query evaluation.')
    else:
        if not a.query_image.is_file():
            parser.error(f'Query file not found: {a.query_image}')
        query,query_bgr,digest=prepare_query_image(a.query_image)
        excluded=set(sha_to_ids.get(digest,set()))
        if excluded:
            print(f'NOTE: Exact duplicate query file found inside gallery; excluding IDs {sorted(excluded)}')
        source=str(a.query_image)
    stds, scales = calibration(matrices)
    raw={name:distances(query[name],matrices[name],name,stds[name]) for name in FAMILIES}
    normalized={name:raw[name]/scales[name] for name in FAMILIES}
    used=METHODS[a.method]
    combined=np.mean(np.stack([normalized[name] for name in used]),axis=0)
    for iid in excluded: combined[id_to_ix[iid]]=np.inf
    winners=np.argsort(combined,kind='stable')[:5]
    output=[]
    for rank,idx in enumerate(winners,1):
        if not np.isfinite(combined[idx]):
            continue
        row={'rank':rank,'image_id':ids[idx], 'distance_total':float(combined[idx])}
        row.update({f'distance_{name}':float(normalized[name][idx]) for name in FAMILIES})
        output.append(row)
    if len(output)!=5:
        raise RuntimeError('Fewer than 5 matches')
    a.out.mkdir(parents=True,exist_ok=True)
    prefix=f"{a.method}_{a.query_id or a.query_image.stem}"
    csv_path=a.out/f'{prefix}_top5.csv'
    with csv_path.open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(output[0]));writer.writeheader();writer.writerows(output)
    img_path=a.out/f'{prefix}_top5.png'
    make_collage(query_bgr,output,a.zip,members,img_path,prefix)
    meta={'query':source,'method':a.method,'used_groups':used,'calibration_medians':scales,
          'method_note':'Equal-weight GROUP distances after gallery median calibration. Lower means closer. Not calibrated similarity percentage.',
          'evaluation_note':'No human-rated ground truth; cannot claim perceptual accuracy from these outputs.',
          'top5':output}
    (a.out/f'{prefix}_details.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8')
    print(f'Query: {source}; method: {a.method}; used: {", ".join(used)}')
    for x in output:
        print(f"  {x['rank']}. {x['image_id']}.png | D={x['distance_total']:.5f}")
    print(f'Saved: {csv_path}\nSaved: {img_path}\nSaved: {a.out / (prefix+"_details.json")}')


if __name__=='__main__':
    main()
