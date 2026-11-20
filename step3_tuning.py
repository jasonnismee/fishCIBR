"""Step 3 parameter study for FishCBIR. Does not modify original fish.db.

Commands (run from fish_step3 root):
  python step3_tuning.py plan
  python step3_tuning.py extract --zip ../images.zip --limit 25
  python step3_tuning.py extract --zip ../images.zip
  python step3_tuning.py status
  python step3_tuning.py preview --zip ../images.zip --query-id 0334 --family hsv

All candidate vectors are in outputs/step3_tuning/tuning_features.db. This tool
extracts descriptors only; it does not claim perceptual optimality without labels.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sqlite3
import time
import zipfile
from collections import defaultdict
from dataclasses import dataclass, replace
from pathlib import Path

import cv2
import numpy as np

from fishcbir.database import config_key
from fishcbir.features import (
    FeatureParameters, masked_glcm, uniform_lbp, geometry, contour_profile,
    fourier_descriptor, hsv_histogram, color_moments, spatial_histogram,
)
from fishcbir.preprocessing import preprocess

FAMILIES = ('glcm', 'lbp', 'geometry', 'contour', 'fourier', 'hsv', 'moments', 'spatial')
BASE = FeatureParameters()
STUDY_VERSION = 'step3-tuning-v1'; PREPROCESS_VERSION = 'white-mask-v1:threshold250'
DEFAULT_ORIG_DB = Path('data/fish.db')
DEFAULT_OUT = Path('outputs/step3_tuning')

@dataclass(frozen=True)
class Variant:
    name: str
    family: str
    params: FeatureParameters
    @property
    def dim(self):
        return self.params.dimensions()[self.family]
    @property
    def is_default(self):
        return self.params.config_for(self.family) == BASE.config_for(self.family)
    @property
    def fingerprint(self):
        payload = {'version': STUDY_VERSION, 'preprocess': PREPROCESS_VERSION,
                   'family': self.family, 'parameters': self.params.config_for(self.family)}
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:24]
    @property
    def config_json(self):
        return json.dumps(self.params.config_for(self.family), sort_keys=True, ensure_ascii=False)


def variants():
    """Separate, intentionally small grids; no exponential full factorial extraction."""
    v = []
    def add(name, family, **kw):
        v.append(Variant(name, family, replace(BASE, **kw)))
    add('glcm_L16_d12', 'glcm', gray_levels=16)
    add('glcm_L32_d12', 'glcm')
    add('glcm_L64_d12', 'glcm', gray_levels=64)
    add('glcm_L32_d1', 'glcm', glcm_distances=(1,))
    add('glcm_L32_d123', 'glcm', glcm_distances=(1, 2, 3))
    for r in (1, 2, 3): add(f'lbp_P8_R{r}', 'lbp', lbp_radius=r)
    add('geometry_default', 'geometry')
    for n in (32, 64, 128): add(f'contour_N{n}', 'contour', contour_points=n)
    for k in (8, 16, 24, 32): add(f'fourier_K{k}', 'fourier', fourier_coeffs=k)
    for name, bins in (('hsv_8x4x4',(8,4,4)),('hsv_12x4x4',(12,4,4)),
                       ('hsv_16x4x4',(16,4,4)),('hsv_12x6x4',(12,6,4))):
        add(name, 'hsv', hsv_bins=bins)
    add('moments_default', 'moments')
    for name, grid, bins in (
            ('spatial_3x2_8x3x2',(2,3),(8,3,2)),
            ('spatial_3x2_8x4x2',(2,3),(8,4,2)),
            ('spatial_3x2_12x4x2',(2,3),(12,4,2)),
            ('spatial_4x2_8x3x2',(2,4),(8,3,2))):
        add(name, 'spatial', spatial_grid=grid, spatial_bins=bins)
    assert len(v) == len(set(a.name for a in v))
    return v

ALL = {v.name: v for v in variants()}
DEFAULTS = {v.family: v for v in ALL.values() if v.is_default}
assert len(DEFAULTS) == len(FAMILIES)


def feature(prep, v:Variant):
    f = v.family
    if f == 'glcm': value = masked_glcm(prep, v.params)
    elif f == 'lbp': value = uniform_lbp(prep, v.params)
    elif f == 'geometry': value = geometry(prep)
    elif f == 'contour': value = contour_profile(prep, v.params)
    elif f == 'fourier': value = fourier_descriptor(prep, v.params)
    elif f == 'hsv': value = hsv_histogram(prep, v.params.hsv_bins)
    elif f == 'moments': value = color_moments(prep)
    elif f == 'spatial': value = spatial_histogram(prep, v.params)
    else: raise ValueError(f)
    value = np.asarray(value, dtype='<f4')
    if value.shape != (v.dim,) or not np.isfinite(value).all():
        raise ValueError(f'Invalid vector for {v.name}: {value.shape}, expected {(v.dim,)}')
    if f in ('hsv','lbp') and not np.isclose(value.sum(), 1, atol=1e-5):
        raise ValueError(f'Histogram not normalized: {v.name}')
    if f == 'spatial':
        sums = value.reshape(-1, int(np.prod(v.params.spatial_bins))).sum(axis=1)
        if np.any((np.abs(sums-1)>1e-5)&(np.abs(sums)>1e-5)):
            raise ValueError(f'Spatial histogram region not normalized: {v.name}')
    return value


def connect(dbpath:Path):
    dbpath.parent.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(str(dbpath), timeout=60)
    c.execute('PRAGMA foreign_keys=ON')
    c.execute('PRAGMA journal_mode=WAL')
    c.executescript('''
    CREATE TABLE IF NOT EXISTS variants(
      variant_name TEXT PRIMARY KEY,
      family TEXT NOT NULL,
      fingerprint TEXT NOT NULL,
      dimension INTEGER NOT NULL,
      params_json TEXT NOT NULL,
      extraction_version TEXT NOT NULL,
      is_default INTEGER NOT NULL
    );
    CREATE TABLE IF NOT EXISTS vectors(
      variant_name TEXT NOT NULL,
      image_id TEXT NOT NULL,
      sha256 TEXT NOT NULL,
      dimension INTEGER NOT NULL,
      vector BLOB NOT NULL,
      PRIMARY KEY(variant_name,image_id),
      FOREIGN KEY(variant_name) REFERENCES variants(variant_name)
    );
    CREATE TABLE IF NOT EXISTS errors(
      variant_name TEXT,
      image_id TEXT,
      message TEXT,
      created_at TEXT DEFAULT CURRENT_TIMESTAMP,
      UNIQUE(variant_name,image_id)
    );
    CREATE INDEX IF NOT EXISTS idx_vectors_name ON vectors(variant_name);
    ''')
    for v in ALL.values():
        found = c.execute('SELECT fingerprint FROM variants WHERE variant_name=?',(v.name,)).fetchone()
        if found and found[0] != v.fingerprint:
            raise RuntimeError(f'Variant definition changed: {v.name}; do not reuse old vectors. New study DB required.')
        c.execute('INSERT OR IGNORE INTO variants VALUES (?,?,?,?,?,?,?)',
                  (v.name,v.family,v.fingerprint,v.dim,v.config_json,STUDY_VERSION,int(v.is_default)))
    c.commit()
    return c


def gallery_rows(original_db:Path):
    if not original_db.is_file(): raise FileNotFoundError(f'Original DB not found: {original_db}')
    with sqlite3.connect(f'file:{original_db.resolve()}?mode=ro', uri=True) as c:
        rows = c.execute('SELECT image_id, zip_member, sha256 FROM images ORDER BY image_id').fetchall()
    if not rows: raise ValueError('Original gallery DB has no images')
    return rows


def import_defaults(c, original_db:Path, ids):
    """Reuse verified original baseline vectors without recalculating."""
    n_added = 0
    with sqlite3.connect(f'file:{original_db.resolve()}?mode=ro',uri=True) as orig:
        for fam,v in DEFAULTS.items():
            key = config_key(fam, BASE.config_for(fam))
            rows = orig.execute('''SELECT f.image_id, i.sha256, f.dimensions, f.data
                FROM image_features f JOIN images i USING(image_id) WHERE f.config_key=?''',(key,)).fetchall()
            if len(rows) != len(ids):
                raise RuntimeError(f'Cannot reuse baseline {fam}: expected {len(ids)} entries, found {len(rows)}')
            for image_id,sha,dim,blob in rows:
                if ids[image_id] != sha or dim != v.dim or len(blob) != 4*dim:
                    raise RuntimeError(f'Baseline mismatch in {fam} {image_id}')
            before=c.total_changes
            c.executemany('INSERT OR IGNORE INTO vectors VALUES (?,?,?,?,?)',
                          ((v.name,image_id,sha,dim,blob) for image_id,sha,dim,blob in rows))
            n_added += c.total_changes-before
    c.commit()
    return n_added


def select_variants(args):
    if args.variants:
        names = [s.strip() for s in args.variants.split(',') if s.strip()]
        unknown = set(names)-set(ALL)
        if unknown: raise ValueError(f'Unknown variants: {sorted(unknown)}')
        return [ALL[name] for name in dict.fromkeys(names)]
    if args.families:
        fams = [s.strip() for s in args.families.split(',') if s.strip()]
        unknown = set(fams)-set(FAMILIES)
        if unknown: raise ValueError(f'Unknown descriptor families: {sorted(unknown)}')
        return [v for v in ALL.values() if v.family in fams]
    return list(ALL.values())


def run_extraction(a):
    t0=time.perf_counter()
    rows=gallery_rows(a.db)
    ids={id:sha for id,_,sha in rows}
    selected=select_variants(a)
    target=a.out / 'tuning_features.db'
    c=connect(target)
    n_imported=import_defaults(c,a.db,ids)
    remaining = [v for v in selected if not v.is_default]
    print(f'Original: {len(ids)} gallery images; reused baseline vector rows: {n_imported}; variants requested: {len(selected)}',flush=True)
    done_by_var={}
    for v in remaining:
        done_by_var[v.name] = {id for id,sha in c.execute('SELECT image_id,sha256 FROM vectors WHERE variant_name=?',(v.name,)) if ids.get(id)==sha}
    sample=rows[:a.limit] if a.limit else rows
    count=0; errors=[]; vectors_added=0
    with zipfile.ZipFile(a.zip) as z:
        for pos,(image_id,member,expected_sha) in enumerate(sample,1):
            needs=[v for v in remaining if image_id not in done_by_var[v.name]]
            if not needs: continue
            count+=1
            try:
                raw=z.read(member)
                sha=hashlib.sha256(raw).hexdigest()
                if sha != expected_sha: raise ValueError('ZIP file hash differs from existing fish.db')
                bgr=cv2.imdecode(np.frombuffer(raw,np.uint8),cv2.IMREAD_COLOR)
                if bgr is None: raise ValueError('Image decoding failed')
                prep=preprocess(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB))
            except Exception as exc:
                errors.append((image_id,'all',str(exc)))
                if a.fail_fast: raise
                continue
            for v in needs:
                try:
                    vec=feature(prep,v)
                    c.execute('INSERT OR REPLACE INTO vectors VALUES(?,?,?,?,?)',
                              (v.name,image_id,expected_sha,v.dim,sqlite3.Binary(vec.tobytes())))
                    vectors_added+=1
                    c.execute('DELETE FROM errors WHERE variant_name=? AND image_id=?',(v.name,image_id))
                except Exception as exc:
                    errors.append((image_id,v.name,str(exc)))
                    c.execute('INSERT OR REPLACE INTO errors(variant_name,image_id,message) VALUES(?,?,?)',
                              (v.name,image_id,str(exc)))
                    if a.fail_fast: raise
            if pos%25==0 or pos==len(sample):
                c.commit()
                print(f'Processed {pos}/{len(sample)}, new vectors {vectors_added}, errors {len(errors)}',flush=True)
    c.commit()
    reports=[]
    for v in selected:
        n=c.execute('SELECT COUNT(*) FROM vectors WHERE variant_name=?',(v.name,)).fetchone()[0]
        reports.append(dict(variant=v.name,family=v.family,dimension=v.dim,rows=n,total=len(rows),complete=(n==len(rows)),params=v.config_json))
    a.out.mkdir(parents=True,exist_ok=True)
    write_csv(a.out/'variant_status.csv', reports)
    summary={'original_images':len(rows),'processed_images_this_run':count,
             'new_vectors_this_run':vectors_added,'baseline_rows_imported':n_imported,
             'errors_current_run':errors[:100], 'elapsed_seconds':round(time.perf_counter()-t0,2),
             'variant_db':str(target),'note':'Descriptor extraction only: no human-relevance optimization performed.'}
    (a.out/'extraction_summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding='utf-8')
    c.close()
    print(json.dumps(summary,indent=2,ensure_ascii=False),flush=True)
    print(f'Saved {a.out / "variant_status.csv"}',flush=True)


def write_csv(path, rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    if not rows: return
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0].keys()))
        w.writeheader();w.writerows(rows)


def status(a):
    rows=gallery_rows(a.db); ids={id:sha for id,_,sha in rows}
    db=a.out/'tuning_features.db'
    if not db.is_file(): raise FileNotFoundError(f'Run extract first; no study DB: {db}')
    with connect(db) as c:
        all_rows=[]
        for v in ALL.values():
            recs=c.execute('SELECT image_id, sha256, dimension, length(vector) FROM vectors WHERE variant_name=?',(v.name,)).fetchall()
            valid=sum(ids.get(i)==sha and dim==v.dim and n==4*dim for i,sha,dim,n in recs)
            all_rows.append(dict(variant=v.name,family=v.family,dimension=v.dim,valid=valid,total=len(rows),ok=(valid==len(rows))))
    write_csv(a.out/'variant_status.csv',all_rows)
    for t in all_rows:print(f"{t['variant']:<28} dim={t['dimension']:<4} {t['valid']:>4}/{t['total']} {'OK' if t['ok'] else 'INCOMPLETE'}")
    print('Saved:', a.out/'variant_status.csv')


def load_variant_matrix(c, v:Variant, gallery):
    items=c.execute('SELECT image_id,sha256,dimension,vector FROM vectors WHERE variant_name=? ORDER BY image_id',(v.name,)).fetchall()
    if len(items)!=len(gallery):
        raise RuntimeError(f'Variant {v.name} not complete: {len(items)}/{len(gallery)}. Run extract without --limit.')
    arr=[]
    for (iid,member,sha),(id2,sha2,dim,blob) in zip(gallery,items):
        if iid!=id2 or sha!=sha2 or dim!=v.dim or len(blob)!=4*dim:
            raise RuntimeError(f'Variant {v.name} has inconsistent gallery or dimension at {iid}')
        x=np.frombuffer(blob,dtype='<f4').copy()
        if not np.isfinite(x).all(): raise ValueError(f'Nonfinite values in {v.name} {iid}')
        arr.append(x)
    return np.asarray(arr,dtype=np.float32)


def row_distances(q, gallery, v:Variant, sd):
    """The exact family-specific metrics used in step4_baseline.py, with variable spatial grid."""
    eps=1e-12
    x=np.asarray(gallery,dtype=np.float64)
    q=np.asarray(q,dtype=np.float64)
    if v.family=='lbp':
        return 0.5*np.sum((x-q)**2/(x+q+eps),axis=-1)
    if v.family=='hsv':
        return np.sqrt(np.sum((np.sqrt(np.maximum(x,0))-np.sqrt(np.maximum(q,0)))**2,axis=-1)/2)
    if v.family=='spatial':
        regions=int(np.prod(v.params.spatial_grid))
        xx=x.reshape(len(x),regions,-1)
        qq=q.reshape(1,regions,-1)
        return np.sqrt(np.sum((np.sqrt(np.maximum(xx,0))-np.sqrt(np.maximum(qq,0)))**2,axis=2)/2).mean(axis=1)
    return np.sqrt(np.mean(((x-q)/sd)**2,axis=-1))


def calibration(X, v, seed=2026, n_pairs=1200):
    """Identical 1200-pair gallery-only calibration protocol to step4 baseline."""
    sd=np.maximum(X.astype(np.float64).std(axis=0),1e-6)
    rng=np.random.default_rng(seed)
    n=len(X)
    ix=rng.integers(0,n,size=n_pairs)
    jx=rng.integers(0,n,size=n_pairs)
    good=ix!=jx; ix=ix[good];jx=jx[good]
    d=np.asarray([row_distances(X[i],X[j:j+1],v,sd)[0] for i,j in zip(ix,jx)])
    ok=d[np.isfinite(d)&(d>1e-9)]
    if not ok.size:raise RuntimeError(f'Degenerate distances for {v.name}')
    return sd,float(np.median(ok))


def preview(a):
    v=ALL.get(a.variant)
    if a.variant and v is None: raise ValueError(f'Unknown variant {a.variant}; run plan')
    if not v:
        candidates=[u for u in ALL.values() if u.family==a.family]
        if not candidates: raise ValueError('Choose --family or --variant')
        v=next((u for u in candidates if not u.is_default),candidates[0])
    gallery=gallery_rows(a.db); ids=[x[0] for x in gallery]
    c=connect(a.out/'tuning_features.db')
    X=load_variant_matrix(c,v,gallery)
    c.close()
    if a.query_id:
        if a.query_id not in ids:raise ValueError(f'Unknown gallery ID {a.query_id}')
        q=X[ids.index(a.query_id)]
    elif a.query_image:
        # Use the same white-padding resize policy as the step 4 baseline.
        bgr=cv2.imdecode(np.fromfile(str(a.query_image),dtype=np.uint8),cv2.IMREAD_COLOR)
        if bgr is None: raise ValueError(f'Cannot read {a.query_image}')
        h,w=bgr.shape[:2]
        if (w,h)!=(640,320):
            s=min(640/w,320/h);nw=max(1,round(w*s));nh=max(1,round(h*s))
            resized=cv2.resize(bgr,(nw,nh),interpolation=cv2.INTER_AREA if s<1 else cv2.INTER_LINEAR)
            canvas=np.full((320,640,3),255,np.uint8)
            canvas[(320-nh)//2:(320-nh)//2+nh,(640-nw)//2:(640-nw)//2+nw]=resized
            bgr=canvas
        q=feature(preprocess(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)),v)
    else:raise ValueError('Provide --query-id or --query-image')
    sd,scale=calibration(X,v)
    dist=row_distances(q,X,v,sd)/scale
    if a.query_id:dist[ids.index(a.query_id)]=np.inf
    elif a.query_image:
        digest=hashlib.sha256(a.query_image.read_bytes()).hexdigest()
        for j,(_,_,sha) in enumerate(gallery):
            if sha==digest:dist[j]=np.inf
    winners=np.argsort(dist,kind='stable')[:5]
    output=[{'rank':n+1,'image_id':ids[i],'distance':round(float(dist[i]),7), 'variant':v.name} for n,i in enumerate(winners)]
    out=a.out/'previews';out.mkdir(parents=True,exist_ok=True)
    stem=a.query_id or a.query_image.stem
    csvfile=out/f'{v.name}_{stem}_top5.csv';write_csv(csvfile,output)
    canvas=np.full((700,1920,3),245,np.uint8)
    with zipfile.ZipFile(a.zip) as z:
        member=dict((id,m) for id,m,_ in gallery)
        picks=[a.query_id]+[ids[i] for i in winners] if a.query_id else [None]+[ids[i] for i in winners]
        for k,id in enumerate(picks):
            raw=a.query_image.read_bytes() if id is None else z.read(member[id])
            im=cv2.imdecode(np.frombuffer(raw,np.uint8),cv2.IMREAD_COLOR)
            if im is None:raise ValueError(f'Cannot draw image {id}')
            tile=cv2.resize(im,(640,320))
            y,x=(k//3)*350,(k%3)*640
            canvas[y:y+320,x:x+640]=tile
            label=f'QUERY {stem}' if k==0 else f'#{k} {id} D={dist[winners[k-1]]:.4f}'
            cv2.putText(canvas,label,(x+12,y+340),cv2.FONT_HERSHEY_SIMPLEX,0.65,(10,10,10),2,cv2.LINE_AA)
    png=out/f'{v.name}_{stem}_top5.png'
    if not cv2.imwrite(str(png),canvas):raise IOError(f'Unable to save {png}')
    print(f'Query: {stem} | family: {v.family} | variant: {v.name} | dim={v.dim}')
    for rec in output:print(f"  {rec['rank']}. {rec['image_id']} | D={rec['distance']:.5f}")
    print('Saved:',png,'and',csvfile)
    print('EXPLORATORY ONLY: no human-perceptual quality conclusion from one query.')



def make_pool(a):
    """Create a blind candidate pool from ALL 25 variants, not just old baselines.

    This is for human rating, not a measure of retrieval accuracy. Round-robin
    selection reduces the chance that one family monopolizes the judged pool.
    """
    gallery=gallery_rows(a.db);ids=[id for id,_,_ in gallery]
    item_index={id:i for i,id in enumerate(ids)}
    with connect(a.out/'tuning_features.db') as c:
        mats={v.name:load_variant_matrix(c,v,gallery) for v in ALL.values()}
    if a.query_id:
        if a.query_id not in item_index:raise ValueError(f'Unknown image ID {a.query_id}')
        stem=a.query_id;key='gallery:'+stem;qi=item_index[stem]
        prep=None
    else:
        stem=a.query_image.stem
        key='external:'+hashlib.sha256(a.query_image.read_bytes()).hexdigest()
        bgr=cv2.imdecode(np.fromfile(str(a.query_image),dtype=np.uint8),cv2.IMREAD_COLOR)
        if bgr is None:raise ValueError('Cannot decode query image')
        h,w=bgr.shape[:2]
        if (w,h)!=(640,320):
            scale=min(640/w,320/h);nw=max(1,round(w*scale));nh=max(1,round(h*scale))
            resized=cv2.resize(bgr,(nw,nh),interpolation=cv2.INTER_AREA if scale<1 else cv2.INTER_LINEAR)
            canvas=np.full((320,640,3),255,np.uint8);x=(640-nw)//2;y=(320-nh)//2
            canvas[y:y+nh,x:x+nw]=resized;bgr=canvas
        prep=preprocess(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB));qi=None
    # Old ratings are preserved and must not be lost when expanding the pool.
    existing=set()
    if a.existing_ratings and a.existing_ratings.is_file():
        with a.existing_ratings.open(encoding='utf-8-sig',newline='') as f:
            for row in csv.DictReader(f):
                if row.get('query_key')==key and row.get('candidate_image_id') in item_index:
                    existing.add(row['candidate_image_id'])
    results={};origins=defaultdict(set)
    for v in ALL.values():
        X=mats[v.name]
        q=X[qi] if qi is not None else feature(prep,v)
        std,scale=calibration(X,v)
        d=row_distances(q,X,v,std)/scale
        if qi is not None:d[qi]=np.inf
        else:
            digest=hashlib.sha256(a.query_image.read_bytes()).hexdigest()
            for j,(_,_,sha) in enumerate(gallery):
                if digest==sha:d[j]=np.inf
        candidates=[ids[i] for i in np.argsort(d,kind='stable')[:a.top_per_variant] if np.isfinite(d[i])]
        results[v.name]=candidates
        for cand in candidates: origins[cand].add(v.name)
    chosen=list(sorted(existing));used=set(chosen)
    # round-robin across family and variant, then rank, for diversity
    by_family={f:[v for v in ALL.values() if v.family==f] for f in FAMILIES}
    for rank in range(a.top_per_variant):
        for fam in FAMILIES:
            for v in by_family[fam]:
                l=results[v.name]
                if rank>=len(l):continue
                id=l[rank]
                if id not in used:
                    chosen.append(id);used.add(id)
                if len(chosen)>=a.max_candidates:break
            if len(chosen)>=a.max_candidates:break
        if len(chosen)>=a.max_candidates:break
    if a.query_id and a.query_id in used:raise RuntimeError('Query leakage into candidate pool')
    rows=[dict(query_key=key,candidate_image_id=id,source_variants=';'.join(sorted(origins[id])),
               already_rated=int(id in existing)) for id in chosen]
    outdir=a.out/'pools';outdir.mkdir(parents=True,exist_ok=True)
    path=outdir/f'{stem}_candidates.csv';write_csv(path,rows)
    print(f'Query: {key} | candidates: {len(chosen)} | already rated: {sum(id in existing for id in chosen)}')
    print('Saved:',path)
    print('Blind ratings must hide source_variants from the human judge.')

def main():
    p=argparse.ArgumentParser(description='FishCBIR: safe per-family descriptor parameter study')
    sub=p.add_subparsers(dest='command',required=True)
    for name in ('plan','extract','status','preview','pool'):
        s=sub.add_parser(name)
        s.add_argument('--db',type=Path,default=DEFAULT_ORIG_DB)
        s.add_argument('--out',type=Path,default=DEFAULT_OUT)
        if name in ('extract','preview','pool'):s.add_argument('--zip',type=Path,default=Path('../images.zip'))
        if name=='extract':
            s.add_argument('--limit',type=int,default=0,help='Quick test on first N images, 0 = all')
            s.add_argument('--families',help='Comma-separated families, e.g. glcm,lbp')
            s.add_argument('--variants',help='Comma-separated variant names')
            s.add_argument('--fail-fast',action='store_true')
        if name=='pool':
            s.add_argument('--query-id')
            s.add_argument('--query-image',type=Path)
            s.add_argument('--top-per-variant',type=int,default=5)
            s.add_argument('--max-candidates',type=int,default=60)
            s.add_argument('--existing-ratings',type=Path,default=Path('outputs/human_ratings.csv'))
        if name=='preview':
            s.add_argument('--query-id',help='Gallery ID; test only, excludes itself')
            s.add_argument('--query-image',type=Path,help='External query, for a formal held-out test')
            s.add_argument('--family',choices=FAMILIES)
            s.add_argument('--variant',choices=tuple(ALL))
    a=p.parse_args()
    if a.command=='plan':
        print(f'{len(ALL)} candidate variants (including {len(DEFAULTS)} reused baseline variants):')
        for v in ALL.values():print(f'{v.name:<28} family={v.family:<9} dims={v.dim:<4} {"BASELINE" if v.is_default else ""} {v.config_json}')
        print('Original fish.db is NEVER modified; generated variants go to outputs/step3_tuning/tuning_features.db')
    elif a.command=='extract':run_extraction(a)
    elif a.command=='status':status(a)
    elif a.command=='pool':
        if bool(a.query_id)==bool(a.query_image):p.error('Choose exactly one query source')
        if a.top_per_variant<1 or a.max_candidates<5:p.error('Invalid pool size')
        make_pool(a)
    elif a.command=='preview':
        if not a.variant and not a.family:p.error('preview requires --family or --variant')
        if bool(a.query_id)==bool(a.query_image):p.error('Choose exactly one of --query-id or --query-image')
        preview(a)

if __name__=='__main__':main()
