"""CLI: python -m fishcbir.cli extract --zip images.zip --db fish.db --out outputs"""
import argparse,csv,hashlib,io,json,re,time,zipfile,traceback
from collections import Counter
from pathlib import Path
import cv2
import numpy as np
from .preprocessing import preprocess
from .features import extract,FeatureParameters
from .database import open_database,register_configs,put_image,put_features,load_feature_matrix


def entries(z:zipfile.ZipFile):
    numeric={Path(n).stem:n for n in z.namelist() if re.fullmatch(r'\d{4}',Path(n).stem) and n.lower().endswith('.png')}
    if not numeric: raise RuntimeError('No 4-digit PNG images found in the ZIP')
    return dict(sorted(numeric.items()))

def aliases(z:zipfile.ZipFile, numeric:dict):
    # Pair the second copy (species_* names) by exact content hash; never use species as a relevance label.
    named=[n for n in z.namelist() if n.endswith('.png') and n not in numeric.values()]
    byhash={hashlib.sha256(z.read(n)).digest():n for n in named}
    return {id:byhash.get(hashlib.sha256(z.read(n)).digest()) for id,n in numeric.items()}


def draw_check(rgb,prep,feature_vectors,id,outdir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    outdir=Path(outdir); outdir.mkdir(parents=True,exist_ok=True)
    fig,ax=plt.subplots(2,3,figsize=(12,6))
    ax[0,0].imshow(rgb);ax[0,0].set_title(f'Original {id}')
    ax[0,1].imshow(prep.mask,cmap='gray');ax[0,1].set_title('Foreground binary mask')
    overlay=rgb.copy();cv2.drawContours(overlay,[prep.contour],-1,(245,10,10),2)
    x0,y0,x1,y1=prep.bbox;cv2.rectangle(overlay,(x0,y0),(x1-1,y1-1),(20,25,220),2)
    ax[0,2].imshow(overlay);ax[0,2].set_title('Contour and bounding box')
    ax[1,0].plot(feature_vectors['lbp']);ax[1,0].set_title('Uniform LBP 59-bin histogram')
    ax[1,1].plot(feature_vectors['hsv']);ax[1,1].set_title('Global HSV color histogram')
    ax[1,2].plot(feature_vectors['contour']);ax[1,2].set_title('Normalized contour profile')
    for a in ax[0]: a.axis('off')
    fig.tight_layout();fig.savefig(outdir/f'features_{id}.png',dpi=130);plt.close(fig)


def audit(zippath):
    with zipfile.ZipFile(zippath) as z:
        fs=[n for n in z.namelist() if n.lower().endswith('.png')]
        num=entries(z)
        names=aliases(z,num)
        resolutions=Counter(); modes=Counter(); channels=Counter()
        for id,n in num.items():
            buf=np.frombuffer(z.read(n),np.uint8);img=cv2.imdecode(buf,cv2.IMREAD_UNCHANGED)
            if img is None: raise ValueError(f'Unreadable {n}')
            resolutions[str((img.shape[1],img.shape[0]))]+=1
            channels[img.shape[2] if img.ndim==3 else 1]+=1
        summary={'zip_entries':len(z.namelist()),'png_files':len(fs),
                 'numbered_unique_images':len(num),
                 'exact_duplicate_pairs_with_original_names':sum(x is not None for x in names.values()),
                 'numeric_first':next(iter(num)), 'numeric_last':next(reversed(num)),
                 'resolutions':dict(resolutions),'channels':dict(channels)}
        return summary


def process(zippath,dbpath,outdir,limit=None,sample_ids=('0000','0032','0187','0334','0917')):
    """Resume-safe extraction. Export matrices from DB, not only current batch."""
    outdir=Path(outdir);outdir.mkdir(parents=True,exist_ok=True)
    p=FeatureParameters()
    con=open_database(dbpath)
    config_keys=register_configs(con,p)
    failures=[]
    with zipfile.ZipFile(zippath) as z:
        num=entries(z)
        aliases_=aliases(z,num)
        items=list(num.items())[:limit] if limit else list(num.items())
        placeholders=','.join('?'*len(config_keys))
        sql=('SELECT i.image_id FROM images i JOIN image_features f ON f.image_id=i.image_id '
             'WHERE f.config_key IN ('+placeholders+') GROUP BY i.image_id '
             'HAVING COUNT(DISTINCT f.config_key)=?')
        done=set(r[0] for r in con.execute(sql,[*config_keys.values(),len(config_keys)]).fetchall())
        print(f'Resuming: {len(done)} images already extracted; {len(items)} in input',flush=True)
        for idx,(image_id,member) in enumerate(items,1):
            if image_id in done: continue
            start=time.perf_counter()
            try:
                data=z.read(member)
                bgr=cv2.imdecode(np.frombuffer(data,np.uint8),cv2.IMREAD_COLOR)
                if bgr is None: raise ValueError(f'Failed to decode image {member}')
                rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
                prep=preprocess(rgb)
                feat=extract(prep,p)
                put_image(con,image_id,member,aliases_.get(image_id),hashlib.sha256(data).hexdigest(),rgb.shape[1],rgb.shape[0],prep)
                put_features(con,image_id,feat,config_keys)
                if image_id in sample_ids:
                    draw_check(rgb,prep,feat,image_id,outdir/'figures')
                dur=time.perf_counter()-start
                con.execute('INSERT INTO extraction_logs(image_id,status,elapsed_seconds) VALUES(?,?,?)',(image_id,'ok',dur))
            except Exception as e:
                failures.append((image_id,str(e)))
                con.execute('INSERT INTO extraction_logs(image_id,status,error_message) VALUES(?,?,?)',
                            (image_id,'error',str(e)))
            if idx%25==0 or idx==len(items):
                con.commit()
                count=con.execute('SELECT COUNT(*) FROM images').fetchone()[0]
                print(f'Input {idx}/{len(items)}: DB images={count}, new failures={len(failures)}',flush=True)
    con.commit()
    rows=con.execute('SELECT image_id,original_filename,mask_area,mask_ratio,mask_status FROM images ORDER BY image_id').fetchall()
    all_ids=[r[0] for r in rows]
    config_matrices={}
    for name,key in config_keys.items():
        group_ids,matrix=load_feature_matrix(con,key)
        if group_ids!=all_ids:
            raise RuntimeError(f'Incomplete feature group {name}: {len(group_ids)} vectors for {len(all_ids)} images')
        config_matrices[name]=matrix
    np.savez_compressed(outdir/'feature_matrices.npz',ids=np.array(all_ids,dtype='U4'),**config_matrices)
    g=config_matrices['geometry']
    latest_times={}
    for iid,t in con.execute("SELECT image_id, elapsed_seconds FROM extraction_logs WHERE status='ok' ORDER BY log_id"):
        latest_times[iid]=t
    qc=[]
    for i,(image_id,alias,area,ratio,status) in enumerate(rows):
        qc.append({'image_id':image_id,'alias_filename':alias,'mask_area':area,
                   'mask_ratio':round(ratio,6),'mask_status':status,
                   'aspect_ratio':round(float(g[i,0]),6),
                   'solidity':round(float(g[i,1]),6),
                   'extent':round(float(g[i,2]),6),
                   'elapsed_seconds':round(latest_times[image_id],4) if image_id in latest_times else None})
    if qc:
        with (outdir/'image_qc.csv').open('w',newline='',encoding='utf-8-sig') as f:
            writer=csv.DictWriter(f,fieldnames=list(qc[0]));writer.writeheader();writer.writerows(qc)
    with (outdir/'feature_samples.csv').open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.writer(f)
        writer.writerow(['image_id']+[f'{name}_{j:03d}' for name,dim in p.dimensions().items() for j in range(dim)])
        for sample in sample_ids:
            if sample not in all_ids: continue
            i=all_ids.index(sample)
            writer.writerow([sample]+[round(float(x),7) for name in p.dimensions() for x in config_matrices[name][i]])
    runtimes=list(latest_times.values())
    summary={'total_images_in_zip':len(num),
        'success_count':len(all_ids),'failure_count_current_run':len(failures),
        'failures_current_run':failures,'dimensions':p.dimensions(),
        'total_dimensions':sum(p.dimensions().values()),
        'mask_status_counts':dict(Counter(a['mask_status'] for a in qc)),
        'mask_area_fraction_min':round(min((a['mask_ratio'] for a in qc),default=0),6),
        'mask_area_fraction_max':round(max((a['mask_ratio'] for a in qc),default=0),6),
        'per_image_seconds_median':round(float(np.median(runtimes)),4) if runtimes else None,
        'elapsed_seconds_sum':round(float(np.sum(runtimes)),2),
        'configs':config_keys,'code_version':'step3-v1',
        'note':'Descriptor extraction validation only; no human-perceptual retrieval evaluation.'}
    (outdir/'run_summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding='utf-8')
    con.close()
    print(json.dumps(summary,ensure_ascii=False,indent=2))
    return summary

def main():
    a=argparse.ArgumentParser(description='Fish CBIR step 3: descriptor extraction and SQLite DB')
    s=a.add_subparsers(dest='action',required=True)
    aud=s.add_parser('audit');aud.add_argument('--zip',required=True)
    ex=s.add_parser('extract');ex.add_argument('--zip',required=True)
    ex.add_argument('--db',default='data/fish.db');ex.add_argument('--out',default='outputs')
    ex.add_argument('--limit',type=int,default=None)
    args=a.parse_args()
    if args.action=='audit': print(json.dumps(audit(args.zip),indent=2,ensure_ascii=False))
    elif args.action=='extract':
        Path(args.db).parent.mkdir(parents=True,exist_ok=True)
        process(args.zip,args.db,args.out,args.limit)

if __name__=='__main__':main()
