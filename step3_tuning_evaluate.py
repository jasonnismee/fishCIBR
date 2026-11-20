"""Human-judgment evaluation and optional Optuna tuning for FishCBIR variants.

Uses the original Step 4.2 / 4.3 CSV formats. All metrics are on the JUDGED
candidate pool, never on all unlabeled gallery items. Does NOT treat unjudged
images as irrelevant and does NOT optimize on held-out test queries.

Run from fish_step3 root after `step3_tuning.py extract` has finished:
  python step3_tuning_evaluate.py compare --ratings outputs/human_ratings.csv \
      --pairwise outputs/pairwise_preferences.csv
  python step3_tuning_evaluate.py optimize --ratings ... --pairwise ... --trials 100

If fewer than 15 queries exist, optimize is blocked unless --exploratory.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sqlite3
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from fishcbir.preprocessing import preprocess
from step3_tuning import (
    ALL, FAMILIES, gallery_rows, connect, load_variant_matrix, feature,
    calibration, row_distances, write_csv, DEFAULT_ORIG_DB, DEFAULT_OUT,
)


def csv_rows(path):
    if not path or not Path(path).exists():return []
    with Path(path).open('r',encoding='utf-8-sig',newline='') as f:
        return list(csv.DictReader(f))


def read_queries_manifest(path):
    paths={}
    for row in csv_rows(path):
        key=row.get('query_key','').strip();p=Path(row.get('image_path','').strip())
        if not key or not p.exists():
            raise ValueError(f'Invalid manifest query_key/image_path: {row}')
        if not key.startswith('external:'):
            raise ValueError(f'External image manifest only accepts external: keys, got {key}')
        if hashlib.sha256(p.read_bytes()).hexdigest()!=key.split(':',1)[1]:
            raise ValueError(f'External image changed: {p}; query hash does not match')
        paths[key]=p
    return paths


def load_judgments(ratings_path, pairwise_path):
    raw_ratings=defaultdict(lambda:defaultdict(dict))
    for r in csv_rows(ratings_path):
        q=r.get('query_key','').strip();id=r.get('candidate_image_id','').strip()
        if not q or not id:continue
        value=int(r['rating'])
        if value not in (0,1,2,3):raise ValueError(f'Bad rating 0-3: {r}')
        raw_ratings[q][id][r.get('annotator','unknown')]=value
    ratings={q:{id:float(np.mean(list(a.values()))) for id,a in vv.items()} for q,vv in raw_ratings.items()}
    pairs=defaultdict(list)
    for p in csv_rows(pairwise_path):
        q=p.get('query_key','').strip()
        if not q:continue
        choice=p.get('choice','').strip()
        if choice not in ('A','B','tie','neither'):raise ValueError(f'Bad pairwise choice: {choice}')
        a=p.get('left_image_id','').strip();b=p.get('right_image_id','').strip()
        if a==b or not a or not b:raise ValueError(f'Bad pairwise pair: {p}')
        pairs[q].append((a,b,choice))
    if not ratings and not pairs:
        raise ValueError('No usable judgments. Supply outputs/human_ratings.csv and/or outputs/pairwise_preferences.csv.')
    return ratings,pairs


def load_external_prepared(p):
    raw=p.read_bytes()
    bgr=cv2.imdecode(np.frombuffer(raw,np.uint8),cv2.IMREAD_COLOR)
    if bgr is None:raise ValueError(f'Cannot read external query image: {p}')
    h,w=bgr.shape[:2]
    if (w,h)!=(640,320):
        scale=min(640/w,320/h)
        nw,nh=max(1,round(w*scale)),max(1,round(h*scale))
        small=cv2.resize(bgr,(nw,nh),interpolation=cv2.INTER_AREA if scale<1 else cv2.INTER_LINEAR)
        canvas=np.full((320,640,3),255,np.uint8)
        x=(640-nw)//2;y=(320-nh)//2;canvas[y:y+nh,x:x+nw]=small;bgr=canvas
    return preprocess(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB))


def prepare_context(db, study_db, ratings, pairs, manifest, allow_partial=False):
    gallery=gallery_rows(db)
    ids=[id for id,member,sha in gallery]
    idx={id:i for i,id in enumerate(ids)}
    with connect(study_db) as c:
        mats={v.name: load_variant_matrix(c,v,gallery) for v in ALL.values()}
    cal={v.name:calibration(mats[v.name],v) for v in ALL.values()}
    keys=sorted(set(ratings)|set(pairs))
    context={}
    for key in keys:
        if key.startswith('gallery:'):
            qid=key.split(':',1)[1]
            if qid not in idx:raise ValueError(f'Unknown gallery query: {qid}')
            qindex=idx[qid]
            prepared=None
        elif key.startswith('external:'):
            if key not in manifest:
                raise ValueError(f'Missing query image for {key}; provide --manifest with query_key,image_path')
            prepared=load_external_prepared(manifest[key]);qindex=None
        else:raise ValueError(f'Unknown query key format: {key}')
        candidates=sorted(set(ratings.get(key,{}))|{a for a,b,_ in pairs.get(key,[])}|{b for a,b,_ in pairs.get(key,[])})
        if qindex is not None:candidates=[id for id in candidates if id!=qid]
        missing=set(candidates)-set(idx)
        if missing:raise ValueError(f'Unknown candidate IDs for {key}: {sorted(missing)}')
        if not candidates:continue
        indexes=np.array([idx[id] for id in candidates],dtype=int)
        dists={}
        for v in ALL.values():
            X=mats[v.name]
            q=X[qindex] if prepared is None else feature(prepared,v)
            sd,scale=cal[v.name]
            dists[v.name]=row_distances(q,X[indexes],v,sd)/scale
        context[key]=dict(ids=candidates,lookup={id:i for i,id in enumerate(candidates)},
                          ratings=ratings.get(key,{}),pairs=pairs.get(key,[]),distances=dists,
                          external=key.startswith('external:'))
    return context


def ndcg5(scores, ideal):
    def dcg(values):
        return sum((2**r-1)/math.log2(j+2) for j,r in enumerate(values[:5]))
    denominator=dcg(ideal)
    return dcg(scores)/denominator if denominator>0 else None


def query_metrics(item, dist):
    lookup=item['lookup'];relevance=item['ratings'];pairs=item['pairs'];ids=item['ids']
    # Among judged candidates only, not a full-gallery NDCG estimate.
    scored=[id for id in ids if id in relevance]
    ndcg=None; p5=None; mean5=None
    if len(scored)>=5:
        ranked=sorted(scored,key=lambda id:(float(dist[lookup[id]]),id))[:5]
        rel=[relevance[id] for id in ranked]
        ndcg=ndcg5(rel,sorted(relevance.values(),reverse=True)[:5])
        p5=float(np.mean([x>=2 for x in rel]))
        mean5=float(np.mean(rel))
    # Exclude 'neither' (both irrelevant); no information about relative preference.
    # Exclude 'tie' from a strict preference accuracy, but count them separately.
    hits=[]
    tie_count=0; neither_count=0
    for a,b,choice in pairs:
        if choice=='tie': tie_count+=1;continue
        if choice=='neither': neither_count+=1;continue
        if a not in lookup or b not in lookup:continue
        da=float(dist[lookup[a]]);db=float(dist[lookup[b]])
        if abs(da-db)<1e-10:hits.append(.5)
        elif choice=='A':hits.append(float(da<db))
        elif choice=='B':hits.append(float(db<da))
    return dict(ndcg=ndcg,p5=p5,mean5=mean5,
                pairwise=float(np.mean(hits)) if hits else None,
                decisive_pairs=len(hits),ties=tie_count,neither=neither_count,
                rated_candidates=len(scored))


def summary(metrics):
    def average(k):
        vals=[x[k] for x in metrics if x[k] is not None]
        return round(float(np.mean(vals)),6) if vals else None
    return dict(queries=len(metrics),queries_with_ndcg=sum(m['ndcg'] is not None for m in metrics),
                ndcg_at_5_pool=average('ndcg'),precision_at_5_pool=average('p5'),
                mean_human_at_5_pool=average('mean5'),pairwise_accuracy=average('pairwise'),
                decisive_pairs=sum(m['decisive_pairs'] for m in metrics),
                ties=sum(m['ties'] for m in metrics),neither=sum(m['neither'] for m in metrics))


def simple_distance(item, vname):return item['distances'][vname]


def score_blended(item, selected, weights):
    total=np.zeros(len(item['ids']),dtype=np.float64)
    for family,vname in selected.items():
        total+=float(weights[family])*item['distances'][vname]
    return total


def blended_fitness(context, query_keys, selected, weights):
    ms=[query_metrics(context[key],score_blended(context[key],selected,weights)) for key in query_keys]
    report=summary(ms)
    nd=report['ndcg_at_5_pool'];pa=report['pairwise_accuracy']
    if nd is None and pa is None:return -1.0,report
    if nd is None:return pa,report
    if pa is None:return nd,report
    return 0.4*nd+0.6*pa,report


def describe_context(context):
    qs=len(context)
    external=sum(item['external'] for item in context.values())
    judged=sum(len(item['ratings']) for item in context.values())
    pairs=sum(len(item['pairs']) for item in context.values())
    print(f'Queries: {qs} | external: {external} | graded candidate pairs: {judged} | pairwise votes: {pairs}')
    if qs < 15 or external < 10:
        print('WARNING: exploratory study only. Not enough held-out external queries to select a perceptually optimal configuration.')
    if qs==1:
        print('WARNING: Single-query 0334-style feedback is not a reliable model selection basis.')


def compare(a):
    ratings,pairs=load_judgments(a.ratings,a.pairwise)
    manifest=read_queries_manifest(a.manifest)
    ctx=prepare_context(a.db,a.out/'tuning_features.db',ratings,pairs,manifest)
    describe_context(ctx)
    rows=[]
    for v in ALL.values():
        report=summary([query_metrics(item,simple_distance(item,v.name)) for item in ctx.values()])
        rows.append(dict(variant=v.name,family=v.family,dimensions=v.dim,**report))
    # Scores should be interpreted ONLY within families (and judged pools), not as a proof
    # that raw GLCM is better than any multi-group combination on full external queries.
    rows.sort(key=lambda x:(x['family'],-(x['pairwise_accuracy'] if x['pairwise_accuracy'] is not None else -1),
                            -(x['ndcg_at_5_pool'] if x['ndcg_at_5_pool'] is not None else -1),x['dimensions']))
    output=a.out/'variant_human_comparison.csv';write_csv(output,rows)
    print('Variant performance on *judged* pools only:')
    for r in rows:
        pa=r['pairwise_accuracy'];nd=r['ndcg_at_5_pool']
        print(f"{r['variant']:<28} dim={r['dimensions']:<4} NDCG@5_pool={str(nd):>7} pairwise={str(pa):>7}")
    print('Saved:',output)
    print('No best configuration is declared until multiple independent external queries are judged.')


def split_queries(keys,seed):
    shuffled=list(sorted(keys))
    np.random.default_rng(seed).shuffle(shuffled)
    n=len(shuffled)
    n_test=max(1,round(n*.2));n_val=max(1,round(n*.2))
    if n-n_test-n_val<1:raise ValueError('Need >= 3 distinct queries for train/validation/test')
    return shuffled[n_test+n_val:],shuffled[n_test:n_test+n_val],shuffled[:n_test]


def optimize(a):
    try:import optuna
    except ImportError as exc:
        raise RuntimeError('Optuna missing. Install: python -m pip install optuna') from exc
    ratings,pairs=load_judgments(a.ratings,a.pairwise)
    manifest=read_queries_manifest(a.manifest)
    ctx=prepare_context(a.db,a.out/'tuning_features.db',ratings,pairs,manifest)
    describe_context(ctx)
    n=len(ctx);external=sum(x['external'] for x in ctx.values())
    decisive=sum(sum(c in ('A','B') for _,_,c in x['pairs']) for x in ctx.values())
    if not a.exploratory and (n<15 or external<10 or decisive<40):
        raise RuntimeError('Insufficient independent human judgments for trustworthy optimization. '
                           'Need >=15 queries, >=10 external queries and >=40 decisive A/B votes; '
                           'use --exploratory ONLY for debugging (not scientific conclusions).')
    train,val,test=split_queries(ctx.keys(),a.seed)
    # One categorical hyperparameter per feature group plus one nonnegative weight.
    options={f:[v.name for v in ALL.values() if v.family==f] for f in FAMILIES}
    weights_name={fam:f'weight_{fam}' for fam in FAMILIES}
    def pick(trial):
        chosen={f:trial.suggest_categorical(f'variant_{f}',options[f]) for f in FAMILIES}
        raw={f:trial.suggest_float(weights_name[f],0.0,1.0) for f in FAMILIES}
        denom=sum(raw.values())
        weights={f:(raw[f]/denom if denom>0 else 1.0/len(FAMILIES)) for f in FAMILIES}
        return chosen,weights
    def objective(trial):
        chosen,weights=pick(trial)
        fitness,_=blended_fitness(ctx,train,chosen,weights)
        return fitness
    sampler=optuna.samplers.TPESampler(seed=a.seed)
    study=optuna.create_study(direction='maximize',sampler=sampler)
    study.optimize(objective,n_trials=a.trials,show_progress_bar=False)
    # Validation is used only AFTER train optimization to choose among top trials;
    # never expose held-out test outcomes to Optuna.
    top=sorted(study.trials,key=lambda t:t.value if t.value is not None else -1,reverse=True)[:min(10,len(study.trials))]
    candidates=[]
    for t in top:
        selected={f:t.params[f'variant_{f}'] for f in FAMILIES}
        raw={f:t.params[weights_name[f]] for f in FAMILIES}
        denom=sum(raw.values());weights={f:v/denom for f,v in raw.items()}
        validation_score,vm=blended_fitness(ctx,val,selected,weights)
        candidates.append((validation_score,t.number,selected,weights,vm,t.value))
    candidates.sort(key=lambda r:(-r[0],r[1]))
    best_val,number,selected,weights,vm,train_score=candidates[0]
    test_score,tm=blended_fitness(ctx,test,selected,weights)
    report={
        'status':'EXPLORATORY_ONLY' if a.exploratory or n<30 else 'INDEPENDENT_TEST_ON_JUDGED_POOLS',
        'notes':['Pool-relative scores; not full-gallery recall or fully judged NDCG',
                 'Holdout test query IDs were not used to fit or select candidate configurations',
                 'Human judgments on full gallery and query diversity determine real validity'],
        'queries':{'total':n,'external':external,'decisive_votes':decisive,
                   'train':train,'validation':val,'test':test},
        'optimizer':'Optuna TPE, family-config categorical choices and continuous weights',
        'seed':a.seed,'trials':a.trials,'chosen_trial':number,
        'configs':selected,'weights':{k:round(v,7) for k,v in weights.items()},
        'train_fitness':train_score,'validation_fitness':best_val,'validation_metrics':vm,
        'test_fitness':test_score,'test_metrics':tm
    }
    a.out.mkdir(parents=True,exist_ok=True)
    path=a.out/'optuna_judged_pool_result.json'
    path.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2))
    print('Saved:',path)
    print('Do not call the selected model best overall without a larger held-out external-query test set.')



def create_manifest(a):
    paths=[]
    for path in sorted(a.folder.rglob('*')):
        if path.suffix.lower() not in ('.png','.jpg','.jpeg','.webp') or not path.is_file():continue
        digest=hashlib.sha256(path.read_bytes()).hexdigest()
        paths.append({'query_key':'external:'+digest,'image_path':str(path.resolve())})
    if not paths:raise ValueError(f'No query images in {a.folder}')
    write_csv(a.output,paths)
    print(f'Saved {len(paths)} external-query files to {a.output}')

def main():
    p=argparse.ArgumentParser(description='Human-based judged-pool evaluation of step3 parameter variants')
    sub=p.add_subparsers(dest='command',required=True)
    for action in ('compare','optimize','manifest'):
        s=sub.add_parser(action)
        if action=='manifest':
            s.add_argument('--folder',type=Path,required=True)
            s.add_argument('--output',type=Path,default=Path('outputs/external_queries.csv'))
            continue
        s.add_argument('--db',type=Path,default=DEFAULT_ORIG_DB)
        s.add_argument('--out',type=Path,default=DEFAULT_OUT)
        s.add_argument('--ratings',type=Path,default=Path('outputs/human_ratings.csv'))
        s.add_argument('--pairwise',type=Path,default=Path('outputs/pairwise_preferences.csv'))
        s.add_argument('--manifest',type=Path,default=Path('outputs/external_queries.csv'))
        if action=='optimize':
            s.add_argument('--trials',type=int,default=100)
            s.add_argument('--seed',type=int,default=2026)
            s.add_argument('--exploratory',action='store_true')
    a=p.parse_args()
    if a.command=='compare':compare(a)
    elif a.command=='manifest':create_manifest(a)
    else:optimize(a)

if __name__=='__main__':main()
