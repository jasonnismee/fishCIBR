"""Compare Top-5 methods against a FULLY judged candidate pool for one query.

All candidate IDs appearing in method CSVs must have human judgments. Missing scores
cause a hard error (never silently treated as 0). The metrics are pool-relative and
for exploratory indexed queries are NOT external-query generalization estimates.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import math
from collections import defaultdict
from pathlib import Path

from step4_visual_rating import METHODS, load_candidates


def load_labels(path: Path, query_key: str):
    if not path.is_file():
        raise FileNotFoundError('Human rating CSV not found: ' + str(path))
    grouped = defaultdict(dict)
    with path.open(encoding='utf-8-sig', newline='') as f:
        for row in csv.DictReader(f):
            if row['query_key'] != query_key:
                continue
            val = int(row['rating'])
            if val not in (0, 1, 2, 3):
                raise ValueError('Invalid rating: ' + str(val))
            grouped[row['candidate_image_id']][row['annotator']] = val
    return {iid: sum(r.values()) / len(r) for iid, r in grouped.items()}


def ndcg5(rels, ideal_rels):
    def dcg(values):
        return sum((2 ** r - 1) / math.log2(i + 2) for i, r in enumerate(values[:5]))
    ideal = dcg(ideal_rels)
    return dcg(rels) / ideal if ideal > 0 else 1.0


def main():
    p = argparse.ArgumentParser(description='Per-query human-rated comparison across 5 retrieval methods')
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument('--query-id')
    g.add_argument('--query-image', type=Path)
    p.add_argument('--results', type=Path, default=Path('outputs/step4_baseline'))
    p.add_argument('--ratings', type=Path, default=Path('outputs/human_ratings.csv'))
    p.add_argument('--out', type=Path, default=Path('outputs/step4_baseline'))
    a = p.parse_args()
    stem = a.query_id or a.query_image.stem
    key = 'gallery:' + a.query_id if a.query_id else 'external:' + hashlib.sha256(a.query_image.read_bytes()).hexdigest()
    candidates, _, found, missing = load_candidates(a.results, stem)
    labels = load_labels(a.ratings, key)
    unscored = set(candidates) - set(labels)
    if unscored:
        p.error('Human ratings incomplete: ' + str(len(unscored)) + ' unscored candidates; ' + ', '.join(sorted(unscored)))
    ideal = sorted((labels[i] for i in candidates), reverse=True)
    result = []
    for method in found:
        path = a.results / f'{method}_{stem}_top5.csv'
        with path.open(encoding='utf-8-sig', newline='') as f:
            ranks = sorted(list(csv.DictReader(f)), key=lambda r: int(r['rank']))
        ids = [r['image_id'] for r in ranks[:5]]
        if len(ids) != 5:
            raise ValueError('Expected exactly five Top5 results from ' + method)
        scores = [labels[i] for i in ids]
        result.append({'method': method, 'ndcg_at_5_pool': round(ndcg5(scores, ideal), 5),
                       'precision_at_5_rel_ge_2': round(sum(v >= 2 for v in scores) / 5, 5),
                       'human_mean_at_5': round(sum(scores) / 5, 5),
                       'judged_pool_size': len(candidates)})
    result.sort(key=lambda x: (-x['ndcg_at_5_pool'], -x['human_mean_at_5'], x['method']))
    a.out.mkdir(parents=True, exist_ok=True)
    target = a.out / f'comparison_{stem}.csv'
    with target.open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(result[0]))
        w.writeheader()
        w.writerows(result)
    if a.query_id:
        print('PRACTICE QUERY from gallery; NOT official held-out evaluation')
    print('Metrics relative to the human-rated pooled candidates (NOT full-gallery recall).')
    for item in result:
        print(f"{item['method']:8s}  NDCG@5={item['ndcg_at_5_pool']:.3f}  P@5={item['precision_at_5_rel_ge_2']:.2f}  human_mean={item['human_mean_at_5']:.2f}")
    print('Saved:', target)


if __name__ == '__main__':
    main()
