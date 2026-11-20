r"""Step 4.2: RRF fusion across COLOR/SHAPE/TEXTURE rank lists.

Run from fish_step3:
  ..\.venv\Scripts\python.exe .\step4_rank_fusion.py --zip ..\images.zip --query-id 0334
  ..\.venv\Scripts\python.exe .\step4_rank_fusion.py --zip ..\images.zip --query-image ..\query_images\fish01.png

Higher RRF score = better. This is an UNTRAINED alternative to equal distance fusion;
no claim about human-rated relevance is made.
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

from step4_baseline import (
    FAMILIES, METHODS, calibration, distances, load_gallery, prepare_query_image
)


def rr_fuse(normalized: dict[str, np.ndarray], excluded_indices: set[int], k: int = 60):
    """RRF over exactly three *macro-family* ranked lists, each with equal influence."""
    macro = ('color', 'shape', 'texture')
    group_distances = {
        name: np.mean(np.stack([normalized[f] for f in METHODS[name]]), axis=0)
        for name in macro
    }
    n = len(next(iter(group_distances.values())))
    ranks = {}
    score = np.zeros(n, dtype=np.float64)
    for name in macro:
        d = group_distances[name].copy()
        if excluded_indices:
            d[list(excluded_indices)] = np.inf
        order = np.argsort(d, kind='stable')
        rank = np.empty(n, dtype=np.int64)
        rank[order] = np.arange(1, n + 1)
        ranks[name] = rank
        score += 1.0 / (k + rank)
    if excluded_indices:
        score[list(excluded_indices)] = -np.inf
    return score, ranks, group_distances


def create_rrf_collage(query_bgr, results, zip_file, members, output):
    panel_w, panel_h = 640, 350
    canvas = np.full((panel_h * 2, panel_w * 3, 3), 245, np.uint8)
    with zipfile.ZipFile(zip_file) as z:
        tiles = [('QUERY', query_bgr)]
        for result in results:
            raw = z.read(members[result['image_id']])
            bgr = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
            if bgr is None:
                raise ValueError('Cannot decode gallery image ' + result['image_id'])
            label = f"#{result['rank']}  {result['image_id']}  RRF={result['rrf_score']:.5f}"
            tiles.append((label, bgr))
    for idx, (label, bgr) in enumerate(tiles):
        r, c = divmod(idx, 3)
        y, x = r * panel_h, c * panel_w
        canvas[y:y + 320, x:x + 640] = cv2.resize(bgr, (640, 320))
        cv2.putText(canvas, label, (x + 14, y + 341), cv2.FONT_HERSHEY_SIMPLEX,
                    .67, (15, 15, 15), 2, cv2.LINE_AA)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output), canvas):
        raise OSError(f'Cannot write collage: {output}')


def main():
    p = argparse.ArgumentParser(description='Top-5 with Reciprocal Rank Fusion (RRF)')
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument('--query-id', help='Existing gallery image ID, only for exploratory smoke test')
    g.add_argument('--query-image', type=Path, help='External query image')
    p.add_argument('--db', type=Path, default=Path('data/fish.db'))
    p.add_argument('--zip', type=Path, default=Path('../images.zip'))
    p.add_argument('--out', type=Path, default=Path('outputs/step4_baseline'))
    p.add_argument('--rrf-k', type=int, default=60)
    a = p.parse_args()
    if not a.zip.is_file():
        p.error(f'Original images ZIP not found: {a.zip}')
    if a.rrf_k < 1:
        p.error('--rrf-k must be >= 1')
    ids, members, sha_to_ids, matrices = load_gallery(a.db)
    index = {iid: i for i, iid in enumerate(ids)}
    if a.query_id:
        if a.query_id not in index:
            p.error(f'Unknown image ID {a.query_id}')
        print('SMOKE TEST: indexed query; cannot count as external-image evaluation.')
        ix = index[a.query_id]
        query = {name: X[ix] for name, X in matrices.items()}
        with zipfile.ZipFile(a.zip) as z:
            query_raw = z.read(members[a.query_id])
        query_bgr = cv2.imdecode(np.frombuffer(query_raw, np.uint8), cv2.IMREAD_COLOR)
        excluded = {a.query_id}
        source = 'gallery:' + a.query_id
    else:
        if not a.query_image.is_file():
            p.error(f'External query image not found: {a.query_image}')
        query, query_bgr, sha = prepare_query_image(a.query_image)
        excluded = set(sha_to_ids.get(sha, set()))
        if excluded:
            print('NOTE: Exact duplicate found; excluding', sorted(excluded))
        source = str(a.query_image)
    stds, scales = calibration(matrices)
    raw = {name: distances(query[name], matrices[name], name, stds[name]) for name in FAMILIES}
    normalized = {name: raw[name] / scales[name] for name in FAMILIES}
    scores, ranks, macro_distance = rr_fuse(normalized, {index[i] for i in excluded}, a.rrf_k)
    top = np.argsort(-scores, kind='stable')[:5]
    rows = []
    for order, ix in enumerate(top, 1):
        if not np.isfinite(scores[ix]):
            raise RuntimeError('Fewer than five valid matches')
        row = {'rank': order, 'image_id': ids[ix], 'rrf_score': float(scores[ix])}
        for name in ('color', 'shape', 'texture'):
            row['rank_' + name] = int(ranks[name][ix])
            row['distance_' + name] = float(macro_distance[name][ix])
        rows.append(row)
    a.out.mkdir(parents=True, exist_ok=True)
    prefix = 'rrf_' + (a.query_id or a.query_image.stem)
    csv_out = a.out / (prefix + '_top5.csv')
    with csv_out.open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    image_out = a.out / (prefix + '_top5.png')
    create_rrf_collage(query_bgr, rows, a.zip, members, image_out)
    meta = {'query': source, 'method': 'rrf', 'rrf_k': a.rrf_k,
            'fusion_lists': ['color', 'shape', 'texture'],
            'note': 'HIGHER RRF is better; score is NOT perceptual similarity percentage.',
            'calibration_medians': scales, 'top5': rows,
            'evaluation_note': 'Human visual scores must be collected independently.'}
    (a.out / (prefix + '_details.json')).write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding='utf-8')
    print('Query:', source, 'method: RRF over 3 macro-families (color, shape, texture)')
    for row in rows:
        print(f"  {row['rank']}. {row['image_id']}.png | RRF={row['rrf_score']:.6f}")
    print('Saved:', csv_out)
    print('Saved:', image_out)
    print('Saved:', a.out / (prefix + '_details.json'))


if __name__ == '__main__':
    main()
