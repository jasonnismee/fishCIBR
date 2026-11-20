r"""Step 4.3: blind pairwise judgments for human-perceptual fish-image similarity.

Use from fish_step3 root (Windows PowerShell):
  ..\.venv\Scripts\python.exe .\step4_pairwise_rating.py --zip ..\images.zip --query-id 0334 --annotator judge01

The 0-3 ratings from step4_visual_rating.py are NOT changed. The new CSV is
an independent source of paired preferences for later Learning-to-Rank.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import random
import sqlite3
import zipfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

from PIL import Image, ImageTk

METHODS = ('equal', 'color', 'shape', 'texture', 'rrf')
CSV_FIELDS = ('annotator', 'query_key', 'left_image_id', 'right_image_id', 'choice', 'timestamp_utc')
PLAN_FIELDS = ('left_image_id', 'right_image_id')
VALID_CHOICES = {'A', 'B', 'tie', 'neither'}


def read_top5(results: Path, stem: str):
    candidates = set()
    sources = defaultdict(set)
    ranks = defaultdict(dict)
    found = []
    for method in METHODS:
        path = results / f'{method}_{stem}_top5.csv'
        if not path.is_file():
            continue
        with path.open('r', encoding='utf-8-sig', newline='') as f:
            reader = csv.DictReader(f)
            if not {'image_id', 'rank'}.issubset(set(reader.fieldnames or [])):
                raise ValueError(f'Missing image_id/rank columns: {path}')
            for row in reader:
                iid = row['image_id'].strip()
                if not iid:
                    continue
                candidates.add(iid)
                sources[iid].add(method)
                ranks[iid][method] = int(row['rank'])
        found.append(method)
    if not found:
        raise FileNotFoundError(f'No Top 5 CSV in {results} for query {stem}. Run step4_baseline and RRF first.')
    if len(candidates) < 2:
        raise ValueError('At least 2 distinct candidate images are required')
    return sorted(candidates), sources, ranks, found


def load_ratings(path: Path, annotator: str, query_key: str):
    if not path.is_file():
        return {}
    with path.open('r', encoding='utf-8-sig', newline='') as f:
        return {row['candidate_image_id']: int(row['rating'])
                for row in csv.DictReader(f)
                if row.get('annotator') == annotator and row.get('query_key') == query_key
                and row.get('rating') in {'0', '1', '2', '3'}}


def make_pairs(candidates, sources, ranks, ratings, n_pairs, query_key):
    """Coverage-conscious, reproducible paired sampling, without showing algorithm labels.

    Prioritize pairs that the same rater gave the same 0-3 score, to resolve
    the ties observed with query 0334. Also cover candidates across methods.
    """
    seed = int(hashlib.sha256((query_key + ':pairwise-v1').encode()).hexdigest()[:16], 16)
    rng = random.Random(seed)
    pool = list(combinations(candidates, 2))
    if not pool:
        return []
    goal = min(n_pairs, len(pool))
    count = Counter()
    selected = []
    while len(selected) < goal:
        best = None
        best_score = -1e100
        for a, b in pool:
            # Stronger selection of same-rated pairs when already rated 0-3;
            # particularly useful when most images are 0 or 1.
            if a in ratings and b in ratings:
                same = (ratings[a] == ratings[b])
                rating_score = 5.0 if same else -1.5
            else:
                rating_score = 0.0
            is_cross_method = len(sources[a] ^ sources[b]) > 0
            source_bonus = 1.8 if is_cross_method else 0.0
            top_bonus = (1.0 if min(ranks[a].values()) <= 2 or min(ranks[b].values()) <= 2 else 0.0)
            coverage_bonus = 3.0/(1.0+count[a]) + 3.0/(1.0+count[b])
            score = rating_score + source_bonus + top_bonus + coverage_bonus + rng.uniform(0, .00001)
            if score > best_score:
                best_score, best = score, (a, b)
        pool.remove(best)
        a, b = best
        if rng.random() < .5:
            a, b = b, a
        selected.append((a, b))
        count[a] += 1
        count[b] += 1
    return selected


def read_plan(path: Path):
    with path.open('r', encoding='utf-8-sig', newline='') as f:
        return [(row['left_image_id'], row['right_image_id']) for row in csv.DictReader(f)]


def write_plan(path: Path, pairs):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=PLAN_FIELDS)
        writer.writeheader()
        for a, b in pairs:
            writer.writerow({'left_image_id': a, 'right_image_id': b})


def load_votes(path: Path, annotator: str, query_key: str):
    if not path.is_file():
        return {}
    votes = {}
    with path.open('r', encoding='utf-8-sig', newline='') as f:
        for row in csv.DictReader(f):
            if row.get('annotator') == annotator and row.get('query_key') == query_key \
                    and row.get('choice') in VALID_CHOICES:
                votes[(row['left_image_id'], row['right_image_id'])] = row['choice']
    return votes


def save_vote(path: Path, annotator: str, query_key: str, left: str, right: str, choice: str):
    if choice not in VALID_CHOICES:
        raise ValueError(f'Unknown choice {choice}')
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    if path.is_file():
        with path.open('r', encoding='utf-8-sig', newline='') as f:
            rows = list(csv.DictReader(f))
    rows = [r for r in rows if not (r.get('annotator') == annotator and
            r.get('query_key') == query_key and r.get('left_image_id') == left and
            r.get('right_image_id') == right)]
    rows.append({'annotator': annotator, 'query_key': query_key,
                 'left_image_id': left, 'right_image_id': right,
                 'choice': choice, 'timestamp_utc': datetime.now(timezone.utc).isoformat(timespec='seconds')})
    temporary = path.with_name(path.name + '.tmp')
    try:
        with temporary.open('w', encoding='utf-8-sig', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def image_from_zip(z: zipfile.ZipFile, member: str, target):
    with Image.open(io.BytesIO(z.read(member))) as im:
        im = im.convert('RGB')
        im.thumbnail(target, Image.Resampling.LANCZOS)
        canvas = Image.new('RGB', target, 'white')
        canvas.paste(im, ((target[0]-im.width)//2, (target[1]-im.height)//2))
        return canvas


def main():
    p = argparse.ArgumentParser(description='Blind A/B comparisons for fish visual similarity')
    q = p.add_mutually_exclusive_group(required=True)
    q.add_argument('--query-id', help='Existing gallery image ID for exploratory judgment, e.g. 0334')
    q.add_argument('--query-image', type=Path, help='External query image, preferred for formal evaluation')
    p.add_argument('--zip', type=Path, default=Path('../images.zip'))
    p.add_argument('--db', type=Path, default=Path('data/fish.db'))
    p.add_argument('--results', type=Path, default=Path('outputs/step4_baseline'))
    p.add_argument('--ratings', type=Path, default=Path('outputs/human_ratings.csv'))
    p.add_argument('--out', type=Path, default=Path('outputs/pairwise_preferences.csv'))
    p.add_argument('--plan-dir', type=Path, default=Path('outputs/pairwise_plans'))
    p.add_argument('--annotator', required=True)
    p.add_argument('--pairs', type=int, default=25, help='Number of pairs to judge for this query, default 25')
    p.add_argument('--dry-run', action='store_true', help='Check and show sampling without opening GUI')
    a = p.parse_args()
    if a.pairs < 1:
        p.error('--pairs must be positive')
    if not a.annotator.strip():
        p.error('--annotator must not be empty')
    if a.query_image and not a.query_image.is_file():
        p.error(f'External query image does not exist: {a.query_image}')
    stem = a.query_id or a.query_image.stem
    query_key = f'gallery:{stem}' if a.query_id else 'external:' + hashlib.sha256(a.query_image.read_bytes()).hexdigest()
    candidates, sources, ranks, methods = read_top5(a.results, stem)
    ratings = load_ratings(a.ratings, a.annotator, query_key)
    plan_id = hashlib.sha256((query_key + ':' + a.annotator).encode()).hexdigest()[:16]
    plan_file = a.plan_dir / f'{stem}_{plan_id}_pairs.csv'
    new_pairs = make_pairs(candidates, sources, ranks, ratings, a.pairs, query_key)
    pairs = read_plan(plan_file) if plan_file.is_file() else new_pairs
    if len(pairs) != a.pairs and plan_file.is_file():
        print(f'NOTE: Existing pair plan contains {len(pairs)} pairs; --pairs only applies on first run.')
    print('Query:', query_key)
    print('Methods in candidate pool:', ', '.join(methods))
    print('Unique candidate images:', len(candidates), '| Previous 0-3 ratings:', len(ratings))
    print('Pairs to judge:', len(pairs))
    print('Existing completed pairs:', len(load_votes(a.out, a.annotator, query_key)))
    if a.query_id:
        print('PRACTICE ONLY: this query is in the gallery; do not use it as an external test.')
    if a.dry_run:
        print('Plan:', plan_file)
        for idx, (left, right) in enumerate(pairs, 1):
            print(f'  {idx:02d}. {left} vs {right}')
        return
    if not a.zip.is_file() or not a.db.is_file():
        p.error('Missing images.zip or fish.db')
    with sqlite3.connect(a.db) as db:
        members = dict(db.execute('SELECT image_id,zip_member FROM images'))
    unknown = (set(candidates) | ({a.query_id} if a.query_id else set())) - members.keys()
    if unknown:
        p.error('Unknown gallery IDs: ' + ','.join(sorted(unknown)))
    if not plan_file.is_file():
        write_plan(plan_file, pairs)
    votes = load_votes(a.out, a.annotator, query_key)
    if all(pair in votes for pair in pairs):
        print('All pairs already rated. You can re-open the window to review or change scores.')

    import tkinter as tk
    from tkinter import messagebox
    try:
        root = tk.Tk()
    except tk.TclError as exc:
        p.error('Cannot open Tkinter window: ' + str(exc))
    root.title('FishCBIR - Đánh giá thị giác theo cặp A/B')
    screen_w, screen_h = root.winfo_screenwidth(), root.winfo_screenheight()
    small = screen_h < 830
    target = (490, 245) if not small else (420, 210)
    root.geometry(f'{min(screen_w-30, 1160)}x{min(screen_h-60, 830 if not small else 730)}')
    root.configure(bg='#f4f5f6')
    archive = zipfile.ZipFile(a.zip)
    try:
        if a.query_id:
            query_img = image_from_zip(archive, members[a.query_id], target)
        else:
            with Image.open(a.query_image) as im:
                im = im.convert('RGB')
                im.thumbnail(target, Image.Resampling.LANCZOS)
                query_img = Image.new('RGB', target, 'white')
                query_img.paste(im, ((target[0]-im.width)//2, (target[1]-im.height)//2))
        query_photo = ImageTk.PhotoImage(query_img, master=root)
    except Exception:
        archive.close()
        root.destroy()
        raise
    images = {}
    idx = [next((i for i, pair in enumerate(pairs) if pair not in votes), 0)]
    tk.Label(root, text='Ảnh nào giống ẢNH TRUY VẤN hơn về thị giác?',
             font=('Segoe UI', 17, 'bold'), bg='#f4f5f6').pack(pady=(9, 2))
    tk.Label(root, text='Chỉ xét màu sắc, hình dạng, hoa văn; KHÔNG xét loài cá. Không có đáp án đúng bắt buộc.',
             font=('Segoe UI', 10), bg='#f4f5f6').pack(pady=2)
    q_frame = tk.Frame(root, bg='#f4f5f6'); q_frame.pack()
    tk.Label(q_frame, text='ẢNH TRUY VẤN', font=('Segoe UI', 11, 'bold'), bg='#f4f5f6').pack()
    tk.Label(q_frame, image=query_photo, bg='white').pack(padx=3, pady=(1, 4))
    choices = tk.Frame(root, bg='#f4f5f6'); choices.pack(fill='x', expand=False)
    leftbox = tk.Frame(choices, bg='#f4f5f6'); leftbox.pack(side='left', expand=True)
    rightbox = tk.Frame(choices, bg='#f4f5f6'); rightbox.pack(side='left', expand=True)
    tk.Label(leftbox, text='ẢNH A', font=('Segoe UI', 13, 'bold'), bg='#f4f5f6').pack()
    left_label = tk.Label(leftbox, bg='white'); left_label.pack()
    tk.Label(rightbox, text='ẢNH B', font=('Segoe UI', 13, 'bold'), bg='#f4f5f6').pack()
    right_label = tk.Label(rightbox, bg='white'); right_label.pack()
    status = tk.Label(root, font=('Segoe UI', 10), bg='#f4f5f6'); status.pack(pady=(6, 3))
    reminder = tk.Label(root, font=('Segoe UI', 10), bg='#f4f5f6', fg='#555'); reminder.pack()

    def display():
        a_id, b_id = pairs[idx[0]]
        for iid in (a_id, b_id):
            if iid not in images:
                images[iid] = ImageTk.PhotoImage(image_from_zip(archive, members[iid], target), master=root)
        left_label.configure(image=images[a_id]); left_label.image = images[a_id]
        right_label.configure(image=images[b_id]); right_label.image = images[b_id]
        status.configure(text=f'Cặp {idx[0]+1}/{len(pairs)}    |    Đã chấm: {sum(k in votes for k in pairs)}/{len(pairs)}')
        old = votes.get((a_id, b_id))
        display_old = {'A':'Ảnh A giống hơn', 'B':'Ảnh B giống hơn', 'tie':'Tương đương', 'neither':'Cả hai không đáng kể'}
        reminder.configure(text='Đã chọn: ' + display_old[old] if old else 'Phím tắt A / D / T / N. Có thể quay lại để sửa lựa chọn.')

    def save_and_next(value):
        left, right = pairs[idx[0]]
        try:
            save_vote(a.out, a.annotator, query_key, left, right, value)
        except OSError as exc:
            messagebox.showerror('Lỗi ghi dữ liệu', str(exc))
            return
        votes[(left, right)] = value
        if idx[0] < len(pairs) - 1:
            idx[0] += 1
        else:
            not_done = [i for i, pair in enumerate(pairs) if pair not in votes]
            if not_done:
                idx[0] = not_done[0]
            else:
                messagebox.showinfo('Hoàn thành', f'Đã lưu {len(pairs)} lựa chọn vào:\n{a.out}')
        display()

    buttons = tk.Frame(root, bg='#f4f5f6'); buttons.pack(pady=(7, 4))
    options = (
        ('A  —  A giống hơn', 'A', '#d4ece3'),
        ('D  —  B giống hơn', 'B', '#d4ece3'),
        ('T  —  Tương đương', 'tie', '#e7e9ed'),
        ('N  —  Cả hai không giống', 'neither', '#e7e9ed'),
    )
    for label, val, shade in options:
        tk.Button(buttons, text=label, font=('Segoe UI', 10), bg=shade, width=22,
                  command=lambda v=val: save_and_next(v)).pack(side='left', padx=5)
    nav = tk.Frame(root, bg='#f4f5f6'); nav.pack(pady=(1, 6))
    tk.Button(nav, text='← Trước', width=13, command=lambda: navigate(-1)).pack(side='left', padx=12)
    tk.Button(nav, text='Tiếp →', width=13, command=lambda: navigate(1)).pack(side='left', padx=12)
    tk.Label(root, text=f'Tự lưu vào: {a.out}', font=('Segoe UI', 9), bg='#f4f5f6', fg='#777').pack()

    def navigate(delta):
        idx[0] = max(0, min(len(pairs)-1, idx[0]+delta))
        display()

    for key, value in (('a', 'A'), ('d', 'B'), ('t', 'tie'), ('n', 'neither')):
        root.bind(f'<KeyPress-{key}>', lambda e, v=value: save_and_next(v))
        root.bind(f'<KeyPress-{key.upper()}>', lambda e, v=value: save_and_next(v))
    root.bind('<Left>', lambda _e: navigate(-1))
    root.bind('<Right>', lambda _e: navigate(1))

    def close():
        archive.close()
        root.destroy()
    root.protocol('WM_DELETE_WINDOW', close)
    display()
    root.mainloop()


if __name__ == '__main__':
    main()
