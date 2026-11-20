"""Step 4.2 human rating: judge pooled visual matches (0-3) without species labels.

No server or extra web package required; uses standard tkinter + Pillow.
Run after equal/color/shape/texture and step4_rank_fusion.py produced top5 CSV files.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import random
import sqlite3
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image, ImageTk

METHODS = ('equal', 'color', 'shape', 'texture', 'rrf')
CSV_FIELDS = ('annotator', 'query_key', 'candidate_image_id', 'rating', 'timestamp_utc')


def load_candidates(results_dir: Path, query_stem: str, methods=METHODS):
    """Union (deduplicate) candidate images from methods; do not expose their origins to raters."""
    pool = set()
    sources = {}
    found = []
    missing = []
    for method in methods:
        file = results_dir / f'{method}_{query_stem}_top5.csv'
        if not file.is_file():
            missing.append(method)
            continue
        found.append(method)
        with file.open(encoding='utf-8-sig', newline='') as f:
            for row in csv.DictReader(f):
                iid = row['image_id'].strip()
                if iid:
                    pool.add(iid)
                    sources.setdefault(iid, []).append(method)
    if not found:
        raise FileNotFoundError(f'No top5 CSV found for query {query_stem} in {results_dir}')
    if not pool:
        raise ValueError('No candidates found in Top5 CSV')
    ordered = sorted(pool)
    seed = int(hashlib.sha256(query_stem.encode()).hexdigest()[:8], 16)
    random.Random(seed).shuffle(ordered)
    return ordered, sources, found, missing


def load_previous(path: Path, annotator: str, query_key: str):
    if not path.is_file():
        return {}
    with path.open(encoding='utf-8-sig', newline='') as f:
        rows = csv.DictReader(f)
        return {r['candidate_image_id']: int(r['rating']) for r in rows
                if r['annotator'] == annotator and r['query_key'] == query_key
                and r.get('rating', '').strip() in ('0', '1', '2', '3')}


def save_rating(path: Path, annotator: str, query_key: str, candidate_id: str, rating: int):
    path.parent.mkdir(parents=True, exist_ok=True)
    all_rows = []
    if path.exists():
        with path.open(encoding='utf-8-sig', newline='') as f:
            all_rows = list(csv.DictReader(f))
    all_rows = [r for r in all_rows if not (r['annotator'] == annotator and
                r['query_key'] == query_key and r['candidate_image_id'] == candidate_id)]
    all_rows.append({'annotator': annotator, 'query_key': query_key,
                     'candidate_image_id': candidate_id, 'rating': int(rating),
                     'timestamp_utc': datetime.now(timezone.utc).isoformat(timespec='seconds')})
    tmp = path.with_name(path.name + '.tmp')
    try:
        with tmp.open('w', encoding='utf-8-sig', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
            writer.writeheader()
            writer.writerows(all_rows)
        tmp.replace(path)
    finally:
        if tmp.exists():
            tmp.unlink()


def img_from_zip(z, member: str, target=(600, 300)):
    raw = z.read(member)
    img = Image.open(io.BytesIO(raw)).convert('RGB')
    img.thumbnail(target, Image.Resampling.LANCZOS)
    canvas = Image.new('RGB', target, 'white')
    canvas.paste(img, ((target[0] - img.width) // 2, (target[1] - img.height) // 2))
    return canvas


def main():
    p = argparse.ArgumentParser(description='Collect blinded human visual ratings for retrieval')
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument('--query-id', help='Gallery ID for a practice session, e.g. 0334')
    g.add_argument('--query-image', type=Path, help='Path to EXTERNAL image; preferred for evaluation')
    p.add_argument('--zip', type=Path, default=Path('../images.zip'))
    p.add_argument('--db', type=Path, default=Path('data/fish.db'))
    p.add_argument('--results', type=Path, default=Path('outputs/step4_baseline'))
    p.add_argument('--out', type=Path, default=Path('outputs/human_ratings.csv'))
    p.add_argument('--annotator', required=True, help='A distinct ID for each person e.g. judge01')
    p.add_argument('--dry-run', action='store_true', help='Print candidate counts without GUI')
    a = p.parse_args()
    query_stem = a.query_id or a.query_image.stem
    query_key = 'gallery:' + query_stem if a.query_id else 'external:' + hashlib.sha256(a.query_image.read_bytes()).hexdigest()
    candidates, sources, found, missing = load_candidates(a.results, query_stem)
    print('Query:', query_key)
    print('Methods:', ', '.join(found))
    if missing:
        print('Missing optional Top5 files:', ', '.join(missing))
    print('Unique candidates:', len(candidates))
    if a.query_id:
        print('PRACTICE ONLY: query is an indexed image; do not report this as an external-image test.')
    if a.dry_run:
        print('Candidate IDs:', ', '.join(sorted(candidates)))
        return
    if not a.zip.is_file() or not a.db.is_file():
        p.error('Missing original image ZIP or database')
    if not a.annotator.strip():
        p.error('--annotator cannot be empty')
    with sqlite3.connect(a.db) as con:
        members = dict(con.execute('SELECT image_id,zip_member FROM images'))
    unknown = set(candidates) - set(members)
    if unknown:
        p.error('Unknown candidate image IDs: ' + ', '.join(sorted(unknown)))
    import tkinter as tk
    from tkinter import messagebox
    try:
        root = tk.Tk()
    except tk.TclError as exc:
        p.error('Cannot open tkinter window: ' + str(exc))
    root.title('Đánh giá tương đồng thị giác - FishCBIR')
    root.geometry('1260x790')
    root.configure(bg='#f4f5f6')
    z = zipfile.ZipFile(a.zip)
    try:
        if a.query_id:
            if a.query_id not in members:
                p.error(f'Unknown query image ID: {a.query_id}')
            query_img = img_from_zip(z, members[a.query_id])
        else:
            query_img = Image.open(a.query_image).convert('RGB')
            query_img.thumbnail((600, 300), Image.Resampling.LANCZOS)
            canvas = Image.new('RGB', (600, 300), 'white')
            canvas.paste(query_img, ((600-query_img.width)//2, (300-query_img.height)//2))
            query_img = canvas
        query_photo = ImageTk.PhotoImage(query_img, master=root)
    except Exception:
        z.close()
        root.destroy()
        raise
    rated = load_previous(a.out, a.annotator, query_key)
    index = [0]
    photo_cache = {}
    header = tk.Label(root, text='Đánh giá mức độ giống nhau về thị giác, KHÔNG xét loài cá',
                      font=('Segoe UI', 18, 'bold'), bg='#f4f5f6')
    header.pack(pady=(12, 4))
    note = '0: Khác rõ  •  1: Giống vài nét  •  2: Khá giống  •  3: Rất giống tổng thể'
    tk.Label(root, text=note, font=('Segoe UI', 11), bg='#f4f5f6').pack(pady=(0, 10))
    image_frame = tk.Frame(root, bg='#f4f5f6')
    image_frame.pack(padx=10, fill='x')
    left = tk.Frame(image_frame, bg='#f4f5f6')
    left.pack(side='left', expand=True)
    right = tk.Frame(image_frame, bg='#f4f5f6')
    right.pack(side='left', expand=True)
    tk.Label(left, text='ẢNH TRUY VẤN', font=('Segoe UI', 13, 'bold'), bg='#f4f5f6').pack()
    tk.Label(left, image=query_photo, bg='white').pack(padx=5, pady=5)
    tk.Label(right, text='ẢNH ỨNG VIÊN', font=('Segoe UI', 13, 'bold'), bg='#f4f5f6').pack()
    candidate_image_label = tk.Label(right, bg='white')
    candidate_image_label.pack(padx=5, pady=5)
    progress = tk.Label(root, font=('Segoe UI', 12), bg='#f4f5f6')
    progress.pack(pady=(15, 4))
    rating_var = tk.IntVar(root, value=-1)
    rating_frame = tk.Frame(root, bg='#f4f5f6')
    rating_frame.pack(pady=5)
    for value, label in enumerate(('0 - Khác rõ', '1 - Giống vài nét', '2 - Khá giống', '3 - Rất giống')):
        tk.Radiobutton(rating_frame, text=label, variable=rating_var, value=value,
                       font=('Segoe UI', 11), bg='#f4f5f6', indicatoron=True).pack(side='left', padx=12)

    def show():
        iid = candidates[index[0]]
        if iid not in photo_cache:
            photo_cache[iid] = ImageTk.PhotoImage(img_from_zip(z, members[iid]), master=root)
        candidate_image_label.configure(image=photo_cache[iid])
        candidate_image_label.image = photo_cache[iid]
        rating_var.set(rated.get(iid, -1))
        progress.configure(text=f'Ứng viên {index[0] + 1}/{len(candidates)}  |  Đã chấm: {len(rated)}/{len(candidates)}')
        if len(rated) == len(candidates):
            finish_msg.configure(text='Đã chấm đủ ứng viên của truy vấn này. Có thể đóng cửa sổ.', fg='#087e52')
        else:
            finish_msg.configure(text='Chọn điểm rồi bấm Lưu & tiếp. Phím tắt: 0 / 1 / 2 / 3.', fg='#555555')

    def go(delta):
        index[0] = max(0, min(len(candidates) - 1, index[0] + delta))
        show()

    def save_and_next(value=None):
        val = rating_var.get() if value is None else value
        if val not in (0, 1, 2, 3):
            messagebox.showwarning('Chưa chấm điểm', 'Hãy chọn mức 0, 1, 2 hoặc 3.')
            return
        iid = candidates[index[0]]
        try:
            save_rating(a.out, a.annotator, query_key, iid, val)
        except Exception as exc:
            messagebox.showerror('Không lưu được điểm', str(exc))
            return
        rated[iid] = val
        if index[0] < len(candidates) - 1:
            index[0] += 1
        else:
            not_rated = [i for i, c in enumerate(candidates) if c not in rated]
            if not_rated:
                index[0] = not_rated[0]
        show()

    actions = tk.Frame(root, bg='#f4f5f6')
    actions.pack(pady=10)
    tk.Button(actions, text='← Trước', font=('Segoe UI', 11), command=lambda: go(-1), width=16).pack(side='left', padx=10)
    tk.Button(actions, text='Lưu điểm & tiếp →', font=('Segoe UI', 11, 'bold'),
              command=save_and_next, width=21, bg='#dcf5e9').pack(side='left', padx=10)
    tk.Button(actions, text='Tiếp (không chấm) →', font=('Segoe UI', 11),
              command=lambda: go(1), width=22).pack(side='left', padx=10)
    finish_msg = tk.Label(root, font=('Segoe UI', 11), bg='#f4f5f6')
    finish_msg.pack(pady=6)
    tk.Label(root, text=f'Kết quả chấm: {a.out}  |  Người chấm: {a.annotator}',
             font=('Segoe UI', 9), bg='#f4f5f6', fg='#777777').pack(pady=6)
    root.bind('<KeyPress-0>', lambda _e: save_and_next(0))
    root.bind('<KeyPress-1>', lambda _e: save_and_next(1))
    root.bind('<KeyPress-2>', lambda _e: save_and_next(2))
    root.bind('<KeyPress-3>', lambda _e: save_and_next(3))
    root.bind('<Left>', lambda _e: go(-1))
    root.bind('<Right>', lambda _e: go(1))

    def close():
        z.close()
        root.destroy()

    root.protocol('WM_DELETE_WINDOW', close)
    show()
    root.mainloop()


if __name__ == '__main__':
    main()
