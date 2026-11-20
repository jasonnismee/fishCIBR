"""Blind human ratings for the candidate pool of ALL parameter variants.

The existing outputs/human_ratings.csv is appended/updated with the SAME schema
used by step4_visual_rating.py. No model/variant name is displayed to the rater.
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

from PIL import Image,ImageTk

FIELDS=['annotator','query_key','candidate_image_id','rating','timestamp_utc']


def load_csv(path):
    if not path.is_file():return []
    with path.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))


def save_rating(path,annotator,key,candidate,value):
    rows=[r for r in load_csv(path) if not (r.get('annotator')==annotator and
           r.get('query_key')==key and r.get('candidate_image_id')==candidate)]
    rows.append(dict(annotator=annotator,query_key=key,candidate_image_id=candidate,
                     rating=str(value),timestamp_utc=datetime.now(timezone.utc).isoformat(timespec='seconds')))
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    with tmp.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=FIELDS);w.writeheader();w.writerows(rows)
    tmp.replace(path)


def fit_image(raw,size=(560,280)):
    with Image.open(io.BytesIO(raw)) as base:
        image=base.convert('RGB')
    image.thumbnail(size,Image.Resampling.LANCZOS)
    bg=Image.new('RGB',size,'white')
    bg.paste(image,((size[0]-image.width)//2,(size[1]-image.height)//2))
    return bg


def main():
    p=argparse.ArgumentParser(description='Blind 0-3 similarity rating for parameter-study candidate pool')
    p.add_argument('--pool',type=Path,required=True)
    q=p.add_mutually_exclusive_group(required=True)
    q.add_argument('--query-id')
    q.add_argument('--query-image',type=Path)
    p.add_argument('--zip',type=Path,default=Path('../images.zip'))
    p.add_argument('--db',type=Path,default=Path('data/fish.db'))
    p.add_argument('--out',type=Path,default=Path('outputs/human_ratings.csv'))
    p.add_argument('--annotator',required=True)
    p.add_argument('--include-rated',action='store_true',help='Revisit previously judged candidates')
    p.add_argument('--dry-run',action='store_true')
    a=p.parse_args()
    if not a.pool.is_file():p.error(f'Pool missing: {a.pool}')
    key='gallery:'+a.query_id if a.query_id else 'external:'+hashlib.sha256(a.query_image.read_bytes()).hexdigest()
    rows=load_csv(a.pool)
    ids=list(dict.fromkeys(r.get('candidate_image_id','').strip() for r in rows if r.get('query_key')==key and r.get('candidate_image_id')))
    if not ids:p.error(f'No candidate rows for query {key} in {a.pool}')
    if a.query_id in ids:p.error('Query leaked into candidate pool')
    rng=random.Random(int(hashlib.sha256(key.encode()).hexdigest()[:12],16))
    rng.shuffle(ids)
    saved={r['candidate_image_id']:int(r['rating']) for r in load_csv(a.out)
           if r.get('annotator')==a.annotator and r.get('query_key')==key and r.get('rating') in ('0','1','2','3')}
    queue=ids if a.include_rated else [i for i in ids if i not in saved]
    print(f'Query: {key}; pool={len(ids)}, previously rated={len(set(ids)&set(saved))}, to review={len(queue)}')
    if not queue:
        print('All candidates were already rated. Use --include-rated to review them.')
        return
    if a.dry_run:
        print('Candidate IDs:',', '.join(queue));return
    with sqlite3.connect(f'file:{a.db.resolve()}?mode=ro',uri=True) as c:
        members=dict(c.execute('SELECT image_id,zip_member FROM images'))
    missing=set(ids)-set(members)
    if missing:p.error(f'Unknown image IDs: {sorted(missing)}')
    import tkinter as tk
    from tkinter import messagebox
    root=tk.Tk();root.title('FishCBIR — Blind human visual similarity ratings (parameter study)')
    root.geometry('1190x710');root.configure(bg='#f5f6f7')
    z=zipfile.ZipFile(a.zip)
    try:
        if a.query_id:
            query_raw=z.read(members[a.query_id])
        else:query_raw=a.query_image.read_bytes()
        qphoto=ImageTk.PhotoImage(fit_image(query_raw),master=root)
        cache={}
        title=tk.Label(root,text='ĐÁNH GIÁ GIỐNG NHAU VỀ THỊ GIÁC — KHÔNG XÉT LOÀI CÁ',
                       bg='#f5f6f7',font=('Segoe UI',16,'bold'))
        title.pack(pady=(15,4))
        tk.Label(root,text='0: khác rõ   •   1: giống vài nét   •   2: khá giống   •   3: rất giống tổng thể',
                 bg='#f5f6f7',font=('Segoe UI',11)).pack(pady=(0,15))
        pane=tk.Frame(root,bg='#f5f6f7');pane.pack()
        lc=tk.Frame(pane,bg='#f5f6f7');lc.pack(side='left',padx=8)
        rc=tk.Frame(pane,bg='#f5f6f7');rc.pack(side='left',padx=8)
        tk.Label(lc,text='ẢNH TRUY VẤN',bg='#f5f6f7',font=('Segoe UI',12,'bold')).pack()
        tk.Label(lc,image=qphoto,bg='white').pack()
        tk.Label(rc,text='ẢNH ỨNG VIÊN (ẩn thuật toán)',bg='#f5f6f7',font=('Segoe UI',12,'bold')).pack()
        candidate_photo_label=tk.Label(rc,bg='white');candidate_photo_label.pack()
        counter=tk.Label(root,bg='#f5f6f7',font=('Segoe UI',11));counter.pack(pady=10)
        idx=[0]
        value=tk.IntVar(value=-1)
        buttons=tk.Frame(root,bg='#f5f6f7');buttons.pack(pady=4)
        def show():
            id=queue[idx[0]]
            if id not in cache:
                cache[id]=ImageTk.PhotoImage(fit_image(z.read(members[id])),master=root)
            candidate_photo_label.configure(image=cache[id]);candidate_photo_label.image=cache[id]
            value.set(saved.get(id,-1))
            counter.configure(text=f'Ảnh {idx[0]+1}/{len(queue)}  |  Đã chấm trong pool: {len(set(ids)&set(saved))}/{len(ids)}')
        def save(n=None):
            v=value.get() if n is None else n
            if v not in (0,1,2,3):messagebox.showwarning('Chưa chấm','Chọn mức 0,1,2,3');return
            id=queue[idx[0]]
            try:save_rating(a.out,a.annotator,key,id,v)
            except Exception as exc:messagebox.showerror('Không lưu được',str(exc));return
            saved[id]=v
            if idx[0] < len(queue)-1:idx[0]+=1
            else:
                todo=[j for j,c in enumerate(queue) if c not in saved]
                if todo:idx[0]=todo[0]
                else:messagebox.showinfo('Hoàn thành','Đã chấm toàn bộ ảnh trong pool.');show();return
            show()
        def step(by):
            idx[0]=max(0,min(len(queue)-1,idx[0]+by));show()
        for n in range(4):
            tk.Button(buttons,text=f'{n} - Chấm',font=('Segoe UI',11),width=13,
                      command=lambda k=n:save(k)).pack(side='left',padx=8)
        nav=tk.Frame(root,bg='#f5f6f7');nav.pack(pady=12)
        tk.Button(nav,text='← Trước',font=('Segoe UI',10),width=16,command=lambda:step(-1)).pack(side='left',padx=8)
        tk.Button(nav,text='Tiếp (bỏ qua) →',font=('Segoe UI',10),width=17,command=lambda:step(1)).pack(side='left',padx=8)
        tk.Label(root,bg='#f5f6f7',fg='#777',text=f'Kết quả được lưu vào: {a.out} | Phím tắt 0,1,2,3, mũi tên trái/phải').pack(pady=6)
        root.bind('<KeyPress-0>',lambda _:save(0));root.bind('<KeyPress-1>',lambda _:save(1))
        root.bind('<KeyPress-2>',lambda _:save(2));root.bind('<KeyPress-3>',lambda _:save(3))
        root.bind('<Left>',lambda _:step(-1));root.bind('<Right>',lambda _:step(1))
        def close():z.close();root.destroy()
        root.protocol('WM_DELETE_WINDOW',close)
        show();root.mainloop()
    finally:
        z.close()

if __name__=='__main__':main()
