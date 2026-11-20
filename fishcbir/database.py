"""SQLite storage: one feature vector per image and immutable configuration key."""
import hashlib,json,sqlite3
import numpy as np
from .features import FeatureParameters

VERSION='step3-v1'

def config_key(name:str,params:dict)->str:
    blob=json.dumps({'version':VERSION,'name':name,'params':params},sort_keys=True,separators=(',',':'))
    return hashlib.sha256(blob.encode('utf-8')).hexdigest()[:20]

def open_database(path:str):
    con=sqlite3.connect(str(path))
    con.execute('PRAGMA foreign_keys=ON')
    con.execute('PRAGMA journal_mode=WAL')
    con.executescript('''
    CREATE TABLE IF NOT EXISTS images (
      image_id TEXT PRIMARY KEY,
      zip_member TEXT NOT NULL,
      original_filename TEXT,
      sha256 TEXT NOT NULL UNIQUE,
      width INTEGER NOT NULL,
      height INTEGER NOT NULL,
      bbox_x0 INTEGER,
      bbox_y0 INTEGER,
      bbox_x1 INTEGER,
      bbox_y1 INTEGER,
      mask_area INTEGER,
      mask_ratio REAL,
      mask_status TEXT,
      created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
    );
    CREATE TABLE IF NOT EXISTS feature_configs (
      config_key TEXT PRIMARY KEY,
      family TEXT NOT NULL,
      dimensions INTEGER NOT NULL,
      parameters_json TEXT NOT NULL,
      version TEXT NOT NULL
    );
    CREATE TABLE IF NOT EXISTS image_features (
      image_id TEXT NOT NULL,
      config_key TEXT NOT NULL,
      dtype TEXT NOT NULL DEFAULT 'float32',
      dimensions INTEGER NOT NULL,
      data BLOB NOT NULL,
      extracted_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
      PRIMARY KEY (image_id, config_key),
      FOREIGN KEY(image_id) REFERENCES images(image_id),
      FOREIGN KEY(config_key) REFERENCES feature_configs(config_key)
    );
    CREATE TABLE IF NOT EXISTS extraction_logs (
      log_id INTEGER PRIMARY KEY AUTOINCREMENT,
      image_id TEXT NOT NULL,
      status TEXT NOT NULL,
      elapsed_seconds REAL,
      error_message TEXT,
      created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
    );
    CREATE INDEX IF NOT EXISTS idx_feature_config ON image_features(config_key);
    CREATE INDEX IF NOT EXISTS idx_logs_image ON extraction_logs(image_id);
    ''')
    return con

def register_configs(con,p:FeatureParameters):
    configs={}
    for name,dim in p.dimensions().items():
        prm=p.config_for(name)
        key=config_key(name,prm)
        con.execute('''INSERT OR IGNORE INTO feature_configs
          (config_key,family,dimensions,parameters_json,version)
          VALUES(?,?,?,?,?)''',(key,name,dim,json.dumps(prm,ensure_ascii=False,sort_keys=True),VERSION))
        configs[name]=key
    con.commit()
    return configs

def put_image(con,id,filename,alias,sha,w,h,prep):
    x0,y0,x1,y1=prep.bbox
    con.execute('''INSERT INTO images(image_id,zip_member,original_filename,sha256,width,height,
      bbox_x0,bbox_y0,bbox_x1,bbox_y1,mask_area,mask_ratio,mask_status)
      VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)
      ON CONFLICT(image_id) DO UPDATE SET
      zip_member=excluded.zip_member,original_filename=excluded.original_filename,
      sha256=excluded.sha256,width=excluded.width,height=excluded.height,
      bbox_x0=excluded.bbox_x0,bbox_y0=excluded.bbox_y0,
      bbox_x1=excluded.bbox_x1,bbox_y1=excluded.bbox_y1,
      mask_area=excluded.mask_area,mask_ratio=excluded.mask_ratio,mask_status=excluded.mask_status
    ''',(id,filename,alias,sha,w,h,x0,y0,x1,y1,prep.area,prep.mask_ratio,prep.status))

def put_features(con,id,features,configs):
    for name,v in features.items():
        v=np.asarray(v,dtype='<f4')
        con.execute('''INSERT OR REPLACE INTO image_features
          (image_id,config_key,dtype,dimensions,data) VALUES(?,?,?,?,?)''',
          (id,configs[name],'float32',len(v),sqlite3.Binary(v.tobytes())))

def load_feature_matrix(con,config_key):
    rows=con.execute('''SELECT image_id,dimensions,data FROM image_features
      WHERE config_key=? ORDER BY image_id''',(config_key,)).fetchall()
    if not rows: return [],np.zeros((0,0),dtype=np.float32)
    return [r[0] for r in rows],np.stack([np.frombuffer(r[2],dtype='<f4',count=r[1]).copy() for r in rows])
