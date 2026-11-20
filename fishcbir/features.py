"""Eight interpretable fish image descriptors, aligned with report Part 2.

Dimensions (default): GLCM 40, Uniform LBP 59, Geometry 3, Contour 64,
Fourier 16, HSV 192, Color Moments 9, Spatial HSV 288 (total = 671).
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
import cv2
import numpy as np
from skimage.feature import local_binary_pattern
from .preprocessing import PreparedImage

@dataclass(frozen=True)
class FeatureParameters:
    gray_levels: int = 32
    glcm_distances: tuple[int, ...] = (1, 2)
    glcm_offsets_unit: tuple[tuple[int,int], ...] = ((0,1),(-1,1),(-1,0),(-1,-1))
    lbp_points: int = 8
    lbp_radius: int = 1
    contour_points: int = 64
    fourier_resample: int = 256
    fourier_coeffs: int = 16
    hsv_bins: tuple[int,int,int] = (12,4,4)
    spatial_bins: tuple[int,int,int] = (8,3,2)
    spatial_grid: tuple[int,int] = (2,3)  # rows, columns

    def dimensions(self):
        p = self.lbp_points
        # Number of non-rotation invariant uniform LBP labels: P(P-1)+3
        return {'glcm':len(self.glcm_distances)*len(self.glcm_offsets_unit)*5,
                'lbp':p*(p-1)+3,
                'geometry':3,
                'contour':self.contour_points,
                'fourier':self.fourier_coeffs,
                'hsv':int(np.prod(self.hsv_bins)),
                'moments':9,
                'spatial':int(np.prod(self.spatial_bins)*np.prod(self.spatial_grid))}

    def config_for(self,name:str)->dict:
        d=asdict(self)
        fields={'glcm':('gray_levels','glcm_distances','glcm_offsets_unit'),
                'lbp':('lbp_points','lbp_radius'),
                'geometry':(), 'contour':('contour_points',),
                'fourier':('fourier_resample','fourier_coeffs'),
                'hsv':('hsv_bins',),'moments':(),
                'spatial':('spatial_bins','spatial_grid')}
        return {k:d[k] for k in fields[name]}


def masked_glcm(prep:PreparedImage, params:FeatureParameters)->np.ndarray:
    q=(prep.gray.astype(np.int32)*params.gray_levels//256).astype(np.int16)
    h,w=q.shape
    mask=prep.mask
    L=params.gray_levels
    i_idx=np.arange(L,dtype=np.float64)[:,None]
    j_idx=np.arange(L,dtype=np.float64)[None,:]
    diff=i_idx-j_idx
    diff2=diff*diff
    inv_homo=1.0/(1.0+diff2)
    vals=[]
    for d in params.glcm_distances:
        for dr0,dc0 in params.glcm_offsets_unit:
            dr,dc=dr0*d,dc0*d
            r0,r1=max(0,-dr),min(h,h-dr)
            c0,c1=max(0,-dc),min(w,w-dc)
            valid=mask[r0:r1,c0:c1] & mask[r0+dr:r1+dr,c0+dc:c1+dc]
            if not np.any(valid): raise ValueError('No valid fish pixel pairs for GLCM')
            a=q[r0:r1,c0:c1][valid].astype(np.int32)
            b=q[r0+dr:r1+dr,c0+dc:c1+dc][valid].astype(np.int32)
            mat=np.bincount(a*L+b,minlength=L*L).reshape(L,L).astype(np.float64)
            mat/=mat.sum()
            asm=float(np.sum(mat**2))
            contrast=float(np.sum(mat*diff2))
            homogeneity=float(np.sum(mat*inv_homo))
            p_i=mat.sum(axis=1); p_j=mat.sum(axis=0)
            mi=float(np.dot(np.arange(L),p_i)); mj=float(np.dot(np.arange(L),p_j))
            si=np.sqrt(float(np.dot((np.arange(L)-mi)**2,p_i)))
            sj=np.sqrt(float(np.dot((np.arange(L)-mj)**2,p_j)))
            corr=float(np.sum(mat*(i_idx-mi)*(j_idx-mj))/(si*sj)) if si>1e-12 and sj>1e-12 else 1.0
            nonzero=mat[mat>0]
            entropy=float(-np.sum(nonzero*np.log2(nonzero)))
            vals.extend((asm,contrast,homogeneity,corr,entropy))
    return np.array(vals,dtype=np.float32)


def uniform_lbp(prep:PreparedImage, params:FeatureParameters)->np.ndarray:
    if params.lbp_points!=8:
        # Function supports other P, but labels are P*(P-1)+3; code paths tested for P=8.
        raise ValueError('Current validated implementation requires P=8')
    codes=local_binary_pattern(prep.gray,params.lbp_points,params.lbp_radius,method='nri_uniform')
    k=2*params.lbp_radius+1
    valid=cv2.erode(prep.mask.astype(np.uint8),np.ones((k,k),np.uint8))>0
    n=params.lbp_points*(params.lbp_points-1)+3
    hist=np.bincount(codes[valid].astype(np.int32),minlength=n).astype(np.float64)[:n]
    if hist.sum()==0: raise ValueError('No valid foreground positions for LBP')
    return (hist/hist.sum()).astype(np.float32)


def geometry(prep:PreparedImage)->np.ndarray:
    x0,y0,x1,y1=prep.bbox
    w,h=x1-x0,y1-y0
    convex=cv2.convexHull(prep.contour)
    hull_area=float(cv2.contourArea(convex))
    solidity=prep.area/hull_area if hull_area>0 else 0.0
    # Pixel mask area / geometric hull area can slightly exceed 1 because of discretization;
    # use a hull mask for a consistent pixel-domain solidity estimate.
    hull_mask=np.zeros(prep.mask.shape,np.uint8)
    cv2.drawContours(hull_mask,[convex],-1,1,-1)
    hull_px=int(hull_mask.sum())
    solidity=prep.area/hull_px if hull_px else 0.0
    return np.asarray([w/h,solidity,prep.area/(w*h)],dtype=np.float32)


def resample_contour(contour:np.ndarray,n:int)->np.ndarray:
    pts=contour.reshape(-1,2).astype(np.float64)
    if len(pts)<3: raise ValueError('Contour too short')
    # Use the upper-leftmost of all leftmost contour points as a stable origin.
    start=int(np.lexsort((pts[:,1],pts[:,0]))[0])
    pts=np.roll(pts,-start,axis=0)
    # A consistent clockwise orientation is important when comparing profiles.
    signed=np.sum(pts[:,0]*np.roll(pts[:,1],-1)-np.roll(pts[:,0],-1)*pts[:,1])
    if signed<0: pts=np.concatenate((pts[:1],pts[:0:-1]),axis=0)
    edges=np.linalg.norm(np.roll(pts,-1,axis=0)-pts,axis=1)
    cum=np.r_[0,np.cumsum(edges)]
    perimeter=cum[-1]
    if perimeter<=0: raise ValueError('Degenerate contour')
    at=np.linspace(0,perimeter,n,endpoint=False)
    xy=np.vstack((np.interp(at,cum,np.r_[pts[:,0],pts[0,0]]),
                  np.interp(at,cum,np.r_[pts[:,1],pts[0,1]]))).T
    return xy


def contour_profile(prep:PreparedImage,params:FeatureParameters)->np.ndarray:
    xy=resample_contour(prep.contour,params.contour_points)
    centroid=np.mean(xy,axis=0)
    x0,y0,x1,y1=prep.bbox
    scale=max(x1-x0,y1-y0)
    return (np.linalg.norm(xy-centroid,axis=1)/scale).astype(np.float32)


def fourier_descriptor(prep:PreparedImage,params:FeatureParameters)->np.ndarray:
    xy=resample_contour(prep.contour,params.fourier_resample)
    z=(xy[:,0]-xy[:,0].mean())+1j*(xy[:,1]-xy[:,1].mean())
    ff=np.fft.fft(z)
    # Absolute coefficients remove rotation and starting-point phase;
    # use F1 as in report when well-conditioned; else use dominant first harmonic.
    scale=max(abs(ff[1]),abs(ff[-1]),1e-12)
    if params.fourier_coeffs+1>=len(ff): raise ValueError('Too many Fourier coefficients')
    return (np.abs(ff[2:2+params.fourier_coeffs])/scale).astype(np.float32)


def color_bin_ids(hsv:np.ndarray,bins:tuple[int,int,int])->np.ndarray:
    h,s,v=(hsv[:,i].astype(np.int32) for i in range(3))
    nh,ns,nv=bins
    hb=np.minimum(h*nh//180,nh-1)
    sb=np.minimum(s*ns//256,ns-1)
    vb=np.minimum(v*nv//256,nv-1)
    return (hb*ns+sb)*nv+vb


def hsv_histogram(prep:PreparedImage,bins:tuple[int,int,int])->np.ndarray:
    n=int(np.prod(bins))
    idx=color_bin_ids(prep.hsv[prep.mask],bins)
    hist=np.bincount(idx,minlength=n).astype(np.float64)
    return (hist/hist.sum()).astype(np.float32)


def color_moments(prep:PreparedImage)->np.ndarray:
    rgb=prep.rgb[prep.mask].astype(np.float64)/255.
    mean=rgb.mean(axis=0)
    z=rgb-mean
    sigma=np.sqrt(np.mean(z*z,axis=0))
    # Signed cube-root of the third central moment (NOT standardized skewness).
    tau=np.cbrt(np.mean(z*z*z,axis=0))
    return np.r_[mean,sigma,tau].astype(np.float32)


def spatial_histogram(prep:PreparedImage,params:FeatureParameters)->np.ndarray:
    rows,cols=params.spatial_grid
    bins=params.spatial_bins
    nb=int(np.prod(bins))
    x0,y0,x1,y1=prep.bbox
    yy,xx=np.nonzero(prep.mask)
    region_r=np.minimum((yy-y0)*rows//(y1-y0),rows-1)
    region_c=np.minimum((xx-x0)*cols//(x1-x0),cols-1)
    regions=region_r*cols+region_c
    colors=color_bin_ids(prep.hsv[prep.mask],bins)
    counts=np.bincount(regions*nb+colors,minlength=rows*cols*nb).reshape(rows*cols,nb).astype(np.float64)
    totals=counts.sum(axis=1,keepdims=True)
    np.divide(counts,totals,out=counts,where=totals>0)
    return counts.reshape(-1).astype(np.float32)


def extract(prep:PreparedImage,params:FeatureParameters|None=None)->dict[str,np.ndarray]:
    p=params or FeatureParameters()
    feats={'glcm':masked_glcm(prep,p),
           'lbp':uniform_lbp(prep,p),
           'geometry':geometry(prep),
           'contour':contour_profile(prep,p),
           'fourier':fourier_descriptor(prep,p),
           'hsv':hsv_histogram(prep,p.hsv_bins),
           'moments':color_moments(prep),
           'spatial':spatial_histogram(prep,p)}
    for name,vector in feats.items():
        if vector.shape!=(p.dimensions()[name],):
            raise AssertionError(f'{name}: invalid length {len(vector)}')
        if not np.isfinite(vector).all():
            raise ValueError(f'{name}: NaN or infinity in descriptor')
    if not np.isclose(feats['lbp'].sum(),1,atol=1e-5): raise ValueError('LBP histogram sum invalid')
    if not np.isclose(feats['hsv'].sum(),1,atol=1e-5): raise ValueError('HSV histogram sum invalid')
    return feats
