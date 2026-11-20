import numpy as np
from fishcbir.features import FeatureParameters,resample_contour,color_bin_ids
from fishcbir.preprocessing import preprocess
from fishcbir.features import extract


def test_dimensions():
    p=FeatureParameters()
    assert sum(p.dimensions().values())==671
    assert p.dimensions()['lbp']==59


def test_on_synthetic_fish():
    import cv2
    img=np.full((320,640,3),255,np.uint8)
    cv2.ellipse(img,(320,160),(180,65),0,0,360,(155,60,35),-1)
    cv2.fillPoly(img,[np.array([(485,160),(590,100),(590,220)],np.int32)],(155,60,35))
    prep=preprocess(img)
    feats=extract(prep)
    assert sum(len(v) for v in feats.values())==671
    assert np.isclose(feats['lbp'].sum(),1,atol=1e-5)
    assert np.isclose(feats['hsv'].sum(),1,atol=1e-5)
    assert all(np.isfinite(v).all() for v in feats.values())
