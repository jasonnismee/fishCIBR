"""White-background fish segmentation and contour extraction.

Foreground is extracted from non-nearly-white pixels; largest component is filled
so bright fish body regions are not mistaken for background. All features consume
exactly the same foreground mask. Errors are reported, not silently hidden.
"""
from dataclasses import dataclass
import cv2
import numpy as np

@dataclass
class PreparedImage:
    rgb: np.ndarray
    gray: np.ndarray
    hsv: np.ndarray
    mask: np.ndarray
    contour: np.ndarray
    bbox: tuple[int, int, int, int]
    area: int
    mask_ratio: float
    status: str


def preprocess(rgb: np.ndarray, whiteness_threshold: int = 250) -> PreparedImage:
    if rgb is None or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError('Expected an H×W×3 RGB image')
    if rgb.dtype != np.uint8:
        raise ValueError('Expected uint8 RGB image')

    h, w = rgb.shape[:2]
    raw = np.any(rgb < whiteness_threshold, axis=2).astype(np.uint8) * 255
    raw = cv2.morphologyEx(raw, cv2.MORPH_CLOSE,
                           cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    contours, _ = cv2.findContours(raw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        raise ValueError('Fish silhouette not found; all pixels are near-white')
    contour = max(contours, key=cv2.contourArea)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)
    # Thin detached tips may not belong to main contour; do not artificially attach them.
    ys, xs = np.nonzero(mask)
    if len(xs) < 20:
        raise ValueError('Fish mask too small')
    x0, x1 = int(xs.min()), int(xs.max()+1)
    y0, y1 = int(ys.min()), int(ys.max()+1)
    area = int(len(xs))
    ratio = area/(h*w)
    flags=[]
    if ratio < .04 or ratio > .8: flags.append('unusual_area')
    if x0 == 0 or y0 == 0 or x1 == w or y1 == h: flags.append('touches_border')
    # Warn when raw foreground is poorly retained by the main contour.
    raw_n=int((raw>0).sum())
    if raw_n and int(np.logical_and(raw>0,mask>0).sum())/raw_n < .9:
        flags.append('disconnected_foreground')
    gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)
    hsv=cv2.cvtColor(rgb,cv2.COLOR_RGB2HSV)
    return PreparedImage(rgb,gray,hsv,mask>0,contour,(x0,y0,x1,y1),area,ratio,
                         'ok' if not flags else ','.join(flags))
