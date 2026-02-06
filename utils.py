
import os
import cv2
import numpy as np
from PIL import ImageFont


def load_font(size=20):
   
    candidates = [
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/segoeui.ttf",
        "C:/Windows/Fonts/tahoma.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for p in candidates:
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def cv2_imread_unicode(path, flags=cv2.IMREAD_COLOR):
    
    data = np.fromfile(path, dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, flags)


def cv2_imwrite_unicode(path, img, params=None):
   
    ext = os.path.splitext(path)[1]
    if not ext:
        ext = ".png"
    ok, buf = cv2.imencode(ext, img, [] if params is None else params)
    if not ok:
        return False
    buf.tofile(path)
    return True


def preprocess_face(frame_bgr, x, y, w, h):
    
    face = frame_bgr[y:y+h, x:x+w]
    gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (48, 48), interpolation=cv2.INTER_AREA)
    norm = (resized.astype(np.float32) / 255.0)[..., None]
    return norm, resized
