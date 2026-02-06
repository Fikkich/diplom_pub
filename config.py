
import os
import sys

WINDOW_W, WINDOW_H = 1100, 700
VIEW_W, VIEW_H = 900, 600

if getattr(sys, 'frozen', False):
    base_dir = os.path.dirname(sys.executable)
else:
    base_dir = os.path.dirname(__file__)

MODEL_PATH = os.path.join(base_dir, "emotion_model_fer2013_best.h5")

EMOTION_LABELS_UA = ['злість', 'відраза', 'страх', 'радість', 'смуток', 'подив', 'нейтрально']

SMOOTH_WINDOW = 15

PRED_INTERVAL_MS = 150
MAX_CATCHUP_GRABS = 10
