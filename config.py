"""
Конфігурація та константи додатку для розпізнавання емоцій
"""
import os
import sys

# Розміри вікна
WINDOW_W, WINDOW_H = 1100, 700
VIEW_W, VIEW_H = 900, 600

# Визначення базової директорії
if getattr(sys, 'frozen', False):
    base_dir = os.path.dirname(sys.executable)
else:
    base_dir = os.path.dirname(__file__)

# Шлях до моделі
MODEL_PATH = os.path.join(base_dir, "emotion_model_fer2013_best.h5")

# Мітки емоцій українською
EMOTION_LABELS_UA = ['злість', 'відраза', 'страх', 'радість', 'смуток', 'подив', 'нейтрально']

# Параметри згладжування
SMOOTH_WINDOW = 15

# Параметри детекції облич
PRED_INTERVAL_MS = 150
MAX_CATCHUP_GRABS = 10
