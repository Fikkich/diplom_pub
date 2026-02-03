"""
Модуль для роботи з моделлю розпізнавання емоцій
"""
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from tensorflow import keras
import numpy as np
from config import MODEL_PATH


# Глобальна змінна для моделі
_emotion_model = None


def load_model():
    """
    Завантажує модель розпізнавання емоцій
    
    Returns:
        Завантажена модель Keras
    """
    global _emotion_model
    if _emotion_model is None:
        _emotion_model = keras.models.load_model(MODEL_PATH)
    return _emotion_model


def predict_emotion(face_img_48x48: np.ndarray) -> np.ndarray:
    """
    Передбачає емоції на обличчі
    
    Args:
        face_img_48x48: Зображення обличчя 48x48 пікселів
        
    Returns:
        Масив ймовірностей для кожної емоції
    """
    model = load_model()
    inp = np.expand_dims(face_img_48x48, axis=0)
    probs = model.predict(inp, verbose=0)[0]
    probs = probs / np.sum(probs)  # Нормалізація
    return probs
