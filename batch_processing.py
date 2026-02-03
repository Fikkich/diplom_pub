"""
Модуль для пакетної обробки зображень
"""
import os
import csv
import threading
import numpy as np
from tkinter import filedialog, messagebox
import customtkinter as ctk
import cv2
from utils import cv2_imread_unicode
from face_detection import detect_faces, get_largest_face
from utils import preprocess_face
from emotion_model import predict_emotion
from config import EMOTION_LABELS_UA


def find_image_files(folder):
    """
    Знаходить всі файли зображень у папці
    
    Args:
        folder: Шлях до папки
        
    Returns:
        Список шляхів до файлів
    """
    exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")
    file_list = []
    for root_dir, _, files in os.walk(folder):
        for fname in files:
            if fname.lower().endswith(exts):
                file_list.append(os.path.join(root_dir, fname))
    return file_list


def analyze_image(img_path, model_predict_func):
    """
    Аналізує одне зображення
    
    Args:
        img_path: Шлях до зображення
        model_predict_func: Функція для передбачення емоцій
        
    Returns:
        Словник з результатами аналізу
    """
    img = cv2_imread_unicode(img_path, cv2.IMREAD_COLOR)
    if img is None:
        return {
            "has_face": 0,
            "pred_emotion": "",
            "pred_conf": "",
            "top_emotions": [("", 0.0), ("", 0.0), ("", 0.0)]
        }

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detect_faces(gray, scale_factor=1.2, min_neighbors=5, min_size=(60, 60))

    if len(faces) == 0:
        return {
            "has_face": 0,
            "pred_emotion": "",
            "pred_conf": "",
            "top_emotions": [("", 0.0), ("", 0.0), ("", 0.0)]
        }

    x, y, w, h = get_largest_face(faces)
    face_input, _ = preprocess_face(img, x, y, w, h)
    probs = model_predict_func(face_input)

    idx = int(np.argmax(probs))
    emotion = EMOTION_LABELS_UA[idx]
    conf = float(probs[idx])

    top_indices = np.argsort(probs)[-3:][::-1]
    top = [(EMOTION_LABELS_UA[i], float(probs[i])) for i in top_indices]
    while len(top) < 3:
        top.append(("", 0.0))

    return {
        "has_face": 1,
        "pred_emotion": emotion,
        "pred_conf": f"{conf:.4f}",
        "top_emotions": top
    }


def create_progress_window(parent):
    """
    Створює вікно прогресу для пакетної обробки
    
    Args:
        parent: Батьківське вікно
        
    Returns:
        Кортеж (вікно, мітка інфо, прогрес-бар, мітка відсотків)
    """
    progress_win = ctk.CTkToplevel(parent)
    progress_win.title("Обробка папки")
    progress_win.geometry("400x120")
    progress_win.resizable(False, False)
    progress_win.transient(parent)
    progress_win.grab_set()
    progress_win.attributes("-topmost", True)

    lbl_info = ctk.CTkLabel(progress_win, text="Обробка зображень...")
    lbl_info.pack(pady=(15, 5))

    progress_bar = ctk.CTkProgressBar(progress_win, width=320)
    progress_bar.pack(pady=5)
    progress_bar.set(0.0)

    lbl_percent = ctk.CTkLabel(progress_win, text="0 %")
    lbl_percent.pack(pady=(5, 10))
    progress_win.update_idletasks()

    return progress_win, lbl_info, progress_bar, lbl_percent


def process_batch_worker(file_list, out_csv, model_predict_func, progress_callback):
    """
    Робоча функція для обробки пакету зображень
    
    Args:
        file_list: Список шляхів до файлів
        out_csv: Шлях до вихідного CSV файлу
        model_predict_func: Функція для передбачення емоцій
        progress_callback: Функція для оновлення прогресу
    """
    try:
        with open(out_csv, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f, delimiter=';')
            writer.writerow([
                "file",
                "has_face",
                "pred_emotion",
                "pred_conf",
                "top1_emotion", "top1_conf",
                "top2_emotion", "top2_conf",
                "top3_emotion", "top3_conf",
            ])

            for i, path in enumerate(file_list):
                result = analyze_image(path, model_predict_func)
                top = result["top_emotions"]

                writer.writerow([
                    path,
                    result["has_face"],
                    result["pred_emotion"],
                    result["pred_conf"],
                    top[0][0], f"{top[0][1]:.4f}",
                    top[1][0], f"{top[1][1]:.4f}",
                    top[2][0], f"{top[2][1]:.4f}",
                ])

                if progress_callback:
                    progress_callback(i + 1, len(file_list))
    except Exception as e:
        if progress_callback:
            progress_callback(-1, -1, str(e))
