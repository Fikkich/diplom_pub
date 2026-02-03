"""
Модуль для детекції облич на зображеннях та відео
"""
import cv2

# Ініціалізація детектора облич
CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
FACE_DETECTOR = cv2.CascadeClassifier(CASCADE_PATH)


def detect_faces(gray_image, scale_factor=1.2, min_neighbors=5, min_size=(60, 60)):
    """
    Виявляє обличчя на зображенні
    
    Args:
        gray_image: Сіре зображення
        scale_factor: Коефіцієнт масштабування
        min_neighbors: Мінімальна кількість сусідів
        min_size: Мінімальний розмір обличчя
        
    Returns:
        Список прямокутників (x, y, w, h) для знайдених облич
    """
    faces = FACE_DETECTOR.detectMultiScale(
        gray_image,
        scaleFactor=scale_factor,
        minNeighbors=min_neighbors,
        minSize=min_size
    )
    return faces


def get_largest_face(faces):
    """
    Повертає найбільше обличчя зі списку
    
    Args:
        faces: Список прямокутників облич
        
    Returns:
        Прямокутник найбільшого обличчя (x, y, w, h) або None
    """
    if len(faces) == 0:
        return None
    return max(faces, key=lambda b: b[2] * b[3])
