
import cv2


CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
FACE_DETECTOR = cv2.CascadeClassifier(CASCADE_PATH)


def detect_faces(gray_image, scale_factor=1.2, min_neighbors=5, min_size=(60, 60)):
   
    faces = FACE_DETECTOR.detectMultiScale(
        gray_image,
        scaleFactor=scale_factor,
        minNeighbors=min_neighbors,
        minSize=min_size
    )
    return faces


def get_largest_face(faces):
    
    if len(faces) == 0:
        return None
    return max(faces, key=lambda b: b[2] * b[3])
