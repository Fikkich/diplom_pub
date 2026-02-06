
import cv2
import time
from tkinter import messagebox
from config import MAX_CATCHUP_GRABS


def refresh_cameras():
    
    indices, labels = [], []
    for i in range(0, 7):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                indices.append(i)
                labels.append(f"Камера {i}")
        cap.release()

    if not indices:
        indices, labels = [0], ["Камера 0"]

    return indices, labels


def open_camera(device_index):
   
    cap = cv2.VideoCapture(device_index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        return None

    ret, _ = cap.read()
    if not ret:
        cap.release()
        return None

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    return cap


def open_video(video_path):
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, None

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps and fps > 1:
        fps = float(fps)
    else:
        fps = 30.0

    return cap, fps


def read_frame_camera(cap):
   
    ret, frame = cap.read()
    if not ret:
        return None
    return frame


def read_frame_video_realtime(cap, video_fps, video_t0, video_frame_index):
    
    if video_t0 is None:
        video_t0 = time.perf_counter()

    elapsed = time.perf_counter() - video_t0
    target_frame = int(elapsed * video_fps)

    catchup = target_frame - video_frame_index
    if catchup > 0:
        grabs = min(catchup, MAX_CATCHUP_GRABS)
        for _ in range(grabs):
            ok = cap.grab()
            if not ok:
                return None, video_frame_index
            video_frame_index += 1

    ok, frame = cap.retrieve()
    if not ok or frame is None:
        ok2, frame2 = cap.read()
        if not ok2:
            return None, video_frame_index
        video_frame_index += 1
        return frame2, video_frame_index

    if catchup <= 0:
        video_frame_index += 1

    return frame, video_frame_index
