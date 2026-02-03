import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import customtkinter as ctk
from tkinter import filedialog, messagebox, colorchooser
import cv2
import time
import numpy as np
from collections import deque
from PIL import Image, ImageTk, ImageDraw, ImageFont
from tensorflow import keras
import csv
import threading
import sys

WINDOW_W, WINDOW_H = 1100, 700
VIEW_W, VIEW_H = 900, 600

if getattr(sys, 'frozen', False):
    base_dir = os.path.dirname(sys.executable)
else:
    base_dir = os.path.dirname(__file__)

model_path = os.path.join(base_dir, "emotion_model_fer2013_best.h5")
emotion_model = keras.models.load_model(model_path)
EMOTION_LABELS_UA = ['злість','відраза','страх','радість','смуток','подив','нейтрально']

SMOOTH_WINDOW = 15
CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
FACE_DETECTOR = cv2.CascadeClassifier(CASCADE_PATH)

PRED_INTERVAL_MS = 150
MAX_CATCHUP_GRABS = 10

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

class EmotionApp:
    def __init__(self, root):
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        self.root = root
        self.root.title("Розпізнавання емоцій")
        self.root.geometry(f"{WINDOW_W}x{WINDOW_H}")
        self.root.minsize(WINDOW_W, WINDOW_H)
        self.root.maxsize(WINDOW_W, WINDOW_H)
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        icon_path = os.path.join(base_dir, "app_icon.ico")
        if os.path.exists(icon_path):
            try:
                self.root.iconbitmap(icon_path)
            except Exception:
                pass

        self.cap = None
        self.running = False

        self.current_device = ctk.StringVar(value="0")

        self.prob_buffer = deque(maxlen=SMOOTH_WINDOW)

        self.last_time = time.time()
        self.fps = 0.0

        self.source_mode = "camera"
        self.video_path = None
        self.video_fps = 30.0
        self.video_t0 = None
        self.video_frame_index = 0

        self.last_pred_ts = 0.0
        self.last_emotion = None
        self.last_conf = 0.0
        self.last_smoothed = None

        self.conf_threshold_var = ctk.DoubleVar(value=0.6)

        self.rectangle_color_bgr = (0, 200, 100)
        self.text_color_rgb = (0, 200, 100)

        self.batch_thread = None
        self.batch_total = 0
        self.batch_done = 0
        self.batch_running = False
        self.progress_win = None

        self.setup_ui()
        self.show_blank()
        self.refresh_cameras()

    def setup_ui(self):
        top = ctk.CTkFrame(self.root)
        top.pack(side=ctk.TOP, fill=ctk.X, padx=8, pady=8)

        ctk.CTkLabel(top, text="Камера:").pack(side=ctk.LEFT, padx=5)
        self.device_combo = ctk.CTkComboBox(top, state="readonly", width=200,
                                            values=["Завантаження..."])
        self.device_combo.pack(side=ctk.LEFT, padx=5)
        self.device_combo.bind("<<ComboboxSelected>>", self.on_camera_selected)

        self.btn_refresh = ctk.CTkButton(top, text="Оновити",
                                         command=self.refresh_cameras, width=100)
        self.btn_refresh.pack(side=ctk.LEFT, padx=5)

        self.btn_start = ctk.CTkButton(top, text="Старт", command=self.start,
                                       width=100, fg_color="green",
                                       hover_color="darkgreen")
        self.btn_start.pack(side=ctk.LEFT, padx=5)

        self.btn_stop = ctk.CTkButton(top, text="Стоп", command=self.stop,
                                      width=100, state=ctk.DISABLED,
                                      fg_color="red", hover_color="darkred")
        self.btn_stop.pack(side=ctk.LEFT, padx=5)

        self.btn_load = ctk.CTkButton(top, text="Завантажити фото",
                                      command=self.load_image, width=150)
        self.btn_load.pack(side=ctk.LEFT, padx=10)

        self.btn_load_video = ctk.CTkButton(top, text="Завантажити відео",
                                            command=self.load_video, width=150)
        self.btn_load_video.pack(side=ctk.LEFT, padx=5)

        self.btn_batch = ctk.CTkButton(top, text="Аналізувати папку",
                                       command=self.batch_analyze_folder, width=150)
        self.btn_batch.pack(side=ctk.LEFT, padx=5)

        self.fps_label = ctk.CTkLabel(top, text="FPS: 0.0")
        self.fps_label.pack(side=ctk.RIGHT, padx=10)

        middle = ctk.CTkFrame(self.root)
        middle.pack(fill=ctk.BOTH, expand=True, padx=8, pady=8)

        sidebar = ctk.CTkFrame(middle, width=150)
        sidebar.pack(side=ctk.LEFT, fill=ctk.Y, padx=(0, 8))
        sidebar.pack_propagate(False)

        ctk.CTkLabel(sidebar, text="Налаштування",
                     font=ctk.CTkFont(size=16, weight="bold")).pack(pady=(15, 20))

        ctk.CTkLabel(sidebar, text="Колір рамки:",
                     font=ctk.CTkFont(size=12)).pack(pady=(10, 5))
        rect_color_hex = "#%02x%02x%02x" % (
            self.rectangle_color_bgr[2],
            self.rectangle_color_bgr[1],
            self.rectangle_color_bgr[0]
        )
        self.btn_rect_color = ctk.CTkButton(sidebar, text="", width=16, height=16,
                                            command=self.choose_rectangle_color,
                                            fg_color=rect_color_hex,
                                            hover_color=rect_color_hex)
        self.btn_rect_color.pack(pady=5)

        ctk.CTkLabel(sidebar, text="Колір тексту:",
                     font=ctk.CTkFont(size=12)).pack(pady=(20, 5))
        text_color_hex = "#%02x%02x%02x" % self.text_color_rgb
        self.btn_text_color = ctk.CTkButton(sidebar, text="", width=16, height=16,
                                            command=self.choose_text_color,
                                            fg_color=text_color_hex,
                                            hover_color=text_color_hex)
        self.btn_text_color.pack(pady=5)

        ctk.CTkLabel(sidebar, text="Мін. впевненість:",
                     font=ctk.CTkFont(size=12)).pack(pady=(20, 5))
        self.conf_slider = ctk.CTkSlider(
            sidebar, from_=0.4, to=0.95, number_of_steps=11,
            variable=self.conf_threshold_var, width=120
        )
        self.conf_slider.pack(pady=5)
        self.conf_label = ctk.CTkLabel(sidebar, text="60%",
                                       font=ctk.CTkFont(size=10))
        self.conf_label.pack(pady=(0, 10))
        self.conf_threshold_var.trace('w', self.update_conf_label)

        ctk.CTkLabel(sidebar, text="Ймовірності емоцій:",
                     font=ctk.CTkFont(size=12, weight="bold")).pack(pady=(10, 5))

        self.emotion_bars = []
        for emo in EMOTION_LABELS_UA:
            row_frame = ctk.CTkFrame(sidebar)
            row_frame.pack(fill=ctk.X, pady=2)

            lbl = ctk.CTkLabel(row_frame, text=emo, width=70, anchor="w")
            lbl.pack(side=ctk.LEFT, padx=2)

            bar = ctk.CTkProgressBar(row_frame, width=60, height=10)
            bar.pack(side=ctk.LEFT, expand=True, fill=ctk.X, padx=2)
            bar.set(0.0)
            self.emotion_bars.append(bar)

        self.video_frame = ctk.CTkFrame(middle, width=VIEW_W, height=VIEW_H)
        self.video_frame.pack_propagate(False)
        self.video_frame.pack(side=ctk.LEFT, fill=ctk.BOTH, expand=True)

        self.video_label = ctk.CTkLabel(self.video_frame, text="")
        self.video_label.pack(fill=ctk.BOTH, expand=True)

        bottom = ctk.CTkFrame(self.root)
        bottom.pack(side=ctk.BOTTOM, fill=ctk.X, padx=8, pady=8)

        self.emotion_var = ctk.StringVar(value="—")
        self.conf_var = ctk.StringVar(value="—")
        ctk.CTkLabel(bottom, text="Емоція:").pack(side=ctk.LEFT, padx=5)
        ctk.CTkLabel(bottom, textvariable=self.emotion_var, width=150).pack(side=ctk.LEFT, padx=5)
        ctk.CTkLabel(bottom, text="Впевненість:").pack(side=ctk.LEFT, padx=10)
        ctk.CTkLabel(bottom, textvariable=self.conf_var, width=80).pack(side=ctk.LEFT)

        self.btn_save_annot = ctk.CTkButton(
            bottom, text="Зберегти фото з позначкою",
            command=self.save_annotated, state=ctk.DISABLED, width=200
        )
        self.btn_save_annot.pack(side=ctk.RIGHT)

        self.last_annotated_bgr = None
        self.update_job = None

    def update_conf_label(self, *args):
        val = self.conf_threshold_var.get()
        self.conf_label.configure(text=f"{val:.0%}")

    def model_predict(self, face_img_48x48: np.ndarray) -> np.ndarray:
        inp = np.expand_dims(face_img_48x48, axis=0)
        probs = emotion_model.predict(inp, verbose=0)[0]
        probs = probs / np.sum(probs)
        return probs

    def refresh_cameras(self):
        was_running = self.running
        if was_running:
            self.stop()

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
            indices, labels = [0], ["Камера 0 (типово)"]

        self.device_indices = indices
        self.device_combo.configure(values=labels)
        try:
            current_idx = self.device_indices.index(int(self.current_device.get()))
            self.device_combo.set(labels[current_idx])
        except Exception:
            self.device_combo.set(labels[0])
            self.current_device.set(str(self.device_indices[0]))

    def on_camera_selected(self, event=None):
        selected = self.device_combo.get()
        try:
            cam_num = int(selected.split()[-1])
            if cam_num in self.device_indices:
                self.current_device.set(str(cam_num))
        except Exception:
            pass

    def choose_rectangle_color(self):
        rgb_color = (
            self.rectangle_color_bgr[2],
            self.rectangle_color_bgr[1],
            self.rectangle_color_bgr[0]
        )
        color = colorchooser.askcolor(title="Оберіть колір рамки", color=rgb_color)
        if color[1] is not None:
            hex_color = color[1].lstrip('#')
            rgb = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
            self.rectangle_color_bgr = (rgb[2], rgb[1], rgb[0])
            self.btn_rect_color.configure(fg_color=color[1], hover_color=color[1])

    def choose_text_color(self):
        rgb_color = self.text_color_rgb
        color = colorchooser.askcolor(title="Оберіть колір тексту", color=rgb_color)
        if color[1] is not None:
            hex_color = color[1].lstrip('#')
            self.text_color_rgb = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
            self.btn_text_color.configure(fg_color=color[1], hover_color=color[1])

    def fit_to_view(self, pil_img: Image.Image) -> Image.Image:
        img = pil_img.copy()
        img.thumbnail((VIEW_W, VIEW_H))
        return img

    def compose_on_white(self, pil_img: Image.Image) -> Image.Image:
        bg_color = (40, 40, 40) if ctk.get_appearance_mode() == "Dark" else (255, 255, 255)
        bg = Image.new("RGB", (VIEW_W, VIEW_H), bg_color)
        x0 = (VIEW_W - pil_img.width) // 2
        y0 = (VIEW_H - pil_img.height) // 2
        bg.paste(pil_img, (x0, y0))
        return bg

    def show_blank(self):
        bg_color = (40, 40, 40) if ctk.get_appearance_mode() == "Dark" else (255, 255, 255)
        blank = Image.new("RGB", (VIEW_W, VIEW_H), bg_color)
        imgtk = ImageTk.PhotoImage(blank)
        self.video_label.imgtk = imgtk
        self.video_label.configure(image=imgtk)

    def _reset_runtime_state(self):
        self.prob_buffer.clear()
        self.last_pred_ts = 0.0
        self.last_emotion = None
        self.last_conf = 0.0
        self.last_smoothed = None
        self.last_time = time.time()
        self.fps = 0.0
        self.fps_label.configure(text="FPS: 0.0")
        self.emotion_var.set("—")
        self.conf_var.set("—")
        for bar in self.emotion_bars:
            bar.set(0.0)

    def start(self):
        if self.running:
            return

        self.source_mode = "camera"
        dev = int(self.current_device.get())
        self.cap = cv2.VideoCapture(dev, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            messagebox.showerror("Помилка", f"Не вдалося відкрити камеру {dev}. Оберіть іншу.")
            return

        ret, _ = self.cap.read()
        if not ret:
            self.cap.release()
            self.cap = None
            messagebox.showerror("Помилка", f"Камера {dev} не віддає кадри. Оберіть іншу.")
            return

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        self.running = True
        self.btn_start.configure(state=ctk.DISABLED)
        self.btn_stop.configure(state=ctk.NORMAL)
        self.btn_load.configure(state=ctk.DISABLED)
        self.btn_load_video.configure(state=ctk.DISABLED)
        self.btn_batch.configure(state=ctk.DISABLED)

        self._reset_runtime_state()
        self.loop()

    def load_video(self):
        if self.running:
            self.stop()

        fname = filedialog.askopenfilename(
            title="Оберіть відео",
            filetypes=[
                ("Відео", "*.mp4 *.avi *.mkv *.mov *.webm"),
                ("Усі файли", "*.*"),
            ]
        )
        if not fname:
            return

        cap = cv2.VideoCapture(fname)
        if not cap.isOpened():
            messagebox.showerror("Помилка", "Не вдалося відкрити відео.")
            return

        self.cap = cap
        self.source_mode = "video"
        self.video_path = fname

        fps = self.cap.get(cv2.CAP_PROP_FPS)
        if fps and fps > 1:
            self.video_fps = float(fps)
        else:
            self.video_fps = 30.0

        self.video_t0 = time.perf_counter()
        self.video_frame_index = 0

        self.running = True
        self.btn_start.configure(state=ctk.DISABLED)
        self.btn_stop.configure(state=ctk.NORMAL)
        self.btn_load.configure(state=ctk.DISABLED)
        self.btn_load_video.configure(state=ctk.DISABLED)
        self.btn_batch.configure(state=ctk.DISABLED)

        self._reset_runtime_state()
        self.loop()

    def stop(self):
        self.running = False
        if self.update_job is not None:
            self.root.after_cancel(self.update_job)
            self.update_job = None
        if self.cap is not None:
            self.cap.release()
            self.cap = None

        self.btn_start.configure(state=ctk.NORMAL)
        self.btn_stop.configure(state=ctk.DISABLED)
        self.btn_load.configure(state=ctk.NORMAL)
        self.btn_load_video.configure(state=ctk.NORMAL)
        self.btn_batch.configure(state=ctk.NORMAL)
        self.btn_save_annot.configure(state=ctk.DISABLED)

        self.last_annotated_bgr = None
        self.video_t0 = None
        self.video_frame_index = 0

    def on_close(self):
        self.stop()
        self.root.destroy()

    def overlay_annotation(self, frame_bgr, x, y, w, h, emotion_text, conf):
        out = frame_bgr.copy()
        cv2.rectangle(out, (x, y), (x+w, y+h), self.rectangle_color_bgr, 2)
        font_size = max(20, min(int(h * 0.23), 50))
        label = f"{emotion_text}: {conf:.1%}"

        rgb = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        draw = ImageDraw.Draw(pil_img)
        font = load_font(size=font_size)
        tx = x
        ty = max(0, y - font_size - 8)
        for dx, dy in [(1,1), (1,0), (0,1), (-1,1)]:
            draw.text((tx+dx, ty+dy), label, font=font, fill=(0, 0, 0))
        draw.text((tx, ty), label, font=font, fill=self.text_color_rgb)
        out_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        return out_bgr

    def _read_frame_camera(self):
        ret, frame = self.cap.read()
        if not ret:
            return None
        return frame

    def _read_frame_video_realtime(self):
        if self.video_t0 is None:
            self.video_t0 = time.perf_counter()

        elapsed = time.perf_counter() - self.video_t0
        target_frame = int(elapsed * self.video_fps)

        catchup = target_frame - self.video_frame_index
        if catchup > 0:
            grabs = min(catchup, MAX_CATCHUP_GRABS)
            for _ in range(grabs):
                ok = self.cap.grab()
                if not ok:
                    return None
                self.video_frame_index += 1

        ok, frame = self.cap.retrieve()
        if not ok or frame is None:
            ok2, frame2 = self.cap.read()
            if not ok2:
                return None
            self.video_frame_index += 1
            return frame2

        if catchup <= 0:
            self.video_frame_index += 1

        return frame

    def loop(self):
        if not self.running or self.cap is None:
            return

        if self.source_mode == "video":
            frame = self._read_frame_video_realtime()
        else:
            frame = self._read_frame_camera()

        if frame is None:
            self.stop()
            return

        gray_full = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = FACE_DETECTOR.detectMultiScale(
            gray_full, scaleFactor=1.2, minNeighbors=5, minSize=(60, 60)
        )

        now_ms = time.perf_counter() * 1000.0
        do_predict = (now_ms - self.last_pred_ts) >= PRED_INTERVAL_MS

        annotated = frame

        if len(faces) > 0:
            x, y, w, h = max(faces, key=lambda b: b[2]*b[3])

            if do_predict:
                face_input, _ = preprocess_face(frame, x, y, w, h)
                probs = self.model_predict(face_input)
                self.prob_buffer.append(probs)
                smoothed = np.mean(self.prob_buffer, axis=0)

                idx = int(np.argmax(smoothed))
                emotion = EMOTION_LABELS_UA[idx]
                conf = float(smoothed[idx])

                self.last_emotion = emotion
                self.last_conf = conf
                self.last_smoothed = smoothed
                self.last_pred_ts = now_ms

            if self.last_smoothed is not None:
                for i, bar in enumerate(self.emotion_bars):
                    bar.set(float(self.last_smoothed[i]))
            else:
                for bar in self.emotion_bars:
                    bar.set(0.0)

            threshold = self.conf_threshold_var.get()
            if (self.last_emotion is not None) and (self.last_conf >= threshold):
                self.emotion_var.set(self.last_emotion)
                self.conf_var.set(f"{self.last_conf:.1%}")
                annotated = self.overlay_annotation(frame, x, y, w, h, self.last_emotion, self.last_conf)
                self.btn_save_annot.configure(state=ctk.NORMAL)
            else:
                self.emotion_var.set("🤔 Невідомо" if self.last_emotion is not None else "—")
                if self.last_emotion is not None:
                    self.conf_var.set(f"{self.last_conf:.1%} < {threshold:.0%}")
                else:
                    self.conf_var.set("—")
                annotated = frame
                self.btn_save_annot.configure(state=ctk.DISABLED)
        else:
            self.emotion_var.set("—")
            self.conf_var.set("—")
            self.btn_save_annot.configure(state=ctk.DISABLED)
            for bar in self.emotion_bars:
                bar.set(0.0)

        self.last_annotated_bgr = annotated

        now = time.time()
        dt = now - self.last_time
        if dt > 0:
            self.fps = 1.0 / dt
        self.last_time = now
        self.fps_label.configure(text=f"FPS: {self.fps:.1f}")

        frame_rgb = cv2.cvtColor(self.last_annotated_bgr, cv2.COLOR_BGR2RGB)
        im = Image.fromarray(frame_rgb)
        im = self.fit_to_view(im)
        bg = self.compose_on_white(im)
        imgtk = ImageTk.PhotoImage(image=bg)
        self.video_label.imgtk = imgtk
        self.video_label.configure(image=imgtk)

        delay = 10
        self.update_job = self.root.after(delay, self.loop)

    def load_image(self):
        fname = filedialog.askopenfilename(
            title="Оберіть зображення",
            filetypes=[
                ("Зображення", "*.jpg *.jpeg *.png *.bmp *.webp *.tif *.tiff"),
                ("Усі файли", "*.*"),
            ]
        )
        if not fname:
            return
        img = cv2_imread_unicode(fname, cv2.IMREAD_COLOR)
        if img is None:
            messagebox.showerror("Помилка", "Не вдалося відкрити фото. Перевірте шлях або формат.")
            return
        self.predict_image(img)

    def predict_image(self, img):
        frame = img.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = FACE_DETECTOR.detectMultiScale(
            gray, scaleFactor=1.2, minNeighbors=5, minSize=(60,60)
        )
        if len(faces) == 0:
            self.emotion_var.set("—")
            self.conf_var.set("—")
            self.btn_save_annot.configure(state=ctk.DISABLED)
            annotated = frame
            for bar in self.emotion_bars:
                bar.set(0.0)
        else:
            x, y, w, h = max(faces, key=lambda b: b[2]*b[3])
            face_input, _ = preprocess_face(frame, x, y, w, h)
            probs = self.model_predict(face_input)
            idx = int(np.argmax(probs))
            emotion = EMOTION_LABELS_UA[idx]
            conf = float(probs[idx])

            for i, bar in enumerate(self.emotion_bars):
                bar.set(float(probs[i]))

            threshold = self.conf_threshold_var.get()
            if conf >= threshold:
                self.emotion_var.set(emotion)
                self.conf_var.set(f"{conf:.1%}")
                annotated = self.overlay_annotation(frame, x, y, w, h, emotion, conf)
                self.btn_save_annot.configure(state=ctk.NORMAL)
            else:
                self.emotion_var.set("🤔 Невідомо")
                self.conf_var.set(f"{conf:.1%} < {threshold:.0%}")
                annotated = frame
                self.btn_save_annot.configure(state=ctk.DISABLED)

        self.last_annotated_bgr = annotated
        im = Image.fromarray(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
        im = self.fit_to_view(im)
        bg = self.compose_on_white(im)
        imgtk = ImageTk.PhotoImage(image=bg)
        self.video_label.imgtk = imgtk
        self.video_label.configure(image=imgtk)

    def batch_analyze_folder(self):
        folder = filedialog.askdirectory(title="Оберіть папку з зображеннями")
        if not folder:
            return

        exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")
        file_list = []
        for root_dir, _, files in os.walk(folder):
            for fname in files:
                if fname.lower().endswith(exts):
                    file_list.append(os.path.join(root_dir, fname))

        if not file_list:
            messagebox.showwarning("Немає зображень", "У вибраній папці не знайдено зображень.")
            return

        out_csv = filedialog.asksaveasfilename(
            title="Зберегти звіт",
            defaultextension=".csv",
            filetypes=[("CSV", "*.csv"), ("Усі файли", "*.*")]
        )
        if not out_csv:
            return

        self.btn_batch.configure(state=ctk.DISABLED)
        self.btn_start.configure(state=ctk.DISABLED)
        self.btn_load.configure(state=ctk.DISABLED)
        self.btn_load_video.configure(state=ctk.DISABLED)

        self.progress_win = ctk.CTkToplevel(self.root)
        self.progress_win.title("Обробка папки")
        self.progress_win.geometry("400x120")
        self.progress_win.resizable(False, False)
        self.progress_win.transient(self.root)
        self.progress_win.grab_set()
        self.progress_win.attributes("-topmost", True)

        self.lbl_info = ctk.CTkLabel(self.progress_win, text="Обробка зображень...")
        self.lbl_info.pack(pady=(15, 5))

        self.progress_bar = ctk.CTkProgressBar(self.progress_win, width=320)
        self.progress_bar.pack(pady=5)
        self.progress_bar.set(0.0)

        self.lbl_percent = ctk.CTkLabel(self.progress_win, text="0 %")
        self.lbl_percent.pack(pady=(5, 10))
        self.progress_win.update_idletasks()

        self.batch_total = len(file_list)
        self.batch_done = 0
        self.batch_running = True

        def worker():
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

                    for path in file_list:
                        img = cv2_imread_unicode(path, cv2.IMREAD_COLOR)
                        if img is None:
                            writer.writerow([path, 0, "", "", "", "", "", "", "", "", ""])
                        else:
                            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                            faces = FACE_DETECTOR.detectMultiScale(
                                gray, scaleFactor=1.2, minNeighbors=5, minSize=(60, 60)
                            )
                            if len(faces) == 0:
                                writer.writerow([path, 0, "", "", "", "", "", "", "", "", ""])
                            else:
                                x, y, w, h = max(faces, key=lambda b: b[2] * b[3])
                                face_input, _ = preprocess_face(img, x, y, w, h)
                                probs = self.model_predict(face_input)

                                idx = int(np.argmax(probs))
                                emotion = EMOTION_LABELS_UA[idx]
                                conf = float(probs[idx])

                                top_indices = np.argsort(probs)[-3:][::-1]
                                top = [(EMOTION_LABELS_UA[i], float(probs[i])) for i in top_indices]
                                while len(top) < 3:
                                    top.append(("", 0.0))

                                writer.writerow([
                                    path, 1, emotion, f"{conf:.4f}",
                                    top[0][0], f"{top[0][1]:.4f}",
                                    top[1][0], f"{top[1][1]:.4f}",
                                    top[2][0], f"{top[2][1]:.4f}",
                                ])

                        self.batch_done += 1
            finally:
                self.batch_running = False

        self.batch_thread = threading.Thread(target=worker, daemon=True)
        self.batch_thread.start()
        self.root.after(100, self._update_batch_progress, out_csv)

    def _update_batch_progress(self, out_csv):
        frac = 0.0 if self.batch_total == 0 else self.batch_done / self.batch_total
        if frac > 1.0:
            frac = 1.0

        if self.progress_win is not None:
            self.progress_bar.set(frac)
            self.lbl_percent.configure(text=f"{frac*100:.1f} %")
            self.progress_win.update_idletasks()

        if self.batch_running:
            self.root.after(100, self._update_batch_progress, out_csv)
        else:
            if self.progress_win is not None:
                self.progress_win.destroy()
                self.progress_win = None

            self.btn_batch.configure(state=ctk.NORMAL)
            self.btn_start.configure(state=ctk.NORMAL)
            self.btn_load.configure(state=ctk.NORMAL)
            self.btn_load_video.configure(state=ctk.NORMAL)

            messagebox.showinfo("Готово",
                                f"Звіт по {self.batch_total} зображеннях збережено:\n{out_csv}")

    def save_annotated(self):
        if self.last_annotated_bgr is None:
            return
        fname = filedialog.asksaveasfilename(
            title="Зберегти зображення",
            defaultextension=".png",
            filetypes=[("PNG", "*.png"),
                       ("JPEG", "*.jpg;*.jpeg"),
                       ("Усі файли", "*.*")]
        )
        if not fname:
            return
        ok = cv2_imwrite_unicode(fname, self.last_annotated_bgr)
        if ok:
            messagebox.showinfo("Збережено",
                                f"Зображення з позначкою збережено:\n{fname}")
        else:
            messagebox.showerror("Помилка", "Не вдалося зберегти файл.")

if __name__ == "__main__":
    root = ctk.CTk()
    app = EmotionApp(root)
    root.mainloop()
