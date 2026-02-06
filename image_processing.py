
import cv2
import numpy as np
import customtkinter as ctk
from PIL import Image, ImageTk, ImageDraw
from config import VIEW_W, VIEW_H
from utils import load_font


def fit_to_view(pil_img: Image.Image) -> Image.Image:
    
    img = pil_img.copy()
    img.thumbnail((VIEW_W, VIEW_H))
    return img


def compose_on_white(pil_img: Image.Image) -> Image.Image:
   
    bg_color = (40, 40, 40) if ctk.get_appearance_mode() == "Dark" else (255, 255, 255)
    bg = Image.new("RGB", (VIEW_W, VIEW_H), bg_color)
    x0 = (VIEW_W - pil_img.width) // 2
    y0 = (VIEW_H - pil_img.height) // 2
    bg.paste(pil_img, (x0, y0))
    return bg


def create_blank_image():
   
    bg_color = (40, 40, 40) if ctk.get_appearance_mode() == "Dark" else (255, 255, 255)
    blank = Image.new("RGB", (VIEW_W, VIEW_H), bg_color)
    return blank


def overlay_annotation(frame_bgr, x, y, w, h, emotion_text, conf, 
                       rectangle_color_bgr, text_color_rgb):
    
    out = frame_bgr.copy()
    cv2.rectangle(out, (x, y), (x+w, y+h), rectangle_color_bgr, 2)
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
    draw.text((tx, ty), label, font=font, fill=text_color_rgb)
    out_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    return out_bgr


def convert_frame_to_tk_image(frame_bgr):
   
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    im = Image.fromarray(frame_rgb)
    im = fit_to_view(im)
    bg = compose_on_white(im)
    imgtk = ImageTk.PhotoImage(image=bg)
    return imgtk
