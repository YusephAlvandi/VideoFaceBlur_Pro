"""
VideoFaceBlur v2 — Dual Mode (Face-only / Full Body)
Author: Yuseph Alvandi
Description: Choose between face-only or full-body blurring with YOLOv8.
"""

import customtkinter as ctk
from tkinter import filedialog, messagebox
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image, ImageTk
import os
import subprocess
import tempfile
import shutil

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

class VideoFaceBlurApp:
    def __init__(self):
        self.window = ctk.CTk()
        self.window.title("Video Face Blur Pro")
        self.window.geometry("1000x750")
        self.window.configure(fg_color="#0a0a0a")
        
        self.video_path = None
        
        face_model = os.path.expanduser("~/python_projects/image_processing/models/yolov8n-face.pt")
        body_model = os.path.expanduser("~/python_projects/image_processing/models/yolov8m.pt")
        self.face_model = YOLO(face_model)
        self.body_model = YOLO(body_model)
        
        self.blur_strength = ctk.IntVar(value=25)
        self.confidence = ctk.DoubleVar(value=0.5)
        self.blur_mode = ctk.StringVar(value="face")
        
        self.setup_ui()
    
    def setup_ui(self):
        header = ctk.CTkFrame(self.window, fg_color="transparent")
        header.pack(fill="x", pady=(20, 10), padx=30)
        ctk.CTkLabel(header, text="Video Face Blur Pro", font=ctk.CTkFont(size=28, weight="bold"), text_color="#1E90FF").pack()
        ctk.CTkLabel(header, text="Dual mode: Face-only or Full-body blurring", font=ctk.CTkFont(size=14), text_color="#AAAAAA").pack()
        
        content = ctk.CTkFrame(self.window, fg_color="transparent")
        content.pack(fill="both", expand=True, padx=30, pady=10)
        
        left = ctk.CTkFrame(content, fg_color="#1a1a1a", corner_radius=12, width=380)
        left.pack(side="left", fill="y", padx=(0, 10))
        left.pack_propagate(False)
        
        ctk.CTkLabel(left, text="Controls", font=ctk.CTkFont(size=18, weight="bold"), text_color="#1E90FF").pack(pady=(20, 15))
        
        ctk.CTkButton(left, text="Open Video", command=self.open_video, height=40, font=ctk.CTkFont(size=14)).pack(pady=10, padx=20, fill="x")
        self.file_label = ctk.CTkLabel(left, text="No video selected", text_color="#888888", font=ctk.CTkFont(size=12))
        self.file_label.pack(pady=5)
        
        ctk.CTkLabel(left, text="Blur Mode", font=ctk.CTkFont(size=14, weight="bold"), text_color="#CCCCCC").pack(pady=(15, 5))
        ctk.CTkRadioButton(left, text="Face Only", variable=self.blur_mode, value="face").pack(anchor="w", padx=40, pady=3)
        ctk.CTkRadioButton(left, text="Full Body", variable=self.blur_mode, value="body").pack(anchor="w", padx=40, pady=3)
        
        ctk.CTkLabel(left, text="Blur Strength", font=ctk.CTkFont(size=14, weight="bold"), text_color="#CCCCCC").pack(pady=(15, 5))
        ctk.CTkSlider(left, from_=5, to=55, variable=self.blur_strength, width=250, command=self.update_blur_label).pack()
        self.blur_label = ctk.CTkLabel(left, text="25", font=ctk.CTkFont(size=11), text_color="#1E90FF")
        self.blur_label.pack()
        
        ctk.CTkLabel(left, text="Confidence", font=ctk.CTkFont(size=14, weight="bold"), text_color="#CCCCCC").pack(pady=(15, 5))
        ctk.CTkSlider(left, from_=0.1, to=0.9, variable=self.confidence, width=250, command=self.update_conf_label).pack()
        self.conf_label = ctk.CTkLabel(left, text="0.5", font=ctk.CTkFont(size=11), text_color="#1E90FF")
        self.conf_label.pack()
        
        self.btn_process = ctk.CTkButton(left, text="Start Processing", command=self.process_video, height=45, fg_color="#E67E22", font=ctk.CTkFont(size=15, weight="bold"))
        self.btn_process.pack(pady=20, padx=20, fill="x")
        
        self.progress_label = ctk.CTkLabel(left, text="", text_color="#FFAA33", font=ctk.CTkFont(size=12))
        self.progress_label.pack(pady=5)
        
        self.status_label = ctk.CTkLabel(left, text="Ready — Dual model loaded", text_color="#4CAF50", font=ctk.CTkFont(size=11))
        self.status_label.pack(pady=10)
        
        right = ctk.CTkFrame(content, fg_color="#1a1a1a", corner_radius=12)
        right.pack(side="right", fill="both", expand=True)
        self.preview_label = ctk.CTkLabel(right, text="No Video Loaded", font=ctk.CTkFont(size=16), text_color="#555555")
        self.preview_label.pack(expand=True)
    
    def open_video(self):
        path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv")])
        if not path: return
        self.video_path = path
        self.file_label.configure(text=os.path.basename(path))
        cap = cv2.VideoCapture(path)
        ret, frame = cap.read()
        if ret: self.show_frame(frame)
        cap.release()
    
    def update_blur_label(self, value):
        v = int(float(value))
        if v % 2 == 0: v += 1
        self.blur_label.configure(text=str(v))
    
    def update_conf_label(self, value):
        self.conf_label.configure(text=f"{float(value):.1f}")
    
    def show_frame(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        preview_w = 550
        ratio = preview_w / pil_img.width
        preview_h = int(pil_img.height * ratio)
        tk_img = ImageTk.PhotoImage(pil_img.resize((preview_w, preview_h)))
        self.preview_label.configure(image=tk_img, text="")
        self.preview_label.image = tk_img
    
    def process_video(self):
        if not self.video_path:
            messagebox.showerror("Error", "Please select a video file!"); return
        
        output = filedialog.asksaveasfilename(defaultextension=".mp4", filetypes=[("MP4", "*.mp4"), ("AVI", "*.avi")])
        if not output: return
        
        self.btn_process.configure(state="disabled", text="Processing...")
        self.window.update()
        
        try:
            cap = cv2.VideoCapture(self.video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            temp_video = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(temp_video, fourcc, fps, (width, height))
            
            blur_k = self.blur_strength.get()
            if blur_k % 2 == 0: blur_k += 1
            
            model = self.face_model if self.blur_mode.get() == "face" else self.body_model
            
            frame_count = 0
            
            while True:
                ret, frame = cap.read()
                if not ret: break
                
                try:
                    results = model.track(frame, conf=self.confidence.get(), persist=True, verbose=False)
                    
                    if results[0].boxes is not None:
                        for box in results[0].boxes:
                            if self.blur_mode.get() == "body" and int(box.cls[0]) != 0:
                                continue
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            x1, y1 = max(0, x1), max(0, y1)
                            x2, y2 = min(width, x2), min(height, y2)
                            
                            if x2 > x1 and y2 > y1:
                                roi = frame[y1:y2, x1:x2]
                                roi = cv2.GaussianBlur(roi, (blur_k, blur_k), 30)
                                frame[y1:y2, x1:x2] = roi
                    
                    out.write(frame)
                    frame_count += 1
                    
                    if frame_count % 10 == 0:
                        progress = min(int((frame_count / total_frames) * 100), 100) if total_frames > 0 else 0
                        self.progress_label.configure(text=f"Frame {frame_count} — {progress}%")
                        self.show_frame(frame)
                        self.window.update()
                        
                except Exception:
                    out.write(frame)
                    frame_count += 1
                    continue
            
            cap.release()
            out.release()
            
            self.progress_label.configure(text="Merging audio...")
            self.window.update()
            
            try:
                temp_audio = tempfile.NamedTemporaryFile(suffix=".aac", delete=False).name
                subprocess.run(["ffmpeg", "-i", self.video_path, "-vn", "-acodec", "copy", temp_audio, "-y"], check=True, capture_output=True)
                subprocess.run(["ffmpeg", "-i", temp_video, "-i", temp_audio, "-c:v", "copy", "-c:a", "aac", "-shortest", output, "-y"], check=True, capture_output=True)
                os.unlink(temp_audio)
            except:
                shutil.copy2(temp_video, output)
            
            os.unlink(temp_video)
            
            self.btn_process.configure(state="normal", text="Start Processing")
            self.progress_label.configure(text="")
            self.status_label.configure(text=f"Done! {frame_count} frames with audio.", text_color="#4CAF50")
            messagebox.showinfo("Complete", f"Video processed!\n{frame_count} frames\nAudio preserved\nOutput: {os.path.basename(output)}")
            
        except Exception as e:
            self.btn_process.configure(state="normal", text="Start Processing")
            self.progress_label.configure(text="")
            self.status_label.configure(text=f"Error: {str(e)[:50]}", text_color="#FF5555")
            messagebox.showerror("Error", f"Processing failed:\n{e}")
    
    def run(self): self.window.mainloop()

if __name__ == "__main__":
    app = VideoFaceBlurApp()
    app.run()
