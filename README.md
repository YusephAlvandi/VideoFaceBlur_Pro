Video Face Blur Pro

An AI-powered tool for video blurring in two different modes (Face-Only or Full-Body), based on the level of privacy they need. Two state-of-the-art convolutional deep learning models (YOLOv8m for Full-Body and YOLOv8n-face for Face-Only) have been used to develop this tool. Audio is preserved via ffmpeg and through frame-by-frame processing of the video, tracking reduces flickering in the output and makes it smoother.

The YOLO models have significantly better performance compared to older classical models (Haar Cascade or Caffe SSD) and consequently the corresponding tools have better fidelity and consistency than those using older models, such as my older version (https://github.com/YusephAlvandi/VideoFaceBlur).

Different angles of a human face or the motion of the person do not lead to missing frames or missed blurring.
It works completely offline.
Built with Python, OpenCV, CustomTkinter, and Ultralytics.


HOW TO RUN

python3.11 video_face_blur.py

1. Open a video file
2. Select blur mode (Face Only or Full Body)
3. Adjust blur strength and confidence if needed
4. Click Start Processing


DEPENDENCIES

Python libraries:
pip install ultralytics opencv-python pillow customtkinter numpy

System tools:
sudo apt install ffmpeg

Model files (auto-downloaded on first run):
- yolov8n-face.pt (6 MB)
- yolov8m.pt (48 MB)


AUTHOR

Yuseph Alvandi
PhD in Optics and Laser Physics
Python Developer and Image Processing Specialist

GitHub: https://github.com/YusephAlvandi


LICENSE

MIT License
