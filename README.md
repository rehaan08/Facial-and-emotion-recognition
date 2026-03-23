#  Facial Recognition & Emotion Detection

A real-time facial recognition and emotion detection system built with Python. 
The system detects faces, identifies known individuals, and displays their 
emotional state live through a webcam feed.

![Python](https://img.shields.io/badge/Python-3.10-blue)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green)
![DeepFace](https://img.shields.io/badge/DeepFace-latest-orange)

---

##  Features

-  Real-time face detection and tracking
-  Facial recognition of known individuals
-  Emotion detection (Happy, Sad, Angry, Neutral, Surprise, Fear, Disgust)
-  Color-coded bounding box based on detected emotion
-  Smooth, fluid face tracking
-  Multi-threaded for performance

---

##  Built With

- **Python 3.10**
- **OpenCV** — Real-time face detection via Haar Cascade
- **DeepFace** — Emotion analysis and facial recognition
- **PyGame** — Live video display
- **NumPy** — Frame processing

---

##  Installation

**1. Clone the repository:**
```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
cd YOUR_REPO_NAME
```

**2. Create a virtual environment:**
```bash
python3.10 -m venv .venv
source .venv/bin/activate
```

**3. Install dependencies:**
```bash
pip install opencv-python-headless deepface pygame numpy
```

---

##  How to Run

**1. Add reference images to the `known_faces` folder:**
```
known_faces/
├── john.jpg
├── alice.jpg
└── rehaan.jpg
```

**2. Run the program:**
```bash
python face.py
```

**3. Close the window to quit**

---

##  Emotion Color Guide

| Emotion | Color |
|---|---|
|  Happy | Green |
|  Sad | Blue |
|  Angry | Red |
|  Fear | Purple |
|  Surprise | Orange |
|  Disgust | Dark Green |
|  Neutral | Grey |

---

##  Project Structure
```
├── face.py              # Main application
├── known_faces/         # Reference images for recognition
│   └── rehaan.jpg
├── camera_test.py       # Camera diagnostic tool
└── README.md
```

---

##  Troubleshooting

**Camera not opening?**
- The system auto-detects your camera index
- Make sure PyCharm/Terminal has camera permissions in System Settings

**Module not found?**
- Make sure you're using Python 3.10
- Run `pip install opencv-python-headless deepface pygame numpy`

**Recognition not working?**
- Use a clear, well-lit JPEG photo
- Face should be looking straight at camera
- Convert HEIC photos to JPG using Preview on Mac

---

##  How It Works

The system uses two detection mechanisms running in parallel:
- **Haar Cascade** runs every frame for instant box tracking
- **DeepFace** runs in a background thread every 10 frames for 
emotion and recognition without blocking the video feed

---

##  Author

**Rehaan**  
Built with Python on MacBook Air M2
