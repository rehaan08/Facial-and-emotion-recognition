# Facial-and-emotion-recognition
Facial recognition software

Here's a 250-word write-up of how the code works:

---

## How the Facial Recognition System Works

This facial recognition system is built in Python using several libraries working together to deliver real-time face detection, emotion analysis, and identity recognition through a live webcam feed.

When the program starts, it first scans the `known_faces` folder and loads any reference images found there, storing each person's name and image path for later comparison. It then automatically searches through available camera indices to find a working webcam, making it robust across different hardware configurations.

The system uses two separate face detection mechanisms working in parallel. The first is OpenCV's Haar Cascade classifier, a classical computer vision algorithm that runs on every single frame. This is extremely fast and lightweight, allowing the bounding box to track and follow faces in real time with virtually no lag. The second is DeepFace, a deep learning framework that runs in a background thread every ten frames. DeepFace handles the heavier tasks of emotion analysis and identity recognition without blocking the video feed.

Threading is the key architectural decision that makes the system smooth. By running DeepFace in a separate thread, the main loop continues capturing and displaying frames at 60 frames per second regardless of how long DeepFace takes to process. A `processing` flag prevents multiple threads from firing simultaneously.

The display is handled by PyGame, which renders each camera frame and overlays the bounding box, name label, and emotion label on top. The box color changes dynamically based on the detected emotion, giving immediate visual feedback. The frame is horizontally flipped to create a natural mirror-like experience for the user.
