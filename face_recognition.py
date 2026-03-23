import cv2
import os
import time
import numpy as np
import pygame
from deepface import DeepFace
import threading

KNOWN_FACES_DIR = "known_faces"
TOLERANCE = 0.6
os.environ["OPENCV_AVFOUNDATION_SKIP_AUTH"] = "1"
DISPLAY_WIDTH = 640
DISPLAY_HEIGHT = 480

EMOTION_COLORS = {
    "HAPPY": (0, 255, 0),
    "SAD": (0, 0, 255),
    "ANGRY": (255, 0, 0),
    "FEAR": (128, 0, 128),
    "SURPRISE": (255, 165, 0),
    "DISGUST": (0, 128, 0),
    "NEUTRAL": (200, 200, 200),
}

emotion_label = "Scanning..."
name_label = "Scanning..."
processing = False
call_count = 0
box_color = (200, 200, 200)

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


def load_known_faces():
    known_faces = []
    if not os.path.exists(KNOWN_FACES_DIR):
        os.makedirs(KNOWN_FACES_DIR)
        print(f"Created '{KNOWN_FACES_DIR}' folder.")
        return known_faces
    for filename in os.listdir(KNOWN_FACES_DIR):
        if filename.endswith((".jpg", ".jpeg", ".png")):
            name = os.path.splitext(filename)[0]
            path = os.path.join(KNOWN_FACES_DIR, filename)
            known_faces.append({"name": name, "path": path})
            print(f"Loaded: {name}")
    if not known_faces:
        print("No images found in known_faces folder!")
    return known_faces


def find_camera():
    print("Searching for camera...")
    for i in range(5):
        cap = cv2.VideoCapture(i, cv2.CAP_AVFOUNDATION)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                print(f"Found camera at index {i}")
                cap.release()
                return i
        cap.release()
    return 0


def deepface_process(frame, known_faces):
    """Runs in background — only does emotion + recognition."""
    global emotion_label, name_label, processing, call_count, box_color
    call_count += 1

    try:
        results = DeepFace.analyze(
            frame,
            actions=["emotion"],
            enforce_detection=False,
            detector_backend="opencv",
            silent=True
        )
        if results:
            emotion = results[0]["dominant_emotion"].upper()
            emotion_label = emotion
            box_color = EMOTION_COLORS.get(emotion, (255, 255, 255))

        if call_count % 5 == 0 and known_faces:
            matched = False
            for person in known_faces:
                try:
                    result = DeepFace.verify(
                        img1_path=frame,
                        img2_path=person["path"],
                        enforce_detection=False,
                        model_name="VGG-Face",
                        silent=True
                    )
                    if result["verified"] and result["distance"] < TOLERANCE:
                        name_label = person["name"].upper()
                        print(f"✅ Recognized: {name_label}")
                        matched = True
                        break
                except Exception:
                    pass
            if not matched:
                name_label = "Unknown"

    except Exception:
        pass
    finally:
        processing = False


def draw_text(surface, text, pos, font, color=(0, 255, 0)):
    text_surface = font.render(text, True, color)
    surface.blit(text_surface, pos)


def run_recognition():
    global processing, box_color

    known_faces = load_known_faces()
    camera_index = find_camera()

    cap = cv2.VideoCapture(camera_index, cv2.CAP_AVFOUNDATION)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    print("Warming up camera...")
    time.sleep(1)
    for _ in range(3):
        cap.read()

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Camera ready!")

    pygame.init()
    screen = pygame.display.set_mode((DISPLAY_WIDTH, DISPLAY_HEIGHT))
    pygame.display.set_caption("Facial Recognition")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("Arial", 24)
    small_font = pygame.font.SysFont("Arial", 16)

    print("Running... Close window to quit.")

    frame_count = 0

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                cap.release()
                pygame.quit()
                return

        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.flip(frame, 1)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=4,
            minSize=(60, 60)
        )

        if frame_count % 10 == 0 and not processing:
            processing = True
            small_frame = cv2.resize(frame, (320, 240))
            thread = threading.Thread(
                target=deepface_process,
                args=(small_frame.copy(), known_faces),
                daemon=True
            )
            thread.start()

        frame_count += 1

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (DISPLAY_WIDTH, DISPLAY_HEIGHT))
        surf = pygame.surfarray.make_surface(frame_resized.transpose(1, 0, 2))
        screen.blit(surf, (0, 0))

        # ── Draw Face Boxes (instant, from Haar)
        for (x, y, w, h) in faces:
            # Draw box with emotion color
            pygame.draw.rect(screen, box_color, (x, y, w, h), 3)

            # Corner accents
            corner = 15
            for cx, cy in [(x, y), (x + w, y), (x, y + h), (x + w, y + h)]:
                dx = -corner if cx == x + w else corner
                dy = -corner if cy == y + h else corner
                pygame.draw.line(screen, box_color, (cx, cy), (cx + dx, cy), 3)
                pygame.draw.line(screen, box_color, (cx, cy), (cx, cy + dy), 3)

            # Name label
            name_bg_w = max(len(name_label) * 13, w)
            pygame.draw.rect(screen, (0, 0, 0),
                             (x, max(0, y - 60), name_bg_w, 28))
            draw_text(screen, name_label,
                      (x + 5, max(0, y - 58)), font, box_color)

            # Emotion label
            emotion_bg_w = max(len(emotion_label) * 13, w)
            pygame.draw.rect(screen, (0, 0, 0),
                             (x, max(0, y - 30), emotion_bg_w, 28))
            draw_text(screen, emotion_label,
                      (x + 5, max(0, y - 28)), font, box_color)

        draw_text(screen, "Close window to quit",
                  (10, DISPLAY_HEIGHT - 25), small_font, (200, 200, 200))

        pygame.display.flip()
        clock.tick(60)

    cap.release()
    pygame.quit()


if __name__ == "__main__":
    run_recognition()
