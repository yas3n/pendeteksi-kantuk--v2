import cv2
import numpy as np
from ultralytics import YOLO

# Muat model YOLOv8
model = YOLO('best.pt')  # Ganti dengan path model Anda

# Fungsi untuk mendeteksi kantuk
def detect_sleepiness(frame):
    results = model(frame)
    for result in results:
        boxes = result.boxes.xyxy  # Mendapatkan bounding boxes
        confs = result.boxes.conf  # Mendapatkan confidence
        for box, conf in zip(boxes, confs):
            if conf > 0.5:  # Ambang batas confidence
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, 'Sleepy', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return frame

# Menggunakan kamera untuk mendeteksi kantuk
cap = cv2.VideoCapture(0)  # Ganti dengan path video jika diperlukan

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = detect_sleepiness(frame)
    cv2.imshow('Pendeteksi Kantuk', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()