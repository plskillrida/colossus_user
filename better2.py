import cv2
import torch
import numpy as np
import mediapipe as mp
import time
from torchvision import models, transforms
from sklearn.svm import SVC
import joblib
import os

# 📐 Image transforms
preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ⚙️ Setup device and MobileNetV2 for feature extraction
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
feature_extractor = models.mobilenet_v2(pretrained=True).features.to(device).eval()

# 🎯 Load SVM model & scaler
clf = joblib.load("svm_model.pkl")
scaler = joblib.load("scaler.pkl")
class_names = ["A", "E", "F", "H", "Hello", "I love you", "N", "R", "Victory"]

# 🤚 MediaPipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=1,
                       min_detection_confidence=0.7,
                       min_tracking_confidence=0.5)

# 🎬 Webcam
cap = cv2.VideoCapture(0)

# 🧠 Sentence state
last_prediction = None
last_written_label = ""
last_detected_time = 0
no_hand_start_time = None
sentence = ""
output_file = "detected_text.txt"
blink_start_time = None
blink_duration = 0.3  # seconds

# 💡 Check if imshow is available (fallback if running in headless)
can_show = True
try:
    cv2.namedWindow("Test")
    cv2.destroyWindow("Test")
except cv2.error:
    can_show = False
    print("⚠️ GUI backend not supported. Display window disabled. Output will still be saved.")

print("🟢 Real-time gesture detection started. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)
    h, w, _ = frame.shape

    detected = False
    label = ""
    x1 = y1 = x2 = y2 = 0

    if results.multi_hand_landmarks:
        no_hand_start_time = None
        for hand_landmarks in results.multi_hand_landmarks:
            detected = True
            x_min, y_min = w, h
            x_max = y_max = 0

            for lm in hand_landmarks.landmark:
                x, y = int(lm.x * w), int(lm.y * h)
                x_min = min(x_min, x)
                y_min = min(y_min, y)
                x_max = max(x_max, x)
                y_max = max(y_max, y)

            margin = 20
            x1 = max(0, x_min - margin)
            y1 = max(0, y_min - margin)
            x2 = min(w, x_max + margin)
            y2 = min(h, y_max + margin)

            roi = frame[y1:y2, x1:x2]
            if roi.size == 0:
                continue

            input_img = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
            input_tensor = preprocess(input_img).unsqueeze(0).to(device)

            with torch.no_grad():
                features = feature_extractor(input_tensor)
                features = torch.nn.functional.adaptive_avg_pool2d(features, (1, 1))
                features = features.view(features.size(0), -1).cpu().numpy()

            features_scaled = scaler.transform(features)
            prediction = clf.predict(features_scaled)[0]
            label = class_names[prediction]

            current_time = time.time()
            if label == last_prediction:
                if current_time - last_detected_time > 2:
                    if label != last_written_label:
                        print(f"[📝 Recorded]: {label}")
                        sentence += label + " "
                        with open(output_file, "w") as f:
                            f.write(sentence.strip())
                        blink_start_time = current_time
                        last_written_label = label
                    last_detected_time = current_time
            else:
                last_prediction = label
                last_detected_time = current_time

    else:
        if no_hand_start_time is None:
            no_hand_start_time = time.time()
        elif time.time() - no_hand_start_time > 3:
            if not sentence.endswith(" "):
                print("⏸️ Gap detected: adding space.")
                sentence += " "
                with open(output_file, "w") as f:
                    f.write(sentence.strip())
            no_hand_start_time = time.time()

        last_prediction = None
        last_written_label = ""

    if can_show:
        # 🧃 Overlay flash box
        overlay = frame.copy()
        if blink_start_time and time.time() - blink_start_time < blink_duration:
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 255, 255), -1)
        elif detected:
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), -1)
        else:
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 255), -1)

        alpha = 0.3
        frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

        if x2 > x1 and y2 > y1:
            color = (0, 255, 0) if detected else (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        if detected and label:
            cv2.putText(frame, f"Prediction: {label}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        cv2.putText(frame, f"Sentence: {sentence.strip()}", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        try:
            cv2.imshow("Sign Language Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        except cv2.error:
            print("⚠️ Could not open window during runtime.")
            can_show = False
    else:
        # Optionally save frames (comment out if not needed)
        # cv2.imwrite("last_frame.jpg", frame)
        time.sleep(0.05)  # Small delay to reduce CPU usage

# 🧹 Clean up
cap.release()
if can_show:
    cv2.destroyAllWindows()
