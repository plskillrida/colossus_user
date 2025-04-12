import cv2
import torch
import numpy as np
import mediapipe as mp
from torchvision import models, transforms
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import joblib

# ✅ Load trained models
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load feature extractor (MobileNetV2 without classifier)
feature_extractor = models.mobilenet_v2(pretrained=True).features.to(device)
feature_extractor.eval()

# Load scaler and SVM
scaler = joblib.load("scaler.pkl")
clf = joblib.load("svm_model.pkl")
class_names = ["A", "E", "F", "H", "Hello", "I love you", "N", "R", "Victory"]

# Preprocessing for hand image
preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# 🎥 Capture webcam
cap = cv2.VideoCapture(0)
print("🔴 Starting real-time prediction... Press 'q' to quit.")

# Green box setup
box_size = 224
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
x1 = frame_width // 2 - box_size // 2
y1 = frame_height // 2 - box_size // 2
x2 = x1 + box_size
y2 = y1 + box_size

# 📦 MediaPipe hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=1,
                       min_detection_confidence=0.5,
                       min_tracking_confidence=0.5)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip and convert color for MediaPipe
    frame_rgb = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
    result = hands.process(frame_rgb)
    frame_display = cv2.flip(frame, 1)  # flip again for display

    # Draw green box
    cv2.rectangle(frame_display, (x1, y1), (x2, y2), (0, 255, 0), 2)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # Get wrist landmark as an anchor point (landmark 0)
            h, w, _ = frame.shape
            wrist = hand_landmarks.landmark[0]
            wrist_x = int(wrist.x * w)
            wrist_y = int(wrist.y * h)

            # Check if wrist is inside green box
            if x1 <= wrist_x <= x2 and y1 <= wrist_y <= y2:
                roi = frame[y1:y2, x1:x2]
                img = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
                img_tensor = preprocess(img).unsqueeze(0).to(device)

                with torch.no_grad():
                    features = feature_extractor(img_tensor)
                    features = torch.nn.functional.adaptive_avg_pool2d(features, (1, 1))
                    features = features.view(features.size(0), -1).cpu().numpy()

                features_scaled = scaler.transform(features)
                prediction = clf.predict(features_scaled)[0]
                label = class_names[prediction]

                cv2.putText(frame_display, f"Prediction: {label}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Optional: Draw a small circle at wrist
            cv2.circle(frame_display, (wrist_x, wrist_y), 5, (255, 0, 0), -1)

    cv2.imshow("Webcam - Real-Time Classification", frame_display)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
