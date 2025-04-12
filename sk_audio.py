import os
import time
import cv2
import torch
import numpy as np
import mediapipe as mp
from torchvision import models, transforms
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import joblib
from openai import OpenAI

# 💬 LLM client setup
client = OpenAI(
    base_url="https://api.studio.nebius.com/v1/",
    api_key="eyJhbGciOiJIUzI1NiIsImtpZCI6IlV6SXJWd1h0dnprLVRvdzlLZWstc0M1akptWXBvX1VaVkxUZlpnMDRlOFUiLCJ0eXAiOiJKV1QifQ.eyJzdWIiOiJnb29nbGUtb2F1dGgyfDExNzYyNzU5MzIwODYyNDA2MTAwNiIsInNjb3BlIjoib3BlbmlkIG9mZmxpbmVfYWNjZXNzIiwiaXNzIjoiYXBpX2tleV9pc3N1ZXIiLCJhdWQiOlsiaHR0cHM6Ly9uZWJpdXMtaW5mZXJlbmNlLmV1LmF1dGgwLmNvbS9hcGkvdjIvIl0sImV4cCI6MTkwMjA3NjI3NywidXVpZCI6Ijc0YmY0ZDAyLTg4MDgtNDRlMC1hYjIxLTkwNzZhM2QxMmNmMyIsIm5hbWUiOiJDb2xvc3N1cyIsImV4cGlyZXNfYXQiOiIyMDMwLTA0LTEwVDE4OjMxOjE3KzAwMDAifQ.uN8CdREKmyGVMOfzHYIq2TPwYu3s7lXt0BSPNky5hxo"  # Replace with your actual Nebius API key
)

os.environ["PATH"] += os.pathsep + r"C:\ffmpeg-7.1.1-essentials_build\bin"

# ✅ Load trained models
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

feature_extractor = models.mobilenet_v2(pretrained=True).features.to(device)
feature_extractor.eval()

scaler = joblib.load("scaler.pkl")
clf = joblib.load("svm_model.pkl")
class_names = ["A", "E", "F", "H", "Hello", "I love you", "N", "R", "Victory"]

preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# 🎥 Webcam setup
cap = cv2.VideoCapture(0)
print("🔴 Starting real-time prediction... Hold gesture for 2s to capture. 'q' to quit.")

box_size = 224
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
x1 = frame_width // 2 - box_size // 2
y1 = frame_height // 2 - box_size // 2
x2 = x1 + box_size
y2 = y1 + box_size

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=1,
                       min_detection_confidence=0.5,
                       min_tracking_confidence=0.5)

gesture_start_time = None
no_hand_start_time = None
captured_text = []
current_prediction = ""

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
    result = hands.process(frame_rgb)
    frame_display = cv2.flip(frame, 1)

    cv2.rectangle(frame_display, (x1, y1), (x2, y2), (0, 255, 0), 2)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            h, w, _ = frame.shape
            wrist = hand_landmarks.landmark[0]
            wrist_x = int(wrist.x * w)
            wrist_y = int(wrist.y * h)

            if x1 <= wrist_x <= x2 and y1 <= wrist_y <= y2:
                if gesture_start_time is None:
                    gesture_start_time = time.time()
                elif time.time() - gesture_start_time >= 2:
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
                    captured_text.append(label)
                    current_prediction = label
                    print(f"[Captured] {label}")
                    gesture_start_time = None

                no_hand_start_time = None
            else:
                gesture_start_time = None

            cv2.circle(frame_display, (wrist_x, wrist_y), 5, (255, 0, 0), -1)

    else:
        if no_hand_start_time is None:
            no_hand_start_time = time.time()
        elif time.time() - no_hand_start_time >= 3:
            captured_text.append(" ")
            current_prediction = "<space>"
            print("[Captured] <space>")
            no_hand_start_time = None
        gesture_start_time = None

    cv2.putText(
        frame_display,
        f"Current: {current_prediction}",
        (10, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 255, 255),
        2,
        cv2.LINE_AA
    )

    sentence = ''.join(captured_text)[-40:]
    cv2.putText(
        frame_display,
        f"Sentence: {sentence}",
        (10, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )

    cv2.imshow("Webcam - Real-Time Classification", frame_display)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()

# 📝 Save raw text
raw_text = ''.join(captured_text)
with open("captured_text.txt", "w") as f:
    f.write(raw_text)

print(f"\n📝 Raw captured text:\n{raw_text}")

# 🤖 Rephrase using LLaMA 3.3 70B
print("\n🧠 Sending to LLaMA 3.3 70B for rephrasing...")
response = client.chat.completions.create(
    model="meta-llama/Llama-3.3-70B-Instruct",
    temperature=0.5,
    messages=[
        {"role": "system", "content": "You are a helpful assistant who rewrites jumbled or informal text into clear, readable English. If there are too many 'Ns' or 'Rs' it means, it's trying to tell the name Afreen, so use the name Afreen. In final, just output the rephrased text."},
        {"role": "user", "content": raw_text}
    ]
)

rephrased = response.choices[0].message.content.strip()
print("\n🪄 Rephrased Output:\n")
print(rephrased)

# 💾 Save rephrased text
with open("rephrased_output.txt", "w") as f:
    f.write(rephrased)

# 🔊 Speak using ElevenLabs (client-based syntax)
print("🗣️ Speaking the rephrased text using ElevenLabs...")

from elevenlabs import ElevenLabs, play

# Initialize the ElevenLabs client
tts_client = ElevenLabs(api_key="sk_a8cd37c9fbe82a7c7fa9132647f6cc4a2597f333518881e8")  # Replace this with your actual ElevenLabs API key

# Generate audio
audio = tts_client.generate(
    text=rephrased,
    voice="Fin",  # You can change the voice to "Rachel", "Bella", "Antoni", etc.
    model="eleven_monolingual_v1"
)

# Play audio
play(audio)
