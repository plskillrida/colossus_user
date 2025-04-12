import os
import time
import cv2
import torch
import numpy as np
import mediapipe as mp
from torchvision import models, transforms
from torchvision.models import MobileNet_V2_Weights
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import joblib
from openai import OpenAI
from elevenlabs import ElevenLabs, play, save

# 🛠️ Set FFMPEG path for ElevenLabs (Windows users)
os.environ["PATH"] += os.pathsep + r"C:\ffmpeg-7.1.1-essentials_build\bin"

# 💬 LLM client setup (LLaMA)
llm_client = OpenAI(
    base_url="https://api.studio.nebius.com/v1/",
    api_key=""
)

# 🔊 ElevenLabs client setup
eleven_client = ElevenLabs(
    api_key=""
)

# ✅ Load models
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

weights = MobileNet_V2_Weights.DEFAULT
feature_extractor = models.mobilenet_v2(weights=weights).features.to(device)
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
if not cap.isOpened():
    raise IOError("Cannot open webcam")

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

print("🔴 Starting real-time prediction... Hold gesture for 2s to capture. 'q' to quit.")

box_size = 300

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
    h, w, _ = frame.shape

    live_prediction = ""

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            wrist = hand_landmarks.landmark[0]
            wrist_x = int(wrist.x * w)
            wrist_y = int(wrist.y * h)

            x1 = max(0, wrist_x - box_size // 2)
            y1 = max(0, wrist_y - box_size // 2)
            x2 = min(w, wrist_x + box_size // 2)
            y2 = min(h, wrist_y + box_size // 2)

            roi = frame[y1:y2, x1:x2]
            img = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
            img_tensor = preprocess(img).unsqueeze(0).to(device)

            with torch.no_grad():
                features = feature_extractor(img_tensor)
                features = torch.nn.functional.adaptive_avg_pool2d(features, (1, 1))
                features = features.view(features.size(0), -1).cpu().numpy()

            features_scaled = scaler.transform(features)
            prediction = clf.predict(features_scaled)[0]
            live_prediction = class_names[prediction]

            if gesture_start_time is None:
                gesture_start_time = time.time()
            elif time.time() - gesture_start_time >= 7:
                captured_text.append(live_prediction)
                current_prediction = live_prediction
                print(f"[Captured] {live_prediction}")
                gesture_start_time = None

            no_hand_start_time = None

            cv2.rectangle(frame_display, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(frame_display, (wrist_x, wrist_y), 5, (255, 0, 0), -1)

    else:
        if no_hand_start_time is None:
            no_hand_start_time = time.time()
        elif time.time() - no_hand_start_time >= 9:
            captured_text.append(" ")
            current_prediction = "<space>"
            print("[Captured] <space>")
            no_hand_start_time = None
        gesture_start_time = None

    # ✨ Display predictions and sentence
    cv2.putText(frame_display, f"Captured: {current_prediction}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)

    margin = int(w * 0.1)
    text = f"Live: {live_prediction}"
    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)[0]
    text_x = w - text_size[0] - margin
    cv2.putText(frame_display, text, (text_x, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (100, 255, 100), 2)

    sentence = ''.join(captured_text)[-60:]
    cv2.putText(frame_display, f"Sentence: {sentence}", (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    cv2.imshow("Webcam - Real-Time Classification", frame_display)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 🔚 Cleanup
cap.release()
cv2.destroyAllWindows()

# 📝 Save raw gesture text
raw_text = ''.join(captured_text)
with open("captured_text.txt", "w") as f:
    f.write(raw_text)

print(f"\n📝 Raw captured text:\n{raw_text}")

# 🤖 LLaMA rephrasing
print("\n🧠 Sending to LLaMA 3.3 70B for rephrasing...")
response = llm_client.chat.completions.create(
    model="meta-llama/Llama-3.3-70B-Instruct",
    temperature=0.5,
    messages=[
        {"role": "system", "content": "You are a helpful assistant who rewrites jumbled or informal text into clear, readable English. If there are too many 'Ns' or 'Rs' it means it's trying to say the name Afreen, so interpret it that way. In the end, just return the rephrased text."},
        {"role": "user", "content": raw_text}
    ]
)

rephrased = response.choices[0].message.content
print("\n🪄 Rephrased Output:\n")
print(rephrased)

# 💾 Save rephrased version
with open("rephrased_output.txt", "w") as f:
    f.write(rephrased)

# 🔊 ElevenLabs TTS (Fin voice)
print("\n🔊 Generating audio with ElevenLabs...")

audio = eleven_client.generate(
    text=rephrased,
    voice="Fin"
)

play(audio)
