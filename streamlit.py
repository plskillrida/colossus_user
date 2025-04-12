import streamlit as st
import cv2
import torch
import numpy as np
import mediapipe as mp
from torchvision import models, transforms
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import joblib
from openai import OpenAI
from elevenlabs import ElevenLabs, play, save
import tempfile
import time
from PIL import Image

# Config
st.set_page_config(page_title="Gesture to Speech", layout="centered")

# Title
st.title("🤟 Sign Language to Speech")


# Load models
@st.cache_resource
def load_models():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feature_extractor = models.mobilenet_v2(pretrained=True).features.to(device).eval()
    scaler = joblib.load("scaler.pkl")
    clf = joblib.load("svm_model.pkl")
    return feature_extractor, scaler, clf, device

for key, default in {
    "stop_capture": False,
    "capturing": False,
    "captured_text": "",
    "rephrased_text": "",
    "audio_bytes": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default
        
feature_extractor, scaler, clf, device = load_models()
class_names = ["A", "E", "F", "H", "Hello", "I love you", "N", "R", "Victory"]

# Preprocessing
preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# OpenAI LLaMA 3.3 client via Nebius
openai_client = OpenAI(
    base_url="https://api.studio.nebius.com/v1/",
    api_key="YOUR_NEBIUS_API_KEY"
)

# ElevenLabs client
tts_client = ElevenLabs(api_key="YOUR_ELEVENLABS_API_KEY")

# Webcam Detection
def run_detection():
    cap = cv2.VideoCapture(0)
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=False,
                           max_num_hands=1,
                           min_detection_confidence=0.5,
                           min_tracking_confidence=0.5)

    gesture_start_time = None
    no_hand_start_time = None
    captured_text = []
    current_prediction = ""

    box_size = 224
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    x1 = frame_width // 2 - box_size // 2
    y1 = frame_height // 2 - box_size // 2
    x2 = x1 + box_size
    y2 = y1 + box_size

    placeholder = st.empty()

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
                        gesture_start_time = None
                        st.info(f"Captured: {label}")

                    no_hand_start_time = None
                else:
                    gesture_start_time = None

        else:
            if no_hand_start_time is None:
                no_hand_start_time = time.time()
            elif time.time() - no_hand_start_time >= 3:
                captured_text.append(" ")
                current_prediction = "<space>"
                no_hand_start_time = None
                st.info("Captured: <space>")
            gesture_start_time = None

        sentence = ''.join(captured_text)[-40:]
        cv2.putText(frame_display, f"Sentence: {sentence}", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        placeholder.image(frame_display, channels="BGR")

        # Check stop flag
        if st.session_state.get("stop_capture", False):
            break

    cap.release()
    cv2.destroyAllWindows()
    return ''.join(captured_text)

# Rephrase
def rephrase_text(raw_text):
    st.subheader("🧠 Rephrasing with LLaMA 3.3")
    response = openai_client.chat.completions.create(
        model="meta-llama/Llama-3.3-70B-Instruct",
        temperature=0.5,
        messages=[
            {"role": "system", "content": "You are a helpful assistant who rewrites jumbled or informal text into clear, readable English. If there are too many 'Ns' or 'Rs' it means it's trying to say the name Afreen, so use the name Afreen. Just output the rephrased text."},
            {"role": "user", "content": raw_text}
        ]
    )
    return response.choices[0].message.content.strip()

# Speak
def speak_text(text):
    st.subheader("🗣️ Speaking the Text")
    audio = tts_client.generate(
        text=text,
        voice="Fin",
        model="eleven_monolingual_v1"
    )
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_file:
        save(audio, tmp_file.name)
        audio_bytes = open(tmp_file.name, 'rb').read()
    return audio_bytes



# Init state
if "stop_capture" not in st.session_state:
    st.session_state.stop_capture = False

# Init state
if "stop_capture" not in st.session_state:
    st.session_state.stop_capture = False
if "capturing" not in st.session_state:
    st.session_state.capturing = False
if "captured_text" not in st.session_state:
    st.session_state.captured_text = ""

# Start capture
if st.button("📸 Start Webcam & Capture"):
    st.session_state.stop_capture = False
    st.session_state.capturing = True
    st.rerun()  # Restart the script with updated state

# Stop capture
if st.button("⏹️ Stop Capture"):
    st.session_state.stop_capture = True
    st.session_state.capturing = False
    st.rerun()

# Perform capture if in capturing mode
if st.session_state.capturing:
    st.session_state.captured_text = run_detection()
    st.session_state.capturing = False
    st.rerun()

# After capture, rephrase and speak
if st.session_state.captured_text and not st.session_state.rephrased_text:
    raw = st.session_state.captured_text
    st.success("Captured Raw Text:")
    st.code(raw)
    st.session_state.rephrased_text = rephrase_text(raw)
    st.rerun()  # Rerun to separate rephrasing from speaking

if st.session_state.rephrased_text and not st.session_state.audio_bytes:
    st.success("🪄 Rephrased Output:")
    st.write(st.session_state.rephrased_text)
    st.session_state.audio_bytes = speak_text(st.session_state.rephrased_text)
    st.rerun()

if st.session_state.audio_bytes:
    st.audio(st.session_state.audio_bytes, format="audio/mp3")

    # Reset session state for next round
    st.session_state.captured_text = ""
    st.session_state.rephrased_text = ""
    st.session_state.audio_bytes = None


