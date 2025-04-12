import cv2
import torch
import numpy as np
import time
import mediapipe as mp
from torchvision import models, transforms
from sklearn.svm import SVC
import joblib

# 🎯 Load SVM model & scaler
clf = joblib.load("svm_model.pkl")
scaler = joblib.load("scaler.pkl")
class_names = ["A", "E", "F", "H", "Hello", "I love you", "N", "R", "Victory"]

# ⚙️ Setup device and MobileNetV2 for feature extraction
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
feature_extractor = models.mobilenet_v2(pretrained=True).features.to(device).eval()

# 📐 Image transforms
preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# 🤚 MediaPipe setup
mp_hands = mp.solutions.hands
hands = mp_hands_
