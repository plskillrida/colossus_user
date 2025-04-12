import os
import torch
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# 📁 Paths and config
train_dir = "../dataset/train"
val_dir = "../dataset/test"
img_size = 224
batch_size = 32
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ Using device: {device}")

# 🌀 Image transforms
common_transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# 📦 Dataset loaders
train_dataset = datasets.ImageFolder(train_dir, transform=common_transform)
val_dataset = datasets.ImageFolder(val_dir, transform=common_transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
class_names = train_dataset.classes
print("Classes:", class_names)

# 🧠 Load pretrained feature extractor (MobileNetV2 without classifier)
model = models.mobilenet_v2(pretrained=True).features.to(device)
model.eval()

# 🔍 Feature extraction
def extract_features(dataloader):
    features = []
    labels = []

    with torch.no_grad():
        for inputs, targets in tqdm(dataloader, desc="Extracting features"):
            inputs = inputs.to(device)
            x = model(inputs)                      # shape: [batch, 1280, 7, 7]
            x = torch.nn.functional.adaptive_avg_pool2d(x, (1, 1))  # -> [batch, 1280, 1, 1]
            x = x.view(x.size(0), -1)              # -> [batch, 1280]
            features.append(x.cpu().numpy())
            labels.append(targets.numpy())

    return np.concatenate(features), np.concatenate(labels)

train_feats, train_labels = extract_features(train_loader)
val_feats, val_labels = extract_features(val_loader)

# 📏 Scale features
scaler = StandardScaler()
train_feats = scaler.fit_transform(train_feats)
val_feats = scaler.transform(val_feats)

# 🧪 Train and evaluate SVM
clf = SVC(kernel='linear', C=1.0)
clf.fit(train_feats, train_labels)
val_preds = clf.predict(val_feats)
acc = accuracy_score(val_labels, val_preds)
print(f"🎯 Sklearn SVM Validation Accuracy: {acc:.4f}")

import joblib

# 💾 Save the trained scaler and SVM model
joblib.dump(scaler, "scaler.pkl")
joblib.dump(clf, "svm_model.pkl")

print("💾 Scaler and SVM model saved as 'scaler.pkl' and 'svm_model.pkl'")
