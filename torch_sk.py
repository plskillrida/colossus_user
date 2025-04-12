import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# 📁 Paths and configs
train_dir = "../dataset/train"
val_dir = "../dataset/test"
img_size = 224
batch_size = 32
num_epochs = 5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 🌀 Transforms
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(img_size),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], 
                         [0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], 
                         [0.229, 0.224, 0.225])
])

# 📦 Datasets and loaders
train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
val_dataset = datasets.ImageFolder(val_dir, transform=val_transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
class_names = train_dataset.classes
print("Classes:", class_names)

# 🧠 Model setup
base_model = models.mobilenet_v2(pretrained=True)
for param in base_model.features.parameters():
    param.requires_grad = False

# Replace classifier
base_model.classifier = nn.Sequential(
    nn.Linear(base_model.last_channel, 512),
    nn.LayerNorm(512),
    nn.LeakyReLU(),
    nn.Dropout(0.4),
    nn.Linear(512, 256),
    nn.LayerNorm(256),
    nn.LeakyReLU(),
    nn.Dropout(0.3),
    nn.Linear(256, len(class_names))
)

model = base_model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5, verbose=True)

# 🔁 Training loop
best_acc = 0.0
patience, patience_counter = 7, 0

print("🚀 Starting training...")
for epoch in range(num_epochs):
    model.train()
    train_loss, train_correct = 0.0, 0

    for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Training]"):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        _, preds = torch.max(outputs, 1)
        train_loss += loss.item() * inputs.size(0)
        train_correct += torch.sum(preds == labels)

    train_acc = train_correct.double() / len(train_dataset)

    # 🔍 Validation
    model.eval()
    val_loss, val_correct = 0.0, 0
    with torch.no_grad():
        for inputs, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Validation]"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            _, preds = torch.max(outputs, 1)
            val_loss += loss.item() * inputs.size(0)
            val_correct += torch.sum(preds == labels)

    val_acc = val_correct.double() / len(val_dataset)
    scheduler.step(val_loss)

    print(f"📊 Epoch {epoch+1}: Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")

    # 💾 Save best model
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), "model0_torch_best.pth")
        print("✅ Best model saved")
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print("⏹️ Early stopping triggered.")
            break

# ===========================================
# 💡 Sklearn Integration
# ===========================================
print("\n🔍 Extracting features for sklearn classification...")

def extract_features(model, dataloader, device):
    model.eval()
    features, labels_list = [], []

    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Extracting features"):
            inputs = inputs.to(device)
            x = model.features(inputs)
            x = model.avgpool(x)
            x = torch.flatten(x, 1)  # (batch, 1280)
            features.append(x.cpu().numpy())
            labels_list.append(labels.cpu().numpy())

    return np.concatenate(features), np.concatenate(labels_list)

# Feature extraction
train_feats, train_labels = extract_features(model, train_loader, device)
val_feats, val_labels = extract_features(model, val_loader, device)

# Scale features
scaler = StandardScaler()
train_feats = scaler.fit_transform(train_feats)
val_feats = scaler.transform(val_feats)

# Train scikit-learn model
clf = SVC(kernel='linear', C=1.0)
clf.fit(train_feats, train_labels)
val_preds = clf.predict(val_feats)
acc = accuracy_score(val_labels, val_preds)

print(f"🎯 Sklearn SVM Validation Accuracy: {acc:.4f}")
