import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from tqdm import tqdm
import os

# Paths & config
train_dir = "../dataset/train"
val_dir = "../dataset/test"
img_size = 224
batch_size = 32
num_epochs = 5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(device)

# ⚙️ Data augmentation
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(img_size),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load datasets
train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
val_dataset = datasets.ImageFolder(val_dir, transform=val_transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

class_names = train_dataset.classes
print("Classes:", class_names)

# Load base model
base_model = models.mobilenet_v2(pretrained=True)
for param in base_model.features.parameters():
    param.requires_grad = False

# 🧠 Enhanced head
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

# Loss, optimizer, scheduler
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5, verbose=True)

# Training with early stopping
best_acc = 0.0
patience, patience_counter = 7, 0

print("starting training")
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

    # Validation
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

    print(f"Epoch {epoch+1}: Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")

    # Checkpoint best model
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
