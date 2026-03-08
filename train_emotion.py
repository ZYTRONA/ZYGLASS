#!/usr/bin/env python3
"""
Emotion Recognition Training Script using PyTorch + ResNet50
For Arch Linux with RTX 4050 GPU
Transfer Learning on FER2013 Dataset
"""

import torch  # type: ignore
import torch.nn as nn  # type: ignore
import torch.optim as optim  # type: ignore
from torch.utils.data import DataLoader  # type: ignore
from torchvision import transforms, datasets, models  # type: ignore
import os
from pathlib import Path
import time

print("🔍 Checking PyTorch Setup...")
print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
print("")

# 1. Setup Data Paths
project_root = Path('/run/media/aadhiasarana/E/AI GLASS')
train_dir = project_root / 'datasets' / 'train' / 'images'
val_dir = project_root / 'datasets' / 'val' / 'images'

print(f"📂 Dataset Paths:")
print(f"   Train: {train_dir}")
print(f"   Val: {val_dir}")
print("")

# Verify dataset exists
if not train_dir.exists():
    print(f"❌ Error: Training directory not found: {train_dir}")
    exit(1)

# 2. Data Augmentation & Preprocessing
print("🎨 Setting up Data Augmentation Pipeline...")

# Transforms for training (with stronger augmentation)
train_transforms = transforms.Compose([
    transforms.Resize((48, 48)),
    transforms.RandomRotation(30),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.2),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
    transforms.RandomAffine(degrees=0, translate=(0.15, 0.15), scale=(0.85, 1.15)),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])

# Transforms for validation (no augmentation)
val_transforms = transforms.Compose([
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])

# Load datasets
train_dataset = datasets.ImageFolder(str(train_dir), transform=train_transforms)
val_dataset = datasets.ImageFolder(str(val_dir), transform=val_transforms)

# Create data loaders (num_workers=0 to avoid multiprocessing issues)
batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

print(f"✅ Training samples: {len(train_dataset)}")
print(f"✅ Validation samples: {len(val_dataset)}")
print(f"✅ Classes: {train_dataset.classes}")
print(f"✅ Batch size: {batch_size}")
print("")

# 3. Build Model using Transfer Learning (ResNet50)
print("🏗️  Building ResNet50 Transfer Learning Model...")

# Load pre-trained ResNet50
base_model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

# Freeze the base layers initially
for param in base_model.parameters():
    param.requires_grad = False

# Replace the final fully connected layer for 7 emotions
num_ftrs = base_model.fc.in_features
base_model.fc = nn.Sequential(  # type: ignore[assignment]
    nn.Linear(num_ftrs, 512),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(512, 256),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(256, 7)  # 7 emotion classes
)

model = base_model

# Move model to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

print(f"   Base Model: ResNet50 (frozen backbone)")
print(f"   Custom Head: Dense(512) → ReLU → Dropout(0.5) → Dense(256) → ReLU → Dropout(0.3) → Dense(7)")
print(f"   Device: {device}")
print("")

# 4. Setup Training
print("⚙️  Setting up Training Pipeline...")

criterion = nn.CrossEntropyLoss()

# Only optimize the new FC layers in the first 30 epochs
optimizer = optim.Adam(model.fc.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=5)

print("✅ Loss Function: CrossEntropyLoss")
print(f"   Optimizer: Adam (lr=0.0001)")
print(f"   Scheduler: ReduceLROnPlateau")
print("")

# 4.5 Check for existing checkpoint to resume
output_dir = project_root / 'runs' / 'emotion_model_v1'
checkpoint_path = output_dir / 'emotion_model_best.pth'
start_epoch = 0
best_val_acc = 0.0

if checkpoint_path.exists():
    print("📦 Found existing checkpoint! Resuming training...")
    checkpoint = torch.load(str(checkpoint_path), map_location=device)
    
    # If checkpoint contains state_dict
    if isinstance(checkpoint, dict) and 'model_state' in checkpoint:
        model.load_state_dict(checkpoint['model_state'])
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        start_epoch = checkpoint.get('epoch', 0) + 1
        best_val_acc = checkpoint.get('best_acc', 0.0)
        print(f"   Resuming from Epoch {start_epoch}")
        print(f"   Best Accuracy so far: {best_val_acc:.2f}%")
    else:
        # Just model weights (compatibility with old checkpoint format)
        model.load_state_dict(checkpoint)
        print("   Loaded model weights (restarting optimizer from epoch 0)")
        start_epoch = 0
    print("")
else:
    output_dir.mkdir(parents=True, exist_ok=True)
    print("✅ Starting fresh training (no checkpoint found)")
    print("")

# 5. Training Loop
print("🚀 Starting Emotion Model Training on RTX 4050...")
print(f"   Phase 1 (Epochs 1-30): Train only FC head (backbone frozen)")
print(f"   Phase 2 (Epochs 31-50): Fine-tune backbone + FC head")
print(f"   Epochs: 50")
print(f"   Batch Size: {batch_size}")
print(f"   Image Size: 48x48 (ESP32-CAM compatible)")
print(f"   Starting from Epoch: {start_epoch + 1}")
print("")

num_epochs = 50

for epoch in range(start_epoch, num_epochs):
    print(f"\nEpoch {epoch+1}/{num_epochs}")
    print("-" * 50)
    
    # Unfreeze backbone for fine-tuning after epoch 30
    if epoch == 30:
        print("🔓 Unfreezing ResNet50 backbone for fine-tuning...")
        for param in model.layer4.parameters():
            param.requires_grad = True
        for param in model.layer3.parameters():
            param.requires_grad = True
        # Create new optimizer including all parameters
        optimizer = optim.Adam(model.parameters(), lr=0.00001)
        print("   Using lower learning rate (0.00001) for backbone fine-tuning")
    
    # Training phase
    model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0
    
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Statistics
        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        train_total += labels.size(0)
        train_correct += (predicted == labels).sum().item()
    
    train_acc = 100 * train_correct / train_total
    train_loss = train_loss / len(train_loader)
    
    # Validation phase
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()
    
    val_acc = 100 * val_correct / val_total
    val_loss = val_loss / len(val_loader)
    
    # Print epoch results
    print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
    print(f"Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
    
    # Save best model with checkpoint info for resumption
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        
        model_path = output_dir / 'emotion_model_best.pth'
        
        # Save complete checkpoint (model + optimizer + epoch info)
        checkpoint = {
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'epoch': epoch,
            'best_acc': best_val_acc
        }
        torch.save(checkpoint, str(model_path))
        print(f"💾 Best model saved! (Val Acc: {val_acc:.2f}%)")
    
    # Learning rate scheduling
    scheduler.step(val_acc)

# 6. Save Final Model
print("\n" + "="*60)
print("✅ EMOTION RECOGNITION MODEL TRAINING COMPLETE!")
print("="*60)

output_dir = project_root / 'runs' / 'emotion_model_v1'
output_dir.mkdir(parents=True, exist_ok=True)

# Save final model
final_model_path = output_dir / 'emotion_model_final.pth'
torch.save(model.state_dict(), str(final_model_path))
print(f"\n💾 Final Model saved: {final_model_path}")

# Save model architecture info
info_path = output_dir / 'model_info.txt'
with open(info_path, 'w') as f:
    f.write("Emotion Recognition Model\n")
    f.write("="*60 + "\n")
    f.write(f"Base Model: ResNet50\n")
    f.write(f"Number of Classes: 7 (angry, disgust, fear, happy, neutral, sad, surprise)\n")
    f.write(f"Input Size: 48x48 (ESP32-CAM compatible)\n")
    f.write(f"Best Validation Accuracy: {best_val_acc:.2f}%\n")
    f.write(f"\nModel Weights Files:\n")
    f.write(f"  - emotion_model_best.pth (best validation accuracy)\n")
    f.write(f"  - emotion_model_final.pth (final epoch)\n")

print(f"\n📊 Training Summary:")
print(f"   Best Validation Accuracy: {best_val_acc:.2f}%")
print(f"\n💾 Model Location: {output_dir}")
print(f"\n🎯 Ready for deployment on ESP32-CAM!")
print(f"\n📝 Next Steps:")
print(f"   1. Load emotion_model_best.pth in your backend")
print(f"   2. Update backend.py to use emotion detection")
print(f"   3. Deploy to ESP32-CAM hardware")
