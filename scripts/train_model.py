"""Train breed recognition model using transfer learning with ResNet."""
import os, json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(BASE_DIR, 'processed_dataset')
MODEL_DIR = os.path.join(BASE_DIR, 'models')
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE, EPOCHS, LR, IMG_SIZE = 8, 20, 0.001, 224

def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(0.5), transforms.RandomRotation(15),
        transforms.ColorJitter(0.2, 0.2, 0.2), transforms.ToTensor(), norm])
    val_tf = transforms.Compose([transforms.Resize((IMG_SIZE, IMG_SIZE)), transforms.ToTensor(), norm])

    train_ds = datasets.ImageFolder(os.path.join(PROCESSED_DIR, 'train'), transform=train_tf)
    val_ds = datasets.ImageFolder(os.path.join(PROCESSED_DIR, 'val'), transform=val_tf)
    class_names = train_ds.classes
    print(f"Classes: {class_names}")
    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}")

    train_loader = DataLoader(train_ds, BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, BATCH_SIZE, shuffle=False, num_workers=0)

    print(f"Loading ResNet-18 on {DEVICE}...")
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    for p in model.parameters():
        p.requires_grad = False
    model.fc = nn.Sequential(
        nn.Dropout(0.5), nn.Linear(model.fc.in_features, 256), nn.ReLU(),
        nn.Dropout(0.3), nn.Linear(256, len(class_names)))
    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.5)

    best_acc, history = 0.0, {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    model_path = os.path.join(MODEL_DIR, 'breed_model.pth')

    for epoch in range(EPOCHS):
        model.train()
        tr_loss, tr_corr, tr_tot = 0.0, 0, 0
        bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]")
        for images, labels in bar:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            out = model(images)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            tr_loss += loss.item() * images.size(0)
            _, pred = torch.max(out, 1)
            tr_tot += labels.size(0)
            tr_corr += (pred == labels).sum().item()
        scheduler.step()

        model.eval()
        va_loss, va_corr, va_tot = 0.0, 0, 0
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Val]"):
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                out = model(images)
                loss = criterion(out, labels)
                va_loss += loss.item() * images.size(0)
                _, pred = torch.max(out, 1)
                va_tot += labels.size(0)
                va_corr += (pred == labels).sum().item()

        tr_loss /= len(train_ds); tr_acc = tr_corr / tr_tot
        va_loss /= len(val_ds); va_acc = va_corr / va_tot
        history['train_loss'].append(tr_loss); history['val_loss'].append(va_loss)
        history['train_acc'].append(tr_acc); history['val_acc'].append(va_acc)
        print(f"  Train Loss: {tr_loss:.4f}, Train Acc: {tr_acc:.4f} | Val Loss: {va_loss:.4f}, Val Acc: {va_acc:.4f}")

        if va_acc > best_acc:
            best_acc = va_acc
            torch.save({'model_state_dict': model.state_dict(), 'class_names': class_names, 'val_acc': best_acc}, model_path)
            print(f"  ** Best model saved (Val Acc: {best_acc:.4f})")

    with open(os.path.join(MODEL_DIR, 'training_history.json'), 'w') as f:
        json.dump(history, f, indent=2)

    try:
        import matplotlib; matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        ax1.plot(history['train_loss'], label='Train'); ax1.plot(history['val_loss'], label='Val')
        ax1.set_xlabel('Epoch'); ax1.set_ylabel('Loss'); ax1.legend(); ax1.set_title('Loss')
        ax2.plot(history['train_acc'], label='Train'); ax2.plot(history['val_acc'], label='Val')
        ax2.set_xlabel('Epoch'); ax2.set_ylabel('Accuracy'); ax2.legend(); ax2.set_title('Accuracy')
        plt.tight_layout(); plt.savefig(os.path.join(MODEL_DIR, 'training_curves.png'))
        print("Training curves saved.")
    except Exception as e:
        print(f"Plot error: {e}")

    print(f"\nDone! Best Val Acc: {best_acc:.4f}")
    print(f"Model: {model_path}")

if __name__ == '__main__':
    main()