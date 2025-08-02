import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from src.models.swin import SSI_SwinFusionNet
from src.dataset.fusion_dataset import FusionDataset

# -------------------------------
# 1. PARAMÈTRES DE BASE
# -------------------------------
csv_path = "/content/drive/MyDrive/MonProjet_SSI_Swin/dataset_fusion_weighted.csv"   # CSV listant image_path, ssi_path, label, split
batch_size = 16
num_epochs = 10
learning_rate = 1e-4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------------
# 2. DATASETS ET DATALOADERS
# -------------------------------
train_ds = FusionDataset(csv_path, split="train")
val_ds = FusionDataset(csv_path, split="val", label_map=train_ds.label_map)

train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)

# -------------------------------
# 3. MODÈLE, PERTE, OPTIMISEUR
# -------------------------------
model = SSI_SwinFusionNet(num_classes=len(train_ds.label_map), ssi_input_dim=10, pretrained=True)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# -------------------------------
# 4. FONCTION D'ÉVALUATION
# -------------------------------
def evaluate(loader):
    model.eval()
    correct, total, running_loss = 0, 0, 0.0
    with torch.no_grad():
        for images, ssi, labels in loader:
            images, ssi, labels = images.to(device), ssi.to(device), labels.to(device)
            outputs = model(images, ssi)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * labels.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return running_loss / total, correct / total

# -------------------------------
# 5. BOUCLE D'ENTRAÎNEMENT
# -------------------------------
for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0

    for images, ssi, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
        images, ssi, labels = images.to(device), ssi.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images, ssi)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * labels.size(0)

    # Évaluation
    val_loss, val_acc = evaluate(val_loader)
    train_loss /= len(train_ds)

    print(f"Epoch [{epoch+1}/{num_epochs}] "
          f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

# -------------------------------
# 6. SAUVEGARDE DU MODÈLE
# -------------------------------
torch.save(model.state_dict(), "ssi_swin_fusion.pth")
print("✅ Modèle entraîné et sauvegardé.")

