import os
import pandas as pd
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import vit_b_16, ViT_B_16_Weights
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import matplotlib.pyplot as plt

# ======= CONFIG =======
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
csv_file = r"C:\Users\DELL\Desktop\reg\empreintes_comparaison0.csv"
df = pd.read_csv(csv_file)
df = df[["Image", "Score"]]  # Garde uniquement les colonnes utiles

image_folder = r"C:\Users\DELL\Desktop\reg\data"
batch_size = 16
num_epochs = 25
learning_rate = 1e-4
image_size = 224
model_path = "vit_regression_model.pth"

# ======= DATASET =======
class FingerprintDataset(Dataset):
    def __init__(self, dataframe, image_folder, transform=None):
        self.dataframe = dataframe
        self.image_folder = image_folder
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_folder, self.dataframe.iloc[idx, 0])
        image = Image.open(img_name).convert("L")
        image = image.convert("RGB")  # ViT attend 3 canaux
        label = float(self.dataframe.iloc[idx, 1])
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.float32)

# ======= LOAD CSV =======
df = pd.read_csv(csv_file)
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

train_dataset = FingerprintDataset(train_df, image_folder, transform=transform)
test_dataset = FingerprintDataset(test_df, image_folder, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# ======= MODEL =======
weights = ViT_B_16_Weights.DEFAULT
vit = vit_b_16(weights=weights)
in_features = vit.heads.head.in_features
vit.heads.head = nn.Linear(in_features, 1)  # Sortie unique pour régression
vit.to(device)

# ======= TRAINING =======
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(vit.parameters(), lr=learning_rate)

train_losses = []

for epoch in range(num_epochs):
    vit.train()
    running_loss = 0.0
    for images, scores in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
        images, scores = images.to(device), scores.to(device).unsqueeze(1)
        optimizer.zero_grad()
        outputs = vit(images)
        loss = criterion(outputs, scores)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    epoch_loss = running_loss / len(train_loader)
    train_losses.append(epoch_loss)
    print(f"Epoch {epoch+1}: Loss = {epoch_loss:.4f}")

torch.save(vit.state_dict(), model_path)
print("✅ Modèle sauvegardé.")

# ======= COURBE DE PERTE =======
plt.plot(train_losses, label="Train Loss")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.title("Courbe de perte")
plt.grid()
plt.show()

# ======= EVALUATION SIMPLE =======
vit.eval()
preds = []
targets = []

with torch.no_grad():
    for images, scores in test_loader:
        images = images.to(device)
        outputs = vit(images).squeeze(1).cpu().numpy()
        preds.extend(outputs)
        targets.extend(scores.numpy())

from sklearn.metrics import mean_squared_error, r2_score

mse = mean_squared_error(targets, preds)
r2 = r2_score(targets, preds)

print(f"\n🎯 MSE: {mse:.4f}")
print(f"📈 R² Score: {r2:.4f}")
