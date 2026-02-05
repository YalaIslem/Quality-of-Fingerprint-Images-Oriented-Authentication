import os
import shutil
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.models import vit_b_16, ViT_B_16_Weights
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.preprocessing import label_binarize
import seaborn as sns
import numpy as np

# ========= CONFIG =========
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
base_dir = "C:/Users/RANA SOUKEUR/Desktop/VIT"
original_dir = os.path.join(base_dir, "DATA3")  # dossier source avec toutes les images classées
data_dir = os.path.join(base_dir, "DATASEPAREE3")            # dossier de sortie avec train1/ et test1/
train_dir = os.path.join(data_dir, "train1")
test_dir = os.path.join(data_dir, "test1")
batch_size = 32
num_epochs = 5
image_size = 224
learning_rate = 1e-4
model_path = "vit_Arthrose_best_model_multiclasse.pth"
num_classes = 5  # à adapter selon ton jeu de données

# ========= PREPARE FOLDERS (Split 2/3 - 1/3) =========
os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

for class_name in os.listdir(original_dir):
    class_path = os.path.join(original_dir, class_name)
    if not os.path.isdir(class_path):
        continue

    images = os.listdir(class_path)
    train_imgs, test_imgs = train_test_split(images, test_size=1/3, random_state=42)

    train_class_dir = os.path.join(train_dir, class_name)
    test_class_dir = os.path.join(test_dir, class_name)
    os.makedirs(train_class_dir, exist_ok=True)
    os.makedirs(test_class_dir, exist_ok=True)

    for img in train_imgs:
        src = os.path.join(class_path, img)
        dst = os.path.join(train_class_dir, img)
        shutil.copy2(src, dst)

    for img in test_imgs:
        src = os.path.join(class_path, img)
        dst = os.path.join(test_class_dir, img)
        shutil.copy2(src, dst)

print("✅ Répartition 2/3 entraînement, 1/3 test effectuée.")

# ========= TRANSFORMS =========
transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# ========= DATA LOADING =========
train_dataset = datasets.ImageFolder(train_dir, transform=transform)
test_dataset = datasets.ImageFolder(test_dir, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

print(f"Classes détectées : {train_dataset.classes}")
print(f"Nombre de classes : {len(train_dataset.classes)}")

# ========= MODEL =========
weights = ViT_B_16_Weights.DEFAULT
vit = vit_b_16(weights=weights)
in_features = vit.heads[-1].in_features
vit.heads = nn.Linear(in_features, num_classes)
vit.to(device)

# ========= OPTIMIZER & LOSS =========
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(vit.parameters(), lr=learning_rate)

# ========= ENTRAINEMENT =========
train_losses, train_accuracies = [], []
best_f1 = 0.0

for epoch in range(num_epochs):
    vit.train()
    running_loss = 0.0
    all_preds, all_labels = [], []

    for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = vit(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro')

    train_losses.append(running_loss / len(train_loader))
    train_accuracies.append(acc)

    print(f"[Epoch {epoch+1}] Loss: {running_loss:.4f} - Acc: {acc:.4f} - F1-macro: {f1:.4f}")

    if f1 > best_f1:
        best_f1 = f1
        torch.save(vit.state_dict(), model_path)
        print("✅ Modèle sauvegardé (meilleur F1-macro)")

# ========= COURBES =========
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.plot(train_losses, label="Loss")
plt.title("Courbe de perte")
plt.xlabel("Époque")
plt.ylabel("Loss")
plt.grid(True)

plt.subplot(1,2,2)
plt.plot(train_accuracies, label="Accuracy", color="green")
plt.title("Courbe de précision")
plt.xlabel("Époque")
plt.ylabel("Accuracy")
plt.grid(True)

plt.tight_layout()
plt.show()

# ========= ÉVALUATION =========
vit.load_state_dict(torch.load(model_path))
vit.eval()
y_true, y_pred, y_scores = [], [], []

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = vit(images)
        probs = torch.softmax(outputs, dim=1)

        preds = torch.argmax(outputs, dim=1)
        y_pred.extend(preds.cpu().numpy())
        y_true.extend(labels.cpu().numpy())
        y_scores.extend(probs.cpu().numpy())

y_scores = np.array(y_scores)
y_true_bin = label_binarize(y_true, classes=list(range(num_classes)))

test_acc = accuracy_score(y_true, y_pred)
test_f1 = f1_score(y_true, y_pred, average='macro')
test_auc = roc_auc_score(y_true_bin, y_scores, average='macro', multi_class='ovo')

print(f"\n🎯 [TEST] Accuracy: {test_acc:.4f}, F1-macro: {test_f1:.4f}, AUC-ovo: {test_auc:.4f}")

# ========= MATRICE DE CONFUSION =========
cm = confusion_matrix(y_true, y_pred)
class_names = test_dataset.classes

plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Prédit')
plt.ylabel('Réel')
plt.title('Matrice de confusion')
plt.tight_layout()
plt.show()
