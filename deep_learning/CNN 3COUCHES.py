import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import classification_report
from matplotlib import pyplot as plt

# === Paramètres ===
dataset_path = "C:/Users/RANA SOUKEUR/Desktop/VIT/DATA3"  # <-- modifie ici
input_size = (128, 128)
batch_size = 32
num_epochs = 25
num_classes = len(os.listdir(dataset_path))

# === Transformations sur les images ===
transform = transforms.Compose([
    transforms.Resize(input_size),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])  # pour des images en niveaux de gris, change [0.5]*3 pour RGB
])

# === Chargement du dataset ===
dataset = datasets.ImageFolder(root=dataset_path, transform=transform)

# === Split: 2/3 train, 1/3 test (pas de validation) ===
total_size = len(dataset)
train_size = int(2/3 * total_size)
test_size = total_size - train_size

train_set, test_set = random_split(dataset, [train_size, test_size])
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_set, batch_size=batch_size)

# === Définition du CNN (avec couche supplémentaire) ===
class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)  # Nouvelle couche conv
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128 * (input_size[0] // 8) * (input_size[1] // 8), 256)  # Modifié pour nouvelle architecture
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, 128)  # Nouvelle couche FC
        self.fc3 = nn.Linear(128, num_classes)  # Couche finale

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))  # Nouvelle couche
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))  # Nouvelle couche FC
        return self.fc3(x)

# === Initialisation ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleCNN(num_classes=num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# === Entraînement ===
train_loss_list = []
for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    
    avg_train_loss = train_loss / len(train_loader)
    train_loss_list.append(avg_train_loss)
    
    print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_train_loss:.4f}")

# === Évaluation ===
def evaluate_model(loader, name):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
    print(f"\nClassification Report - {name}")
    print(classification_report(all_labels, all_preds, zero_division=0))

evaluate_model(train_loader, "Train")
evaluate_model(test_loader, "Test")

# === Visualisation de la perte ===
plt.plot(train_loss_list, label='Train Loss')
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.legend()
plt.grid()
plt.show()