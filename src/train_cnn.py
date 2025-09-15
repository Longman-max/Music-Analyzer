import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from pathlib import Path


def train_cnn(data_dir="data/processed/melspecs",
              model_out="models/cnn_model.pth",
              epochs=10,
              batch_size=16,
              lr=0.001):
    """
    Train a CNN on mel-spectrogram images stored in class-labeled subfolders.
    Ex: data/processed/melspecs/<genre>/<image>.png
    """

    data_dir = Path(data_dir)
    model_out = Path(model_out)

    if not data_dir.exists():
        raise FileNotFoundError(f"❌ Data directory not found: {data_dir}. Run dataset_builder.py first.")

    # Data transforms (resize + normalize like ImageNet)
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    # Load dataset
    dataset = datasets.ImageFolder(data_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    num_classes = len(dataset.classes)
    print(f"✅ Loaded dataset from {data_dir}, classes = {dataset.classes}, samples = {len(dataset)}")

    # Use pretrained ResNet18 (transfer learning)
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Loss & optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Training loop
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct, total = 0, 0

        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        acc = 100 * correct / total
        print(f"📀 Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(dataloader):.4f}, Acc: {acc:.2f}%")

    # Save trained model
    model_out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), model_out)
    print(f"💾 Saved CNN model to {model_out}")

    return model, dataset.classes


if __name__ == "__main__":
    train_cnn()
