"""
A minimal training loop using torchvision datasets.ImageFolder where mel-spectrograms
are saved into class subfolders (e.g., processed/melspecs/<label>/*.png)
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models


def train_cnn(data_dir, epochs=10, batch_size=32, lr=1e-4, model_out='cnn.pth'):
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    dataset = datasets.ImageFolder(data_dir, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = models.resnet18(pretrained=False)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc = nn.Linear(model.fc.in_features, len(dataset.classes))
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        running = 0
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            out = model(imgs)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            running += loss.item()
        print(f'Epoch {epoch+1}/{epochs} loss {running/len(loader):.4f}')

    torch.save(model.state_dict(), model_out)
    print('Saved', model_out)

if __name__ == '__main__':
    train_cnn('processed/melspecs')
