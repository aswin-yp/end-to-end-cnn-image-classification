import torch
from torch import nn, optim
from model import TinyVGG
from dataset import get_dataloaders

device = "cuda" if torch.cuda.is_available() else "cpu"

train_dir = "data/train"
test_dir = "data/test"

train_loader, _, classes = get_dataloaders(train_dir, test_dir)

model = TinyVGG(num_classes=len(classes)).to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 5

for epoch in range(epochs):
    model.train()
    total_loss = 0

    for X, y in train_loader:
        X, y = X.to(device), y.to(device)

        preds = model(X)
        loss = loss_fn(preds, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss:.4f}")

torch.save(model.state_dict(), "cnn_model.pt")
print("✅ Model trained and saved")
