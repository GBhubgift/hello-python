import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)

# ===== 超参数 =====
EPOCHS = 30
BATCH_SIZE = 128
LR = 1e-3

# ===== 数据集 =====
tfm = transforms.Compose([
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
train_data = datasets.MNIST(root="./data", train=True, transform=tfm, download=True)
test_data  = datasets.MNIST(root="./data", train=False, transform=tfm, download=True)

train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
test_loader  = DataLoader(test_data, batch_size=1000, shuffle=False)

# ===== 超级 CNN 模型 =====
class SuperCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout(0.25),

            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout(0.25),

            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout(0.5),
        )
        self.classifier = nn.Sequential(
            nn.Linear(256*3*3, 512), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.feature(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

model = SuperCNN().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
criterion = nn.CrossEntropyLoss()

# ===== 训练 + 验证 =====
def evaluate():
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for tx, ty in test_loader:
            tx, ty = tx.to(device), ty.to(device)
            out = model(tx)
            pred = out.argmax(1)
            correct += (pred == ty).sum().item()
            total += ty.size(0)
    return correct / total

best_acc = 0.0
for epoch in range(EPOCHS):
    model.train()
    for bx, by in train_loader:
        bx, by = bx.to(device), by.to(device)
        out = model(bx)
        loss = criterion(out, by)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    acc = evaluate()
    best_acc = max(best_acc, acc)
    scheduler.step()
    print(f"Epoch {epoch+1}/{EPOCHS} | loss={loss.item():.4f} | acc={acc*100:.2f}% | best={best_acc*100:.2f}%")

print("训练完成！最佳准确率:", best_acc*100)
torch.save(model.state_dict(), "super_cnn_mnist.pt")