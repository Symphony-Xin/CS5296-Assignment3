import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import time

# Transformations and Data Loading
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)

epoches = 10
results = []

# Linear Regression Model
class LinearRegression(nn.Module):
    def __init__(self):
        super(LinearRegression, self).__init__()
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(28*28, 10)  # Predicting 10 classes directly

    def forward(self, x):
        x = self.flatten(x)
        out = self.linear(x)
        return out

model = LinearRegression()
loss_fn = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Training
def train(dataloader, model, loss_fn, optimizer):
    model.train()
    size = len(dataloader.dataset)
    train_loss, train_acc = 0, 0
    for batch, (X, y) in enumerate(dataloader):
        y = torch.nn.functional.one_hot(y, num_classes=10).float()  # One-hot encode y
        optimizer.zero_grad()
        pred = model(X)
        _, success = torch.max(pred.data, 1)
        success = torch.nn.functional.one_hot(success, num_classes=10).float()
        loss = loss_fn(pred, y)
        train_loss += loss.item()
        train_acc += (success == y).type(torch.float).sum().item()
        loss.backward()
        optimizer.step()
    train_loss /= size
    train_acc /= size
    results.append(f"Train Accuracy: {(10 * train_acc):>0.2f}%, Train Avg loss: {train_loss:>8f} \n")
    print(f"Accuracy: {(10 * train_acc):>0.2f}%, Avg loss: {train_loss:>8f} \n")

# Testing
def test(dataloader, model, criterion):
    model.eval()
    size = len(dataloader.dataset)
    total_loss, total, test_acc = 0, 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            y = torch.nn.functional.one_hot(y, num_classes=10).float()  # One-hot encode y
            pred = model(X)
            _, success = torch.max(pred.data, 1)
            success = torch.nn.functional.one_hot(success,num_classes=10).float()
            test_acc += (success == y).sum().item()
            total_loss += criterion(pred, y).item()
    test_acc /= size
    avg_loss = total_loss / size
    results.append((f"Test Accuracy: {(10*test_acc):>0.2f}%, Test Avg Loss: {avg_loss:>8f}\n"))
    print(f"Accuracy: {(100*test_acc):>0.2f}%, Avg Loss: {avg_loss}")

# Run training and testing
start = time.time()
for epoch in range(epoches):
    results.append(f"----------------epoch {epoch+1}------------------\n")
    print("----------------epoch {:d}------------------".format(epoch+1))
    train(train_loader, model, loss_fn, optimizer)
    test(test_loader, model, loss_fn)
end = time.time()
results.append(f'{epoches} epochs done!\nRuntime:{end-start}')
print('{:d} epochs done!'.format(epoches))

with open('./LR.txt','w') as f:
    for result in results:
        f.write(result + '\n')