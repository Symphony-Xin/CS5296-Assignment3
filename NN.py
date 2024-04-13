import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import time

# Transformations applied on each image
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,))])

# Loading MNIST dataset
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# create dataloaders
train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)

epochs = 5
results = []

# Define the Neural Network model
class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

# Define Model, Loss, and Optimizer
model = NeuralNet()
loss_fn = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training the model
def train(dataloader, model, loss_fn, optimizer):
    model.train()
    size = len(dataloader.dataset)
    train_loss, train_acc = 0, 0
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction，acc and loss
        y = torch.nn.functional.one_hot(y, num_classes=10).float()
        pred = model(X)
        _, success = torch.max(pred.data, 1)
        success = torch.nn.functional.one_hot(success, num_classes=10).float()
        loss = loss_fn(pred, y)
        train_loss += loss.item()
        train_acc += (success == y).type(torch.float).sum().item()
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    train_loss /= size
    train_acc /= size
    results.append(f"Train Accuracy: {(10 * train_acc):>0.2f}%, Train Avg loss: {train_loss:>8f} \n")
    print(f"Accuracy: {(10 * train_acc):>0.2f}%, Avg loss: {train_loss:>8f} \n")

# Testing the model
def test(dataloader, model, loss_fn):
    model.eval()
    size = len(dataloader.dataset)
    test_loss, test_acc = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            y = torch.nn.functional.one_hot(y, num_classes=10).float()
            pred = model(X)
            _, success = torch.max(pred.data, 1)
            success = torch.nn.functional.one_hot(success,num_classes=10).float()
            test_loss += loss_fn(pred, y).item()
            test_acc += (success == y).type(torch.float).sum().item()
    test_loss /= size
    test_acc /= size
    results.append(f"Test Accuracy: {(10*test_acc):>0.2f}%, Test Avg loss: {test_loss:>8f} \n")
    print(f"Accuracy: {(10*test_acc):>0.2f}%, Avg loss: {test_loss:>8f} \n")


start = time.time()
for epoch in range(epochs):
    results.append("-----------epoch {:d}--------------\n".format(epoch+1))
    print("-----------epoch {:d}--------------".format(epoch+1))
    train(train_loader, model, loss_fn, optimizer)
    test(test_loader, model, loss_fn)

end = time.time()
results.append(f'{epochs} epochs done!\nRuntime:{end-start}')
print('{:d} epochs done!'.format(epochs))

with open('./NN.txt','w') as f:
    for result in results:
        f.write(result)
f.close()