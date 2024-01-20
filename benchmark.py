import torchvision
import torch
import tqdm

full_dataset = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=torchvision.transforms.ToTensor())
train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [int(len(full_dataset) * 0.8), len(full_dataset) - int(len(full_dataset) * 0.8)])

class Resnet18(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet = torchvision.models.resnet18(pretrained=True)
        self.resnet.fc = torch.nn.Linear(512, 10)

    def forward(self, x):
        return self.resnet(x)

def train(loader, model, epochs, lr, device):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()
    for epoch in range(epochs):
        model.train()
        for x, y in tqdm.tqdm(loader):
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            y_hat = model(x)
            loss = criterion(y_hat, y)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}/{epochs} loss: {loss.item():.4f}")

def test(loader, model, device):
    model = model.to(device)
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in tqdm.tqdm(loader):
            x = x.to(device)
            y = y.to(device)
            y_hat = model(x)
            correct += (y_hat > 0).eq(y).sum().item()
            total += y.shape[0]
    print(f"Accuracy: {correct / total * 100:.2f}%")


model = Resnet18()
batch_size = 64
epochs = 10
lr = 1e-4

def main():
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train(train_loader, model, epochs, lr, device)
    test(test_loader, model, device)

if __name__ == "__main__":
    main()