from matplotlib import pyplot as plt
import torchvision
import torch
import tqdm

train_dataset = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=torchvision.transforms.ToTensor())
test_dataset = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=torchvision.transforms.ToTensor())

class MagicActivate(torch.nn.Module):
    def __init__(self, n_in):
        super(MagicActivate, self).__init__()
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(n_in * 6, n_in),
        )
    def forward(self, x):
        return self.fc(torch.cat([
            torch.sin(x),
            torch.cos(x),
            torch.exp(x),
            -torch.exp(x),
            torch.ceil(x),
            torch.relu(x),
        ], dim=-1))

class BasicCNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = torch.nn.Sequential(
            torch.nn.Conv2d(3, 16, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),

            torch.nn.Conv2d(16, 32, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),

            torch.nn.Conv2d(32, 64, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),

            torch.nn.Conv2d(64, 128, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),

            torch.nn.Flatten(),
            torch.nn.Linear(512, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 10)
        )
    def forward(self, x):
        x = self.cnn(x)
        return x
    
class BasicCNNM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = torch.nn.Sequential(
            torch.nn.Conv2d(3, 16, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),

            torch.nn.Conv2d(16, 32, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),

            torch.nn.Conv2d(32, 64, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),

            torch.nn.Conv2d(64, 128, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),

            torch.nn.Flatten(),
            torch.nn.Linear(512, 64),
            MagicActivate(64),
            torch.nn.Linear(64, 10)
        )
    def forward(self, x):
        x = self.cnn(x)
        return x

def train(tr_loader, va_loader , model, epochs, lr, device):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()
    best_loss = float("inf")
    all_losses = []
    all_accuracies = []
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for x, y in tqdm.tqdm(tr_loader):
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            y_hat = model(x)
            loss = criterion(y_hat, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        total_loss /= len(tr_loader)
        all_losses.append(total_loss)
        print(f"Epoch {epoch+1}/{epochs} loss: {total_loss :.4f}")

        if total_loss < best_loss:
            best_loss = total_loss
            torch.save(model.state_dict(), "best_model.pth")
            print("Saved best model")

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in tqdm.tqdm(va_loader):
                x = x.to(device)
                y = y.to(device)
                y_hat = model(x)
                _, predicted = torch.max(y_hat.data, 1)
                total += y.size(0)
                correct += (predicted == y).sum().item()
        all_accuracies.append(correct / total * 100)
        print(f"Accuracy: {correct / total * 100:.2f}%")
    
    axis[0].plot(all_losses, label=f"{model.__class__.__name__} Loss")
    axis[1].plot(all_accuracies, label=f"{model.__class__.__name__} Accuracy")

models = [ BasicCNN(), BasicCNNM() ]
batch_size = 128
epochs = 20
lr = 1e-4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
figure, axis = plt.subplots(1, 2)
figure.set_size_inches(12, 6)
axis[0].set_title("Loss")
axis[0].set_xlabel("Epochs")
axis[0].set_ylabel("Loss")
axis[1].set_title("Accuracy")
axis[1].set_xlabel("Epochs")
axis[1].set_ylabel("Accuracy")

def main():
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    for model in models:
        train(train_loader, test_loader, model, epochs, lr, device)

if __name__ == "__main__":
    main()
    axis[0].legend()
    axis[1].legend()
    plt.savefig("benchmark.png")