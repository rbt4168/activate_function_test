import torch
import numpy as np
import random
import tqdm
import matplotlib.pyplot as plt

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

class ModifiedActivate(torch.nn.Module):
    def __init__(self, n_in, n_out):
        super(ModifiedActivate, self).__init__()
        self.magic_unit = torch.nn.Sequential(
            torch.nn.Linear(n_in, 64),
            MagicActivate(64),
            torch.nn.Linear(64, n_out),
        )
        self.color = "#33A6B8"
    def forward(self, x):
        return self.magic_unit(x)

class MagicActivate2(torch.nn.Module):
    def __init__(self, n_in):
        super(MagicActivate2, self).__init__()
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(n_in * 6, n_in),
        )
    def forward(self, x):
        return self.fc(torch.cat([
            torch.relu(x),
            torch.relu(x),
            torch.relu(x),
            torch.relu(x),
            torch.relu(x),
            torch.relu(x),
        ], dim=-1))

class PureReLU(torch.nn.Module):
    def __init__(self, n_in, n_out):
        super(PureReLU, self).__init__()
        self.magic_unit = torch.nn.Sequential(
            torch.nn.Linear(n_in, 64),
            MagicActivate2(64),
            torch.nn.Linear(64, n_out),
        )
        self.color = "#8F77B5"
    def forward(self, x):
        return self.magic_unit(x)

def tar_function(x):
    # furrier series with 2pi
    return np.sin(x * 2 * np.pi) + np.sin(x * 4 * np.pi) + np.sin(x * 6 * np.pi) + np.sin(x * 8 * np.pi) + np.sin(x * 10 * np.pi) + np.sin(x * 12 * np.pi) + np.sin(x * 14 * np.pi) + np.sin(x * 16 * np.pi) + np.sin(x * 18 * np.pi) + np.sin(x * 20 * np.pi)

class MyDataset(torch.utils.data.Dataset):
    def __init__(self):
        super(MyDataset, self).__init__()
        self.data = torch.rand(2000, 1) * 6 - 3

        # ==================== #
        self.target = torch.tensor(tar_function(self.data)).float()
        # ==================== #

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.target[idx]

def train(model, dataset, optimizer, loss_fn, epochs=150, batch_size=50):
    all_loss = []
    for epoch in range(epochs):
        # shuffle dataset
        indices = list(range(len(dataset)))
        random.shuffle(indices)
        dataset = torch.utils.data.Subset(dataset, indices)
        # train
        model.train()
        epoch_loss = 0
        for i in tqdm.tqdm(range(0, len(dataset), batch_size)):
            # get batch
            batch = dataset[i:i+batch_size]
            batch_x, batch_y = batch

            # forward
            pred = model(batch_x)
            loss = loss_fn(pred, batch_y)

            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # record loss
            epoch_loss += torch.log(loss.sum()).item()
        all_loss.append(epoch_loss / (len(dataset) // batch_size))
        print(f"Epoch {epoch+1}/{epochs} loss: {loss.item():.4f}")
    
    axis[1].plot(all_loss, label=f"{model.__class__.__name__} Loss", color=model.color)



# test for different optimizers
def main():
    # create dataset
    dataset = MyDataset()

    # create optimizers
    optimizer = torch.optim.Adam

    # create loss function
    loss_fn = torch.nn.MSELoss()

    # plot real sin(x)
    all_x = np.linspace(-3, 3, 1000)


    # ==================== #
    real_pred = tar_function(all_x)
    # ==================== #



    axis[0].plot(all_x, real_pred, label="Target", color="#000000")
    axis[0].set_title("Target Function")
    axis[1].set_title("Log Loss")

    # models 
    models = [
        ModifiedActivate(1, 1),
        PureReLU(1, 1),
    ]

    # train
    for model in models:
        train(model, dataset, optimizer(model.parameters(), lr=1e-4), loss_fn)

        all_pred = []
        for x in all_x:
            x = torch.tensor([x]).float()
            all_pred.append(model(x).detach().numpy())
        all_pred = np.array(all_pred).flatten()
        axis[0].plot(all_x, all_pred, label=model.__class__.__name__, color=model.color)

    for model in models:
        pytorch_total_params = sum(p.numel() for p in model.parameters())
        print(f"Model {model.__class__.__name__} has {pytorch_total_params} parameters")



figure, axis = plt.subplots(2)
if __name__ == "__main__":
    main()
    axis[0].legend()
    axis[1].legend()
    plt.savefig("result.png")