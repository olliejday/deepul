import numpy as np
import pickle
import time
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import trange, tqdm
import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision
from torchvision.utils import save_image, make_grid
import torchvision.transforms as transforms
import IPython
import math

from models import LeNet

device = torch.device("cuda:0")


class Solver(object):
    def __init__(self, n_epochs=100, batch_size=128, train_data_amount=1.0):
        self.log_interval = 100
        self.batch_size = batch_size
        self.train_data_amount = train_data_amount
        self.train_loader, self.test_loader = self.create_loaders()
        self.n_batches_in_epoch = len(self.train_loader)
        self.n_epochs = n_epochs
        self.curr_itr = 0

    def build(self, part_name):
        self.d = LeNet().to(device)
        self.d_optimizer = torch.optim.Adam(self.d.parameters(), lr=1e-3)
        self.part_name = part_name

    def create_loaders(self):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

        full_train_set = torchvision.datasets.MNIST(root="./data", train=True, download=True, transform=transform)
        end_index = math.ceil(60000 * self.train_data_amount)
        # train_data = full_train_set.data[:end_index]
        # train_labels = full_train_set.targets[:end_index]
        train_set = torch.utils.data.Subset(full_train_set, range(end_index))
        test_set = torchvision.datasets.MNIST(root="./data", train=False, download=True, transform=transform)
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=self.batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=self.batch_size, shuffle=False)

        return train_loader, test_loader

    def train(self):
        train_losses = []
        val_losses = []
        for epoch_i in trange(self.n_epochs, desc='Epoch', ncols=80):
            epoch_i += 1

            self.d.train()
            self.batch_loss_history = []

            for batch_i, (x, y) in enumerate(tqdm(self.train_loader, desc='Batch', ncols=80, leave=False)):
                batch_i += 1
                self.curr_itr += 1

                x = x.to(device).float()
                y = y.to(device)

                # do a discriminator update
                self.d_optimizer.zero_grad()
                pred = self.d(x)

                pred_indices = torch.argmax(pred, dim=1)
                accuracy = (torch.sum(pred_indices == y).float() / pred.shape[0]).data  # not sure this is right
                d_loss = F.cross_entropy(pred, y)
                d_loss.backward()
                self.d_optimizer.step()

                self.batch_loss_history.append(d_loss.item())

                if batch_i > 1 and batch_i % self.log_interval == 0:
                    log_string = f'Epoch: {epoch_i} | Itr: {self.curr_itr} | '
                    log_string += f'Accuracy: {accuracy:.3f}'
                    tqdm.write(log_string)
                # if self.curr_itr % 50000 == 0:  # todo: implement model saving with just the statedict
                #     self.save_model(f"{self.part_name}_epoch{self.curr_itr}.model")
            val_loss = self.val_loss()
            val_losses.append(val_loss)
            epoch_loss = np.mean(self.batch_loss_history)
            tqdm.write(f'Epoch Loss: {epoch_loss:.2f}, val accuracy: {val_loss:.3f}')
            train_losses.append(epoch_loss)
            # self.sample(100, f"{self.part_name}_samples{epoch_i}.png")
            np.save("train_losses.npy", np.array(train_losses))

        train_losses = np.array(train_losses)
        val_losses = np.array(val_losses)
        # self.save_model(f"{self.part_name}.model")
        self.plot_losses(train_losses, val_losses)


        return train_losses

    def val_loss(self):
        self.d.eval()

        val_loss_total = 0
        val_items = 0
        with torch.no_grad():
            for (inputs, labels) in self.test_loader:
                inputs = inputs.to(device).float()
                labels = labels.to(device)
                logits = self.d(inputs)
                predictions = torch.argmax(logits, dim=1)
                num_correct = torch.sum(predictions == labels).float()
                val_loss_total += num_correct
                val_items += inputs.shape[0]

        return val_loss_total / val_items

    def plot_losses(self, train_losses, test_losses):
        plt.figure()
        plt.plot(np.arange(len(train_losses)), train_losses, label="Train")
        plt.plot(np.arange(len(test_losses)), test_losses, label="Test")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.title(self.part_name)
        plt.savefig(f"{self.part_name}_loss.png")
        plt.close()


if __name__ == "__main__":
    solver = Solver(n_epochs=1000, batch_size=32, train_data_amount=0.0025)
    solver.build("bigan")
    solver.train()
    IPython.embed()
