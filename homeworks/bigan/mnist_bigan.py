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

from models import LeNet, Encoder, Generator, Discriminator, Generator1, Discriminator1, Encoder1, TestEncoder

device = torch.device("cuda:0")


class Solver(object):
    def __init__(self, n_epochs=100, batch_size=128, train_data_amount=1.0, latent_dim=50):
        self.log_interval = 100
        self.batch_size = batch_size
        self.train_data_amount = train_data_amount
        self.train_loader, self.test_loader = self.create_loaders()
        self.n_batches_in_epoch = len(self.train_loader)
        self.n_epochs = n_epochs
        self.curr_itr = 0
        self.latent_dim = latent_dim

    def build(self, part_name):
        self.d = Discriminator1(784 + self.latent_dim).to(device)  #Discriminator().to(device)
        self.e = Encoder1(784, self.latent_dim).to(device) # Encoder().to(device)
        self.g = Generator1(self.latent_dim, 784).to(device)  #Generator().to(device)
        self.d_optimizer = torch.optim.Adam(self.d.parameters(), lr=2e-4, betas=(0.5, 0.999), weight_decay=2.5e-5)
        self.g_optimizer = torch.optim.Adam(list(self.e.parameters()) + list(self.g.parameters()), lr=2e-4, betas=(0.5, 0.999), weight_decay=2.5e-5)
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
            self.g.train()
            self.e.train()
            self.batch_loss_history = []

            for batch_i, (x, y) in enumerate(tqdm(self.train_loader, desc='Batch', ncols=80, leave=False)):
                batch_i += 1
                self.curr_itr += 1
                x = x.to(device).float() # * 2 - 1

                # do a minibatch update
                self.d_optimizer.zero_grad()

                # z_fake = torch.normal(torch.zeros(x.shape[0], self.latent_dim), torch.ones(x.shape[0], self.latent_dim)).to(device)
                z_fake = (torch.rand(x.shape[0], self.latent_dim).to(device) - 0.5) * 2
                z_real = self.e(x).reshape(x.shape[0], self.latent_dim)
                x_fake = self.g(z_fake).reshape(x.shape[0], -1)
                x_real = x.view(x.shape[0], -1)

                z_fake = torch.zeros_like(z_fake).to(device)  #todo: for testing if my gan is broken
                z_real = torch.zeros_like(z_real).to(device)

                d_loss = - 0.5 * (self.d(z_real, x_real)).log().mean() - 0.5 * (1 - self.d(z_fake, x_fake)).log().mean()
                d_loss.backward()
                self.d_optimizer.step()
                for _ in range(1):
                    self.g_optimizer.zero_grad()
                    # z_fake = torch.normal(torch.zeros(x.shape[0], self.latent_dim),
                    #                       torch.ones(x.shape[0], self.latent_dim)).to(device)
                    z_fake = (torch.rand(x.shape[0], self.latent_dim).to(device) - 0.5) * 2
                    x_fake = self.g(z_fake).view(x.shape[0], -1)
                    z_fake = torch.zeros_like(z_fake).to(device)  #todo: testing if gan is broken

                    # g_loss = (1 - self.d(fakes)).log().mean()
                    g_loss = - (self.d(z_fake, x_fake)).log().mean()
                    g_loss.backward()
                    self.g_optimizer.step()

                self.batch_loss_history.append(d_loss.item())

                if batch_i > 1 and batch_i % self.log_interval == 0:
                    log_string = f'Epoch: {epoch_i} | Itr: {self.curr_itr} | '
                    log_string += f'd_loss: {d_loss:.3f}'
                    tqdm.write(log_string)
                # if self.curr_itr % 50000 == 0:
                #     self.save_model(f"{self.part_name}_epoch{self.curr_itr}.model")
            epoch_loss = np.mean(self.batch_loss_history)
            tqdm.write(f'Epoch Loss: {epoch_loss:.2f}')
            train_losses.append(epoch_loss)
            self.sample(100, f"{self.part_name}_samples{epoch_i}.png")
            np.save("train_losses.npy", np.array(train_losses))

        self.save_models('weights')
        train_losses = np.array(train_losses)
        # self.save_model(f"{self.part_name}.model")
        self.plot_losses(train_losses)

        return train_losses


    def sample(self, n, filename):
        self.g.eval()
        # z = torch.normal(torch.zeros(n, self.latent_dim), torch.ones(n, self.latent_dim)).to(device)
        z = (torch.rand(n, self.latent_dim).to(device) - 0.5) * 2
        samples = self.g(z).reshape(-1, 1, 28, 28) * 0.5 + 1
        save_image(samples, filename, nrow=10, normalize=True)

    def plot_losses(self, train_losses):
        plt.figure()
        plt.plot(np.arange(len(train_losses)), train_losses, label="Train")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.title(self.part_name)
        plt.savefig(f"{self.part_name}_loss.png")
        plt.close()

    def save_models(self, filename):
        torch.save(self.g.state_dict(), "g_" + filename)
        torch.save(self.d.state_dict(), "d_" + filename)
        torch.save(self.e.state_dict(), "e_" + filename)

    def load_models(self, filename):
        self.g.load_state_dict(torch.load("g_" + filename))
        self.d.load_state_dict(torch.load("d_" + filename))
        self.e.load_state_dict(torch.load("e_" + filename))



if __name__ == "__main__":
    solver = Solver(n_epochs=100, batch_size=128, train_data_amount=1, latent_dim=32)
    solver.build("bigan")
    solver.train()
    IPython.embed()
