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
import IPython

from models import Generator, Discriminator

device = torch.device("cuda:0")


class Solver(object):
    def __init__(self, n_iterations=50000, batch_size=128, n_filters=256):
        self.n_critic = 5
        self.log_interval = 100
        self.batch_size = batch_size
        self.n_filters = n_filters
        self.train_loader = self.create_loaders()
        self.n_batches_in_epoch = len(self.train_loader)
        self.n_epochs = self.n_critic * n_iterations // self.n_batches_in_epoch
        self.curr_itr = 0

    def build(self, part_name):
        self.g = Generator(n_filters=self.n_filters).to(device)
        self.d = Discriminator().to(device)
        self.g_optimizer = torch.optim.Adam(self.g.parameters(), lr=2e-4, betas=(0, 0.9))
        self.g_scheduler = torch.optim.lr_scheduler.LambdaLR(self.g_optimizer,
                                                             lambda epoch: (100000 - epoch) / 100000,
                                                             last_epoch=-1)
        self.d_optimizer = torch.optim.Adam(self.d.parameters(), lr=2e-4, betas=(0, 0.9))
        self.d_scheduler = torch.optim.lr_scheduler.LambdaLR(self.d_optimizer,
                                                             lambda epoch: (100000 - epoch) / 100000,
                                                             last_epoch=-1)
        self.part_name = part_name

    def create_loaders(self):
        train_data = torchvision.datasets.CIFAR10("./data", transform=torchvision.transforms.ToTensor())

        train_loader = torch.utils.data.DataLoader(train_data, batch_size=self.batch_size, shuffle=True)
        return train_loader

    def gradient_penalty(self, real_data, fake_data):
        batch_size = real_data.shape[0]

        # Calculate interpolation
        eps = torch.rand(batch_size, 1, 1, 1)
        eps = eps.expand_as(real_data).to(device)
        interpolated = eps * real_data.data + (1 - eps) * fake_data.data
        interpolated.requires_grad = True

        d_output = self.d(interpolated)
        gradients = torch.autograd.grad(outputs=d_output, inputs=interpolated,
                                        grad_outputs=torch.ones(d_output.size()).to(device),
                                        create_graph=True, retain_graph=True)[0]

        gradients = gradients.view(batch_size, -1)
        gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)
        return ((gradients_norm - 1) ** 2).mean()

    def train(self):
        train_losses = []
        for epoch_i in trange(self.n_epochs, desc='Epoch', ncols=80):
            epoch_i += 1

            self.d.train()
            self.g.train()
            self.batch_loss_history = []

            for batch_i, x in enumerate(tqdm(self.train_loader, desc='Batch', ncols=80, leave=False)):
                batch_i += 1
                self.curr_itr += 1
                x = x[0].to(device).float()
                x = 2 * (x - 0.5)

                # do a critic update
                self.d_optimizer.zero_grad()
                fake_data = self.g.sample(x.shape[0])
                gp = self.gradient_penalty(x, fake_data)
                d_loss = self.d(fake_data).mean() - self.d(x).mean() + 10*gp
                d_loss.backward()
                self.d_optimizer.step()
                # generator update
                if self.curr_itr % self.n_critic == 0:
                    self.g_optimizer.zero_grad()
                    fake_data = self.g.sample(self.batch_size)
                    g_loss = -self.d(fake_data).mean()
                    g_loss.backward()
                    self.g_optimizer.step()

                    # step the learning rate
                    self.g_scheduler.step()
                    self.d_scheduler.step()

                    self.batch_loss_history.append(g_loss.data.cpu().numpy())

                    if batch_i > 1 and batch_i % self.log_interval == 0:
                        log_string = f'Epoch: {epoch_i} | Itr: {self.curr_itr} | '
                        log_string += f'Generator Loss: {g_loss:.3f}, Critic Loss: {d_loss.data:.3f}'
                        tqdm.write(log_string)
                if self.curr_itr % 50000 == 0:  # todo: implement model saving with just the statedict
                    self.save_model(f"{self.part_name}_epoch{self.curr_itr}.model")
            epoch_loss = np.mean(self.batch_loss_history)
            tqdm.write(f'Epoch Loss: {epoch_loss:.2f}')
            train_losses.append(epoch_loss)
            self.sample(100, f"{self.part_name}_samples{self.curr_itr}.png")
            np.save("train_losses.npy", np.array(train_losses))

        train_losses = np.array(train_losses)
        self.save_model(f"{self.part_name}.model")
        self.plot_losses(train_losses)


        return train_losses

    def sample(self, num_samples, filename):
        self.g.eval()
        samples = self.g.sample(num_samples)
        save_image(samples, filename, nrow=10, normalize=True)

    def sample2(self, num_samples):
        import ipdb; ipdb.set_trace()
        self.g.eval()
        output = []
        for i in range(num_samples//100):
            output.append(self.g.sample(100).detach().cpu().numpy())
        return np.concatenate(output, axis=0)

    def save_model(self, filename):
        torch.save(self.g, "g_" + filename)
        torch.save(self.d, "d_" + filename)

    def load_model(self, filename):
        if '/' in filename:
            path_lst = filename.split('/')
            assert len(path_lst) == 2
            d_path = path_lst[0] + '/d_' + path_lst[1]
            g_path = path_lst[0] + '/g_' + path_lst[1]
        else:
            d_path = "d_" + filename
            g_path = "g_" + filename

        self.d = torch.load(d_path).to(device)
        self.g = torch.load(g_path).to(device)
        # self.vae = torch.vae(filename, map_location="cpu")

    def plot_losses(self, train_losses):
        plt.figure()
        plt.plot(np.arange(len(train_losses)), train_losses, label="Train")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(self.part_name)
        plt.savefig(f"{self.part_name}_loss.png")
        plt.close()


if __name__ == "__main__":
    solver = Solver(n_iterations=50000)
    solver.build("1-1")
    # solver = Solver(n_iterations=50000, batch_size=256, n_filters=128)
    # solver.build("1-2")
    # solver.train()
    solver.load_model("1-1results/1-1.model")
    # solver.load_model("1-2results/1-2.model")
    IPython.embed()
