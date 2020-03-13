"""
todos:
1. lenet 1 on full mnist
2. lenet 1 on 10% of mnist (see if the labels are permuted to begin with)
3. gan on mnist
4. bigan on mnist
5. try the reconstructions
6. try the representation learning stuff
"""
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
from torchvision.utils import save_image, make_grid
import IPython

device = torch.device("cuda:0")


class PrintLayer(nn.Module):
    def __init__(self):
        super(PrintLayer, self).__init__()

    def forward(self, x):
        # Do your print / debug stuff here
        print(x.shape)
        return x

class LeNet(nn.Module):

    def __init__(self):
        super().__init__()

        self.model = Encoder()
        self.fc = nn.Linear(50, 10)
        self.criterion = nn.CrossEntropyLoss()


    def forward(self, x):
        out = self.model(x)
        out = out.view(x.size(0), -1)
        out = self.fc(out)
        return out


class Encoder(nn.Module):
    # from https://github.com/grvk/lenet-1/blob/master/LeNet-1.ipynb

    def __init__(self):
        super().__init__()

        # input is Nx1x28x28
        layers = [
            # params: 4*(5*5*1 + 1) = 104
            # output is (28 - 5) + 1 = 24 => Nx4x24x24
            nn.Conv2d(1, 4, 5),
            nn.Tanh(),
            # output is 24/2 = 12 => Nx4x12x12
            nn.AvgPool2d(2),
            # params: (5*5*4 + 1) * 12 = 1212
            # output: 12 - 5 + 1 = 8 => Nx12x8x8
            nn.Conv2d(4, 12, 5),
            nn.Tanh(),
            # output: 8/2 = 4 => Nx12x4x4
            nn.AvgPool2d(2)
        ]
        self.model = nn.Sequential(*layers)
        self.fc = nn.Linear(12 * 4 * 4, 50)

    def forward(self, x):
        x = self.model(x).view(x.shape[0], -1)
        return self.fc(x)


class Discriminator(nn.Module):

    def __init__(self):
        super().__init__()
        # layers = [
        #     nn.Linear(192 + 784, 1024),
        #     nn.LeakyReLU(negative_slope=0.2),
        #     nn.Linear(1024, 1024),
        #     nn.BatchNorm1d(1024),
        #     nn.LeakyReLU(negative_slope=0.2),
        #     nn.Linear(1024, 1),
        #     nn.Sigmoid()
        # ]
        # self.model = nn.Sequential(*layers)

        self.convs = nn.Sequential(
            nn.Conv2d(1, 64, (3,3), stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Dropout(0.4),
            # PrintLayer(),
            nn.Conv2d(64, 64, (3, 3), stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Dropout(0.4),
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * 7 * 7, 1),
            nn.Sigmoid()
        )

        # model = Sequential()
        # model.add(Conv2D(64, (3, 3), strides=(2, 2), padding='same', input_shape=in_shape))
        # model.add(LeakyReLU(alpha=0.2))
        # model.add(Dropout(0.4))
        # model.add(Conv2D(64, (3, 3), strides=(2, 2), padding='same'))
        # model.add(LeakyReLU(alpha=0.2))
        # model.add(Dropout(0.4))
        # model.add(Flatten())
        # model.add(Dense(1, activation='sigmoid'))

    def forward(self, x):
        x = self.convs(x)
        # print(x.shape)
        x = x.view(x.shape[0], -1)
        return self.fc(x)

class Generator(nn.Module):

    def __init__(self):
        super().__init__()
        # layers = [
        #     nn.Linear(192, 1024),
        #     nn.LeakyReLU(negative_slope=0.2),
        #     nn.Linear(1024, 1024),
        #     nn.BatchNorm1d(1024),
        #     nn.LeakyReLU(negative_slope=0.2),
        #     nn.Linear(1024, 784),
        #     nn.Tanh()
        # ]
        #
        # self.model = nn.Sequential(*layers)
        #
        # model = Sequential()
        # # foundation for 7x7 image
        # n_nodes = 128 * 7 * 7

        self.fc = nn.Sequential(nn.Linear(50, 128 * 7 * 7),
                            nn.LeakyReLU(negative_slope=0.2))

        self.convs = nn.Sequential(
            nn.ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.2),
            nn.ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv2d(128, 1, kernel_size=7, stride=1, padding=3),
            nn.Sigmoid()
        )
        # model.add(Dense(n_nodes, input_dim=latent_dim))
        # model.add(LeakyReLU(alpha=0.2))
        # model.add(Reshape((7, 7, 128)))
        # # upsample to 14x14
        # model.add(Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same'))
        # model.add(LeakyReLU(alpha=0.2))
        # # upsample to 28x28
        # model.add(Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same'))
        # model.add(LeakyReLU(alpha=0.2))
        # model.add(Conv2D(1, (7, 7), activation='sigmoid', padding='same'))

    def forward(self, x):
        x = self.fc(x)
        x = x.view(x.shape[0], 128, 7, 7)
        return self.convs(x)

class Generator1(nn.Module):
    def __init__(self, g_input_dim, g_output_dim):
        super(Generator1, self).__init__()
        # self.fc1 = nn.Linear(g_input_dim, 256)
        # self.fc2 = nn.Linear(self.fc1.out_features, self.fc1.out_features * 2)
        # self.fc3 = nn.Linear(self.fc2.out_features, self.fc2.out_features * 2)
        # self.fc4 = nn.Linear(self.fc3.out_features, g_output_dim)
        self.fc = nn.Sequential(
            nn.Linear(g_input_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024, affine=False),
            nn.ReLU(),
            nn.Linear(1024, 784),
            nn.Tanh()
        )


    # forward method
    def forward(self, x):
        return self.fc(x).reshape(x.shape[0], 1, 28, 28)
        # x = F.leaky_relu(self.fc1(x), 0.2)
        # x = F.leaky_relu(self.fc2(x), 0.2)
        # x = F.leaky_relu(self.fc3(x), 0.2)
        # return torch.tanh(self.fc4(x)).reshape(x.shape[0], 1, 28, 28) # tanh

class Discriminator1(nn.Module):
    def __init__(self, z_dim, x_dim):
        super(Discriminator1, self).__init__()
        # self.fc = nn.Sequential(
        #     nn.Linear(z_dim + x_dim, 1024),
        #     nn.Dropout(0.5),
        #     nn.LeakyReLU(0.2),
        #     nn.Linear(1024, 1024),
        #     nn.BatchNorm1d(1024, affine=False),
        #     nn.Dropout(0.5),
        #     nn.LeakyReLU(0.2),
        #     nn.Linear(1024, 1),
        #     nn.Sigmoid()
        # )

        self.fc_z = nn.Sequential(
            nn.Linear(z_dim, 5),
            nn.Dropout(0.5),
            nn.BatchNorm1d(5, affine=False),
            nn.Dropout(0.5),
            nn.LeakyReLU(0.2),
        )

        self.fc_x = nn.Sequential(
            nn.Linear(x_dim, 1024),
            # nn.Dropout(0.5),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512, affine=False),
            # nn.Dropout(0.5),
            nn.LeakyReLU(0.2),
        )

        self.fc_combined = nn.Sequential(
            nn.Linear(512 + 5, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    # forward method
    def forward(self, z, x):
        # x = torch.cat((z, x), dim=1)
        # return self.fc(x)
        h_z = self.fc_z(z)
        h_x = self.fc_x(x.view(-1, 784))
        h = torch.cat((h_z, h_x), dim=1)
        return self.fc_combined(h)

class Encoder1(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Encoder1, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024, affine=False),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, output_dim),
            # nn.Tanh()
        )
        # self.fc1 = nn.Linear(input_dim, 1024)
        # self.fc2 = nn.Linear(self.fc1.out_features, self.fc1.out_features // 2)
        # self.fc3 = nn.Linear(self.fc2.out_features, self.fc2.out_features // 2)
        # self.fc4 = nn.Linear(self.fc3.out_features, output_dim)

    # forward method
    def forward(self, x):
        x = x.view(x.shape[0], -1)
        return self.fc(x)
        # x = F.leaky_relu(self.fc1(x), 0.2)
        # x = F.leaky_relu(self.fc2(x), 0.2)
        # x = F.leaky_relu(self.fc3(x), 0.2)
        # return torch.tanh(self.fc4(x))

class TestEncoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.latent_dim = latent_dim

    def forward(self, x):
        return torch.rand(x.shape[0], self.latent_dim).to(x.device) * 2 - 1


