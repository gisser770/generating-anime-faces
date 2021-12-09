from abc import ABCMeta

import numpy as np
import torch
from torch import nn, Tensor


def init_normal(m):
    if type(m) == nn.Conv2d:
        nn.init.normal_(m.weight, mean=0, std=0.02)
    if type(m) == nn.BatchNorm2d or type(m) == nn.BatchNorm1d:
        nn.init.normal_(m.weight, mean=1., std=0.02)


class VAE(nn.Module, metaclass=ABCMeta):
    def __init__(self, latent_dim: int = 512, batch_size=64):
        super(VAE, self).__init__()
        self.is_cuda = False
        if torch.cuda.is_available():
            self.is_cuda = True
        self.batch_size = batch_size
        self.image_size = 64
        self.s2, self.s4, self.s8, self.s16 = int(self.image_size / 2), int(self.image_size / 4),\
                                              int(self.image_size / 8), int(self.image_size / 16)  # 32,16,8,4
        self.gf_dim = 64
        self.c_dim = 3
        self.z_dim = latent_dim  # 512
        self.ef_dim = 64  # encoder filter number

        # Encoder start
        self.ennet_ho = nn.Sequential(
            nn.Conv2d(in_channels=self.c_dim, out_channels=self.ef_dim,
                      kernel_size=(5, 5), stride=(2, 2), padding=2),
            nn.BatchNorm2d(self.ef_dim),
            nn.ReLU()
        )
        self.ennet_h1 = nn.Sequential(
            nn.Conv2d(in_channels=self.ef_dim, out_channels=self.ef_dim * 2,
                      kernel_size=(5, 5), stride=(2, 2), padding=2),
            nn.BatchNorm2d(self.ef_dim * 2),
            nn.ReLU()
        )
        self.ennet_h2 = nn.Sequential(
            nn.Conv2d(in_channels=self.ef_dim * 2, out_channels=self.ef_dim * 4,
                      kernel_size=(5, 5), stride=(2, 2), padding=2),
            nn.BatchNorm2d(self.ef_dim * 4),
            nn.ReLU()
        )
        self.ennet_h3 = nn.Sequential(
            nn.Conv2d(in_channels=self.ef_dim * 4, out_channels=self.ef_dim * 8,
                      kernel_size=(5, 5), stride=(2, 2), padding=2),
            nn.BatchNorm2d(self.ef_dim * 8),
            nn.ReLU()
        )
        self.ennet_out = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.ef_dim * 8 * (self.image_size//16)**2, self.z_dim * 2),
            nn.Identity(),
            # nn.Linear(self.ef_dim * 8, self.z_dim * 2),
            nn.BatchNorm1d(self.z_dim * 2),
            nn.Identity()
        )

        # Decoder start
        self.denet_in = nn.Sequential(
            nn.Linear(self.z_dim, self.gf_dim * 4 * self.s8 ** 2),
            nn.Identity()
        )
        self.denet_h0 = nn.Sequential(
            nn.BatchNorm2d(self.gf_dim * 4),
            nn.ReLU()
        )
        self.denet_h1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=self.gf_dim * 4,  out_channels=self.gf_dim * 4,
                               kernel_size=6, stride=2, padding=2),
            nn.BatchNorm2d(self.gf_dim * 4),
            nn.ReLU()
        )
        self.denet_h2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=self.gf_dim * 4, out_channels=self.gf_dim * 2,
                               kernel_size=6, stride=2, padding=2),
            nn.BatchNorm2d(self.gf_dim * 2),
            nn.ReLU()
        )
        self.denet_h3 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=self.gf_dim * 2, out_channels=self.gf_dim // 2,
                               kernel_size=6, stride=2, padding=2),
            nn.BatchNorm2d(self.gf_dim // 2),
            nn.ReLU()
        )
        self.denet_h4 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=self.gf_dim // 2, out_channels=self.c_dim,
                               kernel_size=(5, 5), stride=(1, 1), padding=2),
            nn.Tanh(),
            nn.Flatten()
        )

    def encoder(self, x: Tensor) -> Tensor:
        out_x = self.ennet_ho(x)
        out_x = self.ennet_h1(out_x)
        out_x = self.ennet_h2(out_x)
        out_x = self.ennet_h3(out_x)
        out_x = self.ennet_out(out_x)
        return out_x

    def reparameterize(self, mu, logvar):
        stddev = torch.sqrt(torch.exp(logvar))
        epsilon = torch.randn(size=[self.batch_size, self.z_dim], device='cuda')
        # if self.is_cuda:
        #     stddev = stddev.cuda()
        #     epsilon = epsilon.cuda()
        return mu + stddev * epsilon

    def decoder(self, z: Tensor) -> Tensor:
        out_z = self.denet_in(z)
        out_z = torch.reshape(out_z, [-1, self.gf_dim * 4, self.s8, self.s8])
        out_z = self.denet_h0(out_z)
        out_z = self.denet_h1(out_z)
        out_z = self.denet_h2(out_z)
        out_z = self.denet_h3(out_z)
        out_z = self.denet_h4(out_z)
        return out_z

    def forward(self, x: Tensor) -> Tensor:
        self.batch_size = x.shape[0]
        encoded = self.encoder(x)
        mu = encoded[:, :self.z_dim]
        logvar = encoded[:, self.z_dim:]
        reparametrized = self.reparameterize(mu, logvar)
        output = self.decoder(reparametrized)
        return output, mu, logvar

