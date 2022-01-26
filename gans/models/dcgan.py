"""
Much of the code in this script is based on the programming assignments
from Coursera's "Generative Adversarial Networks (GANs) Specialization"
by Sharon Zhou et al.
(https://www.coursera.org/specializations/generative-adversarial-networks-gans)
"""

import torch
from torch import nn

from gans import utils


class Generator(nn.Module):
    def __init__(self, z_dim=10, im_channels=4, kernel_size=4, stride=2, hidden_dim=256):
        super(Generator, self).__init__()
        self.z_dim = z_dim
        self.gen = nn.Sequential(
            self.get_generator_block(z_dim, hidden_dim * 8, kernel_size, stride),
            self.get_generator_block(hidden_dim * 8, hidden_dim * 4, kernel_size, stride),
            self.get_generator_block(hidden_dim * 4, hidden_dim * 2, kernel_size, stride),
            self.get_generator_block(hidden_dim * 2, hidden_dim, kernel_size, stride),
            nn.ConvTranspose2d(hidden_dim, im_channels, kernel_size, stride=stride),
            nn.Tanh()
        )

    @staticmethod
    def get_generator_block(input_channels, output_channels, kernel_size, stride):
        return nn.Sequential(
            nn.ConvTranspose2d(input_channels, output_channels, kernel_size, stride=stride),
            nn.BatchNorm2d(output_channels),
            nn.ReLU()
        )

    def forward(self, noise):
        x = noise.view(len(noise), self.z_dim, 1, 1)
        return self.gen(x)


class Discriminator(nn.Module):
    def __init__(self, im_channels=4, hidden_dim=512, kernel_size=4, stride=2):
        super(Discriminator, self).__init__()
        self.disc = nn.Sequential(
            self.get_discriminator_block(im_channels, hidden_dim, kernel_size, stride),
            self.get_discriminator_block(hidden_dim, hidden_dim * 2, kernel_size, stride),
            nn.Conv2d(hidden_dim * 2, 1, kernel_size, stride=stride)
        )

    @staticmethod
    def get_discriminator_block(input_channels, output_channels, kernel_size, stride):
        return nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size, stride=stride),
            nn.BatchNorm2d(output_channels),
            nn.LeakyReLU(0.2)
        )

    def forward(self, image):
        x = self.disc(image)
        return x.view(len(x), -1)


def get_disc_loss(gen, disc, criterion, real, num_images, z_dim, device):
    noise = utils.get_noise(num_images, z_dim, device=device)
    fake = gen(noise).detach()
    disc_pred_fake = disc(fake)
    fake_loss = criterion(disc_pred_fake, torch.zeros_like(disc_pred_fake))
    disc_pred_real = disc(real)
    real_loss = criterion(disc_pred_real, torch.ones_like(disc_pred_real))
    disc_loss = (real_loss + fake_loss) / 2
    return disc_loss


def get_gen_loss(gen, disc, criterion, num_images, z_dim, device):
    noise = utils.get_noise(num_images, z_dim, device=device)
    fake = gen(noise)
    disc_pred = disc(fake)
    gen_loss = criterion(disc_pred, torch.ones_like(disc_pred))
    return gen_loss








