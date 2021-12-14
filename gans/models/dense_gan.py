"""
Much of the code in this script is based on the programming assignments
from Coursera's "Generative Adversarial Networks (GANs) Specialization"
by Sharon Zhou et al.
(https://www.coursera.org/specializations/generative-adversarial-networks-gans)
"""

import torch
from torch import nn


def get_generator_block(input_dim, output_dim):
    return nn.Sequential(
        nn.Linear(input_dim, output_dim),
        nn.BatchNorm1d(output_dim),
        nn.ReLU(inplace=True)
    )


class Generator(nn.Module):
    def __init__(self, z_dim=10, im_dim=4*96*96, hidden_dim=512):
        super(Generator, self).__init__()
        self.gen = nn.Sequential(
            get_generator_block(z_dim, hidden_dim),
            get_generator_block(hidden_dim, hidden_dim * 2),
            get_generator_block(hidden_dim * 2, hidden_dim * 4),
            get_generator_block(hidden_dim * 4, hidden_dim * 8),
            nn.Linear(hidden_dim * 8, im_dim),
            nn.Sigmoid()
        )

    def forward(self, noise):
        return self.gen(noise)


def get_noise(n_samples, z_dim, device='cpu'):
    return torch.randn(n_samples, z_dim, device=device)


def get_discriminator_block(input_dim, output_dim):
    return nn.Sequential(
        nn.Linear(input_dim, output_dim),
        nn.LeakyReLU(0.2)
    )


class Discriminator(nn.Module):
    def __init__(self, im_dim=4*96*96, hidden_dim=512):
        super(Discriminator, self).__init__()
        self.disc = nn.Sequential(
            get_discriminator_block(im_dim, hidden_dim * 4),
            get_discriminator_block(hidden_dim * 4, hidden_dim * 2),
            get_discriminator_block(hidden_dim * 2, hidden_dim),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, image):
        return self.disc(image)


def get_disc_loss(gen, disc, criterion, real, num_images, z_dim, device):
    noise = get_noise(num_images, z_dim, device=device)
    fake = gen(noise).detach()
    disc_pred_fake = disc(fake)
    fake_loss = criterion(disc_pred_fake, torch.zeros_like(disc_pred_fake))
    disc_pred_real = disc(real)
    real_loss = criterion(disc_pred_real, torch.ones_like(disc_pred_real))
    disc_loss = (real_loss + fake_loss) / 2
    return disc_loss


def get_gen_loss(gen, disc, criterion, num_images, z_dim, device):
    noise = get_noise(num_images, z_dim, device=device)
    fake = gen(noise)
    disc_pred = disc(fake)
    gen_loss = criterion(disc_pred, torch.ones_like(disc_pred))
    return gen_loss








