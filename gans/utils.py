import matplotlib.pyplot as plt
import torch
from torchvision.utils import make_grid


def get_noise(n_samples, z_dim, device='cpu'):
    return torch.randn(n_samples, z_dim, device=device)


def show_images(images, nmax=64):
    _, ax = plt.subplots(figsize=(8, 8))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.imshow(make_grid((images.detach()[:nmax]), nrow=8).permute(1, 2, 0))


def show_batch(dl, nmax=64):
    for images in dl:
        show_images(images, nmax)
        break


def show_tensor_images(image_tensor, num_images=9, size=(4, 96, 96)):
    image_unflat = image_tensor.detach().cpu().view(-1, *size)
    image_grid = make_grid(image_unflat[:num_images], nrow=3)
    plt.imshow(image_grid.permute(1, 2, 0).squeeze())
    plt.show()
