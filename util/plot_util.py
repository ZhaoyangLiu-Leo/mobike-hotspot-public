# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

plt.switch_backend('agg')


def plot_losses(source_losses, target_losses, save_path):
    plt.figure(figsize=(8, 6))
    plot_x_axis = range(len(source_losses))

    plt.plot(plot_x_axis, source_losses, '--*', plot_x_axis, target_losses, '-+')
    plt.legend(['source_city', 'target_city'])
    plt.savefig(save_path)


def plot_embedding(X, y, d, title=None):
    """Plot an embedding X with the class label y colored by the domain d."""
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)

    # Plot colors numbers
    plt.figure(figsize=(6,6))
    ax = plt.subplot(111)
    for i in range(X.shape[0]):
        # plot colored number
        plt.text(X[i, 0], X[i, 1], str(y[i]),
                 color=plt.cm.bwr(d[i] / 1.),
                 fontdict={'weight': 'bold', 'size': 9})

    # plt.xticks([]), plt.yticks([])
    if title is not None:
        plt.title(title)


def imshow_grid(images, shape=(2, 8)):
    """Plot images in a grid of a given shape."""
    fig = plt.figure(1)
    grid = ImageGrid(fig, 111, nrows_ncols=shape, axes_pad=0.05)

    size = shape[0] * shape[1]
    for i in range(size):
        grid[i].axis('off')
        grid[i].imshow(images[i])  # The AxesGrid object work as a list of axes.

    plt.show()

