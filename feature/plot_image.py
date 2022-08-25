import matplotlib.pyplot as plt
import torch
from torchvision import transforms
import numpy as np

classes = ['a', 'b', 'c', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
mean, std = torch.tensor([0.485, 0.456, 0.406]), torch.tensor([0.229, 0.224, 0.225])


def denormalize(image):
    image = transforms.Normalize(-mean / std, 1 / std)(image)  # denormalize
    image = image.permute(1, 2, 0)  # Changing from 3x224x224 to 224x224x3
    image = torch.clamp(image, 0, 1)
    return image


# helper function to un-normalize and display an image
def imshow2(img):
    img = denormalize(img)
    plt.imshow(img)


def plot_img(train_images, train_labels, val_images, val_labels):
    # plot img
    fig = plt.figure(figsize=(50, 8))
    # display 20 images
    for idx in np.arange(8):
        ax = fig.add_subplot(2, int(8 / 2), idx + 1, xticks=[], yticks=[])
        # import pdb
        # pdb.set_trace()
        imshow2(train_images[idx])
        ax.set_title("{} ".format(classes[train_labels[idx]]))
    plt.show()

    # plot img
    fig2 = plt.figure(figsize=(50, 8))
    # display 20 images
    for idx in np.arange(8):
        ax2 = fig2.add_subplot(2, int(8 / 2), idx + 1, xticks=[], yticks=[])
        # import pdb
        # pdb.set_trace()
        imshow2(val_images[idx])
        ax2.set_title("{} ".format(classes[val_labels[idx]]))
    plt.show()
