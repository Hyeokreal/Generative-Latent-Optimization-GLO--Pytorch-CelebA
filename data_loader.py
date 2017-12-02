import os
from torch.utils import data
from torchvision import transforms
from PIL import Image
from args import get_opt
from util import pca_feature
import numpy as np
import torch

opt = get_opt()
image_path = os.path.join(os.getcwd(), 'CelebA', '128_crop')


class ImageFolder(data.Dataset):
    """Custom Dataset compatible with prebuilt DataLoader.

    This is just for tutorial. You can use the prebuilt torchvision.datasets.ImageFolder.
    """

    def __init__(self, root, transform=None):
        """Initializes image paths and preprocessing module."""
        self.image_paths = list(map(lambda x: os.path.join(root, x), os.listdir(root)))

        # for i in range(32):
        #     print(self.image_paths[i])
        self.img = root
        self.transform = transform

    def __getitem__(self, index):
        """Reads an image from a file and preprocesses it and returns."""
        image_path = self.image_paths[index]
        print(image_path)
        image = Image.open(image_path).convert('RGB')
        # image = self.img[index].convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        return image

    def __len__(self):
        """Returns the total number of image files."""
        return len(self.image_paths)


def get_loader(image_path, image_size, batch_size, num_workers=2, image_folder=False,
               shuffle=False):
    """Builds and returns Dataloader."""

    transform = transforms.Compose([
        transforms.Scale(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    dataset = ImageFolder(image_path, transform)

    if image_folder:
        return dataset

    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=shuffle,
                                  num_workers=num_workers)

    return data_loader


def save_image_as_numpy(image_size, latent_size, batch_size, save_image_path='/',
                        save_latent_path='/'):
    transform = transforms.Compose([
        transforms.Scale(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    dataset = ImageFolder(image_path, transform)

    epoch = []
    print(len(dataset))

    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=opt.batch_size,
                                  shuffle=False)

    print("creating image files to numpy..")
    print("It will take few minutes.")

    for i, x in enumerate(data_loader):
        epoch.extend(x.numpy())

    # for i in range(len(dataset)):
    #     epoch.append(dataset[i].numpy())

    image_data = np.array(epoch)

    print("npdata shape", image_data.shape)
    np.save('celeb_a_image.npy', image_data)
    print("creating processed the image numpy file done")


def process_z(image_size, latent_size, batch_size, save_image_path='/', save_latent_path='/'):
    if not os.path.isfile('celeb_a_image.npy'):
        print("no image numpy files. create one.")
        save_image_as_numpy(opt.x_dim, opt.z_dim, opt.batch_size, './', './')

    print("now creating processed latent code z .")

    images = np.load('celeb_a_image.npy')
    print(images.shape)

    z = list()

    print(len(images) // opt.z_dim)
    for i in range(len(images) // opt.z_dim):
        chopped_z = pca_feature(images[i * opt.z_dim:(i + 1) * opt.z_dim], opt.z_dim)
        for i in range(opt.z_dim // opt.batch_size):
            z.append(chopped_z[i * opt.batch_size: (i + 1) * opt.batch_size])

    z = np.asarray(z)

    print("pca done z shape is : ", z.shape)

    for i in range(z.shape[0]):
        z[i] = z[i, :] / np.linalg.norm(z[i, :], 2)

    print("re-processed after pca z shape is  : ", z.shape)
    print(z.shape)

    z_in_ball = torch.FloatTensor(z).view(-1, batch_size, opt.z_dim, 1, 1)

    z_file_name = 'processed_z'
    z_file_name += '_b' + str(opt.batch_size)
    z_file_name += '_z' + str(opt.z_dim)
    z_file_name += '.pt'

    torch.save(z_in_ball, z_file_name)
    print("done with creating z, tensor z size is : ", z_in_ball.size())


def denorm(x):
    """Convert range (-1, 1) to (0, 1)"""
    out = (x + 1) / 2
    return out.clamp(0, 1)


if __name__ == "__main__":
    # process_z(opt.x_dim, opt.z_dim, opt.batch_size, './', './')
    image_path = os.path.join(os.getcwd(), 'CelebA', '128_crop')
    train_loader = get_loader(image_path=image_path,
                              image_size=opt.x_dim,
                              batch_size=opt.batch_size,
                              num_workers=2)


