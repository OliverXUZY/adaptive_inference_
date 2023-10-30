import os

from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder, CIFAR10, CIFAR100
from PIL import Image

from .data_utils import *


class ImageNetDataset(ImageFolder):

    def __init__(self, root, split='train', downsample=True):

        assert split in ('train', 'val')
        super(ImageNetDataset, self).__init__(os.path.join(root, split))

        if split == 'train':
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
            ])

        if downsample:
            self.downsample = transforms.Resize(112)
        else:
            self.downsample = None

        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406), 
                std=(0.229, 0.224, 0.225),
            )
        ])

    def __getitem__(self, index):
        path, target = self.samples[index]
        img = self.loader(path)
        img = self.transform(img)
        img_large = self.normalize(img)
        if self.downsample is not None:
            img_small = self.downsample(img)
            img_small = self.normalize(img_small)
        else:
            img_small = img_large

        return img_large, img_small, target


class CIFAR10Dataset(CIFAR10):

    def __init__(self, root, split='train', downsample=True):
        assert split in ('train', 'test')
        train = split == 'train'
        # print("zhuoyan: ", root)
        super(CIFAR10Dataset, self).__init__(root, train, download=True)
        if split == 'train':
            self.transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
            ])
        else:
            self.transform = lambda x: x

        if downsample:
            self.downsample = transforms.Resize(16)
        else:
            self.downsample = None

        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.4914, 0.4822, 0.4465), 
                std=(0.2023, 0.1994, 0.2010),
            )
        ])

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)
        img = self.transform(img)
        img_large = self.normalize(img)
        if self.downsample is not None:
            img_small = self.downsample(img)
            img_small = self.normalize(img_small)
        else:
            img_small = img_large

        return img_large, img_small, target


class CIFAR100Dataset(CIFAR100):

    def __init__(self, root, split='train', downsample=True):
        assert split in ('train', 'test')
        train = split == 'train'
        super(CIFAR100Dataset, self).__init__(root, train, download=True)

        if split == 'train':
            self.transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
            ])
        else:
            self.transform = lambda x: x

        if downsample:
            self.downsample = transforms.Resize(16)
        else:
            self.downsample = None

        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.4914, 0.4822, 0.4465), 
                std=(0.2023, 0.1994, 0.2010),
            )
        ])

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)
        img = self.transform(img)
        img_large = self.normalize(img)
        if self.downsample is not None:
            img_small = self.downsample(img)
            img_small = self.normalize(img_small)
        else:
            img_small = img_large

        return img_large, img_small, target


def make_dataset(dataset, root, split='train', downsample=True):
    if dataset == 'imagenet':
        return ImageNetDataset(root, split, downsample)
    elif dataset == 'cifar10':
        return CIFAR10Dataset(root, split, downsample)
    elif dataset == 'cifar100':
        return CIFAR100Dataset(root, split, downsample)
    else:
        raise NotImplementedError('invalid dataset: {:s}'.format(dataset))


def make_data_loader(dataset, generator, batch_size, num_workers, is_training):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        worker_init_fn=(worker_init_reset_seed if is_training else None),
        shuffle=is_training,
        drop_last=is_training,
        generator=generator,
        persistent_workers=True,
    )
