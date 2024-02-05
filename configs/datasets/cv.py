import os

import torchvision


__all__ = ['cifar10_dir', 'cifar10_channel', 'cifar10_n_classes', 'cifar10_norm', 'cifar10_aug_trans',
           'cifar100_dir', 'cifar100_channel', 'cifar100_n_classes', 'cifar100_norm', 'cifar100_aug_trans',
           'imagenet1k_dir', 'imagenet1k_channel', 'imagenet1k_n_classes', 'imagenet1k_norm', 'imagenet1k_aug_trans',
           'caltech101_dir', 'caltech101_channel', 'caltech101_n_classes', 'caltech101_norm', 'caltech101_aug_trans',
           'caltech256_dir', 'caltech256_channel', 'caltech256_n_classes', 'caltech256_norm', 'caltech256_aug_trans',
           'noise_transform', 'caltech101_transform', 'caltech256_transform']


# >>>>>>>>>> CV Dataset Diractory
CVDATASET_DIR = '/home/wupj/data/cv'
cifar10_dir = os.path.join(CVDATASET_DIR, 'CIFAR-10')
cifar100_dir = os.path.join(CVDATASET_DIR, 'CIFAR-100')
imagenet1k_dir = os.path.join(CVDATASET_DIR, 'ImageNet')
caltech101_dir = os.path.join(CVDATASET_DIR, 'CalTech-101')
caltech256_dir = os.path.join(CVDATASET_DIR, 'CalTech-256')
# <<<<<<<<<< CV Dataset Diractory


# >>>>>>>>>> CV Dataset Setting
cifar10_channel   = 3
cifar10_n_classes = 10

cifar100_channel   = 3
cifar100_n_classes = 100

imagenet1k_channel   = 3
imagenet1k_n_classes = 1000

caltech101_channel   = 3
caltech101_n_classes = 101

caltech256_channel   = 3
caltech256_n_classes = 257
# <<<<<<<<<< CV Dataset Setting


# >>>>>>>>>> CV Dataset Transform
noise_transform = [
        torchvision.transforms.ColorJitter(brightness=.5, hue=.3),
        torchvision.transforms.RandomVerticalFlip(p=0.5),
        torchvision.transforms.RandomHorizontalFlip(p=0.5),
        torchvision.transforms.RandomRotation(degrees=(0, 359)),
        torchvision.transforms.ElasticTransform(alpha=200.)
]
caltech101_transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize(256),
    torchvision.transforms.CenterCrop(224),
])
caltech256_transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize(256),
    torchvision.transforms.CenterCrop(224),
])

cifar10_norm = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])
cifar10_aug_trans = torchvision.transforms.Compose([
    torchvision.transforms.RandomCrop(32, padding=4),
    torchvision.transforms.Resize(32),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

cifar100_norm = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])
cifar100_aug_trans = torchvision.transforms.Compose([
    torchvision.transforms.RandomCrop(32, padding=4),
    torchvision.transforms.Resize(32),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

imagenet1k_norm = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])
imagenet1k_aug_trans = torchvision.transforms.Compose([
    torchvision.transforms.RandomCrop(224, padding=28),
    torchvision.transforms.Resize(224),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

caltech101_norm = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])
caltech101_aug_trans = torchvision.transforms.Compose([
    torchvision.transforms.Resize(256),
    torchvision.transforms.CenterCrop(224),
    torchvision.transforms.RandomCrop(224, padding=28),
    torchvision.transforms.Resize(224),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

caltech256_norm = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])
caltech256_aug_trans = torchvision.transforms.Compose([
    torchvision.transforms.Resize(256),
    torchvision.transforms.CenterCrop(224),
    torchvision.transforms.RandomCrop(224, padding=28),
    torchvision.transforms.Resize(224),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])
# <<<<<<<<<< CV Dataset Transform