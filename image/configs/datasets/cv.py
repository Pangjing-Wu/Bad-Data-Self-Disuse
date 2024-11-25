import os

import torchvision


# >>>>>>>>>> CV Dataset Diractory

# <<<<<<<<<< CV Dataset Diractory


# >>>>>>>>>> CV Dataset Setting
cifar10_channel   = 3
cifar10_n_classes = 10

cifar100_channel   = 3
cifar100_n_classes = 100

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

cifar10_norm = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])
cifar10_aug_trans = torchvision.transforms.Compose([
    torchvision.transforms.RandomResizedCrop(32),
    torchvision.transforms.RandomHorizontalFlip(p=0.5),
    torchvision.transforms.RandomApply([torchvision.transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
    torchvision.transforms.RandomGrayscale(p=0.2),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
])

cifar100_norm = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])
cifar100_aug_trans = torchvision.transforms.Compose([
    torchvision.transforms.RandomResizedCrop(32),
    torchvision.transforms.RandomHorizontalFlip(p=0.5),
    torchvision.transforms.RandomApply([torchvision.transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
    torchvision.transforms.RandomGrayscale(p=0.2),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
])

caltech101_norm = torchvision.transforms.Compose([
    torchvision.transforms.Resize(256),
    torchvision.transforms.CenterCrop(224),
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
    torchvision.transforms.Resize(256),
    torchvision.transforms.CenterCrop(224),
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