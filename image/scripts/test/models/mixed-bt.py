import argparse
import sys

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.append('.')
from utils.io.dataset import load_cv_dataset
from utils.io.embedding import ImageEmbeddingManager
sys.path.pop()


def parse_args():
    parser = argparse.ArgumentParser(description="Test Linear Classification on the Embedding Trained by Mixed Barlow Twins")
    # data parameters
    parser.add_argument("--dataset", default='cifar10', type=str, help="dataset name")
    parser.add_argument("--n_classes", default=10, type=int, help="number of class")
    parser.add_argument("--backbone", default="resnet50", type=str, help="backbone architecture name")
    parser.add_argument("--backbone_channel", default=64, type=int, help="number of backbone output channels")
    parser.add_argument("--backbone_kernel_size", default=3, type=int, help="backbone convolution kernel size (suggested `3` for CIFAR, `7` for ImageNet)")
    parser.add_argument("--backbone_stride", default=1, type=int, help="stride of the convolution operation (suggested `1` for CIFAR, `2` for ImageNet)")
    parser.add_argument("--backbone_padding", default=1, type=int, help="backbone padding numbers  (suggested `1` for CIFAR, `3` for ImageNet)")
    parser.add_argument("--backbone_maxpool", default=False, type=bool, help="whether max pooling is used after a convolutional block (suggested `False` for CIFAR)")
    parser.add_argument("--cuda", default=0, type=int, help="cuda device id")
    return parser.parse_args()


def main(args):
    device = f'cuda:{args.cuda}'
    trainset = load_cv_dataset(args.dataset, train=True, augment=False, resize=False, include_index=False)
    testset  = load_cv_dataset(args.dataset, train=False, augment=False, resize=False, include_index=False)
    train_loader = DataLoader(trainset, batch_size=500, num_workers=0)
    test_loader  = DataLoader(testset, batch_size=500, num_workers=0)
    
    backbone_kwargs = dict(
        out_channel=args.backbone_channel,
        kernel=args.backbone_kernel_size,
        stride=args.backbone_stride, 
        padding=args.backbone_padding,
        maxpool=args.backbone_maxpool
    )
    manager = ImageEmbeddingManager(method='mixed-bt', dataset=args.dataset, backbone=args.backbone, backbone_kwargs=backbone_kwargs)
    model   = manager.load().to(device)
    linear  = nn.Linear(model.embedding_dim, args.n_classes).to(device)
    
    # finetune linear
    optimizer = torch.optim.Adam(linear.parameters(), lr=1e-2, weight_decay=5e-6)
    criterion = nn.CrossEntropyLoss()
    for image, label in train_loader:
        image, label = image.to(device), label.to(device)
        embed = model(image)
        prob  = linear(embed)
        loss = criterion(prob, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # rest on training set.
    correct, total = 0, 0    
    for image, label in train_loader:
        with torch.no_grad():
            image, label = image.to(device), label.to(device)
            pred = torch.argmax(linear(model(image)), dim=1)
        total += label.size(0)
        correct += (pred == label).sum().item()
    train_acc = correct / total
    print(f'train accuracy: {train_acc*100:.3f}')
    correct, total = 0, 0
    for image, label in test_loader:
        with torch.no_grad():
            image, label = image.to(device), label.to(device)
            pred = torch.argmax(linear(model(image)), dim=1)
        total += label.size(0)
        correct += (pred == label).sum().item()
    test_acc = correct / total
    print(f'test accuracy: {test_acc*100:.3f}')
    print('Done!')
    

if __name__ == '__main__':
    args = parse_args()
    main(args)