directory="./results/embeddings/mixed-bt"
mkdir -p $directory
wget https://github.com/wgcban/mix-bt/releases/download/v1.0.0/v3gwgusq_0.0078125_1024_256_cifar10_model.pth -O "$directory/cifar10-resnet50.pth"
wget https://github.com/wgcban/mix-bt/releases/download/v1.0.0/z6ngefw7_0.0078125_1024_256_cifar100_model.pth -O "$directory/cifar100-resnet50.pth"
wget https://github.com/wgcban/mix-bt/releases/download/v1.0.0/kxlkigsv_0.0009765_1024_256_tiny_imagenet_model.pth -O "$directory/tinyimagenet-resnet50.pth"