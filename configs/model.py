from .datasets import *


# >>>>>>>>>>>> Pre-trained Model Name
resnet18_dataset = 'imagenet1k'
resnet18_weights = 'torchvision.models.ResNet18_Weights.IMAGENET1K_V1'
resnet34_dataset = 'imagenet1k'
resnet34_weights = 'torchvision.models.ResNet34_Weights.IMAGENET1K_V1'
resnet50_dataset = 'imagenet1k'
resnet50_weights = 'torchvision.models.ResNet50_Weights.IMAGENET1K_V2'

vit_dataset  = 'imagenet21k'
vit_name     = 'google/vit-base-patch16-224-in21k'
mae_dataset  = 'imagenet21k'
mae_name     = 'facebook/vit-mae-base'
beit_dataset = 'imagenet21k'
beit_name    = 'microsoft/beit-base-patch16-224'

# >>>>>>>>>>>> NN Projection Layer
resnet_proj_layer_name = 'fc'
vit_proj_layer_name    = 'classifier'
# <<<<<<<<<<<< NN Projection Layer

# >>>>>>>>>>>> NN Parameters
resnet18_cifar10_params    = dict(n_classes=cifar10_n_classes, n_channel=cifar10_channel, kernel=3, stride=1, padding=1, maxpool=False, input_size=32)
resnet18_cifar100_params   = dict(n_classes=cifar100_n_classes, n_channel=cifar100_channel, kernel=3, stride=1, padding=1, maxpool=False, input_size=32)
resnet18_imagenet1k_params = dict(n_classes=imagenet1k_n_classes, n_channel=imagenet1k_channel, kernel=7, stride=2, padding=3, maxpool=True, input_size=224)
resnet18_caltech101_params = dict(n_classes=caltech101_n_classes, n_channel=imagenet1k_channel, kernel=7, stride=2, padding=3, maxpool=True, input_size=224)
resnet18_caltech256_params = dict(n_classes=caltech256_n_classes, n_channel=imagenet1k_channel, kernel=7, stride=2, padding=3, maxpool=True, input_size=224)

resnet34_cifar10_params    = dict(n_classes=cifar10_n_classes, n_channel=cifar10_channel, kernel=3, stride=1, padding=1, maxpool=False, input_size=32)
resnet34_cifar100_params   = dict(n_classes=cifar100_n_classes, n_channel=cifar100_channel, kernel=3, stride=1, padding=1, maxpool=False, input_size=32)
resnet34_imagenet1k_params = dict(n_classes=imagenet1k_n_classes, n_channel=imagenet1k_channel, kernel=7, stride=2, padding=3, maxpool=True, input_size=224)
resnet34_caltech101_params = dict(n_classes=caltech101_n_classes, n_channel=imagenet1k_channel, kernel=7, stride=2, padding=3, maxpool=True, input_size=224)
resnet34_caltech256_params = dict(n_classes=caltech256_n_classes, n_channel=imagenet1k_channel, kernel=7, stride=2, padding=3, maxpool=True, input_size=224)

resnet50_cifar10_params    = dict(n_classes=cifar10_n_classes, n_channel=cifar10_channel, kernel=3, stride=1, padding=1, maxpool=False, input_size=32)
resnet50_cifar100_params   = dict(n_classes=cifar100_n_classes, n_channel=cifar100_channel, kernel=3, stride=1, padding=1, maxpool=False, input_size=32)
resnet50_imagenet1k_params = dict(n_classes=imagenet1k_n_classes, n_channel=imagenet1k_channel, kernel=7, stride=2, padding=3, maxpool=True, input_size=224)
resnet50_caltech101_params = dict(n_classes=caltech101_n_classes, n_channel=imagenet1k_channel, kernel=7, stride=2, padding=3, maxpool=True, input_size=224)
resnet50_caltech256_params = dict(n_classes=caltech256_n_classes, n_channel=imagenet1k_channel, kernel=7, stride=2, padding=3, maxpool=True, input_size=224)

vit_imagenet21k_params     = dict(name=vit_name, input_size=224)
mae_imagenet21k_params     = dict(name=mae_name, input_size=224)
beit_imagenet21k_params    = dict(name=beit_name, input_size=224)
# <<<<<<<<<<<< NN Parameters


# >>>>>>>>>>>> ML Parameters
random_forest_cifar10_params  = dict(n_estimators=100, criterion='gini', n_jobs=10)
random_forest_cifar100_params = dict(n_estimators=100, criterion='gini', n_jobs=10)

label_spreading_cifar10_params  = dict(kernel='rbf', gamma=1.0, alpha=0.05, max_iter=100, n_jobs=10)
label_spreading_cifar100_params = dict(kernel='rbf', gamma=1.0, alpha=0.05, max_iter=100, n_jobs=10)
# label_spreading_cifar10_params  = dict(kernel='knn', n_neighbors=10, alpha=0.05, max_iter=100, n_jobs=10)
# label_spreading_cifar100_params = dict(kernel='knn', n_neighbors=10, alpha=0.05, max_iter=100, n_jobs=10)
# <<<<<<<<<<<< ML Parameters