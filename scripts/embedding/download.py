import argparse
import sys

import torch.nn as nn
import torchvision
import torchvision.models.resnet as resnet
from transformers import BeitForImageClassification, ViTForImageClassification, ViTMAEModel

sys.path.append('.')
import configs
from utils.nn.io import save_embedding_state


def parse_args():
    parser = argparse.ArgumentParser(description='Supervised Model Embedding')
    parser.add_argument('nets', type=str, nargs='+', help='DNN model name {resnet18/resnet50/vit/mae/beit}')
    parser.add_argument('-o', '--override', action='store_true', help='override existing models')
    return parser.parse_args()


def main(args):
    AVAILABLE_MODELS = set(('resnet18', 'resnet34', 'resnet50', 'vit', 'mae', 'beit'))
    __format = lambda text: text.lower().replace('-', '').replace('_', '')
    nets = set(map(__format, args.nets))
    # check input models.
    for model in nets:
        if __format(model) not in AVAILABLE_MODELS:
            raise ValueError(f"available models are `{', '.join(AVAILABLE_MODELS)}`, but got an unknown model `{model}`.")
    # save pre-trained models.
    if 'resnet18' in nets:
        model  = resnet.resnet18(weights=eval(configs.resnet18_weights))
        model.fc = nn.Identity()
        state  = model.state_dict()
        metric = eval(configs.resnet18_weights).meta['_metrics']['ImageNet-1K']['acc@1']
        save_embedding_state(state, 'resnet18', configs.resnet18_dataset, algo='pt', date='666666-6666', 
                             epoch=0, metric=f'acc{int(metric*100)}', best=True, override=args.override)
    if 'resnet34' in nets:
        model  = resnet.resnet34(weights=eval(configs.resnet34_weights))
        model.fc = nn.Identity()
        state  = model.state_dict()
        metric = eval(configs.resnet34_weights).meta['_metrics']['ImageNet-1K']['acc@1']
        save_embedding_state(state, 'resnet34', configs.resnet34_dataset, algo='pt', date='666666-6666', 
                             epoch=0, metric=f'acc{int(metric*100)}', best=True, override=args.override)
    if 'resnet50' in nets:
        model  = resnet.resnet50(weights=eval(configs.resnet50_weights))
        model.fc = nn.Identity()
        state  = model.state_dict()
        metric = eval(configs.resnet50_weights).meta['_metrics']['ImageNet-1K']['acc@1']
        save_embedding_state(state, 'resnet50', configs.resnet34_dataset, algo='pt', date='666666-6666', 
                             epoch=0, metric=f'acc{int(metric*100)}', best=True, override=args.override)
    if 'vit' in nets:
        model = ViTForImageClassification.from_pretrained(configs.vit_name)
        model.classifier = nn.Identity()
        state = model.state_dict()
        save_embedding_state(state, 'vit', configs.vit_dataset, algo='pt', date='666666-6666', 
                             epoch=0, metric=f'acc0000', best=True, override=args.override)
    if 'mae' in nets:
        state = ViTMAEModel.from_pretrained(configs.mae_name).state_dict()
        save_embedding_state(state, 'mae', configs.mae_dataset, algo='pt', date='666666-6666', 
                             epoch=0, metric=f'acc0000', best=True, override=args.override)
    if 'beit' in nets:
        model = BeitForImageClassification.from_pretrained(configs.beit_name)
        model.classifier = nn.Identity()
        state = model.state_dict()
        save_embedding_state(state, 'beit', configs.beit_dataset, algo='pt', date='666666-6666', 
                             epoch=0, metric=f'acc0000', best=True, override=args.override)
            
            
if  __name__ == '__main__':
    args = parse_args()
    main(args)