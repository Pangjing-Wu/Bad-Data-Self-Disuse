import argparse
import datetime
import sys
import traceback

import torch
import torch.nn as nn

sys.path.append('.')
import configs
from models.resnet import *
from utils.dataset import load_cv_dataset
from utils.logger import set_logger
from utils.nn import set_seed
from utils.nn.classifier import train, test
from utils.nn.io import save_embedding_state
from utils.path import set_embedding_log_path


device = torch.device('cuda:0')


def parse_args():
    parser = argparse.ArgumentParser(description='Supervised Model Embedding')
    parser.add_argument('--dataset', required=True, help='dataset name {cifar10/cifar100}')
    parser.add_argument('--net', required=True, type=str, help='DNN model name {resnet18/resnet50}')
    parser.add_argument('--epoch', type=int, default=50, help='training epoch (default: 50)')
    parser.add_argument('--lr', type=float, default=1e-3, help='initial learning rate (default: 0.001)')
    parser.add_argument('--bs', type=int, default=128, help='training batch size (default: 128)')
    parser.add_argument('--ckp', type=int, default=0, help='save checkpoint per epoch (default: 0)')
    parser.add_argument('--n_jobs', type=int, default=0, help='n workers for dataloader (default: 0)')
    parser.add_argument('--seed', type=int, default=0, help='random seed (default: 0)')
    parser.add_argument('--augment', action='store_true', help='augment training data')
    parser.add_argument('--debug', action='store_true', help='running without saving model')
    return parser.parse_args()


def main(args, logger):
    # set random seed.
    set_seed(args.seed)
    
    # log parameters.
    logger.info(f'====== Training Supervised Embedding ======')
    logger.info('\n'.join([f'{k} = {str(v)}' for k, v in sorted(dict(vars(args)).items())]))
    
    # load dataset.
    trainset = load_cv_dataset(args.dataset, train=True, augment=args.augment)
    testset  = load_cv_dataset(args.dataset, train=False, augment=False)
    logger.info(f'load dataset {args.dataset} with batch size {args.bs}.') 
    logger.info(f'trainloader shuffle = True, testloader shuffle = False.')
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.bs, shuffle=True, num_workers=args.n_jobs)
    testloader  = torch.utils.data.DataLoader(testset, batch_size=args.bs, shuffle=False, num_workers=args.n_jobs)
    logger.info('training transform:\n' + '\n'.join([f'==> {str(t)}' for t in trainset.transform.transforms]))
    logger.info('test transform:\n' + '\n'.join([f'==> {str(t)}' for t in testset.transform.transforms]))

    # set model.
    param = getattr(configs, f'{args.net}_{args.dataset}_params')
    if args.net == 'resnet18':
        model = resnet18(**param)
    elif args.net == 'resnet34':
        model = resnet34(**param)
    elif args.net == 'resnet50':
        model = resnet50(**param)
    else:
        raise ValueError(f"unknown model '{args.net}'.")
    logger.info(f"set model {args.net} with {param['n_channel']} input channel and {param['n_classes']} output classes.")
    
    # set scaler and model.
    model     = model.to(device)
    criterion = nn.CrossEntropyLoss(reduction='mean')
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    schedule  = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epoch)
    logger.info(f"criterion = CrossEntropyLoss; optimizer = Adam; use scaler = True.")
    logger.info(f"use learning rate schedule: CosineAnnealingLR(T_max={args.epoch}).")

    # training model.
    best_epoch, best_acc = 0, 0
    for e in range(1, args.epoch + 1):
        ret = train(model, trainloader, optimizer, criterion, schedule)
        logger.info(f"Epoch {e:3}/{args.epoch}: train loss:{ret.loss:.5f}, train acc: {ret.acc:.3f}, lr: {ret.lr:.3e}.")
        ret = test(model, testloader, criterion)
        logger.info(f"Epoch {e:3}/{args.epoch}: test loss:{ret.loss:.5f}, acc: {ret.acc:.3f}.") 
        # saving checkpoint.
        if not args.debug and args.ckp > 0 and e % args.ckp == 0:
            state = model.cpu().state_dict()
            save_embedding_state(state, args.net, args.dataset, algo='sl', date=date, 
                           epoch=e, metric=f'acc{int(ret.acc*100)}', best=False)
            model.to(device)
        # cache best model.
        if ret.acc > best_acc:
            best_epoch, best_acc = e, ret.acc
            best_state = model.cpu().state_dict()
            model.to(device)
    # save best model.
    logger.debug(f'obtained best model at epoch {best_epoch} with acc {best_acc:.2f}%, saved.')
    if not args.debug:
        logger.info(f'obtained best model at epoch {best_epoch} with acc {best_acc:.2f}%, saved.')
        save_embedding_state(best_state, args.net, args.dataset, algo='sl', date=date, 
                             epoch=best_epoch, metric=f'acc{int(best_acc*100)}', best=True)
    logger.info('done!')


if __name__ == '__main__':
    # python ./scripts/embedding/supervised.py --dataset cifar10 --net resnet18 --bs 128 --lr 1e-3 --ckp 10 --epoch 50 --debug
    args    = parse_args()
    date    = datetime.datetime.now().strftime('%y%m%d-%H%M')
    logpath = set_embedding_log_path(args.net, args.dataset, algo='sl', date=date)
    logger  = set_logger(name='sl-embed', level='DEBUG') if args.debug else set_logger(name='sl-embed', logfile=logpath)
    logger.debug(f"save log to: '{logpath}'.")
    try:
        main(args, logger)
    except:
        logger.error(traceback.format_exc())