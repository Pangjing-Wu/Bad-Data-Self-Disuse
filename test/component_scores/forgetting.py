import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.append('.')
from component_scores import ForgettingEvent


class RandomDataset(object):
    
    def __init__(self, n_classes=2, length=5) -> None:
        self.classes   = list(range(n_classes))
        self.length    = length
        self.labels    = torch.from_numpy(np.random.randint(0, n_classes, length))
    
    def __len__(self):
        return self.length

    def __getitem__(self, index):
        x = torch.softmax(torch.Tensor(np.random.randn(len(self.classes))), dim=-1)
        return index, x, self.labels[index]

n  = 4
bs = 4
n_classes = 2
epoch = 5

model = nn.Linear(n_classes, n_classes, bias=False)
setattr(model, 'fc', None)

dataset    = RandomDataset(n_classes, n)
scorer     = ForgettingEvent(dataset=dataset)
dataloader = DataLoader(dataset, batch_size=bs, shuffle=True)

print(f'forgetting event: {scorer.forgetting}')
print(f'unlearned: {scorer.unlearned}')
print(f'prev_acc: {scorer.prev_acc}')

for e in range(epoch):
    print(f'********** epoch #{e+1} **********')
    for b, (i, x, y) in enumerate(dataloader):
        print(f'>>>>>>>>>> batch {b+1}/{len(dataloader)}')
        print(f'index = {i}')
        pred = torch.argmax(model(x), dim=-1)
        print(f'pred == y: {pred.eq(y)}')
        print(f'prev_acc: {scorer.prev_acc}')
        prev_acc = scorer.prev_acc.clone()
        scorer.update(model, i, x, y)
        print(f'acc: {scorer.prev_acc}')
        print(f'acc < prev_acc: {scorer.prev_acc < prev_acc}')
        print(f'forgetting event: {scorer.forgetting}')
        print(f'unlearned: {scorer.unlearned}')
print(f'final forgetting event: {scorer.scores()}')