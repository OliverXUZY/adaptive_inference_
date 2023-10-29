import random

import torch
import numpy as np

from libs.datasets import make_dataset, make_data_loader
from libs.model import make_resnet
from libs.utils import fix_random_seed


rng = fix_random_seed(2022)

net = make_resnet('resnet50', 'imagenet', False).cuda()
net.eval()

val_set = make_dataset(
    dataset='imagenet',
    root='./data/imagenet',
    split='val',
)
val_loader = make_data_loader(
    val_set, 
    generator=rng,
    batch_size=50,
    num_workers=12,
    is_training=False,
)

masks = np.ones((15, 15))
for i in range(15):
    idx = random.sample(range(15), 2)
    masks[i, idx] = 0
masks = masks.astype(bool)
masks_torch = torch.from_numpy(masks).cuda()

accs = np.zeros((len(masks), 1000))
for k in range(len(masks)):
    for idx, (x, _, y) in enumerate(val_loader):
        x, y = x.cuda(), y.cuda()
        mask = masks_torch[k].repeat(50, 1)
        with torch.no_grad():
            logits = net(x, mask)
        _, pred = logits.max(dim=1)
        is_correct = pred == y
        acc = is_correct.sum() / 50
        accs[k][idx] = acc.item()

np.savez(
    '13.npz', 
    masks=masks,
    accs=accs.astype(float),
)
