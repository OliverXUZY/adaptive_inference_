import random
import argparse
import torch
import numpy as np
from collections import defaultdict 

from libs.datasets import make_dataset, make_data_loader
from libs.model import make_resnet
from libs.utils import fix_random_seed, check_file
from libs.core import load_config

def parse_args():
    parser = argparse.ArgumentParser(description="This is my training script.")
    parser.add_argument('-c', '--config', type=str, help='config file path')
    parser.add_argument('-n', '--name', type=str, help='job name')
    parser.add_argument('-g', '--gpu', type=str, default='0', help='GPU IDs')

    args = parser.parse_args()
    return args

def main(args):
    rng = fix_random_seed(2023)
    # check_file(args.config)
    # cfg = load_config(args.config)
    # print(cfg)
    # assert False
    cfg = defaultdict(dict)
    cfg['data'] = {'dataset': 'cifar10',
                    'root': '/srv/home/zxu444/datasets/cifar10_dataset ',
                    'downsample': True,
                    'train_split': 'train',
                    'val_split': 'test',
                    'batch_size': 64,
                    'num_workers': 8}

    net, macs_brk = make_resnet('resnet18', 'cifar10', True)
    macs_brk = macs_brk.cuda()
    net = net.cuda()
    net.eval()

    val_set = make_dataset(
        dataset=cfg['data']['dataset'],
        root=cfg['data']['root'],
        split=cfg['data']['val_split'],
        downsample=cfg['data'].get('downsample', False),
    )
    val_loader = make_data_loader(
        val_set, 
        generator=rng,
        batch_size=cfg['data']['batch_size'],
        num_workers=cfg['data']['num_workers'],
        is_training=False,
    )

    print('val data size: {:d}'.format(len(val_set)))

    masks = np.zeros((16, 7))
    for i in range(15):
        idx = random.sample(range(7), 5)
        masks[i, idx] = 1
    masks = masks.astype(bool)
    masks_torch = torch.from_numpy(masks).cuda()
    # print(masks)
    # assert False

    accs = np.zeros((len(masks), len(val_loader)))

    over_accs = np.zeros(len(masks))
    # Initialize counters
    total_correct = 0
    total_samples = 0
    macs_total = np.zeros(len(masks))
    for k in range(len(masks)):
        # macss = []
        for idx, (x, _, y) in enumerate(val_loader):
            x, y = x.cuda(), y.cuda()
            # print("y.shape", y.shape)
            mask = masks_torch[k].repeat(y.shape[0], 1)
            with torch.no_grad():
                logits = net(x, mask)
            _, pred = logits.max(dim=1)
            is_correct = pred == y
            acc = is_correct.sum() / cfg['data']['batch_size']
            accs[k][idx] = acc.item()

            # macs = (masks_torch[k] * macs_brk[1:]).sum(dim=-1) + macs_brk[0]
            # macs.clamp_(max=1)
            # print("masks_torch[k] shape", masks_torch[k].shape)
            # print("masks_torch[k]", masks_torch[k])
            # print("macs_brk: ", macs_brk)
            # print("macs", macs)

            # Update counters
            total_correct += (pred == y).sum().item()
            total_samples += y.size(0)
        macs = (masks_torch[k] * macs_brk[1:]).sum(dim=-1) + macs_brk[0]
        macs.clamp_(max=1)
        macs_total[k] = macs.item()
        # assert False
        
         # Compute overall accuracy
        overall_accuracy = total_correct / total_samples
        over_accs[k] = overall_accuracy

    np.savez(
        'resnet18_cifar10_skip5.npz', 
        masks=masks,
        accs=accs.astype(float),
        over_accs = over_accs.astype(float),
        macs_total = macs_total.astype(float)
    )

if __name__ == '__main__':
    args = parse_args()
    
    main(args)