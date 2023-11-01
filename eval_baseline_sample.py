import random
import argparse
import torch
import numpy as np
from collections import defaultdict 

from libs.datasets import make_dataset, make_data_loader
from libs.model import make_resnet
from libs.utils import fix_random_seed, check_file
from libs.core import load_config
import libs.utils as utils

def parse_args():
    parser = argparse.ArgumentParser(description="This is my training script.")
    parser.add_argument('-c', '--config', type=str, help='config file path')
    parser.add_argument('-n', '--name', type=str, help='job name')
    parser.add_argument('-g', '--gpu', type=str, default='0', help='GPU IDs')
    parser.add_argument('-m', '--model', type=str, default='resnet50', help='backbone')
    parser.add_argument('--dataset', type=str, default='imagenet', help='The dataset we used')


    args = parser.parse_args()
    return args

def main(args):
    rng = fix_random_seed(2023)
    # check_file(args.config)
    # cfg = load_config(args.config)
    # print(cfg)
    # assert False
    cfg = defaultdict(dict)
    if args.dataset == 'cifar10':
        cfg['data'] = {'dataset': 'cifar10',
                        'root': '/srv/home/zxu444/datasets/cifar10_dataset ',
                        'downsample': True,
                        'train_split': 'train',
                        'val_split': 'test',
                        'batch_size': 64,
                        'num_workers': 8}

        net, macs_brk = make_resnet(args.model, 'cifar10', True)
    elif args.dataset == 'imagenet':
        cfg['data'] = {'dataset': 'imagenet',
                        'root': '/backup/zhuoyan/datasets/imagenet',
                        'downsample': True,
                        'train_split': 'train',
                        'val_split': 'val',
                        'batch_size': 16,
                        'num_workers': 12}

        net, macs_brk = make_resnet(args.model, 'imagenet', True)

    macs_brk = macs_brk.cuda()
    net = net.cuda()
    net.eval()
    utils.set_log_path(f"log/{args.model}_{cfg['data']['dataset']}")
    utils.ensure_path(f"log/{args.model}_{cfg['data']['dataset']}")

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

    ### construct masks
    # do not skip first block!
    if args.model == "resnet18":
        num_block = 7
    elif args.model == "resnet50":
        num_block = 15
    else:
        raise NotImplementedError

    
    masks = np.ones((128, num_block))
    # Set random seed for the built-in random module
    seed_value = 42  # you can choose any number you like
    random.seed(seed_value)
    skip_block = args.skip_block
    for i in range(128): # the last one is always true, skip no blocks
        idx = random.sample(range(num_block), skip_block)
        masks[i, idx] = 0
    # print(masks)
    masks = masks.astype(bool)
    
    # assert False
    masks_torch = torch.from_numpy(masks).cuda()
    # print(masks)
    # assert False
    log_str = f"skip {skip_block} block | "

    accs = np.zeros((len(masks), len(val_loader)))

    over_accs = np.zeros(len(masks))
   
    macs_total = np.zeros(len(masks))
    for k in range(len(masks)):
        # Initialize counters
        total_correct = 0
        total_samples = 0
        # macss = []
        for idx, (x, _, y) in enumerate(val_loader):
            x, y = x.cuda(), y.cuda()
            # print("y.shape", y.shape)
            mask = masks_torch[k].repeat(y.shape[0], 1)
            with torch.no_grad():
                logits = net(x, mask)
            _, pred = logits.max(dim=1)
            is_correct = pred == y
            acc = is_correct.sum() / y.shape[0]
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
        # print(macs)
        # assert False
        
         # Compute overall accuracy
        overall_accuracy = total_correct / total_samples
        over_accs[k] = overall_accuracy
    log_str += "macs: {:.2f}({:.2f}) | ".format(macs_total.mean()*100, macs_total.std()*100)
    log_str += "accs: {:.2f}({:.2f})".format(over_accs.mean()*100, over_accs.std()*100)
    utils.log(log_str,"baseline.txt")

    np.savez(
        f"log/{args.model}_{cfg['data']['dataset']}/{args.model}_cifar10_skip{skip_block}.npz", 
        masks=masks,
        accs=accs.astype(float),
        over_accs = over_accs.astype(float),
        macs_total = macs_total.astype(float)
    )


if __name__ == '__main__':
    args = parse_args()
    # do not skip first block!
    if args.model == "resnet18":
        num_block = 8
    elif args.model == "resnet50":
        num_block = 16
    else:
        raise NotImplementedError

    for skip_block in range(1,num_block):
        args.skip_block = skip_block
        # print(args)
        main(args)