import random
import argparse
import torch
import numpy as np
from collections import defaultdict 
import itertools

from libs.datasets import make_dataset, make_data_loader
from libs.model import make_resnet
from libs.utils import fix_random_seed
import libs.utils as utils

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
    log_path = f"log/resnet18_{cfg['data']['dataset']}/baseline"
    utils.set_log_path(log_path)
    utils.ensure_path(log_path)

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

    
    

    
    # print(masks)
    masks = np.array([False,  True, False,  True, False, False, False]).reshape(1,7)
    # print(masks)
    # assert False
    
    # assert False
    masks_torch = torch.from_numpy(masks).cuda()
    # print(masks)


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

            macs = (masks_torch[k] * macs_brk[1:]).sum(dim=-1) + macs_brk[0]
            macs.clamp_(max=1)
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
        print(macs)
        # assert False
        
         # Compute overall accuracy
        overall_accuracy = total_correct / total_samples
        print(overall_accuracy)
        over_accs[k] = overall_accuracy
    log_str = ""
    log_str += "macs: {:.2f}({:.2f}) | ".format(macs_total.mean()*100, macs_total.std()*100)
    log_str += "accs: {:.2f}({:.2f})".format(over_accs.mean()*100, over_accs.std()*100)
    print(log_str)

    
    


if __name__ == '__main__':
    args = parse_args()
    main(args)