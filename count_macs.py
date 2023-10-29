import os
import argparse

import torch
import numpy as np
from ptflops import get_model_complexity_info

from libs.model import make_resnet


def input_constructor_factory(mask):
    def input_constructor(size):
        x = torch.empty(size).cuda()
        return {'x': x[None], 'mask': mask[None]}
    return input_constructor

def main(args):
    # load model
    print('Loading model...')
    net = make_resnet(args.arch, args.dataset, return_macs=False)
    net.cuda()

    # define branch masks
    n_knobs = net.n_blocks - 1
    print('Number of knobs: {:d}'.format(n_knobs))
    print('Building masks...')
    masks = np.concatenate(
        [np.zeros((1, n_knobs)), np.tril(np.ones((n_knobs, n_knobs)))]
    )
    masks = torch.from_numpy(masks).bool()
    masks = masks.cuda()

    # input size
    size = (3, 32, 32) if args.dataset == 'cifar' else (3, 224, 224)

    print('Calculating per-block MACs breakdown...')
    all_macs = []
    for mask in masks:
        macs, _ = get_model_complexity_info(
            net, size,
            input_constructor=input_constructor_factory(mask),
            as_strings=False,
            print_per_layer_stat=False,
            verbose=False,
        )
        all_macs.append(macs)

    # normalize relative to full model
    all_macs = [macs / all_macs[-1] for macs in all_macs]

    macs_breakdown = [all_macs[0]]
    for i in range(1, len(all_macs)):
        macs_breakdown.append(all_macs[i] - all_macs[i - 1])
    macs_breakdown = np.array(macs_breakdown, dtype=np.float32)
    
    os.makedirs(args.path, exist_ok=True)
    out_path = os.path.join(
        args.path, '{:s}_{:s}.npy'.format(args.arch, args.dataset)
    )
    np.save(out_path, macs_breakdown)
    print('Done!')

###############################################################################
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-a', '--arch', type=str, help='model architecture',
        choices=('resnet18', 'resnet34', 'resnet50', 'resnet101'),
    )
    parser.add_argument(
        '-d', '--dataset', type=str, help='dataset name',
        choices=('cifar10', 'cifar100', 'imagenet'),
    )
    parser.add_argument('-p', '--path', type=str, help='output path')
    args = parser.parse_args()

    main(args)