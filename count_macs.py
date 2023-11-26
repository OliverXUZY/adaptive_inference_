import os
import argparse

import torch
import numpy as np
from ptflops import get_model_complexity_info

from libs.model import make_resnet, make_vit


def input_constructor_factory(mask):
    '''
    consider size = (3,32,32)
    input_constructor(size) will return dict as res
    res[x] is array with all 0s of shape [1,3,32,32]
    res[mask] is array with mask of shape [1,7] (if mask.shape is (7,))
    '''
    def input_constructor(size):
        x = torch.empty(size)#.cuda()
        return {'x': x[None], 'mask': mask[None]}
    return input_constructor

def main(args):
    # load model
    print('Loading model...')
    if "resnet" in args.arch:
        net = make_resnet(args.arch, args.dataset, return_macs=False, load_from="timm")
        n_knobs = net.n_blocks - 1
    elif "vit" in args.arch:
        net = make_vit(model_card = "timm/vit_small_patch16_224.augreg_in1k")
        n_knobs = len(net.blocks) - 1
    # net.cuda()
    
    # define branch masks
    # n_knobs = net.n_blocks - 1
    print('Number of knobs: {:d}'.format(n_knobs))
    print('Building masks...')
    masks = np.concatenate(
        [np.zeros((1, n_knobs)), np.tril(np.ones((n_knobs, n_knobs)))]
    )
    # print("masks: ", masks)
    # print("masks.shape: ", masks.shape)
    masks = torch.from_numpy(masks).bool()
    # masks = masks.cuda()

    # input size
    size = (3, 32, 32) if args.dataset == 'cifar10' else (3, 224, 224)

    # mask = masks[0]
    # print("mask: ", mask)
    # input_constructor=input_constructor_factory(mask)
    # print("input_constructor_factory(mask): ", input_constructor_factory(mask))
    # res = input_constructor(size)
    # print("x: ",res['x'].shape)
    # print("mask: ",res['mask'].shape)
    # assert False


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
        # print(f"macs: {macs}")
        # assert False
        all_macs.append(macs)
        
    # print("all_macs: ", all_macs)

    # normalize relative to full model
    all_macs = [macs / all_macs[-1] for macs in all_macs]

    macs_breakdown = [all_macs[0]]
    for i in range(1, len(all_macs)):
        macs_breakdown.append(all_macs[i] - all_macs[i - 1])
    macs_breakdown = np.array(macs_breakdown, dtype=np.float32)
    # print(f"macs_breakdown: {macs_breakdown}")
    # for value in macs_breakdown:
    #     print(f"{value:.4f}")
    
    # assert False, "Not saving anything"
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
        choices=('resnet18', 'resnet34', 'resnet50', 'resnet101', "vit"),
    )
    parser.add_argument(
        '-d', '--dataset', type=str, help='dataset name',
        choices=('cifar10', 'cifar100', 'imagenet'),
    )
    parser.add_argument('-p', '--path', type=str, help='output path')
    args = parser.parse_args()

    main(args)