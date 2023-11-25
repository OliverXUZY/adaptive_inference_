import random
import argparse
import torch
import numpy as np
from collections import defaultdict
import torch.nn as nn 
from itertools import combinations
from math import comb

import sys
import os
run_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(run_dir)
if project_dir not in sys.path:
    sys.path.insert(0, project_dir)

from libs.datasets import make_dataset, make_data_loader
from libs.model import make_resnet, make_vit
from libs.utils import fix_random_seed, check_file
from libs.core import load_config
import libs.utils as utils
from libs.utils import Timer, time_str

def parse_args():
    parser = argparse.ArgumentParser(description="This is my training script.")
    parser.add_argument('-c', '--config', type=str, help='config file path')
    parser.add_argument('-n', '--name', type=str, help='job name')
    parser.add_argument('-g', '--gpu', type=str, default=None, help='GPU IDs')
    parser.add_argument('-m', '--model', type=str, default='vit', help='backbone')
    parser.add_argument('--dataset', type=str, default='imagenet', help='The dataset we used')
    parser.add_argument('--skip_block', type=int, default=0, help='how many blocks to skip')


    args = parser.parse_args()
    return args

def main(args):
    timer = Timer()
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

    elif args.dataset == 'imagenet':
        cfg['data'] = {'dataset': 'imagenet',
                        'root': '/srv/home/zxu444/datasets/imagenet/images',
                        'downsample': True,
                        'train_split': 'train',
                        'val_split': 'val',
                        'batch_size': 512,
                        'num_workers': 16}

    
    
    log_path = f"log/testEval_{args.model}_{cfg['data']['dataset']}_test2"
    utils.set_log_path(log_path)
    utils.ensure_path(log_path)

    ### dataset
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

    ### model
    if "resnet" in args.model:
        net, macs_brk = make_resnet(args.model, args.dataset, True, load_from="timm", model_card = "timm/resnet50.a3_in1k")
        macs_brk = macs_brk.cuda()
    elif "vit" in args.model:
        # TODO: tem
        net = make_vit(model_card = "timm/vit_small_patch16_224.augreg_in1k", dataset = args.dataset, return_macs=True)
        macs_brk = torch.from_numpy(np.ones(12).astype(np.float32)).cuda()
    else:
        raise NotImplementedError("Other backbone hasn't been implemented yet")
    net = net.cuda()

    compare_official = False
    if compare_official:
        import timm
        vit = timm.create_model("timm/vit_small_patch16_224.augreg_in1k", pretrained=True)
        vit = vit.cuda()
        vit.eval()
    
    if args._parallel:
        net = nn.DataParallel(net)
    net.eval()

    ### construct masks
    # do not skip first block!
    if args.model == "resnet18":
        num_block = 8-1
    elif args.model == "resnet50":
        num_block = 16-1
    elif "vit" in args.model:
        num_block = 12-1
    else:
        raise NotImplementedError("num block of other backbone hasn't been implemented yet")

    # skip block
    skip_block = args.skip_block
    log_str = f"skip {skip_block} block | "
    num_combinations = comb(num_block, skip_block)

    if num_combinations <= 1:
        all_combinations = list(combinations(range(num_block), skip_block))
        masks = np.ones((len(all_combinations), num_block))
        for i, idx in enumerate(all_combinations):
            masks[i, idx] = 0
    else:

        masks = np.ones((64, num_block))
        # Set random seed for the built-in random module
        seed_value = 42  # you can choose any number you like
        random.seed(seed_value)
        
        for i in range(64): # the last one is always true, skip no blocks
            idx = random.sample(range(num_block), skip_block)
            masks[i, idx] = 0
    print("mask.shape: ", masks.shape)
    # print(masks)
    # assert False
    masks = masks.astype(bool)
    
    # assert False
    masks_torch = torch.from_numpy(masks).cuda()
    
    

    ### start inference
    accs = np.zeros((len(masks), len(val_loader)))
    over_accs = np.zeros(len(masks))
    macs_total = np.zeros(len(masks))

    one_mask_timer = Timer()
    one_loader_timer = Timer()

    for k in range(len(masks)):
        # Initialize counters
        total_correct = 0
        total_samples = 0
        # macss = []
        one_loader_timer.start()
        for idx, (x, _, y) in enumerate(val_loader):
            x, y = x.cuda(), y.cuda()
            # print("y.shape", y.shape)
            mask = masks_torch[k].repeat(y.shape[0], 1)
            # print("mask in one batch: ", mask)
            # print("mask.shape in one batch: ", mask.shape)
            with torch.no_grad():
                logits = net(x, mask)
                # logits_vit = vit(x)
            _, pred = logits.max(dim=1)
            # print("logits.shape: ", logits.shape)
            # print(torch.equal(logits, logits_vit))
            # assert False
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
            if idx % 20 == 0:
                print(f"mask {k} --- loader {idx+1}|{len(val_loader)} --- time elapsed: {time_str(one_loader_timer.end())}")
        macs = (masks_torch[k] * macs_brk[1:]).sum(dim=-1) + macs_brk[0]
        macs.clamp_(max=1)
        macs_total[k] = macs.item()
        # print(macs)
        # assert False
        
         # Compute overall accuracy
        overall_accuracy = total_correct / total_samples
        over_accs[k] = overall_accuracy
        
        utils.log("masks {} | macs: {:.2f} | accs: {:.2f} | time elapsed: {} | {}".format(
                                    k,
                                    macs.item(),
                                    overall_accuracy,
                                    time_str(one_mask_timer.end()), 
                                    time_str(one_mask_timer.end() / (k+1) * len(masks))
                                    ), 
                "log.txt")
    log_str += "macs: {:.2f}({:.2f}) | ".format(macs_total.mean()*100, macs_total.std()*100)
    log_str += "accs: {:.2f}({:.2f}) | ".format(over_accs.mean()*100, over_accs.std()*100)
    
    
    
    np.savez(
        f"{log_path}/{args.model}_{cfg['data']['dataset']}_skip{skip_block}.npz", 
        masks=masks,
        accs=accs.astype(float),
        over_accs = over_accs.astype(float),
        macs_total = macs_total.astype(float)
    )
    log_str += "time elapsed: {}".format(time_str(timer.end()))
    utils.log(log_str,"log.txt")
    utils.log(log_str,"baseline.txt")
    
    print("========== done ==========")



if __name__ == '__main__':
    args = parse_args()
    device_count = torch.cuda.device_count()
    args._parallel = False
    if device_count > 1:
        print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
        args._parallel = True

    
    args.skip_block = 1
    main(args)

    # for skip_block in range(3,num_block):
    #     args.skip_block = skip_block
        # print(args)
        