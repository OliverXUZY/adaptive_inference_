import os
import argparse
from math import comb
from itertools import combinations
import torch

from src.datasets import make_dataset, make_data_loader, subset_dataset
from src.utils import *
from src.core import load_config
from src.model import make_resnet

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--name', type=str, help='job name')
    parser.add_argument('-p', '--load_path', type=str, 
                        help='path to load model')
    parser.add_argument('-c', '--config', type=str, 
                        default="./config/resnet18_cifar100.yaml",
                        help='path to config file')
    
    parser.add_argument(
        '-m', '--macs', type=float, default=1, help='MACs constraint'
    )
    parser.add_argument('-g', '--gpu', type=str, default='0', help='GPU IDs')
    
    parser.add_argument('--limit', type=int, default=0, help='limit sample to evaluate')
    parser.add_argument('--skip_block', type=int, default=-1, help='how many blocks to skip')

    args = parser.parse_args()
    return args

def main(skip_block = 0):
    args = parse_args()
    if args.load_path:
        ckpt_path = args.load_path
        cfg_path = os.path.join(ckpt_path, 'config.yaml')
    else:
        cfg_path = args.config

    check_file(cfg_path)
    cfg = load_config(cfg_path)
    print(f'config loaded from {cfg_path}')


    # configure GPUs
    n_gpus = len(args.gpu.split(','))
    if n_gpus > 1:
        cfg['_parallel'] = True
    set_gpu(args.gpu)

    rng = fix_random_seed(42)
    log_path = "log/look_up_table"
    ensure_path(path = log_path)
    set_log_path(path = log_path)

    ###########################################################################
    """ dataset """

    val_set = make_dataset(
        dataset=cfg['data']['dataset'],
        root=cfg['data']['root'],
        split=cfg['data']['val_split'],
        downsample=cfg['data'].get('downsample', False),
    )
    

    print('val data size: {:d}'.format(len(val_set)))
    if args.limit > 0:
        val_set = subset_dataset(val_set, num_samples = args.limit)
        print('subset val data size: {:d}'.format(len(val_set)))
    
    val_loader = make_data_loader(
        val_set, 
        generator=rng,
        batch_size=cfg['data']['batch_size'],
        num_workers=cfg['data']['num_workers'],
        is_training=False,
    )
    
    ###########################################################################
    """ model """
    if cfg.get('model'):
        if cfg.get('model').get('resnet'):
            arch = cfg.get('model').get('resnet').get('arch')
            net, macs_brk = make_resnet(arch, cfg['data']['dataset'], 
                                        return_macs=True, pretrained = True, load_from = "hugg")
        else:
            raise NotImplementedError("Other backbone hasn't been implemented yet")
    else:
        raise NotImplementedError("need a model")

    if args.load_path:
        # load_path
        load_path = args.load_path
        ckpt_name = os.path.join(load_path, 'last.pth')
        ckpt = torch.load(ckpt_name)
        net.load_state_dict(ckpt['arch'])
        print('model ckpt loaded from {}'.format(ckpt_name))
        log_path = load_path
        ensure_path(path = log_path)
        set_log_path(path = log_path)
        print(f"log look up table to {log_path}")

    net = net.cuda()
    macs_brk = macs_brk.cuda()
    net.eval()

    if arch == "resnet18":
        num_knobs = 8-1
    
    ###########################################################################
    """ eval """
    # skip block
    save_masks = []
    save_accs = []
    save_macs = []
    for skip_block in range(8):
        branches_per_setting = 64
        print(f"skip {skip_block} block | ")
        num_combinations = comb(num_knobs, skip_block)

        #######
        if num_combinations <= branches_per_setting:
            all_combinations = list(combinations(range(num_knobs), skip_block))
            masks = np.ones((len(all_combinations), num_knobs))
            for i, idx in enumerate(all_combinations):
                masks[i, idx] = 0
        else:
            raise NotImplementedError("only numerate countable")

        # print(masks)
        # assert False


        assert masks.shape[1] == num_knobs, f"masks.shape is {masks.shape}"
        if np.issubdtype(masks.dtype, np.number):
            masks = masks.astype(bool)
        masks_torch = torch.from_numpy(masks).cuda()
        
        ### start inference
        
        one_mask_timer = Timer()
        accs_mask = []
        macs_mask = []
        log_str = f"skip {skip_block} block | "

        for k in range(masks_torch.shape[0]):
            mask = masks_torch[k]
            # print("mask.shape", mask.shape)
            # mask = torch.tensor([True, True, False, True, False, True, True])
            mask_ori  = mask.clone()
            assert mask.shape[-1] == num_knobs, f"for one mask, mask shape is {mask.shape}"
            # Initialize counters
            total_correct = 0
            total_samples = 0
            # macss = []
            accs = [] # log the accs for each loader
            
            for idx, (x, _, y) in enumerate(val_loader):
                x, y = x.cuda(), y.cuda()
                if len(mask_ori.shape) == 1:
                    # print("align mask with y")
                    mask = mask_ori.repeat(y.shape[0], 1)
                else:
                    assert len(mask.shape) == 2, f"for one mask in forward, mask shape is {mask.shape}"
                # print("x.shape", x.shape)
                # print("mask.shape", mask.shape)
                # print(mask)
                # assert False
                with torch.no_grad():
                    logits = net(x, mask)
                _, pred = logits.max(dim=1)
                
                is_correct = pred == y
                acc = is_correct.sum() / y.shape[0]
                accs.append(acc.item())
                # Update counters
                total_correct += (pred == y).sum().item()
                total_samples += y.size(0)
                
            macs = (mask_ori * macs_brk[1:]).sum(dim=-1) + macs_brk[0]
            macs.clamp_(max=1)

            print("total_samples", total_samples)

            # Compute overall accuracy for one branch
            overall_accuracy = total_correct / total_samples



            utils.log("masks {} | macs: {:.2f} | accs: {:.2f} | time elapsed: {} | {}".format(
                                        k,
                                        macs.item(),
                                        overall_accuracy,
                                        time_str(one_mask_timer.end()), 
                                        time_str(one_mask_timer.end() / (k+1) * len(masks))
                                        ), 
                    f"log_skip{skip_block}.txt")
            
            accs_mask.append(overall_accuracy)
            macs_mask.append(macs.item())
            
            ## save
            save_masks.append(mask_ori.detach().cpu())
            save_accs.append(overall_accuracy)
            save_macs.append(macs.detach().cpu().item())

        
        macs_total = np.array(macs_mask)
        over_accs = np.array(accs_mask)
        log_str += "macs: {:.2f}({:.2f}) | ".format(macs_total.mean()*100, macs_total.std()*100)
        log_str += "accs: {:.2f}({:.2f}) | ".format(over_accs.mean()*100, over_accs.std()*100)

        log_str += "total time elapsed: {}".format(time_str(one_mask_timer.end()))
        utils.log(log_str,f"log_skip{skip_block}.txt")
        utils.log(log_str,"baseline.txt")
        
        print("========== done ==========")

    np.savez(
        f"{log_path}/look_up_table_{arch}_{cfg['data']['dataset']}.npz", 
        masks=np.array(save_masks),
        accs=np.array(save_accs).astype(float),
        macs = np.array(save_macs).astype(float)
    )



if __name__ == '__main__':
    main()
    
