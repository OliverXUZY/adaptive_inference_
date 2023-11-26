import random
import argparse
import torch
import numpy as np
from collections import defaultdict
import torch.nn as nn 
from itertools import combinations
from math import comb
from torch.utils.data import Subset

from libs.datasets import make_dataset, make_data_loader
from libs.model import make_resnet, make_vit
from libs.utils import fix_random_seed, check_file
from libs.core import load_config
import libs.utils as utils
from libs.utils import Timer, time_str

def subset_dataset(dataset, num_samples = 500):
    
    # Set the random seed for reproducibility
    random_seed = 42
    torch.manual_seed(random_seed)
    # Generate random indices
    indices = torch.randperm(len(dataset))[:num_samples]
    # Create a subset
    subset = Subset(dataset, indices)
    return subset

class Evaluator:
    def __init__(self, model_name = "resnet50", dataset_name = "imagenet", limit = 0, random_seed = None):
        if random_seed:
            rng = fix_random_seed(random_seed)
        self.model_name = model_name
        self.dataset_name = dataset_name

        cfg = defaultdict(dict)
        if dataset_name == 'cifar10':
            cfg['data'] = {'dataset': 'cifar10',
                            'root': '/srv/home/zxu444/datasets/cifar10_dataset ',
                            'downsample': True,
                            'train_split': 'train',
                            'val_split': 'test',
                            'batch_size': 64,
                            'num_workers': 8}

        elif dataset_name == 'imagenet':
            cfg['data'] = {'dataset': 'imagenet',
                            'root': '/srv/home/zxu444/datasets/imagenet/images',
                            'downsample': True,
                            'train_split': 'train',
                            'val_split': 'val',
                            'batch_size': 512,
                            'num_workers': 16}

        
        self.cfg = cfg

        log_path = f"log/subset/{model_name}_{cfg['data']['dataset']}"
        utils.set_log_path(log_path)
        utils.ensure_path(log_path)

        ### dataset
        val_set = make_dataset(
            dataset=cfg['data']['dataset'],
            root=cfg['data']['root'],
            split=cfg['data']['val_split'],
            downsample=cfg['data'].get('downsample', False),
        )
        print('val data size: {:d}'.format(len(val_set)))
        if limit > 0:
            val_set = subset_dataset(val_set, num_samples = limit)
            print('subset val data size: {:d}'.format(len(val_set)))
  

        val_loader = make_data_loader(
            val_set, 
            generator=rng,
            batch_size=cfg['data']['batch_size'],
            num_workers=cfg['data']['num_workers'],
            is_training=False,
        )

        self.val_set, self.val_loader = val_set, val_loader


        ### model
        if "resnet" in self.model_name:
            net, macs_brk = make_resnet(self.model_name, self.dataset_name, True, load_from="timm", model_card = "timm/resnet50.a1_in1k")
            macs_brk = macs_brk.cuda()
        elif "vit" in self.model_name:
            # TODO: tem
            net = make_vit(model_card = "timm/vit_small_patch16_224.augreg_in1k", dataset = self.dataset_name, return_macs=True)
            macs_brk = torch.from_numpy(np.ones(12).astype(np.float32)).cuda()
        else:
            raise NotImplementedError("Other backbone hasn't been implemented yet")
        self.macs_brk = macs_brk
        net = net.cuda()

        compare_official = False
        if compare_official:
            import timm
            vit = timm.create_model("timm/vit_small_patch16_224.augreg_in1k", pretrained=True)
            vit = vit.cuda()
            vit.eval()
        
        device_count = torch.cuda.device_count()
        self._parallel = False
        if device_count > 1:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            self._parallel = True
        if self._parallel:
            net = nn.DataParallel(net)
        net.eval()
        self.net = net

        if self.model_name == "resnet18":
            num_block = 8-1
        elif self.model_name == "resnet50":
            num_block = 16-1
        elif "vit" in self.model_name:
            num_block = 12-1
        else:
            raise NotImplementedError("num block of other backbone hasn't been implemented yet")

        self.num_block = num_block

    def evaluate(self, masks):
        assert masks.shape[1] == self.num_block, f"masks.shape is {masks.shape}"
        if np.issubdtype(masks.dtype, np.number):
            masks = masks.astype(bool)
        # print(masks.dtype)
        masks_torch = torch.from_numpy(masks).cuda()
        

        ### start inference
        # accs = np.zeros((len(masks), len(self.val_loader)))
        # over_accs = np.zeros(len(masks))
        # macs_total = np.zeros(len(masks))

        one_mask_timer = Timer()
        accs_mask_loader = []
        accs_mask = []
        macs_mask = []

        for k in range(masks_torch.shape[0]):
            macs, acc, accs_all_batch = self._evaluation_loop(masks_torch[k])
            utils.log("masks {} | macs: {:.2f} | accs: {:.2f} | time elapsed: {} | {}".format(
                                    k,
                                    macs,
                                    acc,
                                    time_str(one_mask_timer.end()), 
                                    time_str(one_mask_timer.end() / (k+1) * len(masks))
                                    ), 
                "log.txt")
            
            accs_mask.append(acc)
            macs_mask.append(macs)
            accs_mask_loader.append(accs_all_batch)
            
        # log_str += "macs: {:.2f}({:.2f}) | ".format(macs_total.mean()*100, macs_total.std()*100)
        # log_str += "accs: {:.2f}({:.2f}) | ".format(over_accs.mean()*100, over_accs.std()*100)
        
        self.accs_mask = np.array(accs_mask)
        self.macs_mask = np.array(macs_mask)
        self.accs_mask_loader = np.array(accs_mask_loader)
        
        
        log_str = ""
        log_str += "total time elapsed: {}".format(time_str(one_mask_timer.end()))
        utils.log(log_str,"log.txt")
        # utils.log(log_str,"baseline.txt")
        
        print("========== done ==========")
    
    def save(self):
        np.savez(
            f"{self.log_path}/{self.model_name}_{self.cfg['data']['dataset']}.npz", 
            masks=self.masks,
            accs=self.accs_mask_loader.astype(float),
            over_accs = self.accs_mask.astype(float),
            macs_total = self.macs_mask.astype(float)
        )

    def _evaluation_loop(self, mask):
        assert mask.shape[-1] == self.num_block, f"for one mask, mask shape is {mask.shape}"
        # Initialize counters
        total_correct = 0
        total_samples = 0
        # macss = []
        accs = [] # log the accs for each loader
        
        
        for idx, (x, _, y) in enumerate(self.val_loader):
            x, y = x.cuda(), y.cuda()
            if len(mask.shape) == 1:
                mask = mask.repeat(y.shape[0], 1)
            else:
                assert len(mask.shape) == 2, f"for one mask in forward, mask shape is {mask.shape}"
            
            with torch.no_grad():
                logits = self.net(x, mask)
            _, pred = logits.max(dim=1)
            
            is_correct = pred == y
            acc = is_correct.sum() / y.shape[0]
            accs.append(acc.item())


            # Update counters
            total_correct += (pred == y).sum().item()
            total_samples += y.size(0)
            
        macs = (mask * self.macs_brk[1:]).sum(dim=-1) + self.macs_brk[0]
        macs.clamp_(max=1)
        
        # print(macs)
        # assert False
        
        # Compute overall accuracy
        overall_accuracy = total_correct / total_samples
        
        
        return macs.item(), overall_accuracy, accs
    
    def set_log_path(self, log_path = None):
        if not log_path:
            log_path = f"log/{self.model_name}_{self.dataset_name}"
        utils.set_log_path(log_path)
        utils.ensure_path(log_path)
        self.log_path = log_path
