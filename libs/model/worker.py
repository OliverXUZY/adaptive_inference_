import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import combinations
from math import comb
from .resnet import make_resnet
from .modules import BranchEncoder, ContentEncoder, BranchVAE
from .losses import rs_loss_fn, bce_loss_fn, vae_loss_fn


class Worker():

    def __init__(self, model_cfg):

        self.model_cfg = model_cfg
        self.parallel = False

        # unroll model specs
        resnet_cfg = model_cfg['resnet']
        branch_enc_cfg = model_cfg['branch_enc']
        content_enc_cfg = model_cfg['content_enc']
        branch_vae_cfg = model_cfg.get('branch_vae')

        # load pre-trained ResNet
        self.resnet, self.macs_brk = make_resnet(**resnet_cfg)

        ## NOTE: always keep the first residual block in ResNet
        ## This is bacause the first block was never dropped in training
        self.n_knobs = self.resnet.n_blocks - 1
        self.n_branches = 2 ** self.n_knobs
        self.bit_mask = 2 ** torch.arange(self.n_knobs - 1, -1, -1)
        # print("self.n_knobs", self.n_knobs)
        # print("self.n_branches", self.n_branches)
        # print("self.bit_mask", self.bit_mask)

        # build scheduler
        branch_enc_cfg['seq_len'] = self.n_knobs
        assert branch_enc_cfg['out_dim'] == content_enc_cfg['out_dim']
        self.branch_enc = BranchEncoder(**branch_enc_cfg)
        self.content_enc = ContentEncoder(**content_enc_cfg)
        
        # build VAE
        if branch_vae_cfg is not None:
            branch_vae_cfg['in_dim'] = self.n_knobs
            self.branch_vae = BranchVAE(**branch_vae_cfg)
            self.latent_dim = self.branch_vae.latent_dim
        else:
            self.branch_vae = self.latent_dim = None

        # hold inference branches and their embeddings
        self.masks = self.macs = self.bv = None

    def cuda(self, parallel=False):
        self.resnet.cuda()
        self.branch_enc.cuda()
        self.content_enc.cuda()
        if self.branch_vae is not None:
            self.branch_vae.cuda()
        if parallel:
            self.resnet = nn.DataParallel(self.resnet)
            self.branch_enc = nn.DataParallel(self.branch_enc)
            self.content_enc = nn.DataParallel(self.content_enc)
            if self.branch_vae is not None:
                self.branch_vae = nn.DataParallel(self.branch_vae)
            self.parallel = True
        self.macs_brk = self.macs_brk.cuda()

    def _get_module(self, module):
        if module == 'branch_enc':
            m = self.branch_enc.module if self.parallel else self.branch_enc
        elif module == 'content_enc':
            m = self.content_enc.module if self.parallel else self.content_enc
        elif module == 'branch_vae':
            if self.branch_vae is None:
                return
            m = self.branch_vae.module if self.parallel else self.branch_vae
        return m

    def load(self, ckpt):
        branch_enc = self._get_module('branch_enc')
        content_enc = self._get_module('content_enc')
        branch_vae = self._get_module('branch_vae')
        
        branch_enc.load_state_dict(ckpt['branch_enc'])
        content_enc.load_state_dict(ckpt['content_enc'])
        if branch_vae is not None:
            assert ckpt['branch_vae'] is not None
            branch_vae.load_state_dict(ckpt['branch_vae'])

    def save(self):
        branch_enc = self._get_module('branch_enc')
        content_enc = self._get_module('content_enc')
        branch_vae = self._get_module('branch_vae')

        branch_enc_ckpt = branch_enc.state_dict()
        content_enc_ckpt = content_enc.state_dict()
        branch_vae_ckpt = None
        if branch_vae is not None:
            branch_vae_ckpt = branch_vae.state_dict()

        ckpt = {
            'branch_enc': branch_enc_ckpt, 
            'content_enc': content_enc_ckpt,
            'branch_vae': branch_vae_ckpt,
        }
        return ckpt

    def parameters(self, module):
        m = self._get_module(module)
        if m is None:
            return
        return m.parameters()

    def named_parameters(self, module):
        m = self._get_module(module)
        if m is None:
            return
        return m.named_parameters()

    def named_modules(self, module):
        m = self._get_module(module)
        if m is None:
            return
        return m.named_modules()

    @torch.no_grad()
    def _sample_branches(self, n_branches):
        """
        Sample branches for training.

        Args:
            n_branches (int): number of branches to sample.

        Returns:
            masks (bool tensor, (n, kb)): branch masks.
        """
        # print("total branches: ", self.n_branches)
        b_idx = list(range(self.n_branches))
        if n_branches < self.n_branches:
            b_idx = random.sample(b_idx, n_branches)
        b_idx = torch.LongTensor(b_idx)
        # print("b_idx.shape: ", b_idx.shape)
        # print("b_idx: ", b_idx)
        # print("self.bit_mask: ", self.bit_mask)
        # print(b_idx[:, None].shape, b_idx[:, None])
        masks = b_idx[:, None].bitwise_and(self.bit_mask).ne(0)
        # print(b_idx[:, None].bitwise_and(self.bit_mask))
        # print(masks)
        # print("================")
        # assert False
        return masks

    @torch.no_grad()
    def _calculate_macs(self, masks):
        """
        Calculate MACs of branches.

        Args:
            masks (bool tensor, (n, kb)): branch masks.

        Returns:
            macs (float tensor, (n,)): MACs relative to full model.
        """
        macs = (masks * self.macs_brk[1:]).sum(dim=-1) + self.macs_brk[0]
        macs.clamp_(max=1)
        # print("macs.shape: ", macs.shape)
        # assert False
        return macs

    @torch.no_grad()
    def _calculate_target(self, x, y, masks, k=1):
        """
        Calculate targets for Rank & Sort loss.

        Args:
            x (float tensor, (bs, 3, h, w)): input images.
            y (long tensor, (bs,)): ground-truth labels.
            masks (bool tensor, (n, kb)): branch masks.
            k (int): minimum number of positives.

        Returns:
            rs_target (float tensor, (bs, n)): soft targets.
            bce_target (bool tensor, (bs, n)): hard (binary) targets.
            positives (bool tensor, (?, kb)): masks for positive branches.
        """
        bs, n = x.size(0), masks.size(0)
        macs = self._calculate_macs(masks)
        # print(f"macs.shape: {macs.shape}")
        # assert False

        x = x.repeat_interleave(n, dim=0)                       # (bs*n, 3, h, w)
        y = y.repeat_interleave(n, dim=0)                       # (bs*n,)
        masks = masks.repeat(bs, 1)                             # (bs*n, kb) kb = (1,1,0,0,1,0,1) switch for each block

        # run input on sampled branches
        logits = self.resnet(x, masks)                          # (bs*n, c)
        
        # positive branches are those that predict
        ## (1) the correct class, OR
        ## (2) the k highest scores on target class 
        _, pred = logits.max(dim=1)
        is_correct = (pred == y).reshape(bs, n)                 # (bs, n)

        if k == 0:
            bce_target = is_positive = is_correct               # (bs, n)
        else:
            y_score = logits.gather(dim=1, index=y[:, None])    # (bs*n, 1)
            y_score = y_score.reshape(bs, n)                    # (bs, n)
            sorted_idx = y_score.argsort(dim=1, descending=True)
            best_idx = sorted_idx[:, :min(k, n)]                # (bs, k)
            is_best = F.one_hot(best_idx, n).sum(dim=1).bool()  # (bs, n)
            bce_target = is_positive = is_correct | is_best     # (bs, n)

        # calculate soft targets
        ## (1) positive branches have positive targets
        ## (2) cheaper positive branches have larger targets
        rs_target = is_positive * (1 - macs).clamp_(min=1e-3)   # (bs, n)

        # print(f"is_positive.shape: {is_positive.shape}, ")
        # select positive branches
        positives = masks[is_positive.flatten()]                # (?, kb) ? is how many trues in is_positive

        return rs_target, bce_target, positives

    def _embed_content(self, x):
        return F.normalize(self.content_enc(x), dim=-1)

    def _embed_branch(self, masks):
        return F.normalize(self.branch_enc(masks), dim=-1)

    @torch.no_grad()
    def prep_test_branches(self, n_branches, max_macs=1, batch_size=256):
        """
        Pre-compute embeddings of test branches prior to evaluation.

        Args:
            n_branches (int): number of branches to sample.
            max_macs (float): maximum MACs allowed.
            batch_size (int): number of branches to embed at a time.
        """
        if self.branch_vae is not None:
            # sample branches using learned VAE
            ## NOTE: always include full model
            all_masks = torch.ones(1, self.n_knobs, dtype=torch.bool).cuda()
            while len(all_masks) < n_branches:
                z = torch.randn(n_branches, self.latent_dim).cuda()
                p = self.branch_vae(z=z)
                masks = torch.bernoulli(p).bool()
                macs = self._calculate_macs(masks)
                masks = masks[macs <= max_macs]
                all_masks = torch.cat([all_masks, masks]).unique(dim=0)
            all_masks = all_masks[:n_branches]
        else:
            # include all branches
            # assert self.n_branches <= 4096, 'too many branches for inference'
            num_sampled_branches = min(self.n_branches, 4096)
            all_masks = self._sample_branches(num_sampled_branches).cuda() # [n,kb] [128,7]
            macs = self._calculate_macs(all_masks)
            all_masks = all_masks[macs <= max_macs]
        self.masks = all_masks                                  # [n,kb] [128,7]
        self.macs = self._calculate_macs(all_masks)             # [n] [128]
        # print("self.masks.shape: ", self.masks.shape)
        # print("self.masks: ", self.masks)
        # print("self.macs.shape: ", self.macs.shape)
        
        # embed sampled branches
        self.bv = torch.cat(
            [self._embed_branch(m) for m in all_masks.split(batch_size)]
        )
        # print("self.bv.shape: ", self.bv.shape) # [n,branch_enc.out_dim] [128,128]
        # assert False

    def _calculate_logits(self, cv, bv, temperature=1):
        """
        Calculate logits of scheduler for Rank & Sort loss.

        Args:
            cv (float tensor, (bs, d)): content embeddings.
            bv (float tensor, (n, d)): branch embeddings.
            temperature (float): temperature applied on logits.

        Returns:
            logits (float tensor, (bs, n)): similarity logits.
        """
        logits = torch.einsum('ik,jk->ij', cv, bv)              # (bs, n)
        logits = logits * temperature
        return logits
        
    def train(self, rx, cx, y, cfg):
        """
        Args:
            rx (float tensor, (bs, 3, h, w)): input to ResNet.
            cx (float tensor, (bs, 3, h', w')): input to content encoder.
            y (long tensor, (bs,)): target labels.
            cfg (dict): training hyper-parameters.
        """
        self.resnet.eval()
        self.content_enc.train()
        self.branch_enc.train()
        if self.branch_vae is not None:
            self.branch_vae.train()

        rx = rx.cuda(non_blocking=True)
        cx = rx if cx is None else cx.cuda(non_blocking=True)
        y = y.cuda(non_blocking=True)

        # sample branches
        masks = self._sample_branches(cfg['n_branches'])
        masks = masks.cuda(non_blocking=True)                   # (n, kb)

        # calculate targets
        # rs_target: (bs, n), bce_target: (bs, n)
        rs_target, bce_target, positives = \
            self._calculate_target(rx, y, masks, cfg['k'])
        if len(positives) < cfg.get('min_n_positives', 1):
            return

        # calculate scheduler logits
        ## sc_logits: (bs, n)
        cv = self._embed_content(cx)                            # (bs, d)
        bv = self._embed_branch(masks)                          # (n, d)
        # print(f"cv.shape: {cv.shape} | bv.shape: {bv.shape}")
        
        sc_logits = self._calculate_logits(cv, bv, cfg['temperature']) # (bs, n)

        # calculate losses
        ## scheduler and VAE are independently optimized
        loss_dict = dict()

        # rank & sort loss
        rs_loss = sc_logits.new_zeros(1)
        if cfg['rank_weight'] > 0 or cfg['sort_weight'] > 0:
            rank_loss, sort_loss = \
                rs_loss_fn(sc_logits, rs_target, cfg['delta_rs'])
            rs_loss = cfg['rank_weight'] * rank_loss \
                    + cfg['sort_weight'] * sort_loss
            loss_dict['rank'] = rank_loss
            loss_dict['sort'] = sort_loss

        # binary cross-entropy loss
        bce_loss = sc_logits.new_zeros(1)
        if cfg['bce_weight'] > 0:
            bce_loss = bce_loss_fn(sc_logits, bce_target)
            loss_dict['bce'] = bce_loss
            bce_loss = cfg['bce_weight'] * bce_loss
        
        # VAE loss
        vae_loss = sc_logits.new_zeros(1)
        if self.branch_vae is not None:
            # sample positive branches
            idx = torch.randperm(len(positives))[:cfg['vae_batch_size']]
            positives = positives[idx]
            vae_logits, mu, log_var = self.branch_vae(x=positives)
            vae_loss = vae_loss_fn(vae_logits, positives, mu, log_var)
            loss_dict['vae'] = vae_loss

        total_loss = rs_loss + bce_loss + vae_loss
        loss_dict['total'] = total_loss

        return loss_dict

    @torch.no_grad()
    def eval(self, rx, cx, y):
        """
        Args:
            rx (float tensor, (bs, 3, h, w)): input to ResNet.
            cx (float tensor, (bs, 3, h', w')): input to content encoder.
            y (long tensor, (bs,)): target labels.

        Returns:
            acc (float tensor, (1,)): classification accuracy.
            saving (float tensor, (1.)): MACs saving.
        """
        assert self.bv is not None, \
            'call prep_test_branches() before evaluation'

        self.resnet.eval()
        self.content_enc.eval()
        self.branch_enc.eval()
        if self.branch_vae is not None:
            self.branch_vae.eval()

        rx = rx.cuda(non_blocking=True)
        cx = rx if cx is None else cx.cuda(non_blocking=True)
        y = y.cuda(non_blocking=True)

        # select inference branch
        cv = self._embed_content(cx)                            # (bs, d)
        sc_logits = self._calculate_logits(cv, self.bv)         # (bs, n)
        # print("self.masks: ", self.masks)
        
        _, b_idx = sc_logits.max(dim=1)                         # (bs,)
        # print("b_idx: ", b_idx)
        
        masks = self.masks[b_idx]                               # (bs, kb)
        # print("masks: ", masks)
        # assert False
        macs = self.macs[b_idx]                                 # (bs,)

        # run input on selected branch
        logits = self.resnet(rx, masks)                         # (bs, c)
        
        # calculate average accuracy and MACs
        _, pred = logits.max(dim=1)                             # (bs,)
        acc = (pred == y).sum() / len(y)
        macs = macs.sum() / len(y)

        metrics_list = {
            'acc': acc,
            'macs': macs,
        }
        return metrics_list
    

    ####  Zhuoyan Edit
    @torch.no_grad()
    def prep_test_branches_constrain_macs(self, n_branches, max_macs=1, batch_size=256, skip_block = 0, branches_per_setting = 256):
        """
        Pre-compute embeddings of test branches prior to evaluation.

        Args:
            n_branches (int): number of branches to sample.
            max_macs (float): maximum MACs allowed.
            batch_size (int): number of branches to embed at a time.
        """
        if self.branch_vae is not None:
            # sample branches using learned VAE
            ## NOTE: always include full model
            all_masks = torch.ones(1, self.n_knobs, dtype=torch.bool).cuda()
            while len(all_masks) < n_branches:
                z = torch.randn(n_branches, self.latent_dim).cuda()
                p = self.branch_vae(z=z)
                masks = torch.bernoulli(p).bool()
                macs = self._calculate_macs(masks)
                masks = masks[macs <= max_macs]
                all_masks = torch.cat([all_masks, masks]).unique(dim=0)
            all_masks = all_masks[:n_branches]
        else:
            # sample branches for eval()
            num_sampled_branches = min(self.n_branches, branches_per_setting)


            ###33###33###33###33###33
            ## zhuoyan:
            # print(f"skip {skip_block} block | ")
            num_combinations = comb(self.n_knobs, skip_block)

            #######
            if num_combinations <= num_sampled_branches:
                all_combinations = list(combinations(range(self.n_knobs), skip_block))
                all_masks = np.ones((len(all_combinations), self.n_knobs))
                for i, idx in enumerate(all_combinations):
                    all_masks[i, idx] = 0
            else:
                all_masks = np.ones((num_sampled_branches, self.n_knobs))
                # Set random seed for the built-in random module
                seed_value = 42  # you can choose any number you like
                random.seed(seed_value)
                # print("all_masks0: ", all_masks0)
                for i in range(num_sampled_branches): # the last one is always true, skip no blocks
                    idx = random.sample(range(self.n_knobs), skip_block)
                    # print("idx: ", idx)
                    all_masks[i, idx] = 0
            # print("all_masks0: ", all_masks0)
            all_masks = all_masks.astype(bool)
            # print(masks.dtype)
            all_masks = torch.from_numpy(all_masks).cuda()

            ###33###33###33###33###33
            ## original
            # all_masks = self._sample_branches(num_sampled_branches).cuda() # [n,kb] [128,7]
            # macs = self._calculate_macs(all_masks)
            # all_masks = all_masks[macs <= max_macs]

        # print("all_masks0: ", all_masks0)
        # print("all_masks: ", all_masks)
        # print("all_masks0.shape: ", all_masks0.shape)
        # print("all_masks.shape: ", all_masks.shape)
        self.masks = all_masks                                  # [n,kb] [128,7]
        self.macs = self._calculate_macs(all_masks)             # [n] [128]
        # print("self.masks.shape: ", self.masks.shape)
        # print("self.masks: ", self.masks)
        # print("self.macs.shape: ", self.macs.shape)
        # assert False
        
        # embed sampled branches
        self.bv = torch.cat(
            [self._embed_branch(m) for m in all_masks.split(batch_size)]
        )
        # print("self.bv.shape: ", self.bv.shape) # [n,branch_enc.out_dim] [128,128]
        # assert False

    