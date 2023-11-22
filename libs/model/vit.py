""" Simplified implementation of Vision Transformer for adaptive inference """
import os

import torch
import torch.nn as nn
import numpy as np
from functools import partial
from timm.models.vision_transformer import Block, VisionTransformer, checkpoint_filter_fn
from timm.models._builder import build_model_with_cfg
from timm.models._factory import parse_model_name
from timm.models._registry import is_model, model_entrypoint, split_model_name_tag


def drop_path(x, mask=None, scale_by_keep: bool = False):
    """
    similar to timm.layers.drop, instead of take drop_prob, we take masks, make it deterministic
    
    Apply the drop path (Stochastic Depth) per sample in a deterministic manner using masks.

    Args:
        x (Tensor): The input tensor.
        mask (Tensor or None): A boolean tensor of shape (batch_size,) indicating which samples
                               should apply the residual connection.
        scale_by_keep (bool): Whether to scale the output by the keep probability.

    """
    if mask is None:
        return x
    if scale_by_keep:
        # Calculate the keep probability based on the mask.
        keep_prob = mask.float().mean().item()
        scale_factor = 1 / keep_prob if keep_prob > 0. else 0.
    else:
        scale_factor = 1.0

    # Only apply residuals to the masked entries.
    output = torch.zeros_like(x)
    output[mask] = x[mask] * scale_factor

    return output


class ada_Block(Block):
    
    def forward(self, x, mask = None):
        """
        Args:
            x (float tensor, (bs, seq, D)): feature maps.
            mask (bool tensor, (bs,)): mask for residual connection.
        """
        x = x + drop_path(self.ls1(self.attn(self.norm1(x))), mask)
        x = x + drop_path(self.ls2(self.mlp(self.norm2(x))), mask)
        
        return x

class ada_VisionTransformer(VisionTransformer):

    def forward_features(self, x, masks = None):
        x = self.patch_embed(x)
        x = self._pos_embed(x)
        x = self.patch_drop(x)
        x = self.norm_pre(x)
        if self.grad_checkpointing and not torch.jit.is_scripting():
            x = checkpoint_seq(self.blocks, x)
        else:
            for block_idx, block in enumerate(self.blocks):
                assert isinstance(block, ada_Block), "this vit should use customized Block"
                if block_idx == 0 or masks is None:
                    mask = None
                else:
                    mask = masks[:, (block_idx-1)]
                x = block(x, mask)  # call each block with the mask

            # x = self.blocks(x, mask) This one cannot been used since it wrapped by nn.Sequential, which is not applicable with third input
        
        x = self.norm(x)
        return x
    
    def forward(self, x, mask = None):
        x = self.forward_features(x, mask)
        
        x = self.forward_head(x)
        return x

def _create_vision_transformer(variant, pretrained=False, **kwargs):
    '''
    most copied from timm.models.vision_transformer.py, modify to load ada_VisionTransformer
    '''
    if kwargs.get('features_only', None):
        raise RuntimeError('features_only not implemented for Vision Transformer models.')

    if 'flexi' in variant:
        # FIXME Google FlexiViT pretrained models have a strong preference for bilinear patch / embed
        # interpolation, other pretrained models resize better w/ anti-aliased bicubic interpolation.
        _filter_fn = partial(checkpoint_filter_fn, interpolation='bilinear', antialias=False)
    else:
        _filter_fn = checkpoint_filter_fn

    # FIXME attn pool (currently only in siglip) params removed if pool disabled, is there a better soln?
    strict = True
    if 'siglip' in variant and kwargs.get('global_pool', None) != 'map':
        strict = False
    

    kwargs.update({"block_fn": ada_Block})
    # print("########################")
    # print(f"zhuoyan variant: {variant}")
    # print(f"------- pretrained_filter_fn: {_filter_fn}")
    # print(f"------- pretrained_strict: {strict}")
    # print(f"------- kwargs: {kwargs.keys()}")
    # for key,val in kwargs.items():
    #     print(key, val)
    # print("########################")

    return build_model_with_cfg(
        ada_VisionTransformer,
        variant,
        pretrained,
        pretrained_filter_fn=_filter_fn,
        pretrained_strict=strict,
        **kwargs,
    )

def vit_small_patch16_224(pretrained=False, **kwargs) -> VisionTransformer:
    """ ViT-Small (ViT-S/16)
    """
    model_args = dict(patch_size=16, embed_dim=384, depth=12, num_heads=6)
    model = _create_vision_transformer('vit_small_patch16_224', pretrained=pretrained, **dict(model_args, **kwargs))
    return model

def make_vit(model_card = "timm/vit_small_patch16_224.augreg_in1k", dataset = None, return_macs=True):
    '''
    similar to timm.models._factory.py create_models()
    '''
    model_source, model_name = parse_model_name(model_card)
    print("model_source: ", model_source)

    model_name, pretrained_tag = split_model_name_tag(model_name)

    if pretrained_tag:
        # a valid pretrained_cfg argument takes priority over tag in model name
        pretrained_cfg = pretrained_tag
    

    if model_name == "vit_small_patch16_224":
        model = vit_small_patch16_224(
            pretrained=True,
            pretrained_cfg=pretrained_cfg,
            pretrained_cfg_overlay=None,
        )
    else:
        raise NotImplementedError("Not implemented")
    return model
