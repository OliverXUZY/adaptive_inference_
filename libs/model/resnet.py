""" Simplified implementation of ResNets for adaptive inference """
import os

import torch
import torch.nn as nn
import numpy as np
from torchvision.models import get_weight
from timm.models._hub import has_hf_hub, download_cached_file, check_cached_file, load_state_dict_from_hf
from collections import OrderedDict

ckpt_dir = os.path.join(os.path.dirname(__file__), 'ckpt')
macs_dir = os.path.join(os.path.dirname(__file__), 'macs')

n_classes = {'imagenet': 1000, 'cifar10': 10, 'cifar100': 100}


class Block(nn.Module):

    def forward(self, x, mask=None):
        """
        Args:
            x (float tensor, (bs, c, h, w)): feature maps.
            mask (bool tensor, (bs,)): mask for residual connection.
        """
        if mask is None:
            x = self.actv(self.downsample(x) + self._residual(x))
        else:
            res = self._residual(x[mask])
            x = self.downsample(x)
            x[mask] = x[mask] + res
            x = self.actv(x)
        return x


class BasicBlock(Block):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1):
        super(BasicBlock, self).__init__()

        outplanes = planes * self.expansion

        self.conv1 = nn.Conv2d(inplanes, planes, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        
        self.conv2 = nn.Conv2d(planes, outplanes, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(outplanes)

        if stride > 1 or outplanes != inplanes:
            self.downsample = nn.Sequential(
                nn.Conv2d(inplanes, outplanes, 1, stride, 0, bias=False),
                nn.BatchNorm2d(outplanes),
            )
        else:
            self.downsample = nn.Identity()

        self.actv = nn.ReLU(inplace=True)

    def _residual(self, x):
        x = self.actv(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return x


class Bottleneck(Block):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        super(Bottleneck, self).__init__()

        outplanes = planes * self.expansion

        self.conv1 = nn.Conv2d(inplanes, planes, 1, 1, 0, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, 3, stride, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.conv3 = nn.Conv2d(planes, outplanes, 1, 1, 0, bias=False)
        self.bn3 = nn.BatchNorm2d(outplanes)

        if stride > 1 or outplanes != inplanes:
            self.downsample = nn.Sequential(
                nn.Conv2d(inplanes, outplanes, 1, stride, 0, bias=False),
                nn.BatchNorm2d(outplanes),
            )
        else:
            self.downsample = nn.Identity()

        self.actv = nn.ReLU(inplace=True)

    def _residual(self, x):
        x = self.actv(self.bn1(self.conv1(x)))
        x = self.actv(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        return x


class ResNetFeatures(nn.Module):

    def __init__(
        self, 
        block,          # residual block constructor 
        n_blocks,       # number of blocks per stage
        n_planes,       # number of channels at each stage
        dataset,        # dataset name
    ):
        super(ResNetFeatures, self).__init__()

        assert len(n_blocks) in (3, 4)
        assert len(n_planes) == len(n_blocks)
        self.n_blocks = sum(n_blocks)

        # stem
        if dataset == 'imagenet':
            self.conv1 = nn.Conv2d(3, n_planes[0], 7, 2, 3, bias=False)
            self.maxpool = nn.MaxPool2d(3, 2, 1)
        elif dataset in ('cifar', 'cifar10', 'cifar100'):
            self.conv1 = nn.Conv2d(3, n_planes[0], 3, 1, 1, bias=False)
            self.maxpool = nn.Identity()
        else:
            raise NotImplementedError('invalid dataset: {:s}'.format(dataset))
        self.bn1 = nn.BatchNorm2d(n_planes[0])
        self.actv = nn.ReLU(inplace=True)

        # residual layers
        self.layer1 = self._make_stage(
            block, n_blocks[0], n_planes[0], n_planes[0], stride=1
        )
        self.layer2 = self._make_stage(
            block, n_blocks[1], n_planes[0] * block.expansion, n_planes[1]
        )
        self.layer3 = self._make_stage(
            block, n_blocks[2], n_planes[1] * block.expansion, n_planes[2]
        )
        if len(n_blocks) == 4:
            self.layer4 = self._make_stage(
                block, n_blocks[3], n_planes[2] * block.expansion, n_planes[3]
            )
        else:
            self.layer4 = None

    def _make_stage(self, block_fn, n_blocks, inplanes, planes, stride=2):
        blocks = []
        for idx in range(n_blocks):
            blocks.append(block_fn(inplanes, planes, stride))
            inplanes = planes * block_fn.expansion
            stride = 1
        return nn.ModuleList(blocks)

    def forward(self, x, masks=None):
        """
        Args:
            x (float tensor, (bs, c, h, w)): input images.
            masks (bool tensor, (bs, n)): masks for residual connections.
        """
        if masks is not None:
            assert masks.size(1) == self.n_blocks - 1
        mask_idx = 0

        x = self.actv(self.bn1(self.conv1(x)))
        x = self.maxpool(x)

        for block_idx, block in enumerate(self.layer1):
            ## NOTE: never drop the first block
            if block_idx == 0 or masks is None:
                mask = None
            else:
                mask = masks[:, mask_idx]
                mask_idx += 1
            x = block(x, mask)

        for block in self.layer2:
            mask = None if masks is None else masks[:, mask_idx]
            mask_idx += 1
            x = block(x, mask)

        for block in self.layer3:
            mask = None if masks is None else masks[:, mask_idx]
            mask_idx += 1
            x = block(x, mask)

        if self.layer4 is not None:
            for block in self.layer4:
                mask = None if masks is None else masks[:, mask_idx]
                mask_idx += 1
                x = block(x, mask)

        return x


class ResNet(ResNetFeatures):

    def __init__(self, block, n_blocks, n_planes, dataset):
        super(ResNet, self).__init__(block, n_blocks, n_planes, dataset)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(
            n_planes[-1] * block.expansion, n_classes[dataset]
        )

    def forward(self, x, mask=None):
        x = super(ResNet, self).forward(x, mask)
        x = self.avgpool(x).flatten(1)
        x = self.fc(x)
        return x

###############################################################################
# Pre-trained ResNet models
###############################################################################
def resnet18(dataset='imagenet'):
    return ResNet(BasicBlock, [2, 2, 2, 2], [64, 128, 256, 512], dataset)

def resnet34(dataset='imagenet'):
    return ResNet(BasicBlock, [3, 4, 6, 3], [64, 128, 256, 512], dataset)

def resnet50(dataset='imagenet'):
    return ResNet(Bottleneck, [3, 4, 6, 3], [64, 128, 256, 512], dataset)

def resnet101(dataset='imagenet'):
    return ResNet(Bottleneck, [3, 4, 23, 3], [64, 128, 256, 512], dataset)

def make_resnet(arch, dataset, return_macs=True, load_from = "customized", model_card = "timm/resnet50.a1_in1k"):
    if arch == 'resnet18':      model = resnet18(dataset)
    elif arch == 'resnet34':    model = resnet34(dataset)
    elif arch == 'resnet50':    model = resnet50(dataset)
    elif arch == 'resnet101':   model = resnet101(dataset)
    else:
        raise NotImplementedError(
            'invalid ResNet architecture: {:s}'.format(arch)
        )

    # load pre-trained weights
    # TODO: temporarily!
    if load_from == "customized":
        ckpt_path = os.path.join(ckpt_dir, '{:s}_{:s}.pth'.format(arch, dataset))
        if torch.cuda.is_available():
            model.load_state_dict(torch.load(ckpt_path))
        else:
            model.load_state_dict(torch.load(ckpt_path, map_location=torch.device('cpu')))
    elif load_from == "timm":
        state_dict = load_state_dict_from_hf(model_card)
        print(f"load_state_dict_from_hf--{model_card}--length of state_dict{len(state_dict.items())}")
        # Create an ordered dictionary with keys in the same order as resnet50.state_dict()
        ordered_state_dict = OrderedDict((key, state_dict[key]) for key in model.state_dict().keys() if key in state_dict)
        model.load_state_dict(ordered_state_dict)
    else:
        raise NotImplementedError("This feature hasn't been implemented yet.")

    model.eval()

    if not return_macs:
        return model

    # load per-block MACs breakdown
    macs_path = os.path.join(macs_dir, '{:s}_{:s}.npy'.format(arch, dataset))
    macs = torch.from_numpy(np.load(macs_path).astype(np.float32))

    return model, macs

###############################################################################
# Content encoders
###############################################################################
def resnet8_cifar():
    model = ResNetFeatures(BasicBlock, [1, 1, 1], [16, 32, 64], 'cifar')
    return model

def resnet10_imagenet():
    model = ResNetFeatures(
        BasicBlock, [1, 1, 1, 1], [64, 128, 256, 512], 'imagenet'
    )
    return model