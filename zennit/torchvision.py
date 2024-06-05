# This file is part of Zennit
# Copyright (C) 2019-2021 Christopher J. Anders
#
# zennit/torchvision.py
#
# Zennit is free software: you can redistribute it and/or modify it under
# the terms of the GNU Lesser General Public License as published by the Free
# Software Foundation; either version 3 of the License, or (at your option) any
# later version.
#
# Zennit is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for
# more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this library. If not, see <https://www.gnu.org/licenses/>.
'''Specialized Canonizers for models from torchvision.'''
import torch
from torchvision.models import MobileNetV2
from torchvision.models.mobilenetv2 import InvertedResidual
from torchvision.models.resnet import Bottleneck as ResNetBottleneck, BasicBlock

from .canonizers import SequentialMergeBatchNorm, AttributeCanonizer, CompositeCanonizer
from .layer import Sum


class VGGCanonizer(SequentialMergeBatchNorm):
    '''Canonizer for torchvision.models.vgg* type models. This is so far identical to a SequentialMergeBatchNorm'''


class MobileNetV2Canonizer(AttributeCanonizer):
    def __init__(self):
        super().__init__(self._attribute_map)

    @classmethod
    def _attribute_map(cls, name, module):
        if isinstance(module, MobileNetV2):
            attributes = {
                '_forward_impl': cls._forward_impl.__get__(module),
                'canonizer_sum': Sum(),
                'avg_pool2d': torch.nn.AdaptiveAvgPool2d(1),
            }
            return attributes
        if isinstance(module, InvertedResidual):
            attributes = {
                'forward': cls.forward.__get__(module),
                'canonizer_sum': Sum(),
                'avg_pool2d': torch.nn.AdaptiveAvgPool2d(1),
            }
            return attributes
        return None

    @staticmethod
    def _forward_impl(self, x):
        # This exists since TorchScript doesn't support inheritance, so the superclass method
        # (this one) needs to have a name other than `forward` that can be accessed in a subclass
        x = self.features(x)
        # Cannot use "squeeze" as batch-size can be 1 => must use reshape with x.shape[0]
        x = self.avg_pool2d(x).reshape(x.shape[0], -1)
        x = self.classifier(x)
        return x

    @staticmethod
    def forward(self, x):
        if self.use_res_connect:
            out = torch.stack([x, self.conv(x)], dim=-1)
            out = self.canonizer_sum(out)
            return out
        else:
            return self.conv(x)


class ResNetBottleneckCanonizer(AttributeCanonizer):
    '''Canonizer specifically for Bottlenecks of torchvision.models.resnet* type models.'''
    def __init__(self):
        super().__init__(self._attribute_map)

    @classmethod
    def _attribute_map(cls, name, module):
        '''Create a forward function and a Sum module to overload as new attributes for module.

        Parameters
        ----------
        name : string
            Name by which the module is identified.
        module : obj:`torch.nn.Module`
            Instance of a module. If this is a Bottleneck layer, the appropriate attributes to overload are returned.

        Returns
        -------
        None or dict
            None if `module` is not an instance of Bottleneck, otherwise the appropriate attributes to overload onto
            the module instance.
        '''
        if isinstance(module, ResNetBottleneck):
            attributes = {
                'forward': cls.forward.__get__(module),
                'canonizer_sum': Sum(),
            }
            return attributes

        if isinstance(module, BasicBlock):
            attributes = {
                'forward': cls.forward_basic.__get__(module),
                'canonizer_sum': Sum(),
            }
            return attributes
        return None

    @staticmethod
    def forward(self, x):
        '''Modified Bottleneck forward for ResNet.'''
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = torch.stack([identity, out], dim=-1)
        out = self.canonizer_sum(out)

        out = self.relu(out)

        return out

    @staticmethod
    def forward_basic(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = torch.stack([identity, out], dim=-1)
        out = self.canonizer_sum(out)
        out = self.relu(out)

        return out


class ResNetCanonizer(CompositeCanonizer):
    '''Canonizer for torchvision.models.resnet* type models. This applies SequentialMergeBatchNorm, as well as
    add a Sum module to the Bottleneck modules and overload their forward method to use the Sum module instead of
    simply adding two tensors, such that forward and backward hooks may be applied.'''
    def __init__(self):
        super().__init__((
            SequentialMergeBatchNorm(),
            ResNetBottleneckCanonizer(),
        ))


class MobileNetCanonizer(CompositeCanonizer):
    '''Canonizer for torchvision.models.resnet* type models. This applies SequentialMergeBatchNorm, as well as
    add a Sum module to the Bottleneck modules and overload their forward method to use the Sum module instead of
    simply adding two tensors, such that forward and backward hooks may be applied.'''
    def __init__(self):
        super().__init__((
            SequentialMergeBatchNorm(),
            MobileNetV2Canonizer(),
        ))