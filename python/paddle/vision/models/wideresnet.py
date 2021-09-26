# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import division
from __future__ import print_function

import paddle
import paddle.nn as nn

from paddle.utils.download import get_weights_path_from_url
from paddle.vision.models.resnet import BottleneckBlock, ResNet

__all__ = []

model_urls = {'wide_resnet50': ('', ''), 'wide_resnet101': ('', '')}


class WideResNet(nn.Layer):
    """Wide ResNet model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_.

    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same.

    Args:
        Block (BasicBlock|BottleneckBlock): block module of model.
        depth (int): layers of resnet, default: 50.
        num_classes (int): output dim of last fc layer. If num_classes <=0, last fc layer 
                            will not be defined. Default: 1000.
        width_per_group (int): channel nums of each group
        with_pool (bool): use pool before the last fc layer or not. Default: True.

    Examples:
        .. code-block:: python

            from paddle.vision.models import WideResNet

            wide_resnet50 = WideResNet(50)

            wide_resnet101 = WideResNet(101)

    """

    def __init__(self,
                 depth,
                 num_classes=1000,
                 width_per_group=64,
                 with_pool=True):
        super(WideResNet, self).__init__()
        self.layers = ResNet(BottleneckBlock, depth, num_classes,
                             width_per_group * 2, with_pool)

    def forward(self, x):
        return self.layers.forward(x)


def _wide_resnet(arch, depth, pretrained, **kwargs):
    model = WideResNet(depth, **kwargs)
    if pretrained:
        assert arch in model_urls, "{} model do not have a pretrained model now, you should set pretrained=False".format(
            arch)
        weight_path = get_weights_path_from_url(model_urls[arch][0],
                                                model_urls[arch][1])

        param = paddle.load(weight_path)
        model.set_dict(param)

    return model


def wide_resnet50(pretrained=False, **kwargs):
    """Wide ResNet 50-layer model
    
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet

    Examples:
        .. code-block:: python

            from paddle.vision.models import wide_resnet50

            # build model
            model = wide_resnet50()

            # build model and load imagenet pretrained weight
            # model = wide_resnet50(pretrained=True)
    """
    kwargs['width_per_group'] = 64 * 2
    return _wide_resnet('wide_resnet50', 50, pretrained, **kwargs)


def wide_resnet101(pretrained=False, **kwargs):
    """Wide ResNet 101-layer model
    
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet

    Examples:
        .. code-block:: python

            from paddle.vision.models import wide_resnet101

            # build model
            model = wide_resnet101()

            # build model and load imagenet pretrained weight
            # model = wide_resnet101(pretrained=True)
    """
    return _wide_resnet('wide_resnet101', 101, pretrained, **kwargs)
