# copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.fluid.param_attr import ParamAttr
from paddle.nn import AdaptiveAvgPool2D, Linear, MaxPool2D
from paddle.nn.initializer import Uniform
from paddle.utils.download import get_weights_path_from_url

from ..ops import ConvNormActivation

__all__ = []

model_urls = {
    'resnext50_32x4d':
    ('https://bj.bcebos.com/v1/ai-studio-online/1f199442185d4268859cbd72e9ea529ef346ac21a3ae40f5932b1749c10c6227',
     'f848db2216597e5122b43b8e7c7f4832'),
    "resnext50_64x4d":
    ('https://bj.bcebos.com/v1/ai-studio-online/04236a150d9f4eeb9b1e39f0a06bbd87905ca78299ac463b806570a3788d83e9',
     '8c077ffdab55d9fd7df6f2d314f60573'),
    'resnext101_32x4d':
    ('https://bj.bcebos.com/v1/ai-studio-online/265c31d967814419aaac0a66850c617c9f0f25f92ed94a3fbd29d2b938104289',
     '4f9167fca54dd502810975a0b5555711'),
    'resnext101_64x4d':
    ('https://bj.bcebos.com/v1/ai-studio-online/3d60d063ce4447c8a172d81605796f6f3551017dd90e4fc5bcc27364b1cfbf1b',
     'e24397a901f3ecbd3c5b7603c178a7e3'),
    'resnext152_32x4d':
    ('https://bj.bcebos.com/v1/ai-studio-online/29dcfbfcf4df46259f19de8e025879cc1c6a9af97ac840d48585136f84a4e4c0',
     '6efb2388d62cfb9b2dca4bb58d299612'),
    'resnext152_64x4d':
    ('https://bj.bcebos.com/v1/ai-studio-online/8eb2937765154e78afbe9779a27ed9b37211dcb29ce94a5da12ef4ae1f3bae9b',
     '7d2125a165c5678c9cb92f85ef6c287e'),
}


class BottleneckBlock(nn.Layer):
    def __init__(self,
                 num_channels,
                 num_filters,
                 stride,
                 cardinality,
                 shortcut=True):
        super(BottleneckBlock, self).__init__()
        self.conv0 = ConvNormActivation(
            in_channels=num_channels,
            out_channels=num_filters,
            kernel_size=1,
            activation_layer=nn.ReLU)
        self.conv1 = ConvNormActivation(
            in_channels=num_filters,
            out_channels=num_filters,
            kernel_size=3,
            groups=cardinality,
            stride=stride,
            activation_layer=nn.ReLU)
        self.conv2 = ConvNormActivation(
            in_channels=num_filters,
            out_channels=num_filters * 2 if cardinality == 32 else num_filters,
            kernel_size=1,
            activation_layer=None)

        if not shortcut:
            self.short = ConvNormActivation(
                in_channels=num_channels,
                out_channels=num_filters * 2
                if cardinality == 32 else num_filters,
                kernel_size=1,
                stride=stride,
                activation_layer=None)

        self.shortcut = shortcut

    def forward(self, inputs):
        x = self.conv0(inputs)
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)

        if self.shortcut:
            short = inputs
        else:
            short = self.short(inputs)

        x = paddle.add(x=short, y=conv2)
        x = F.relu(x)
        return x


class ResNeXt(nn.Layer):
    """ResNeXt model from
    `"Aggregated Residual Transformations for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_

    Args:
        depth (int, optional): depth of resnext. Default: 50.
        cardinality (int, optional): cardinality of resnext. Default: 32.
        num_classes (int, optional): output dim of last fc layer. If num_classes <=0, last fc layer 
                            will not be defined. Default: 1000.
        with_pool (bool, optional): use pool before the last fc layer or not. Default: True.

    Examples:
        .. code-block:: python

            import paddle
            from paddle.vision.models import ResNeXt

            resnext50_32x4d = ResNeXt(depth=50, cardinality=32)

    """

    def __init__(self,
                 depth=50,
                 cardinality=32,
                 num_classes=1000,
                 with_pool=True):
        super(ResNeXt, self).__init__()

        self.depth = depth
        self.cardinality = cardinality
        self.num_classes = num_classes
        self.with_pool = with_pool

        supported_depth = [50, 101, 152]
        assert depth in supported_depth, \
            "supported layers are {} but input layer is {}".format(
                supported_depth, depth)
        supported_cardinality = [32, 64]
        assert cardinality in supported_cardinality, \
            "supported cardinality is {} but input cardinality is {}" \
            .format(supported_cardinality, cardinality)
        layer_cfg = {50: [3, 4, 6, 3], 101: [3, 4, 23, 3], 152: [3, 8, 36, 3]}
        layers = layer_cfg[depth]
        num_channels = [64, 256, 512, 1024]
        num_filters = [128, 256, 512,
                       1024] if cardinality == 32 else [256, 512, 1024, 2048]

        self.conv = ConvNormActivation(
            in_channels=3,
            out_channels=64,
            kernel_size=7,
            stride=2,
            activation_layer=nn.ReLU)
        self.pool2d_max = MaxPool2D(kernel_size=3, stride=2, padding=1)

        self.block_list = []
        for block in range(len(layers)):
            shortcut = False
            for i in range(layers[block]):
                bottleneck_block = self.add_sublayer(
                    'bb_%d_%d' % (block, i),
                    BottleneckBlock(
                        num_channels=num_channels[block] if i == 0 else
                        num_filters[block] * int(64 // self.cardinality),
                        num_filters=num_filters[block],
                        stride=2 if i == 0 and block != 0 else 1,
                        cardinality=self.cardinality,
                        shortcut=shortcut))
                self.block_list.append(bottleneck_block)
                shortcut = True

        if with_pool:
            self.pool2d_avg = AdaptiveAvgPool2D(1)

        if num_classes > 0:
            self.pool2d_avg_channels = num_channels[-1] * 2
            stdv = 1.0 / math.sqrt(self.pool2d_avg_channels * 1.0)
            self.out = Linear(
                self.pool2d_avg_channels,
                num_classes,
                weight_attr=ParamAttr(initializer=Uniform(-stdv, stdv)))

    def forward(self, inputs):
        with paddle.static.amp.fp16_guard():
            x = self.conv(inputs)
            x = self.pool2d_max(x)
            for block in self.block_list:
                x = block(x)
            if self.with_pool:
                x = self.pool2d_avg(x)
            if self.num_classes > 0:
                x = paddle.reshape(x, shape=[-1, self.pool2d_avg_channels])
                x = self.out(x)
            return x


def _resnext(arch, depth, cardinality, pretrained, **kwargs):
    model = ResNeXt(depth=depth, cardinality=cardinality, **kwargs)
    if pretrained:
        assert arch in model_urls, "{} model do not have a pretrained model now, you should set pretrained=False".format(
            arch)
        weight_path = get_weights_path_from_url(model_urls[arch][0],
                                                model_urls[arch][1])

        param = paddle.load(weight_path)
        model.set_dict(param)

    return model


def resnext50_32x4d(pretrained=False, **kwargs):
    """ResNeXt-50 32x4d model from
    `"Aggregated Residual Transformations for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_
    
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet

    Examples:
        .. code-block:: python

            import paddle
            from paddle.vision.models import resnext50_32x4d

            # build model
            model = resnext50_32x4d()

            # build model and load imagenet pretrained weight
            # model = resnext50_32x4d(pretrained=True)
    """
    return _resnext('resnext50_32x4d', 50, 32, pretrained, **kwargs)


def resnext50_64x4d(pretrained=False, **kwargs):
    """ResNeXt-50 64x4d model from
    `"Aggregated Residual Transformations for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_
    
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet

    Examples:
        .. code-block:: python

            import paddle
            from paddle.vision.models import resnext50_64x4d

            # build model
            model = resnext50_64x4d()

            # build model and load imagenet pretrained weight
            # model = resnext50_64x4d(pretrained=True)
    """
    return _resnext('resnext50_64x4d', 50, 64, pretrained, **kwargs)


def resnext101_32x4d(pretrained=False, **kwargs):
    """ResNeXt-101 32x4d model from
    `"Aggregated Residual Transformations for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_
    
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet

    Examples:
        .. code-block:: python

            import paddle
            from paddle.vision.models import resnext101_32x4d

            # build model
            model = resnext101_32x4d()

            # build model and load imagenet pretrained weight
            # model = resnext101_32x4d(pretrained=True)
    """
    return _resnext('resnext101_32x4d', 101, 32, pretrained, **kwargs)


def resnext101_64x4d(pretrained=False, **kwargs):
    """ResNeXt-101 64x4d model from
    `"Aggregated Residual Transformations for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_
    
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet

    Examples:
        .. code-block:: python

            import paddle
            from paddle.vision.models import resnext101_64x4d

            # build model
            model = resnext101_64x4d()

            # build model and load imagenet pretrained weight
            # model = resnext101_64x4d(pretrained=True)
    """
    return _resnext('resnext101_64x4d', 101, 64, pretrained, **kwargs)


def resnext152_32x4d(pretrained=False, **kwargs):
    """ResNeXt-152 32x4d model from
    `"Aggregated Residual Transformations for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_
    
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet

    Examples:
        .. code-block:: python

            import paddle
            from paddle.vision.models import resnext152_32x4d

            # build model
            model = resnext152_32x4d()

            # build model and load imagenet pretrained weight
            # model = resnext152_32x4d(pretrained=True)
    """
    return _resnext('resnext152_32x4d', 152, 32, pretrained, **kwargs)


def resnext152_64x4d(pretrained=False, **kwargs):
    """ResNeXt-152 64x4d model from
    `"Aggregated Residual Transformations for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_
    
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet

    Examples:
        .. code-block:: python

            import paddle
            from paddle.vision.models import resnext152_64x4d

            # build model
            model = resnext152_64x4d()

            # build model and load imagenet pretrained weight
            # model = resnext152_64x4d(pretrained=True)
    """
    return _resnext('resnext152_64x4d', 152, 64, pretrained, **kwargs)
