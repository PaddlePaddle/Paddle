# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
import paddle.fluid as fluid
import paddle.nn as nn
from paddle.utils.download import get_weights_path_from_url

__all__ = []

model_urls = {
    'densenet121': ('', ''),
    'densenet161': ('', ''),
    'densenet169': ('', ''),
    'densenet201': ('', '')
}


class _DenseLayer(nn.Sequential):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super(_DenseLayer, self).__init__()
        self.add_sublayer('norm1', nn.BatchNorm2D(num_input_features)),
        self.add_sublayer('relu1', nn.ReLU()),
        self.add_sublayer(
            'conv1',
            nn.Conv2D(
                num_input_features,
                bn_size * growth_rate,
                kernel_size=1,
                stride=1)),
        self.add_sublayer('norm2', nn.BatchNorm2D(bn_size * growth_rate)),
        self.add_sublayer('relu2', nn.ReLU()),
        self.add_sublayer(
            'conv2',
            nn.Conv2D(
                bn_size * growth_rate,
                growth_rate,
                kernel_size=3,
                stride=1,
                padding=1)),
        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = super(_DenseLayer, self).forward(x)
        if self.drop_rate > 0:
            new_features = nn.Dropout(
                new_features, p=self.drop_rate, training=self.training)
        return fluid.layers.concat([x, new_features], axis=1)


class _DenseBlock(nn.Sequential):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate,
                 drop_rate):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features + i * growth_rate,
                                growth_rate, bn_size, drop_rate)
            self.add_sublayer('denselayer%d' % (i + 1), layer)


class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.add_sublayer('norm', nn.BatchNorm2D(num_input_features))
        self.add_sublayer('relu', nn.ReLU())
        self.add_sublayer(
            'conv',
            nn.Conv2D(
                num_input_features,
                num_output_features,
                kernel_size=1,
                stride=1))
        self.add_sublayer('pool', nn.AvgPool2D(kernel_size=2, stride=2))


class DenseNet(nn.Layer):
    """Densenet-BC model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

    Args:
        growth_rate (int) - how many filters to add each layer
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes

    Examples:
    .. code-block:: python

        from paddle.vision.models import DenseNet

        config = (6,12,32,32)

        densenet = DenseNet(block_config=config, num_classes=10)
    """

    def __init__(self,
                 growth_rate=32,
                 block_config=(6, 12, 24, 16),
                 num_init_features=64,
                 bn_size=4,
                 drop_rate=0,
                 num_classes=1000):

        super(DenseNet, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2D(
                3, num_init_features, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2D(num_init_features),
            nn.ReLU(),
            nn.MaxPool2D(
                kernel_size=3, stride=2, padding=1), )

        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(
                num_layers=num_layers,
                num_input_features=num_features,
                bn_size=bn_size,
                growth_rate=growth_rate,
                drop_rate=drop_rate)
            self.features.add_sublayer('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(
                    num_input_features=num_features,
                    num_output_features=num_features // 2)
                self.features.add_sublayer('transition%d' % (i + 1), trans)
                num_features = num_features // 2
        self.features.add_sublayer('norm5', nn.BatchNorm2D(num_features))
        self.classifier = nn.Linear(num_features, num_classes)

    def forward(self, x):
        features = self.features(x)
        out = nn.ReLU(features, )
        out = nn.AvgPool2D(
            out, kernel_size=7, stride=1).view(features.size(0), -1)
        out = self.classifier(out)
        return out


def _densenet(arch, block_cfg, pretrained, **kwargs):
    model = DenseNet(block_config=block_cfg, **kwargs)

    if pretrained:
        assert arch in model_urls, "{} model do not have a pretrained model now, you should set pretrained=False".format(
            arch)
        weight_path = get_weights_path_from_url(model_urls[arch][0],
                                                model_urls[arch][1])

        param = paddle.load(weight_path)
        model.load_dict(param)
    return model


def densenet121(pretrained=False, **kwargs):
    """Densenet-121 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet

    Examples:
    .. code-block:: python

        from paddle.vision.models import densenet121

        # build model
        model = densenet121()
    """
    model_name = 'DenseNet121'
    return _densenet(model_name, (6, 12, 24, 16), pretrained, **kwargs)


def densenet161(pretrained=False, **kwargs):
    """Densenet-161 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    
    Examples:
    .. code-block:: python

        from paddle.vision.models import densenet161

        # build model
        model = densenet161()
    """
    model_name = 'DenseNet161'
    return _densenet(model_name, (6, 12, 32, 32), pretrained, **kwargs)


def densenet169(pretrained=False, **kwargs):
    """Densenet-169 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    
    Examples:
    .. code-block:: python

        from paddle.vision.models import densenet169

        # build model
        model = densenet169()
    """
    model_name = 'DenseNet169'
    return _densenet(model_name, (6, 12, 48, 32), pretrained, **kwargs)


def densenet201(pretrained=False, **kwargs):
    """Densenet-201 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    
    Examples:
    .. code-block:: python

        from paddle.vision.models import densenet201

        # build model
        model = densenet201()
    """
    model_name = 'DenseNet201'
    return _densenet(model_name, (6, 12, 64, 48), pretrained, **kwargs)
