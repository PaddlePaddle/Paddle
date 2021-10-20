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

import numpy as np

import paddle
import paddle.fluid as fluid
import paddle.nn as nn
from paddle.utils.download import get_weights_path_from_url

__all__ = []

model_urls = {
    'densenet121':
    ('https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/DenseNet121_pretrained.pdparams',
     'db1b239ed80a905290fd8b01d3af08e4'),
    'densenet161':
    ('https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/DenseNet161_pretrained.pdparams',
     '62158869cb315098bd25ddbfd308a853'),
    'densenet169':
    ('https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/DenseNet169_pretrained.pdparams',
     '82cc7c635c3f19098c748850efb2d796'),
    'densenet201':
    ('https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/DenseNet201_pretrained.pdparams',
     '16ca29565a7712329cf9e36e02caaf58')
}


class BNConvLayer(nn.Layer):
    def __init__(self,
                 num_input_features,
                 num_filters,
                 filter_size,
                 stride=1,
                 pad=0,
                 groups=1,
                 act="relu"):
        super(BNConvLayer, self).__init__()

        self._batch_norm = nn.BatchNorm(num_input_features, act=act)

        self._conv = nn.Conv2D(
            in_channels=num_input_features,
            out_channels=num_filters,
            kernel_size=filter_size,
            stride=stride,
            padding=pad,
            groups=groups,
            bias_attr=False)

    def forward(self, x):
        out = self._batch_norm(x)
        out = self._conv(out)
        return out


class _DenseLayer(nn.Layer):
    def __init__(self, num_input_features, growth_rate, bn_size,
                 drop_rate=None):
        super(_DenseLayer, self).__init__()
        self.bn_ac_func1 = BNConvLayer(
            num_input_features=num_input_features,
            num_filters=bn_size * growth_rate,
            filter_size=1,
            pad=0,
            stride=1)

        self.bn_ac_func2 = BNConvLayer(
            num_input_features=bn_size * growth_rate,
            num_filters=growth_rate,
            filter_size=3,
            pad=1,
            stride=1)

        self.drop_rate = drop_rate
        if self.drop_rate:
            self.dropout_func = nn.Dropout(
                p=self.drop_rate, mode="downscale_in_infer")

    def forward(self, x):
        new_features = self.bn_ac_func1(x)
        out = self.bn_ac_func2(new_features)
        if self.drop_rate > 0:
            out = self.dropout_func(out)
        out = paddle.concat([x, out], axis=1)
        return out


class _DenseBlock(nn.Sequential):
    def __init__(self,
                 num_layers,
                 num_input_features,
                 bn_size,
                 growth_rate,
                 drop_rate,
                 name=None):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features + i * growth_rate,
                                growth_rate, bn_size, drop_rate)
            self.add_sublayer("{}_{}".format(name, i + 1), layer)


class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.conv_ac_func = BNConvLayer(
            num_input_features=num_input_features,
            num_filters=num_output_features,
            filter_size=1,
            pad=0,
            stride=1)
        self.pool2d_avg = nn.AvgPool2D(kernel_size=2, stride=2, padding=0)


class ConvBNLayer(nn.Layer):
    def __init__(self,
                 num_channels,
                 num_filters,
                 filter_size,
                 stride=1,
                 pad=0,
                 groups=1,
                 act="relu"):
        super(ConvBNLayer, self).__init__()

        self._conv = nn.Conv2D(
            in_channels=num_channels,
            out_channels=num_filters,
            kernel_size=filter_size,
            stride=stride,
            padding=pad,
            groups=groups,
            bias_attr=False)
        self._batch_norm = nn.BatchNorm(num_filters, act=act)

    def forward(self, input):
        y = self._conv(input)
        y = self._batch_norm(y)
        return y


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
        with_pool (bool) - use pool before the last fc layer or not

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
                 num_classes=1000,
                 with_pool=True):

        super(DenseNet, self).__init__()
        self.conv1_func = ConvBNLayer(
            num_channels=3,
            num_filters=num_init_features,
            filter_size=7,
            stride=2,
            pad=3,
            act='relu')
        self.pool2d_max = nn.MaxPool2D(kernel_size=3, stride=2, padding=1)

        num_features = num_init_features

        self.block_config = block_config

        self.dense_blocks = []
        self.transition_layers = []

        for i, num_layers in enumerate(block_config):
            self.dense_blocks.append(
                self.add_sublayer(
                    "db_conv_{}".format(i + 2),
                    _DenseBlock(
                        num_layers=num_layers,
                        num_input_features=num_features,
                        bn_size=bn_size,
                        growth_rate=growth_rate,
                        drop_rate=drop_rate,
                        name='conv' + str(i + 2))))
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                self.transition_layers.append(
                    self.add_sublayer(
                        "tr_conv{}_blk".format(i + 2),
                        _Transition(
                            num_input_features=num_features,
                            num_output_features=num_features // 2)))
                num_features = num_features // 2

        self.batch_norm = nn.BatchNorm(num_features, act="relu")
        self.with_pool = with_pool
        if with_pool:
            self.pool2d_avg = nn.AdaptiveAvgPool2D((1, 1))
        self.num_classes = num_classes
        if num_classes > 0:
            stdv = 1.0 / np.sqrt(num_features * 1.0)
            self.out = nn.Linear(
                num_features,
                num_classes,
                weight_attr=paddle.ParamAttr(
                    initializer=nn.initializer.Uniform(-stdv, stdv)))

    def forward(self, x):
        conv = self.conv1_func(x)
        conv = self.pool2d_max(conv)

        for i, num_layers in enumerate(self.block_config):
            conv = self.dense_blocks[i](conv)
            if i != len(self.block_config) - 1:
                conv = self.transition_layers[i](conv)

        conv = self.batch_norm(conv)
        if self.with_pool:
            out = self.pool2d_avg(conv)
        if self.num_classes > 0:
            out = paddle.flatten(out, 1)
            out = self.out(out)
        return out


def _densenet(arch, block_cfg, pretrained, **kwargs):
    model = DenseNet(block_config=block_cfg, **kwargs)

    if pretrained:
        assert arch in model_urls, "{} model do not have a pretrained model now, you should set pretrained=False".format(
            arch)
        weight_path = get_weights_path_from_url(model_urls[arch][0],
                                                model_urls[arch][1])
        param = paddle.load(weight_path)
        model.set_dict(param)
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
    model_name = 'densenet121'
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
    model_name = 'densenet161'
    return _densenet(
        model_name, [6, 12, 36, 24],
        pretrained,
        num_init_features=96,
        growth_rate=48,
        **kwargs)


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
    model_name = 'densenet169'
    return _densenet(model_name, [6, 12, 32, 32], pretrained, **kwargs)


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
    model_name = 'densenet201'
    return _densenet(model_name, [6, 12, 48, 32], pretrained, **kwargs)
