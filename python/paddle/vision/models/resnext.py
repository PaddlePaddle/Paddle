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

import numpy as np
import paddle
from paddle import ParamAttr
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.nn import Conv2D, BatchNorm, Linear, Dropout
from paddle.nn import AdaptiveAvgPool2D, MaxPool2D, AvgPool2D
from paddle.nn.initializer import Uniform

import math

# from ppcls.utils.save_load import load_dygraph_pretrain, load_dygraph_pretrain_from_url
from paddle.utils.download import get_weights_path_from_url
model_urls = {
    'ResNeXt50_32x4d':
    ('https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/ResNeXt50_32x4d_pretrained.pdparams',
    'bf04add2f7fd22efcbe91511bcd1eebe'),
    "ResNeXt50_64x4d":
    ('https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/ResNeXt50_64x4d_pretrained.pdparams',
    '46307df0e2d6d41d3b1c1d22b00abc69'),
    'ResNeXt101_32x4d':
    ('https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/ResNeXt101_32x4d_pretrained.pdparams',
    '078ca145b3bea964ba0544303a43c36d'),
    'ResNeXt101_64x4d':
    ('https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/ResNeXt101_64x4d_pretrained.pdparams',
    '4edc0eb32d3cc5d80eff7cab32cd5c64'),
    'ResNeXt152_32x4d':
    ('https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/ResNeXt152_32x4d_pretrained.pdparams',
    '7971cc994d459af167c502366f866378'),
    'ResNeXt152_64x4d':
    ('https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/ResNeXt152_64x4d_pretrained.pdparams',
    '836943f03709efec364d486c57d132de'),
}

class ConvBNLayer(nn.Layer):
    def __init__(self,
                 num_channels,
                 num_filters,
                 filter_size,
                 stride=1,
                 groups=1,
                 act=None,
                 name=None,
                 data_format="NCHW"):
        super(ConvBNLayer, self).__init__()
        self._conv = Conv2D(
            in_channels=num_channels,
            out_channels=num_filters,
            kernel_size=filter_size,
            stride=stride,
            padding=(filter_size - 1) // 2,
            groups=groups,
            weight_attr=ParamAttr(name=name + "_weights"),
            bias_attr=False,
            data_format=data_format)
        if name == "conv1":
            bn_name = "bn_" + name
        else:
            bn_name = "bn" + name[3:]
        self._batch_norm = BatchNorm(
            num_filters,
            act=act,
            param_attr=ParamAttr(name=bn_name + '_scale'),
            bias_attr=ParamAttr(bn_name + '_offset'),
            moving_mean_name=bn_name + '_mean',
            moving_variance_name=bn_name + '_variance',
            data_layout=data_format)

    def forward(self, inputs):
        y = self._conv(inputs)
        y = self._batch_norm(y)
        return y
class BottleneckBlock(nn.Layer):
    def __init__(self,
                 num_channels,
                 num_filters,
                 stride,
                 cardinality,
                 shortcut=True,
                 name=None,
                 data_format="NCHW"):
        super(BottleneckBlock, self).__init__()
        self.conv0 = ConvBNLayer(
            num_channels=num_channels,
            num_filters=num_filters,
            filter_size=1,
            act='relu',
            name=name + "_branch2a",
            data_format=data_format)
        self.conv1 = ConvBNLayer(
            num_channels=num_filters,
            num_filters=num_filters,
            filter_size=3,
            groups=cardinality,
            stride=stride,
            act='relu',
            name=name + "_branch2b",
            data_format=data_format)
        self.conv2 = ConvBNLayer(
            num_channels=num_filters,
            num_filters=num_filters * 2 if cardinality == 32 else num_filters,
            filter_size=1,
            act=None,
            name=name + "_branch2c",
            data_format=data_format)

        if not shortcut:
            self.short = ConvBNLayer(
                num_channels=num_channels,
                num_filters=num_filters * 2
                if cardinality == 32 else num_filters,
                filter_size=1,
                stride=stride,
                name=name + "_branch1",
                data_format=data_format)

        self.shortcut = shortcut

    def forward(self, inputs):
        y = self.conv0(inputs)
        conv1 = self.conv1(y)
        conv2 = self.conv2(conv1)

        if self.shortcut:
            short = inputs
        else:
            short = self.short(inputs)

        y = paddle.add(x=short, y=conv2)
        y = F.relu(y)
        return y


class ResNeXt(nn.Layer):
    def __init__(self,
                 layers=50,
                 class_num=1000,
                 cardinality=32,
                 input_image_channel=3,
                 data_format="NCHW"):
        super(ResNeXt, self).__init__()

        self.layers = layers
        self.data_format = data_format
        self.input_image_channel = input_image_channel
        self.cardinality = cardinality
        supported_layers = [50, 101, 152]
        assert layers in supported_layers, \
            "supported layers are {} but input layer is {}".format(
                supported_layers, layers)
        supported_cardinality = [32, 64]
        assert cardinality in supported_cardinality, \
            "supported cardinality is {} but input cardinality is {}" \
            .format(supported_cardinality, cardinality)
        if layers == 50:
            depth = [3, 4, 6, 3]
        elif layers == 101:
            depth = [3, 4, 23, 3]
        elif layers == 152:
            depth = [3, 8, 36, 3]
        num_channels = [64, 256, 512, 1024]
        num_filters = [128, 256, 512,
                       1024] if cardinality == 32 else [256, 512, 1024, 2048]

        self.conv = ConvBNLayer(
            num_channels=self.input_image_channel,
            num_filters=64,
            filter_size=7,
            stride=2,
            act='relu',
            name="res_conv1",
            data_format=self.data_format)
        self.pool2d_max = MaxPool2D(
            kernel_size=3, stride=2, padding=1, data_format=self.data_format)

        self.block_list = []
        for block in range(len(depth)):
            shortcut = False
            for i in range(depth[block]):
                if layers in [101, 152] and block == 2:
                    if i == 0:
                        conv_name = "res" + str(block + 2) + "a"
                    else:
                        conv_name = "res" + str(block + 2) + "b" + str(i)
                else:
                    conv_name = "res" + str(block + 2) + chr(97 + i)
                bottleneck_block = self.add_sublayer(
                    'bb_%d_%d' % (block, i),
                    BottleneckBlock(
                        num_channels=num_channels[block] if i == 0 else
                        num_filters[block] * int(64 // self.cardinality),
                        num_filters=num_filters[block],
                        stride=2 if i == 0 and block != 0 else 1,
                        cardinality=self.cardinality,
                        shortcut=shortcut,
                        name=conv_name,
                        data_format=self.data_format))
                self.block_list.append(bottleneck_block)
                shortcut = True

        self.pool2d_avg = AdaptiveAvgPool2D(1, data_format=self.data_format)

        self.pool2d_avg_channels = num_channels[-1] * 2

        stdv = 1.0 / math.sqrt(self.pool2d_avg_channels * 1.0)

        self.out = Linear(
            self.pool2d_avg_channels,
            class_num,
            weight_attr=ParamAttr(
                initializer=Uniform(-stdv, stdv), name="fc_weights"),
            bias_attr=ParamAttr(name="fc_offset"))

    def forward(self, inputs):
        with paddle.static.amp.fp16_guard():
            if self.data_format == "NHWC":
                inputs = paddle.tensor.transpose(inputs, [0, 2, 3, 1])
                inputs.stop_gradient = True
            y = self.conv(inputs)
            y = self.pool2d_max(y)
            for block in self.block_list:
                y = block(y)
            y = self.pool2d_avg(y)
            y = paddle.reshape(y, shape=[-1, self.pool2d_avg_channels])
            y = self.out(y)
            return y


# def _load_pretrained(pretrained, model, model_url, use_ssld=False):
#     if pretrained is False:
#         pass
#     elif pretrained is True:
#         load_dygraph_pretrain_from_url(model, model_url, use_ssld=use_ssld)
#     elif isinstance(pretrained, str):
#         load_dygraph_pretrain(model, pretrained)
#     else:
#         raise RuntimeError(
#             "pretrained type is not available. Please use `string` or `boolean` type."
#         )

def _resnext(arch, layers, cardinality, pretrained, **kwargs):
    model = ResNeXt(layers=layers, cardinality=cardinality, **kwargs)
    if pretrained:
        assert arch in model_urls, "{} model do not have a pretrained model now, you should set pretrained=False".format(
            arch)
        weight_path = get_weights_path_from_url(model_urls[arch][0],
                                                model_urls[arch][1])

        param = paddle.load(weight_path)
        model.set_dict(param)

    return model

def ResNeXt50_32x4d(pretrained=False, **kwargs):


    return _resnext('ResNeXt50_32x4d', 50, 32, pretrained,**kwargs)
# def ResNeXt50_32x4d(pretrained=False, **kwargs):
#     model = ResNeXt(layers=50, cardinality=32, **kwargs)
#     _load_pretrained(
#         pretrained, model, MODEL_URLS["ResNeXt50_32x4d"])
#     return model

def ResNeXt50_64x4d(pretrained=False, **kwargs):


    return _resnext('ResNeXt50_64x4d', 50, 64, pretrained,**kwargs)
# def ResNeXt50_64x4d(pretrained=False, use_ssld=False, **kwargs):
#     model = ResNeXt(layers=50, cardinality=64, **kwargs)
#     _load_pretrained(
#         pretrained, model, MODEL_URLS["ResNeXt50_64x4d"], use_ssld=use_ssld)
#     return model

def ResNeXt101_32x4d(pretrained=False, **kwargs):


    return _resnext('ResNeXt101_32x4d', 101, 32, pretrained,**kwargs)
# def ResNeXt101_32x4d(pretrained=False, use_ssld=False, **kwargs):
#     model = ResNeXt(layers=101, cardinality=32, **kwargs)
#     _load_pretrained(
#         pretrained, model, MODEL_URLS["ResNeXt101_32x4d"], use_ssld=use_ssld)
#     return model

def ResNeXt101_64x4d(pretrained=False, **kwargs):


    return _resnext('ResNeXt101_64x4d', 101, 64, pretrained,**kwargs)
# def ResNeXt101_64x4d(pretrained=False, use_ssld=False, **kwargs):
#     model = ResNeXt(layers=101, cardinality=64, **kwargs)
#     _load_pretrained(
#         pretrained, model, MODEL_URLS["ResNeXt101_64x4d"], use_ssld=use_ssld)
#     return model

def ResNeXt152_32x4d(pretrained=False, **kwargs):


    return _resnext('ResNeXt152_32x4d', 152, 32, pretrained,**kwargs)
# def ResNeXt152_32x4d(pretrained=False, use_ssld=False, **kwargs):
#     model = ResNeXt(layers=152, cardinality=32, **kwargs)
#     _load_pretrained(
#         pretrained, model, MODEL_URLS["ResNeXt152_32x4d"], use_ssld=use_ssld)
#     return model

def ResNeXt152_64x4d(pretrained=False, **kwargs):


    return _resnext('ResNeXt152_64x4d', 152, 64, pretrained,**kwargs)
# def ResNeXt152_64x4d(pretrained=False, use_ssld=False, **kwargs):
#     model = ResNeXt(layers=152, cardinality=64, **kwargs)
#     _load_pretrained(
#         pretrained, model, MODEL_URLS["ResNeXt152_64x4d"], use_ssld=use_ssld)
#     return model
