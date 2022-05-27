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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import paddle
import paddle.nn as nn
from paddle.nn import Linear, Dropout
from paddle.nn import AdaptiveAvgPool2D, MaxPool2D, AvgPool2D
from paddle.nn.initializer import Uniform
from paddle.fluid.param_attr import ParamAttr

from paddle.utils.download import get_weights_path_from_url
from ..ops import ConvNormActivation

__all__ = []

model_urls = {
    "inception_v3":
    ("https://paddle-hapi.bj.bcebos.com/models/inception_v3.pdparams",
     "649a4547c3243e8b59c656f41fe330b8")
}


class InceptionStem(nn.Layer):
    def __init__(self):
        super().__init__()
        self.conv_1a_3x3 = ConvNormActivation(
            in_channels=3,
            out_channels=32,
            kernel_size=3,
            stride=2,
            padding=0,
            activation_layer=nn.ReLU)
        self.conv_2a_3x3 = ConvNormActivation(
            in_channels=32,
            out_channels=32,
            kernel_size=3,
            stride=1,
            padding=0,
            activation_layer=nn.ReLU)
        self.conv_2b_3x3 = ConvNormActivation(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            padding=1,
            activation_layer=nn.ReLU)

        self.max_pool = MaxPool2D(kernel_size=3, stride=2, padding=0)
        self.conv_3b_1x1 = ConvNormActivation(
            in_channels=64,
            out_channels=80,
            kernel_size=1,
            padding=0,
            activation_layer=nn.ReLU)
        self.conv_4a_3x3 = ConvNormActivation(
            in_channels=80,
            out_channels=192,
            kernel_size=3,
            padding=0,
            activation_layer=nn.ReLU)

    def forward(self, x):
        x = self.conv_1a_3x3(x)
        x = self.conv_2a_3x3(x)
        x = self.conv_2b_3x3(x)
        x = self.max_pool(x)
        x = self.conv_3b_1x1(x)
        x = self.conv_4a_3x3(x)
        x = self.max_pool(x)
        return x


class InceptionA(nn.Layer):
    def __init__(self, num_channels, pool_features):
        super().__init__()
        self.branch1x1 = ConvNormActivation(
            in_channels=num_channels,
            out_channels=64,
            kernel_size=1,
            padding=0,
            activation_layer=nn.ReLU)

        self.branch5x5_1 = ConvNormActivation(
            in_channels=num_channels,
            out_channels=48,
            kernel_size=1,
            padding=0,
            activation_layer=nn.ReLU)
        self.branch5x5_2 = ConvNormActivation(
            in_channels=48,
            out_channels=64,
            kernel_size=5,
            padding=2,
            activation_layer=nn.ReLU)

        self.branch3x3dbl_1 = ConvNormActivation(
            in_channels=num_channels,
            out_channels=64,
            kernel_size=1,
            padding=0,
            activation_layer=nn.ReLU)
        self.branch3x3dbl_2 = ConvNormActivation(
            in_channels=64,
            out_channels=96,
            kernel_size=3,
            padding=1,
            activation_layer=nn.ReLU)
        self.branch3x3dbl_3 = ConvNormActivation(
            in_channels=96,
            out_channels=96,
            kernel_size=3,
            padding=1,
            activation_layer=nn.ReLU)

        self.branch_pool = AvgPool2D(
            kernel_size=3, stride=1, padding=1, exclusive=False)
        self.branch_pool_conv = ConvNormActivation(
            in_channels=num_channels,
            out_channels=pool_features,
            kernel_size=1,
            padding=0,
            activation_layer=nn.ReLU)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)
        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

        branch_pool = self.branch_pool(x)
        branch_pool = self.branch_pool_conv(branch_pool)
        x = paddle.concat(
            [branch1x1, branch5x5, branch3x3dbl, branch_pool], axis=1)
        return x


class InceptionB(nn.Layer):
    def __init__(self, num_channels):
        super().__init__()
        self.branch3x3 = ConvNormActivation(
            in_channels=num_channels,
            out_channels=384,
            kernel_size=3,
            stride=2,
            padding=0,
            activation_layer=nn.ReLU)

        self.branch3x3dbl_1 = ConvNormActivation(
            in_channels=num_channels,
            out_channels=64,
            kernel_size=1,
            padding=0,
            activation_layer=nn.ReLU)
        self.branch3x3dbl_2 = ConvNormActivation(
            in_channels=64,
            out_channels=96,
            kernel_size=3,
            padding=1,
            activation_layer=nn.ReLU)
        self.branch3x3dbl_3 = ConvNormActivation(
            in_channels=96,
            out_channels=96,
            kernel_size=3,
            stride=2,
            padding=0,
            activation_layer=nn.ReLU)

        self.branch_pool = MaxPool2D(kernel_size=3, stride=2)

    def forward(self, x):
        branch3x3 = self.branch3x3(x)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

        branch_pool = self.branch_pool(x)

        x = paddle.concat([branch3x3, branch3x3dbl, branch_pool], axis=1)

        return x


class InceptionC(nn.Layer):
    def __init__(self, num_channels, channels_7x7):
        super().__init__()
        self.branch1x1 = ConvNormActivation(
            in_channels=num_channels,
            out_channels=192,
            kernel_size=1,
            padding=0,
            activation_layer=nn.ReLU)

        self.branch7x7_1 = ConvNormActivation(
            in_channels=num_channels,
            out_channels=channels_7x7,
            kernel_size=1,
            stride=1,
            padding=0,
            activation_layer=nn.ReLU)
        self.branch7x7_2 = ConvNormActivation(
            in_channels=channels_7x7,
            out_channels=channels_7x7,
            kernel_size=(1, 7),
            stride=1,
            padding=(0, 3),
            activation_layer=nn.ReLU)
        self.branch7x7_3 = ConvNormActivation(
            in_channels=channels_7x7,
            out_channels=192,
            kernel_size=(7, 1),
            stride=1,
            padding=(3, 0),
            activation_layer=nn.ReLU)

        self.branch7x7dbl_1 = ConvNormActivation(
            in_channels=num_channels,
            out_channels=channels_7x7,
            kernel_size=1,
            padding=0,
            activation_layer=nn.ReLU)
        self.branch7x7dbl_2 = ConvNormActivation(
            in_channels=channels_7x7,
            out_channels=channels_7x7,
            kernel_size=(7, 1),
            padding=(3, 0),
            activation_layer=nn.ReLU)
        self.branch7x7dbl_3 = ConvNormActivation(
            in_channels=channels_7x7,
            out_channels=channels_7x7,
            kernel_size=(1, 7),
            padding=(0, 3),
            activation_layer=nn.ReLU)
        self.branch7x7dbl_4 = ConvNormActivation(
            in_channels=channels_7x7,
            out_channels=channels_7x7,
            kernel_size=(7, 1),
            padding=(3, 0),
            activation_layer=nn.ReLU)
        self.branch7x7dbl_5 = ConvNormActivation(
            in_channels=channels_7x7,
            out_channels=192,
            kernel_size=(1, 7),
            padding=(0, 3),
            activation_layer=nn.ReLU)

        self.branch_pool = AvgPool2D(
            kernel_size=3, stride=1, padding=1, exclusive=False)
        self.branch_pool_conv = ConvNormActivation(
            in_channels=num_channels,
            out_channels=192,
            kernel_size=1,
            padding=0,
            activation_layer=nn.ReLU)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch7x7 = self.branch7x7_1(x)
        branch7x7 = self.branch7x7_2(branch7x7)
        branch7x7 = self.branch7x7_3(branch7x7)

        branch7x7dbl = self.branch7x7dbl_1(x)
        branch7x7dbl = self.branch7x7dbl_2(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_3(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_4(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_5(branch7x7dbl)

        branch_pool = self.branch_pool(x)
        branch_pool = self.branch_pool_conv(branch_pool)

        x = paddle.concat(
            [branch1x1, branch7x7, branch7x7dbl, branch_pool], axis=1)

        return x


class InceptionD(nn.Layer):
    def __init__(self, num_channels):
        super().__init__()
        self.branch3x3_1 = ConvNormActivation(
            in_channels=num_channels,
            out_channels=192,
            kernel_size=1,
            padding=0,
            activation_layer=nn.ReLU)
        self.branch3x3_2 = ConvNormActivation(
            in_channels=192,
            out_channels=320,
            kernel_size=3,
            stride=2,
            padding=0,
            activation_layer=nn.ReLU)

        self.branch7x7x3_1 = ConvNormActivation(
            in_channels=num_channels,
            out_channels=192,
            kernel_size=1,
            padding=0,
            activation_layer=nn.ReLU)
        self.branch7x7x3_2 = ConvNormActivation(
            in_channels=192,
            out_channels=192,
            kernel_size=(1, 7),
            padding=(0, 3),
            activation_layer=nn.ReLU)
        self.branch7x7x3_3 = ConvNormActivation(
            in_channels=192,
            out_channels=192,
            kernel_size=(7, 1),
            padding=(3, 0),
            activation_layer=nn.ReLU)
        self.branch7x7x3_4 = ConvNormActivation(
            in_channels=192,
            out_channels=192,
            kernel_size=3,
            stride=2,
            padding=0,
            activation_layer=nn.ReLU)

        self.branch_pool = MaxPool2D(kernel_size=3, stride=2)

    def forward(self, x):
        branch3x3 = self.branch3x3_1(x)
        branch3x3 = self.branch3x3_2(branch3x3)

        branch7x7x3 = self.branch7x7x3_1(x)
        branch7x7x3 = self.branch7x7x3_2(branch7x7x3)
        branch7x7x3 = self.branch7x7x3_3(branch7x7x3)
        branch7x7x3 = self.branch7x7x3_4(branch7x7x3)

        branch_pool = self.branch_pool(x)

        x = paddle.concat([branch3x3, branch7x7x3, branch_pool], axis=1)
        return x


class InceptionE(nn.Layer):
    def __init__(self, num_channels):
        super().__init__()
        self.branch1x1 = ConvNormActivation(
            in_channels=num_channels,
            out_channels=320,
            kernel_size=1,
            padding=0,
            activation_layer=nn.ReLU)
        self.branch3x3_1 = ConvNormActivation(
            in_channels=num_channels,
            out_channels=384,
            kernel_size=1,
            padding=0,
            activation_layer=nn.ReLU)
        self.branch3x3_2a = ConvNormActivation(
            in_channels=384,
            out_channels=384,
            kernel_size=(1, 3),
            padding=(0, 1),
            activation_layer=nn.ReLU)
        self.branch3x3_2b = ConvNormActivation(
            in_channels=384,
            out_channels=384,
            kernel_size=(3, 1),
            padding=(1, 0),
            activation_layer=nn.ReLU)

        self.branch3x3dbl_1 = ConvNormActivation(
            in_channels=num_channels,
            out_channels=448,
            kernel_size=1,
            padding=0,
            activation_layer=nn.ReLU)
        self.branch3x3dbl_2 = ConvNormActivation(
            in_channels=448,
            out_channels=384,
            kernel_size=3,
            padding=1,
            activation_layer=nn.ReLU)
        self.branch3x3dbl_3a = ConvNormActivation(
            in_channels=384,
            out_channels=384,
            kernel_size=(1, 3),
            padding=(0, 1),
            activation_layer=nn.ReLU)
        self.branch3x3dbl_3b = ConvNormActivation(
            in_channels=384,
            out_channels=384,
            kernel_size=(3, 1),
            padding=(1, 0),
            activation_layer=nn.ReLU)

        self.branch_pool = AvgPool2D(
            kernel_size=3, stride=1, padding=1, exclusive=False)
        self.branch_pool_conv = ConvNormActivation(
            in_channels=num_channels,
            out_channels=192,
            kernel_size=1,
            padding=0,
            activation_layer=nn.ReLU)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch3x3 = self.branch3x3_1(x)
        branch3x3 = [
            self.branch3x3_2a(branch3x3),
            self.branch3x3_2b(branch3x3),
        ]
        branch3x3 = paddle.concat(branch3x3, axis=1)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = [
            self.branch3x3dbl_3a(branch3x3dbl),
            self.branch3x3dbl_3b(branch3x3dbl),
        ]
        branch3x3dbl = paddle.concat(branch3x3dbl, axis=1)

        branch_pool = self.branch_pool(x)
        branch_pool = self.branch_pool_conv(branch_pool)

        x = paddle.concat(
            [branch1x1, branch3x3, branch3x3dbl, branch_pool], axis=1)
        return x


class InceptionV3(nn.Layer):
    """
    InceptionV3
    Args:
        num_classes (int, optional): output dim of last fc layer. If num_classes <=0, last fc layer 
                            will not be defined. Default: 1000. 
        with_pool (bool, optional): use pool before the last fc layer or not. Default: True.

    Examples:
        .. code-block:: python

            import paddle
            from paddle.vision.models import InceptionV3

            inception_v3 = InceptionV3()

            x = paddle.rand([1, 3, 299, 299])
            out = inception_v3(x)

            print(out.shape)
    """

    def __init__(self, num_classes=1000, with_pool=True):
        super().__init__()
        self.num_classes = num_classes
        self.with_pool = with_pool
        self.layers_config = {
            "inception_a": [[192, 256, 288], [32, 64, 64]],
            "inception_b": [288],
            "inception_c": [[768, 768, 768, 768], [128, 160, 160, 192]],
            "inception_d": [768],
            "inception_e": [1280, 2048]
        }

        inception_a_list = self.layers_config["inception_a"]
        inception_c_list = self.layers_config["inception_c"]
        inception_b_list = self.layers_config["inception_b"]
        inception_d_list = self.layers_config["inception_d"]
        inception_e_list = self.layers_config["inception_e"]

        self.inception_stem = InceptionStem()

        self.inception_block_list = nn.LayerList()
        for i in range(len(inception_a_list[0])):
            inception_a = InceptionA(inception_a_list[0][i],
                                     inception_a_list[1][i])
            self.inception_block_list.append(inception_a)

        for i in range(len(inception_b_list)):
            inception_b = InceptionB(inception_b_list[i])
            self.inception_block_list.append(inception_b)

        for i in range(len(inception_c_list[0])):
            inception_c = InceptionC(inception_c_list[0][i],
                                     inception_c_list[1][i])
            self.inception_block_list.append(inception_c)

        for i in range(len(inception_d_list)):
            inception_d = InceptionD(inception_d_list[i])
            self.inception_block_list.append(inception_d)

        for i in range(len(inception_e_list)):
            inception_e = InceptionE(inception_e_list[i])
            self.inception_block_list.append(inception_e)

        if with_pool:
            self.avg_pool = AdaptiveAvgPool2D(1)

        if num_classes > 0:
            self.dropout = Dropout(p=0.2, mode="downscale_in_infer")
            stdv = 1.0 / math.sqrt(2048 * 1.0)
            self.fc = Linear(
                2048,
                num_classes,
                weight_attr=ParamAttr(initializer=Uniform(-stdv, stdv)),
                bias_attr=ParamAttr())

    def forward(self, x):
        x = self.inception_stem(x)
        for inception_block in self.inception_block_list:
            x = inception_block(x)

        if self.with_pool:
            x = self.avg_pool(x)

        if self.num_classes > 0:
            x = paddle.reshape(x, shape=[-1, 2048])
            x = self.dropout(x)
            x = self.fc(x)
        return x


def inception_v3(pretrained=False, **kwargs):
    """
    InceptionV3 model from
    `"Rethinking the Inception Architecture for Computer Vision" <https://arxiv.org/pdf/1512.00567.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    
    Examples:
        .. code-block:: python

            import paddle
            from paddle.vision.models import inception_v3

            # build model
            model = inception_v3()

            # build model and load imagenet pretrained weight
            # model = inception_v3(pretrained=True)

            x = paddle.rand([1, 3, 299, 299])
            out = model(x)

            print(out.shape)
    """
    model = InceptionV3(**kwargs)
    arch = "inception_v3"
    if pretrained:
        assert arch in model_urls, "{} model do not have a pretrained model now, you should set pretrained=False".format(
            arch)
        weight_path = get_weights_path_from_url(model_urls[arch][0],
                                                model_urls[arch][1])

        param = paddle.load(weight_path)
        model.set_dict(param)
    return model
