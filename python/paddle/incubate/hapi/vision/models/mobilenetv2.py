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

import numpy as np
import paddle
import paddle.fluid as fluid
from paddle.fluid.param_attr import ParamAttr
from paddle.fluid.dygraph.nn import Conv2D, Pool2D, BatchNorm, Linear

from ...model import Model
from ...download import get_weights_path_from_url

__all__ = ['MobileNetV2', 'mobilenet_v2']

model_urls = {
    'mobilenetv2_1.0':
    ('https://paddle-hapi.bj.bcebos.com/models/mobilenet_v2_x1.0.pdparams',
     '8ff74f291f72533f2a7956a4efff9d88')
}


class ConvBNLayer(fluid.dygraph.Layer):
    def __init__(self,
                 num_channels,
                 filter_size,
                 num_filters,
                 stride,
                 padding,
                 channels=None,
                 num_groups=1,
                 use_cudnn=True):
        super(ConvBNLayer, self).__init__()

        tmp_param = ParamAttr(name=self.full_name() + "_weights")
        self._conv = Conv2D(
            num_channels=num_channels,
            num_filters=num_filters,
            filter_size=filter_size,
            stride=stride,
            padding=padding,
            groups=num_groups,
            act=None,
            use_cudnn=use_cudnn,
            param_attr=tmp_param,
            bias_attr=False)

        self._batch_norm = BatchNorm(
            num_filters,
            param_attr=ParamAttr(name=self.full_name() + "_bn" + "_scale"),
            bias_attr=ParamAttr(name=self.full_name() + "_bn" + "_offset"),
            moving_mean_name=self.full_name() + "_bn" + '_mean',
            moving_variance_name=self.full_name() + "_bn" + '_variance')

    def forward(self, inputs, if_act=True):
        y = self._conv(inputs)
        y = self._batch_norm(y)
        if if_act:
            y = fluid.layers.relu6(y)
        return y


class InvertedResidualUnit(fluid.dygraph.Layer):
    def __init__(
            self,
            num_channels,
            num_in_filter,
            num_filters,
            stride,
            filter_size,
            padding,
            expansion_factor, ):
        super(InvertedResidualUnit, self).__init__()
        num_expfilter = int(round(num_in_filter * expansion_factor))
        self._expand_conv = ConvBNLayer(
            num_channels=num_channels,
            num_filters=num_expfilter,
            filter_size=1,
            stride=1,
            padding=0,
            num_groups=1)

        self._bottleneck_conv = ConvBNLayer(
            num_channels=num_expfilter,
            num_filters=num_expfilter,
            filter_size=filter_size,
            stride=stride,
            padding=padding,
            num_groups=num_expfilter,
            use_cudnn=False)

        self._linear_conv = ConvBNLayer(
            num_channels=num_expfilter,
            num_filters=num_filters,
            filter_size=1,
            stride=1,
            padding=0,
            num_groups=1)

    def forward(self, inputs, ifshortcut):
        y = self._expand_conv(inputs, if_act=True)
        y = self._bottleneck_conv(y, if_act=True)
        y = self._linear_conv(y, if_act=False)
        if ifshortcut:
            y = fluid.layers.elementwise_add(inputs, y)
        return y


class InvresiBlocks(fluid.dygraph.Layer):
    def __init__(self, in_c, t, c, n, s):
        super(InvresiBlocks, self).__init__()

        self._first_block = InvertedResidualUnit(
            num_channels=in_c,
            num_in_filter=in_c,
            num_filters=c,
            stride=s,
            filter_size=3,
            padding=1,
            expansion_factor=t)

        self._inv_blocks = []
        for i in range(1, n):
            tmp = self.add_sublayer(
                sublayer=InvertedResidualUnit(
                    num_channels=c,
                    num_in_filter=c,
                    num_filters=c,
                    stride=1,
                    filter_size=3,
                    padding=1,
                    expansion_factor=t),
                name=self.full_name() + "_" + str(i + 1))
            self._inv_blocks.append(tmp)

    def forward(self, inputs):
        y = self._first_block(inputs, ifshortcut=False)
        for inv_block in self._inv_blocks:
            y = inv_block(y, ifshortcut=True)
        return y


class MobileNetV2(Model):
    """MobileNetV2 model from
    `"MobileNetV2: Inverted Residuals and Linear Bottlenecks" <https://arxiv.org/abs/1801.04381>`_.

    Args:
        scale (float): scale of channels in each layer. Default: 1.0.
        num_classes (int): output dim of last fc layer. If num_classes <=0, last fc layer 
                            will not be defined. Default: 1000.
        with_pool (bool): use pool before the last fc layer or not. Default: True.
        classifier_activation (str): activation for the last fc layer. Default: 'softmax'.

    Examples:
        .. code-block:: python

            from paddle.incubate.hapi.vision.models import MobileNetV2

            model = MobileNetV2()
    """

    def __init__(self,
                 scale=1.0,
                 num_classes=1000,
                 with_pool=True,
                 classifier_activation='softmax'):
        super(MobileNetV2, self).__init__()
        self.scale = scale
        self.num_classes = num_classes
        self.with_pool = with_pool

        bottleneck_params_list = [
            (1, 16, 1, 1),
            (6, 24, 2, 2),
            (6, 32, 3, 2),
            (6, 64, 4, 2),
            (6, 96, 3, 1),
            (6, 160, 3, 2),
            (6, 320, 1, 1),
        ]

        self._conv1 = ConvBNLayer(
            num_channels=3,
            num_filters=int(32 * scale),
            filter_size=3,
            stride=2,
            padding=1)

        self._invl = []
        i = 1
        in_c = int(32 * scale)
        for layer_setting in bottleneck_params_list:
            t, c, n, s = layer_setting
            i += 1
            tmp = self.add_sublayer(
                sublayer=InvresiBlocks(
                    in_c=in_c, t=t, c=int(c * scale), n=n, s=s),
                name='conv' + str(i))
            self._invl.append(tmp)
            in_c = int(c * scale)

        self._out_c = int(1280 * scale) if scale > 1.0 else 1280
        self._conv9 = ConvBNLayer(
            num_channels=in_c,
            num_filters=self._out_c,
            filter_size=1,
            stride=1,
            padding=0)

        if with_pool:
            self._pool2d_avg = Pool2D(pool_type='avg', global_pooling=True)

        if num_classes > 0:
            tmp_param = ParamAttr(name=self.full_name() + "fc10_weights")
            self._fc = Linear(
                self._out_c,
                num_classes,
                act=classifier_activation,
                param_attr=tmp_param,
                bias_attr=ParamAttr(name="fc10_offset"))

    def forward(self, inputs):
        y = self._conv1(inputs, if_act=True)
        for inv in self._invl:
            y = inv(y)
        y = self._conv9(y, if_act=True)

        if self.with_pool:
            y = self._pool2d_avg(y)
        if self.num_classes > 0:
            y = fluid.layers.reshape(y, shape=[-1, self._out_c])
            y = self._fc(y)
        return y


def _mobilenet(arch, pretrained=False, **kwargs):
    model = MobileNetV2(**kwargs)
    if pretrained:
        assert arch in model_urls, "{} model do not have a pretrained model now, you should set pretrained=False".format(
            arch)
        weight_path = get_weights_path_from_url(model_urls[arch][0],
                                                model_urls[arch][1])
        assert weight_path.endswith(
            '.pdparams'), "suffix of weight must be .pdparams"
        model.load(weight_path)

    return model


def mobilenet_v2(pretrained=False, scale=1.0, **kwargs):
    """MobileNetV2
    
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet. Default: False.
        scale: (float): scale of channels in each layer. Default: 1.0.

    Examples:
        .. code-block:: python

            from paddle.incubate.hapi.vision.models import mobilenet_v2

            # build model
            model = mobilenet_v2()

            # build model and load imagenet pretrained weight
            # model = mobilenet_v2(pretrained=True)

            # build mobilenet v2 with scale=0.5
            model = mobilenet_v2(scale=0.5)
    """
    model = _mobilenet(
        'mobilenetv2_' + str(scale), pretrained, scale=scale, **kwargs)
    return model
