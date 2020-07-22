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

import paddle.fluid as fluid
from paddle.fluid.initializer import MSRA
from paddle.fluid.param_attr import ParamAttr
from paddle.fluid.dygraph.nn import Conv2D, Pool2D, BatchNorm, Linear

from ...model import Model
from ...download import get_weights_path_from_url

__all__ = ['MobileNetV1', 'mobilenet_v1']

model_urls = {
    'mobilenetv1_1.0':
    ('https://paddle-hapi.bj.bcebos.com/models/mobilenet_v1_x1.0.pdparams',
     'bf0d25cb0bed1114d9dac9384ce2b4a6')
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
                 act='relu',
                 use_cudnn=True,
                 name=None):
        super(ConvBNLayer, self).__init__()

        self._conv = Conv2D(
            num_channels=num_channels,
            num_filters=num_filters,
            filter_size=filter_size,
            stride=stride,
            padding=padding,
            groups=num_groups,
            act=None,
            use_cudnn=use_cudnn,
            param_attr=ParamAttr(
                initializer=MSRA(), name=self.full_name() + "_weights"),
            bias_attr=False)

        self._batch_norm = BatchNorm(
            num_filters,
            act=act,
            param_attr=ParamAttr(name=self.full_name() + "_bn" + "_scale"),
            bias_attr=ParamAttr(name=self.full_name() + "_bn" + "_offset"),
            moving_mean_name=self.full_name() + "_bn" + '_mean',
            moving_variance_name=self.full_name() + "_bn" + '_variance')

    def forward(self, inputs):
        y = self._conv(inputs)
        y = self._batch_norm(y)
        return y


class DepthwiseSeparable(fluid.dygraph.Layer):
    def __init__(self,
                 num_channels,
                 num_filters1,
                 num_filters2,
                 num_groups,
                 stride,
                 scale,
                 name=None):
        super(DepthwiseSeparable, self).__init__()

        self._depthwise_conv = ConvBNLayer(
            num_channels=num_channels,
            num_filters=int(num_filters1 * scale),
            filter_size=3,
            stride=stride,
            padding=1,
            num_groups=int(num_groups * scale),
            use_cudnn=False)

        self._pointwise_conv = ConvBNLayer(
            num_channels=int(num_filters1 * scale),
            filter_size=1,
            num_filters=int(num_filters2 * scale),
            stride=1,
            padding=0)

    def forward(self, inputs):
        y = self._depthwise_conv(inputs)
        y = self._pointwise_conv(y)
        return y


class MobileNetV1(Model):
    """MobileNetV1 model from
    `"MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications" <https://arxiv.org/abs/1704.04861>`_.

    Args:
        scale (float): scale of channels in each layer. Default: 1.0.
        num_classes (int): output dim of last fc layer. If num_classes <=0, last fc layer 
                            will not be defined. Default: 1000.
        with_pool (bool): use pool before the last fc layer or not. Default: True.
        classifier_activation (str): activation for the last fc layer. Default: 'softmax'.

    Examples:
        .. code-block:: python

            from paddle.incubate.hapi.vision.models import MobileNetV1

            model = MobileNetV1()
    """

    def __init__(self,
                 scale=1.0,
                 num_classes=1000,
                 with_pool=True,
                 classifier_activation='softmax'):
        super(MobileNetV1, self).__init__()
        self.scale = scale
        self.dwsl = []
        self.num_classes = num_classes
        self.with_pool = with_pool

        self.conv1 = ConvBNLayer(
            num_channels=3,
            filter_size=3,
            channels=3,
            num_filters=int(32 * scale),
            stride=2,
            padding=1)

        dws21 = self.add_sublayer(
            sublayer=DepthwiseSeparable(
                num_channels=int(32 * scale),
                num_filters1=32,
                num_filters2=64,
                num_groups=32,
                stride=1,
                scale=scale),
            name="conv2_1")
        self.dwsl.append(dws21)

        dws22 = self.add_sublayer(
            sublayer=DepthwiseSeparable(
                num_channels=int(64 * scale),
                num_filters1=64,
                num_filters2=128,
                num_groups=64,
                stride=2,
                scale=scale),
            name="conv2_2")
        self.dwsl.append(dws22)

        dws31 = self.add_sublayer(
            sublayer=DepthwiseSeparable(
                num_channels=int(128 * scale),
                num_filters1=128,
                num_filters2=128,
                num_groups=128,
                stride=1,
                scale=scale),
            name="conv3_1")
        self.dwsl.append(dws31)

        dws32 = self.add_sublayer(
            sublayer=DepthwiseSeparable(
                num_channels=int(128 * scale),
                num_filters1=128,
                num_filters2=256,
                num_groups=128,
                stride=2,
                scale=scale),
            name="conv3_2")
        self.dwsl.append(dws32)

        dws41 = self.add_sublayer(
            sublayer=DepthwiseSeparable(
                num_channels=int(256 * scale),
                num_filters1=256,
                num_filters2=256,
                num_groups=256,
                stride=1,
                scale=scale),
            name="conv4_1")
        self.dwsl.append(dws41)

        dws42 = self.add_sublayer(
            sublayer=DepthwiseSeparable(
                num_channels=int(256 * scale),
                num_filters1=256,
                num_filters2=512,
                num_groups=256,
                stride=2,
                scale=scale),
            name="conv4_2")
        self.dwsl.append(dws42)

        for i in range(5):
            tmp = self.add_sublayer(
                sublayer=DepthwiseSeparable(
                    num_channels=int(512 * scale),
                    num_filters1=512,
                    num_filters2=512,
                    num_groups=512,
                    stride=1,
                    scale=scale),
                name="conv5_" + str(i + 1))
            self.dwsl.append(tmp)

        dws56 = self.add_sublayer(
            sublayer=DepthwiseSeparable(
                num_channels=int(512 * scale),
                num_filters1=512,
                num_filters2=1024,
                num_groups=512,
                stride=2,
                scale=scale),
            name="conv5_6")
        self.dwsl.append(dws56)

        dws6 = self.add_sublayer(
            sublayer=DepthwiseSeparable(
                num_channels=int(1024 * scale),
                num_filters1=1024,
                num_filters2=1024,
                num_groups=1024,
                stride=1,
                scale=scale),
            name="conv6")
        self.dwsl.append(dws6)

        if with_pool:
            self.pool2d_avg = Pool2D(pool_type='avg', global_pooling=True)

        if num_classes > -1:
            self.out = Linear(
                int(1024 * scale),
                num_classes,
                act=classifier_activation,
                param_attr=ParamAttr(
                    initializer=MSRA(), name=self.full_name() + "fc7_weights"),
                bias_attr=ParamAttr(name="fc7_offset"))

    def forward(self, inputs):
        y = self.conv1(inputs)
        for dws in self.dwsl:
            y = dws(y)

        if self.with_pool:
            y = self.pool2d_avg(y)

        if self.num_classes > 0:
            y = fluid.layers.reshape(y, shape=[-1, 1024])
            y = self.out(y)
        return y


def _mobilenet(arch, pretrained=False, **kwargs):
    model = MobileNetV1(**kwargs)
    if pretrained:
        assert arch in model_urls, "{} model do not have a pretrained model now, you should set pretrained=False".format(
            arch)
        weight_path = get_weights_path_from_url(model_urls[arch][0],
                                                model_urls[arch][1])
        assert weight_path.endswith(
            '.pdparams'), "suffix of weight must be .pdparams"
        model.load(weight_path)

    return model


def mobilenet_v1(pretrained=False, scale=1.0, **kwargs):
    """MobileNetV1
    
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet. Default: False.
        scale: (float): scale of channels in each layer. Default: 1.0.

    Examples:
        .. code-block:: python

            from paddle.incubate.hapi.vision.models import mobilenet_v1

            # build model
            model = mobilenet_v1()

            # build model and load imagenet pretrained weight
            # model = mobilenet_v1(pretrained=True)

            # build mobilenet v1 with scale=0.5
            model = mobilenet_v1(scale=0.5)
    """
    model = _mobilenet(
        'mobilenetv1_' + str(scale), pretrained, scale=scale, **kwargs)
    return model
