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

import paddle
import paddle.nn as nn
from paddle.fluid.param_attr import ParamAttr
from paddle.nn import AdaptiveAvgPool2D, BatchNorm, Conv2D, Linear, MaxPool2D
from paddle.utils.download import get_weights_path_from_url

__all__ = []

model_urls = {
    "shufflenet_v2_x0_25": (
        "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/ShuffleNetV2_x0_25_pretrained.pdparams",
        "e753404cbd95027759c5f56ecd6c9c4b", ),
    "shufflenet_v2_x0_33": (
        "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/ShuffleNetV2_x0_33_pretrained.pdparams",
        "776e3cf9a4923abdfce789c45b8fe1f2", ),
    "shufflenet_v2_x0_5": (
        "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/ShuffleNetV2_x0_5_pretrained.pdparams",
        "e3649cf531566917e2969487d2bc6b60", ),
    "shufflenet_v2_x1_0": (
        "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/ShuffleNetV2_x1_0_pretrained.pdparams",
        "7821c348ea34e58847c43a08a4ac0bdf", ),
    "shufflenet_v2_x1_5": (
        "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/ShuffleNetV2_x1_5_pretrained.pdparams",
        "93a07fa557ab2d8803550f39e5b6c391", ),
    "shufflenet_v2_x2_0": (
        "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/ShuffleNetV2_x2_0_pretrained.pdparams",
        "4ab1f622fd0d341e0f84b4e057797563", ),
    "shufflenet_v2_swish": (
        "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/ShuffleNetV2_swish_pretrained.pdparams",
        "daff38b3df1b3748fccbb13cfdf02519", ),
}


def channel_shuffle(x, groups):
    batch_size, num_channels, height, width = x.shape[0:4]
    channels_per_group = num_channels // groups

    # reshape
    x = paddle.reshape(
        x, shape=[batch_size, groups, channels_per_group, height, width])

    # transpose
    x = paddle.transpose(x, perm=[0, 2, 1, 3, 4])

    # flatten
    x = paddle.reshape(x, shape=[batch_size, num_channels, height, width])
    return x


class ConvBNLayer(nn.Layer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 padding,
                 groups=1,
                 act=None):
        super(ConvBNLayer, self).__init__()
        self._conv = Conv2D(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            weight_attr=ParamAttr(initializer=nn.initializer.KaimingNormal()),
            bias_attr=False, )

        self._batch_norm = BatchNorm(out_channels, act=act)

    def forward(self, inputs):
        x = self._conv(inputs)
        x = self._batch_norm(x)
        return x


class InvertedResidual(nn.Layer):
    def __init__(self, in_channels, out_channels, stride, act="relu"):
        super(InvertedResidual, self).__init__()
        self._conv_pw = ConvBNLayer(
            in_channels=in_channels // 2,
            out_channels=out_channels // 2,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1,
            act=act)
        self._conv_dw = ConvBNLayer(
            in_channels=out_channels // 2,
            out_channels=out_channels // 2,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=out_channels // 2,
            act=None)
        self._conv_linear = ConvBNLayer(
            in_channels=out_channels // 2,
            out_channels=out_channels // 2,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1,
            act=act)

    def forward(self, inputs):
        x1, x2 = paddle.split(
            inputs,
            num_or_sections=[inputs.shape[1] // 2, inputs.shape[1] // 2],
            axis=1)
        x2 = self._conv_pw(x2)
        x2 = self._conv_dw(x2)
        x2 = self._conv_linear(x2)
        out = paddle.concat([x1, x2], axis=1)
        return channel_shuffle(out, 2)


class InvertedResidualDS(nn.Layer):
    def __init__(self, in_channels, out_channels, stride, act="relu"):
        super(InvertedResidualDS, self).__init__()

        # branch1
        self._conv_dw_1 = ConvBNLayer(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=in_channels,
            act=None)
        self._conv_linear_1 = ConvBNLayer(
            in_channels=in_channels,
            out_channels=out_channels // 2,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1,
            act=act)
        # branch2
        self._conv_pw_2 = ConvBNLayer(
            in_channels=in_channels,
            out_channels=out_channels // 2,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1,
            act=act)
        self._conv_dw_2 = ConvBNLayer(
            in_channels=out_channels // 2,
            out_channels=out_channels // 2,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=out_channels // 2,
            act=None)
        self._conv_linear_2 = ConvBNLayer(
            in_channels=out_channels // 2,
            out_channels=out_channels // 2,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1,
            act=act)

    def forward(self, inputs):
        x1 = self._conv_dw_1(inputs)
        x1 = self._conv_linear_1(x1)
        x2 = self._conv_pw_2(inputs)
        x2 = self._conv_dw_2(x2)
        x2 = self._conv_linear_2(x2)
        out = paddle.concat([x1, x2], axis=1)

        return channel_shuffle(out, 2)


class ShuffleNetV2(nn.Layer):
    """ShuffleNetV2 model from
    `"ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design" <https://arxiv.org/pdf/1807.11164.pdf>`_

    Args:
        scale (float, optional) - scale of output channels. Default: True.
        act (str, optional) - activation function of neural network. Default: "relu".
        num_classes (int, optional): output dim of last fc layer. If num_classes <=0, last fc layer
                            will not be defined. Default: 1000.
        with_pool (bool, optional): use pool before the last fc layer or not. Default: True.

    Examples:
        .. code-block:: python

            import paddle
            from paddle.vision.models import ShuffleNetV2

            shufflenet_v2_swish = ShuffleNetV2(scale=1.0, act="swish")
            x = paddle.rand([1, 3, 224, 224])
            out = shufflenet_v2_swish(x)
            print(out.shape)

    """

    def __init__(self, scale=1.0, act="relu", num_classes=1000, with_pool=True):
        super(ShuffleNetV2, self).__init__()
        self.scale = scale
        self.num_classes = num_classes
        self.with_pool = with_pool
        stage_repeats = [4, 8, 4]

        if scale == 0.25:
            stage_out_channels = [-1, 24, 24, 48, 96, 512]
        elif scale == 0.33:
            stage_out_channels = [-1, 24, 32, 64, 128, 512]
        elif scale == 0.5:
            stage_out_channels = [-1, 24, 48, 96, 192, 1024]
        elif scale == 1.0:
            stage_out_channels = [-1, 24, 116, 232, 464, 1024]
        elif scale == 1.5:
            stage_out_channels = [-1, 24, 176, 352, 704, 1024]
        elif scale == 2.0:
            stage_out_channels = [-1, 24, 224, 488, 976, 2048]
        else:
            raise NotImplementedError("This scale size:[" + str(scale) +
                                      "] is not implemented!")
        # 1. conv1
        self._conv1 = ConvBNLayer(
            in_channels=3,
            out_channels=stage_out_channels[1],
            kernel_size=3,
            stride=2,
            padding=1,
            act=act)
        self._max_pool = MaxPool2D(kernel_size=3, stride=2, padding=1)

        # 2. bottleneck sequences
        self._block_list = []
        for stage_id, num_repeat in enumerate(stage_repeats):
            for i in range(num_repeat):
                if i == 0:
                    block = self.add_sublayer(
                        sublayer=InvertedResidualDS(
                            in_channels=stage_out_channels[stage_id + 1],
                            out_channels=stage_out_channels[stage_id + 2],
                            stride=2,
                            act=act),
                        name=str(stage_id + 2) + "_" + str(i + 1))
                else:
                    block = self.add_sublayer(
                        sublayer=InvertedResidual(
                            in_channels=stage_out_channels[stage_id + 2],
                            out_channels=stage_out_channels[stage_id + 2],
                            stride=1,
                            act=act),
                        name=str(stage_id + 2) + "_" + str(i + 1))
                self._block_list.append(block)
        # 3. last_conv
        self._last_conv = ConvBNLayer(
            in_channels=stage_out_channels[-2],
            out_channels=stage_out_channels[-1],
            kernel_size=1,
            stride=1,
            padding=0,
            act=act)
        # 4. pool
        if with_pool:
            self._pool2d_avg = AdaptiveAvgPool2D(1)

        # 5. fc
        if num_classes > 0:
            self._out_c = stage_out_channels[-1]
            self._fc = Linear(stage_out_channels[-1], num_classes)

    def forward(self, inputs):
        x = self._conv1(inputs)
        x = self._max_pool(x)
        for inv in self._block_list:
            x = inv(x)
        x = self._last_conv(x)

        if self.with_pool:
            x = self._pool2d_avg(x)

        if self.num_classes > 0:
            x = paddle.flatten(x, start_axis=1, stop_axis=-1)
            x = self._fc(x)
        return x


def _shufflenet_v2(arch, pretrained=False, **kwargs):
    model = ShuffleNetV2(**kwargs)
    if pretrained:
        assert (
            arch in model_urls
        ), "{} model do not have a pretrained model now, you should set pretrained=False".format(
            arch)
        weight_path = get_weights_path_from_url(model_urls[arch][0],
                                                model_urls[arch][1])

        param = paddle.load(weight_path)
        model.set_dict(param)
    return model


def shufflenet_v2_x0_25(pretrained=False, **kwargs):
    """ShuffleNetV2 with 0.25x output channels, as described in
    `"ShuffleNet V2: Practical Guidelines for Ecient CNN Architecture Design" <https://arxiv.org/pdf/1807.11164.pdf>`_ 。

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet. Default: False.

    Examples:
        .. code-block:: python

            import paddle
            from paddle.vision.models import shufflenet_v2_x0_25

            # build model
            model = shufflenet_v2_x0_25()

            # build model and load imagenet pretrained weight
            # model = shufflenet_v2_x0_25(pretrained=True)

            x = paddle.rand([1, 3, 224, 224])
            out = model(x)

            print(out.shape)

    """
    return _shufflenet_v2(
        "shufflenet_v2_x0_25", scale=0.25, pretrained=pretrained, **kwargs)


def shufflenet_v2_x0_33(pretrained=False, **kwargs):
    """ShuffleNetV2 with 0.33x output channels, as described in
    `"ShuffleNet V2: Practical Guidelines for Ecient CNN Architecture Design" <https://arxiv.org/pdf/1807.11164.pdf>`_ 。

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet. Default: False.

    Examples:
        .. code-block:: python

            import paddle
            from paddle.vision.models import shufflenet_v2_x0_33

            # build model
            model = shufflenet_v2_x0_33()

            # build model and load imagenet pretrained weight
            # model = shufflenet_v2_x0_33(pretrained=True)

            x = paddle.rand([1, 3, 224, 224])
            out = model(x)

            print(out.shape)

    """
    return _shufflenet_v2(
        "shufflenet_v2_x0_33", scale=0.33, pretrained=pretrained, **kwargs)


def shufflenet_v2_x0_5(pretrained=False, **kwargs):
    """ShuffleNetV2 with 0.5x output channels, as described in
    `"ShuffleNet V2: Practical Guidelines for Ecient CNN Architecture Design" <https://arxiv.org/pdf/1807.11164.pdf>`_ 。

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet. Default: False.

    Examples:
        .. code-block:: python

            import paddle
            from paddle.vision.models import shufflenet_v2_x0_5

            # build model
            model = shufflenet_v2_x0_5()

            # build model and load imagenet pretrained weight
            # model = shufflenet_v2_x0_5(pretrained=True)

            x = paddle.rand([1, 3, 224, 224])
            out = model(x)

            print(out.shape)

    """
    return _shufflenet_v2(
        "shufflenet_v2_x0_5", scale=0.5, pretrained=pretrained, **kwargs)


def shufflenet_v2_x1_0(pretrained=False, **kwargs):
    """ShuffleNetV2 with 1.0x output channels, as described in
    `"ShuffleNet V2: Practical Guidelines for Ecient CNN Architecture Design" <https://arxiv.org/pdf/1807.11164.pdf>`_ 。

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet. Default: False.

    Examples:
        .. code-block:: python

            import paddle
            from paddle.vision.models import shufflenet_v2_x1_0

            # build model
            model = shufflenet_v2_x1_0()

            # build model and load imagenet pretrained weight
            # model = shufflenet_v2_x1_0(pretrained=True)

            x = paddle.rand([1, 3, 224, 224])
            out = model(x)

            print(out.shape)

    """
    return _shufflenet_v2(
        "shufflenet_v2_x1_0", scale=1.0, pretrained=pretrained, **kwargs)


def shufflenet_v2_x1_5(pretrained=False, **kwargs):
    """ShuffleNetV2 with 1.5x output channels, as described in
    `"ShuffleNet V2: Practical Guidelines for Ecient CNN Architecture Design" <https://arxiv.org/pdf/1807.11164.pdf>`_ 。

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet. Default: False.

    Examples:
        .. code-block:: python

            import paddle
            from paddle.vision.models import shufflenet_v2_x1_5

            # build model
            model = shufflenet_v2_x1_5()

            # build model and load imagenet pretrained weight
            # model = shufflenet_v2_x1_5(pretrained=True)

            x = paddle.rand([1, 3, 224, 224])
            out = model(x)

            print(out.shape)

    """
    return _shufflenet_v2(
        "shufflenet_v2_x1_5", scale=1.5, pretrained=pretrained, **kwargs)


def shufflenet_v2_x2_0(pretrained=False, **kwargs):
    """ShuffleNetV2 with 2.0x output channels, as described in
    `"ShuffleNet V2: Practical Guidelines for Ecient CNN Architecture Design" <https://arxiv.org/pdf/1807.11164.pdf>`_ 。

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet. Default: False.

    Examples:
        .. code-block:: python

            import paddle
            from paddle.vision.models import shufflenet_v2_x2_0

            # build model
            model = shufflenet_v2_x2_0()

            # build model and load imagenet pretrained weight
            # model = shufflenet_v2_x2_0(pretrained=True)

            x = paddle.rand([1, 3, 224, 224])
            out = model(x)

            print(out.shape)

    """
    return _shufflenet_v2(
        "shufflenet_v2_x2_0", scale=2.0, pretrained=pretrained, **kwargs)


def shufflenet_v2_swish(pretrained=False, **kwargs):
    """ShuffleNetV2 with 1.0x output channels and swish activation function, as described in
    `"ShuffleNet V2: Practical Guidelines for Ecient CNN Architecture Design" <https://arxiv.org/pdf/1807.11164.pdf>`_ 。

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet. Default: False.

    Examples:
        .. code-block:: python

            import paddle
            from paddle.vision.models import shufflenet_v2_swish

            # build model
            model = shufflenet_v2_swish()

            # build model and load imagenet pretrained weight
            # model = shufflenet_v2_swish(pretrained=True)

            x = paddle.rand([1, 3, 224, 224])
            out = model(x)

            print(out.shape)

    """
    return _shufflenet_v2(
        "shufflenet_v2_swish",
        scale=1.0,
        act="swish",
        pretrained=pretrained,
        **kwargs)
