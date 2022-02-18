# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
from paddle.nn import functional as F
from paddle.regularizer import L2Decay
from paddle.utils.download import get_weights_path_from_url

from ._utils import _make_divisible

__all__ = []

model_urls = {
    "mobilenet_v3_small_x0.35":
    ("https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/MobileNetV3_small_x0_35_pretrained.pdparams",
     "938f11e720fdd04fe46a68322155f7e3"),
    "mobilenet_v3_small_x0.5":
    ("https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/MobileNetV3_small_x0_5_pretrained.pdparams",
     "4ccb3f48b940edacdf37e227c2b77ac2"),
    "mobilenet_v3_small_x0.75":
    ("https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/MobileNetV3_small_x0_75_pretrained.pdparams",
     "ea999201351e094d3e71c3c00584098d"),
    "mobilenet_v3_small_x1.0":
    ("https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/MobileNetV3_small_x1_0_pretrained.pdparams",
     "96d99be8a67d6431ba7de6149f15b23d"),
    "mobilenet_v3_small_x1.25":
    ("https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/MobileNetV3_small_x1_25_pretrained.pdparams",
     "69b75ae25115475af20d0f1af67412c0"),
    "mobilenet_v3_large_x0.35":
    ("https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/MobileNetV3_large_x0_35_pretrained.pdparams",
     "64864e5bff400eca62566c32270f7afc"),
    "mobilenet_v3_large_x0.5":
    ("https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/MobileNetV3_large_x0_5_pretrained.pdparams",
     "d6da6f4fb5d27901be64c7acc26288d0"),
    "mobilenet_v3_large_x0.75":
    ("https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/MobileNetV3_large_x0_75_pretrained.pdparams",
     "5af6995c37d468d8e3ddee69cc238ed7"),
    "mobilenet_v3_large_x1.0":
    ("https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/MobileNetV3_large_x1_0_pretrained.pdparams",
     "8f8b4359af8093191628e410438ca858"),
    "mobilenet_v3_large_x1.25":
    ("https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/MobileNetV3_large_x1_25_pretrained.pdparams",
     "0a092f2906ef0ae8267e4756449cc90a"),
}


def _create_act(act):
    if act == "hardswish":
        return nn.Hardswish()
    elif act == "relu":
        return nn.ReLU()
    elif act is None:
        return None
    else:
        raise RuntimeError(
            "The activation function is not supported: {}".format(act))


class Hardsigmoid(nn.Layer):
    """paddle.nn.Hardsigmoid can't transfer "slope" and "offset" in paddle.nn.functional.hardsigmoid

    Args:
        slope (float, optional): The slope of hardsigmoid function. Default is 0.1666667.
        offset (float, optional): The offset of hardsigmoid function. Default is 0.5.
    """

    def __init__(self, slope=0.1666667, offset=0.5):
        super().__init__()
        self.slope = slope
        self.offset = offset

    def forward(self, x):
        return F.hardsigmoid(x, slope=self.slope, offset=self.offset)


class SqueezeExcitation(nn.Layer):
    def __init__(self, channels, reduction=4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2D(1)
        self.conv1 = nn.Conv2D(
            in_channels=channels,
            out_channels=channels // reduction,
            kernel_size=1,
            stride=1,
            padding=0)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2D(
            in_channels=channels // reduction,
            out_channels=channels,
            kernel_size=1,
            stride=1,
            padding=0)
        self.hardsigmoid = Hardsigmoid(slope=0.2, offset=0.5)

    def forward(self, x):
        identity = x
        x = self.avg_pool(x)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.hardsigmoid(x)
        return paddle.multiply(x=identity, y=x)


class ConvBNLayer(nn.Layer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 filter_size,
                 stride,
                 padding,
                 num_groups=1,
                 act=None):
        super().__init__()

        self.conv = nn.Conv2D(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=filter_size,
            stride=stride,
            padding=padding,
            groups=num_groups,
            bias_attr=False)
        self.bn = nn.BatchNorm(
            num_channels=out_channels,
            act=None,
            param_attr=ParamAttr(regularizer=L2Decay(0.0)),
            bias_attr=ParamAttr(regularizer=L2Decay(0.0)))
        self.act = _create_act(act)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.act is not None:
            x = self.act(x)
        return x


class InvertedResidual(nn.Layer):
    def __init__(self,
                 in_channels,
                 expanded_channels,
                 out_channels,
                 filter_size,
                 stride,
                 use_se,
                 act=None):
        super().__init__()
        self.if_shortcut = stride == 1 and in_channels == out_channels
        self.if_se = use_se

        self.expand_conv = ConvBNLayer(
            in_channels=in_channels,
            out_channels=expanded_channels,
            filter_size=1,
            stride=1,
            padding=0,
            act=act)
        self.bottleneck_conv = ConvBNLayer(
            in_channels=expanded_channels,
            out_channels=expanded_channels,
            filter_size=filter_size,
            stride=stride,
            padding=int((filter_size - 1) // 2),
            num_groups=expanded_channels,
            act=act)
        if self.if_se:
            self.mid_se = SqueezeExcitation(expanded_channels)
        self.linear_conv = ConvBNLayer(
            in_channels=expanded_channels,
            out_channels=out_channels,
            filter_size=1,
            stride=1,
            padding=0,
            act=None)

    def forward(self, x):
        identity = x
        x = self.expand_conv(x)
        x = self.bottleneck_conv(x)
        if self.if_se:
            x = self.mid_se(x)
        x = self.linear_conv(x)
        if self.if_shortcut:
            x = paddle.add(identity, x)
        return x


class InvertedResidualConfig:
    def __init__(self, in_channels, kernel, expanded_channels, out_channels,
                 use_se, activation, stride):
        self.in_channels = in_channels
        self.kernel = kernel
        self.expanded_channels = expanded_channels
        self.out_channels = out_channels
        self.use_se = use_se
        self.activation = activation
        self.stride = stride


class MobileNetV3(nn.Layer):
    """MobileNetV3 model from
    `"Searching for MobileNetV3" <https://arxiv.org/abs/1905.02244>`_.

    Args:
        config (list[InvertedResidualConfig]): MobileNetV3 depthwise blocks config.
        scale (float, optional): Scale of output channels. Default: 1.0.
        last_channel (int, optional): The number of channels on the penultimate layer. Default: 1280.
        num_classes (int, optional): Output dim of last fc layer. If num_classes <=0, last fc layer
                            will not be defined. Default: 1000.
        with_pool (bool, optional): Use pool before the last fc layer or not. Default: True.
    """

    def __init__(self,
                 config,
                 scale=1.0,
                 last_channel=1280,
                 num_classes=1000,
                 with_pool=True):
        super().__init__()

        self.config = config
        self.scale = scale
        self.last_channel = last_channel
        self.num_classes = num_classes
        self.with_pool = with_pool
        self.first_in_channels = config[0].in_channels
        self.last_expanded_channels = config[-1].expanded_channels
        self.last_out_channels = config[-1].out_channels

        self.conv = ConvBNLayer(
            in_channels=3,
            out_channels=_make_divisible(self.first_in_channels * self.scale),
            filter_size=3,
            stride=2,
            padding=1,
            num_groups=1,
            act="hardswish")

        self.blocks = nn.Sequential(*[
            InvertedResidual(
                in_channels=_make_divisible(self.scale * cfg.in_channels),
                expanded_channels=_make_divisible(self.scale *
                                                  cfg.expanded_channels),
                out_channels=_make_divisible(self.scale * cfg.out_channels),
                filter_size=cfg.kernel,
                stride=cfg.stride,
                use_se=cfg.use_se,
                act=cfg.activation) for cfg in self.config
        ])

        self.last_second_conv = ConvBNLayer(
            in_channels=_make_divisible(self.scale * self.last_out_channels),
            out_channels=_make_divisible(self.scale *
                                         self.last_expanded_channels),
            filter_size=1,
            stride=1,
            padding=0,
            num_groups=1,
            act="hardswish")

        if with_pool:
            self.avg_pool = nn.AdaptiveAvgPool2D(1)

        if num_classes > 0:
            self.last_conv = nn.Conv2D(
                in_channels=_make_divisible(self.scale *
                                            self.last_expanded_channels),
                out_channels=self.last_channel,
                kernel_size=1,
                stride=1,
                padding=0,
                bias_attr=False)

            self.hardswish = nn.Hardswish()
            self.dropout = nn.Dropout(p=0.2, mode="downscale_in_infer")
            self.flatten = nn.Flatten(start_axis=1, stop_axis=-1)

            self.fc = nn.Linear(self.last_channel, num_classes)

    def forward(self, x):
        x = self.conv(x)
        x = self.blocks(x)
        x = self.last_second_conv(x)

        if self.with_pool:
            x = self.avg_pool(x)

        if self.num_classes > 0:
            x = self.last_conv(x)
            x = self.hardswish(x)
            x = self.dropout(x)
            x = self.flatten(x)
            x = self.fc(x)

        return x


class MobileNetV3Small(MobileNetV3):
    """MobileNetV3 Small architecture model from
    `"Searching for MobileNetV3" <https://arxiv.org/abs/1905.02244>`_.

    Args:
        scale (float, optional): Scale of channels in each layer. Default: 1.0.
        last_channel (int, optional): The number of channels on the penultimate layer. Default: 1280.
        num_classes (int, optional): Output dim of last fc layer. If num_classes <=0, last fc layer
                            will not be defined. Default: 1000.
        with_pool (bool, optional): Use pool before the last fc layer or not. Default: True.

    Examples:
        .. code-block:: python

            import paddle
            from paddle.vision.models import MobileNetV3Small

            # build model
            model = MobileNetV3Small(scale=1.0)

            x = paddle.rand([1, 3, 224, 224])
            out = model(x)

            print(out.shape)
    """

    def __init__(self,
                 scale=1.0,
                 last_channel=1280,
                 num_classes=1000,
                 with_pool=True):
        config = [
            InvertedResidualConfig(16, 3, 16, 16, True, "relu", 2),
            InvertedResidualConfig(16, 3, 72, 24, False, "relu", 2),
            InvertedResidualConfig(24, 3, 88, 24, False, "relu", 1),
            InvertedResidualConfig(24, 5, 96, 40, True, "hardswish", 2),
            InvertedResidualConfig(40, 5, 240, 40, True, "hardswish", 1),
            InvertedResidualConfig(40, 5, 240, 40, True, "hardswish", 1),
            InvertedResidualConfig(40, 5, 120, 48, True, "hardswish", 1),
            InvertedResidualConfig(48, 5, 144, 48, True, "hardswish", 1),
            InvertedResidualConfig(48, 5, 288, 96, True, "hardswish", 2),
            InvertedResidualConfig(96, 5, 576, 96, True, "hardswish", 1),
            InvertedResidualConfig(96, 5, 576, 96, True, "hardswish", 1),
        ]
        super().__init__(
            config,
            scale=scale,
            last_channel=last_channel,
            with_pool=with_pool,
            num_classes=num_classes)


class MobileNetV3Large(MobileNetV3):
    """MobileNetV3 Large architecture model from
    `"Searching for MobileNetV3" <https://arxiv.org/abs/1905.02244>`_.

    Args:
        scale (float, optional): Scale of channels in each layer. Default: 1.0.
        last_channel (int, optional): The number of channels on the penultimate layer. Default: 1280.
        num_classes (int, optional): Output dim of last fc layer. If num_classes <=0, last fc layer
                            will not be defined. Default: 1000.
        with_pool (bool, optional): Use pool before the last fc layer or not. Default: True.

    Examples:
        .. code-block:: python

            import paddle
            from paddle.vision.models import MobileNetV3Large

            # build model
            model = MobileNetV3Large(scale=1.0)

            x = paddle.rand([1, 3, 224, 224])
            out = model(x)

            print(out.shape)
    """

    def __init__(self,
                 scale=1.0,
                 last_channel=1280,
                 num_classes=1000,
                 with_pool=True):
        config = [
            InvertedResidualConfig(16, 3, 16, 16, False, "relu", 1),
            InvertedResidualConfig(16, 3, 64, 24, False, "relu", 2),
            InvertedResidualConfig(24, 3, 72, 24, False, "relu", 1),
            InvertedResidualConfig(24, 5, 72, 40, True, "relu", 2),
            InvertedResidualConfig(40, 5, 120, 40, True, "relu", 1),
            InvertedResidualConfig(40, 5, 120, 40, True, "relu", 1),
            InvertedResidualConfig(40, 3, 240, 80, False, "hardswish", 2),
            InvertedResidualConfig(80, 3, 200, 80, False, "hardswish", 1),
            InvertedResidualConfig(80, 3, 184, 80, False, "hardswish", 1),
            InvertedResidualConfig(80, 3, 184, 80, False, "hardswish", 1),
            InvertedResidualConfig(80, 3, 480, 112, True, "hardswish", 1),
            InvertedResidualConfig(112, 3, 672, 112, True, "hardswish", 1),
            InvertedResidualConfig(112, 5, 672, 160, True, "hardswish", 2),
            InvertedResidualConfig(160, 5, 960, 160, True, "hardswish", 1),
            InvertedResidualConfig(160, 5, 960, 160, True, "hardswish", 1),
        ]
        super().__init__(
            config,
            scale=scale,
            last_channel=last_channel,
            with_pool=with_pool,
            num_classes=num_classes)


def _mobilenet_v3(arch, pretrained=False, scale=1.0, **kwargs):
    if arch == "mobilenet_v3_large":
        model = MobileNetV3Large(scale=scale, **kwargs)
    else:
        model = MobileNetV3Small(scale=scale, **kwargs)
    if pretrained:
        arch = "{}_x{}".format(arch, scale)
        assert (
            arch in model_urls
        ), "{} model do not have a pretrained model now, you should set pretrained=False".format(
            arch)
        weight_path = get_weights_path_from_url(model_urls[arch][0],
                                                model_urls[arch][1])

        param = paddle.load(weight_path)
        model.set_dict(param)
    return model


def mobilenet_v3_small(pretrained=False, scale=1.0, **kwargs):
    """MobileNetV3 Small architecture model from
    `"Searching for MobileNetV3" <https://arxiv.org/abs/1905.02244>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet. Default: False.
        scale (float, optional): Scale of channels in each layer. Default: 1.0.

    Examples:
        .. code-block:: python

            import paddle
            from paddle.vision.models import mobilenet_v3_small

            # build model
            model = mobilenet_v3_small()

            # build model and load imagenet pretrained weight
            # model = mobilenet_v3_small(pretrained=True)

            # build mobilenet v3 small model with scale=0.5
            model = mobilenet_v3_small(scale=0.5)

            x = paddle.rand([1, 3, 224, 224])
            out = model(x)

            print(out.shape)

    """
    model = _mobilenet_v3(
        "mobilenet_v3_small", scale=scale, pretrained=pretrained, **kwargs)
    return model


def mobilenet_v3_large(pretrained=False, scale=1.0, **kwargs):
    """MobileNetV3 Large architecture model from
    `"Searching for MobileNetV3" <https://arxiv.org/abs/1905.02244>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet. Default: False.
        scale (float, optional): Scale of channels in each layer. Default: 1.0.

    Examples:
        .. code-block:: python

            import paddle
            from paddle.vision.models import mobilenet_v3_large

            # build model
            model = mobilenet_v3_large()

            # build model and load imagenet pretrained weight
            # model = mobilenet_v3_large(pretrained=True)

            # build mobilenet v3 large model with scale=0.5
            model = mobilenet_v3_large(scale=0.5)

            x = paddle.rand([1, 3, 224, 224])
            out = model(x)

            print(out.shape)

    """
    model = _mobilenet_v3(
        "mobilenet_v3_large", scale=scale, pretrained=pretrained, **kwargs)
    return model
