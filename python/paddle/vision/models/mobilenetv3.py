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

from __future__ import annotations

from functools import partial
from typing import (
    TYPE_CHECKING,
    Callable,
    TypedDict,
)

from typing_extensions import NotRequired, Unpack

import paddle
from paddle import nn
from paddle.utils.download import get_weights_path_from_url

from ..ops import ConvNormActivation
from ._utils import _make_divisible

if TYPE_CHECKING:
    from paddle import Tensor

    class _MobileNetV3Options(TypedDict):
        num_classes: NotRequired[int]
        with_pool: NotRequired[bool]


__all__ = []

model_urls = {
    "mobilenet_v3_small_x1.0": (
        "https://paddle-hapi.bj.bcebos.com/models/mobilenet_v3_small_x1.0.pdparams",
        "34fe0e7c1f8b00b2b056ad6788d0590c",
    ),
    "mobilenet_v3_large_x1.0": (
        "https://paddle-hapi.bj.bcebos.com/models/mobilenet_v3_large_x1.0.pdparams",
        "118db5792b4e183b925d8e8e334db3df",
    ),
}


class SqueezeExcitation(nn.Layer):
    """
    This block implements the Squeeze-and-Excitation block from https://arxiv.org/abs/1709.01507 (see Fig. 1).
    Parameters ``activation``, and ``scale_activation`` correspond to ``delta`` and ``sigma`` in eq. 3.
    This code is based on the torchvision code with modifications.
    You can also see at https://github.com/pytorch/vision/blob/main/torchvision/ops/misc.py#L127

    Args:
        input_channels (int): Number of channels in the input image.
        squeeze_channels (int): Number of squeeze channels.
        activation (Callable[..., paddle.nn.Layer], optional): ``delta`` activation. Default: ``paddle.nn.ReLU``.
        scale_activation (Callable[..., paddle.nn.Layer]): ``sigma`` activation. Default: ``paddle.nn.Sigmoid``.
    """

    def __init__(
        self,
        input_channels: int,
        squeeze_channels: int,
        activation: Callable[..., nn.Layer] = nn.ReLU,
        scale_activation: Callable[..., nn.Layer] = nn.Sigmoid,
    ) -> None:
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2D(1)
        self.fc1 = nn.Conv2D(input_channels, squeeze_channels, 1)
        self.fc2 = nn.Conv2D(squeeze_channels, input_channels, 1)
        self.activation = activation()
        self.scale_activation = scale_activation()

    def _scale(self, input: Tensor) -> Tensor:
        scale = self.avgpool(input)
        scale = self.fc1(scale)
        scale = self.activation(scale)
        scale = self.fc2(scale)
        return self.scale_activation(scale)

    def forward(self, input: Tensor) -> Tensor:
        scale = self._scale(input)
        return scale * input


class InvertedResidualConfig:
    def __init__(
        self,
        in_channels: int,
        kernel: int,
        expanded_channels: int,
        out_channels: int,
        use_se: bool,
        activation: str,
        stride: int,
        scale: float = 1.0,
    ):
        self.in_channels = self.adjust_channels(in_channels, scale=scale)
        self.kernel = kernel
        self.expanded_channels = self.adjust_channels(
            expanded_channels, scale=scale
        )
        self.out_channels = self.adjust_channels(out_channels, scale=scale)
        self.use_se = use_se
        if activation is None:
            self.activation_layer = None
        elif activation == "relu":
            self.activation_layer = nn.ReLU
        elif activation == "hardswish":
            self.activation_layer = nn.Hardswish
        else:
            raise RuntimeError(
                f"The activation function is not supported: {activation}"
            )
        self.stride = stride

    @staticmethod
    def adjust_channels(channels, scale=1.0):
        return _make_divisible(channels * scale, 8)


class InvertedResidual(nn.Layer):
    def __init__(
        self,
        in_channels: int,
        expanded_channels: int,
        out_channels: int,
        filter_size: int,
        stride: int,
        use_se: bool,
        activation_layer: Callable[..., nn.Layer],
        norm_layer: Callable[..., nn.Layer],
    ) -> None:
        super().__init__()
        self.use_res_connect = stride == 1 and in_channels == out_channels
        self.use_se = use_se
        self.expand = in_channels != expanded_channels

        if self.expand:
            self.expand_conv = ConvNormActivation(
                in_channels=in_channels,
                out_channels=expanded_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                norm_layer=norm_layer,
                activation_layer=activation_layer,
            )

        self.bottleneck_conv = ConvNormActivation(
            in_channels=expanded_channels,
            out_channels=expanded_channels,
            kernel_size=filter_size,
            stride=stride,
            padding=int((filter_size - 1) // 2),
            groups=expanded_channels,
            norm_layer=norm_layer,
            activation_layer=activation_layer,
        )

        if self.use_se:
            self.mid_se = SqueezeExcitation(
                expanded_channels,
                _make_divisible(expanded_channels // 4),
                scale_activation=nn.Hardsigmoid,
            )

        self.linear_conv = ConvNormActivation(
            in_channels=expanded_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            norm_layer=norm_layer,
            activation_layer=None,
        )

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        if self.expand:
            x = self.expand_conv(x)
        x = self.bottleneck_conv(x)
        if self.use_se:
            x = self.mid_se(x)
        x = self.linear_conv(x)
        if self.use_res_connect:
            x = paddle.add(identity, x)
        return x


class MobileNetV3(nn.Layer):
    """MobileNetV3 model from
    `"Searching for MobileNetV3" <https://arxiv.org/abs/1905.02244>`_.

    Args:
        config (list[InvertedResidualConfig]): MobileNetV3 depthwise blocks config.
        last_channel (int): The number of channels on the penultimate layer.
        scale (float, optional): Scale of channels in each layer. Default: 1.0.
        num_classes (int, optional): Output dim of last fc layer. If num_classes <=0, last fc layer
            will not be defined. Default: 1000.
        with_pool (bool, optional): Use pool before the last fc layer or not. Default: True.
    """

    scale: float
    num_classes: int
    with_pool: bool

    def __init__(
        self,
        config: list[InvertedResidualConfig],
        last_channel: int,
        scale: float = 1.0,
        num_classes: int = 1000,
        with_pool: bool = True,
    ) -> None:
        super().__init__()

        self.config = config
        self.scale = scale
        self.last_channel = last_channel
        self.num_classes = num_classes
        self.with_pool = with_pool
        self.firstconv_in_channels = config[0].in_channels
        self.lastconv_in_channels = config[-1].in_channels
        self.lastconv_out_channels = self.lastconv_in_channels * 6
        norm_layer = partial(nn.BatchNorm2D, epsilon=0.001, momentum=0.99)

        self.conv = ConvNormActivation(
            in_channels=3,
            out_channels=self.firstconv_in_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            groups=1,
            activation_layer=nn.Hardswish,
            norm_layer=norm_layer,
        )

        self.blocks = nn.Sequential(
            *[
                InvertedResidual(
                    in_channels=cfg.in_channels,
                    expanded_channels=cfg.expanded_channels,
                    out_channels=cfg.out_channels,
                    filter_size=cfg.kernel,
                    stride=cfg.stride,
                    use_se=cfg.use_se,
                    activation_layer=cfg.activation_layer,
                    norm_layer=norm_layer,
                )
                for cfg in self.config
            ]
        )

        self.lastconv = ConvNormActivation(
            in_channels=self.lastconv_in_channels,
            out_channels=self.lastconv_out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1,
            norm_layer=norm_layer,
            activation_layer=nn.Hardswish,
        )

        if with_pool:
            self.avgpool = nn.AdaptiveAvgPool2D(1)

        if num_classes > 0:
            self.classifier = nn.Sequential(
                nn.Linear(self.lastconv_out_channels, self.last_channel),
                nn.Hardswish(),
                nn.Dropout(p=0.2),
                nn.Linear(self.last_channel, num_classes),
            )

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = self.blocks(x)
        x = self.lastconv(x)

        if self.with_pool:
            x = self.avgpool(x)

        if self.num_classes > 0:
            x = paddle.flatten(x, 1)
            x = self.classifier(x)

        return x


class MobileNetV3Small(MobileNetV3):
    """MobileNetV3 Small architecture model from
    `"Searching for MobileNetV3" <https://arxiv.org/abs/1905.02244>`_.

    Args:
        scale (float, optional): Scale of channels in each layer. Default: 1.0.
        num_classes (int, optional): Output dim of last fc layer. If num_classes <= 0, last fc layer
            will not be defined. Default: 1000.
        with_pool (bool, optional): Use pool before the last fc layer or not. Default: True.

    Returns:
        :ref:`api_paddle_nn_Layer`. An instance of MobileNetV3 Small architecture model.

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> from paddle.vision.models import MobileNetV3Small

            >>> # Build model
            >>> model = MobileNetV3Small(scale=1.0)

            >>> x = paddle.rand([1, 3, 224, 224])
            >>> out = model(x)

            >>> print(out.shape)
            [1, 1000]
    """

    def __init__(
        self,
        scale: float = 1.0,
        num_classes: int = 1000,
        with_pool: bool = True,
    ) -> None:
        config = [
            InvertedResidualConfig(16, 3, 16, 16, True, "relu", 2, scale),
            InvertedResidualConfig(16, 3, 72, 24, False, "relu", 2, scale),
            InvertedResidualConfig(24, 3, 88, 24, False, "relu", 1, scale),
            InvertedResidualConfig(24, 5, 96, 40, True, "hardswish", 2, scale),
            InvertedResidualConfig(40, 5, 240, 40, True, "hardswish", 1, scale),
            InvertedResidualConfig(40, 5, 240, 40, True, "hardswish", 1, scale),
            InvertedResidualConfig(40, 5, 120, 48, True, "hardswish", 1, scale),
            InvertedResidualConfig(48, 5, 144, 48, True, "hardswish", 1, scale),
            InvertedResidualConfig(48, 5, 288, 96, True, "hardswish", 2, scale),
            InvertedResidualConfig(96, 5, 576, 96, True, "hardswish", 1, scale),
            InvertedResidualConfig(96, 5, 576, 96, True, "hardswish", 1, scale),
        ]
        last_channel = _make_divisible(1024 * scale, 8)
        super().__init__(
            config,
            last_channel=last_channel,
            scale=scale,
            with_pool=with_pool,
            num_classes=num_classes,
        )


class MobileNetV3Large(MobileNetV3):
    """MobileNetV3 Large architecture model from
    `"Searching for MobileNetV3" <https://arxiv.org/abs/1905.02244>`_.

    Args:
        scale (float, optional): Scale of channels in each layer. Default: 1.0.
        num_classes (int, optional): Output dim of last fc layer. If num_classes <= 0, last fc layer
            will not be defined. Default: 1000.
        with_pool (bool, optional): Use pool before the last fc layer or not. Default: True.

    Returns:
        :ref:`api_paddle_nn_Layer`. An instance of MobileNetV3 Large architecture model.

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> from paddle.vision.models import MobileNetV3Large

            >>> # Build model
            >>> model = MobileNetV3Large(scale=1.0)

            >>> x = paddle.rand([1, 3, 224, 224])
            >>> out = model(x)

            >>> print(out.shape)
            [1, 1000]
    """

    def __init__(
        self,
        scale: float = 1.0,
        num_classes: int = 1000,
        with_pool: bool = True,
    ) -> None:
        config = [
            InvertedResidualConfig(16, 3, 16, 16, False, "relu", 1, scale),
            InvertedResidualConfig(16, 3, 64, 24, False, "relu", 2, scale),
            InvertedResidualConfig(24, 3, 72, 24, False, "relu", 1, scale),
            InvertedResidualConfig(24, 5, 72, 40, True, "relu", 2, scale),
            InvertedResidualConfig(40, 5, 120, 40, True, "relu", 1, scale),
            InvertedResidualConfig(40, 5, 120, 40, True, "relu", 1, scale),
            InvertedResidualConfig(
                40, 3, 240, 80, False, "hardswish", 2, scale
            ),
            InvertedResidualConfig(
                80, 3, 200, 80, False, "hardswish", 1, scale
            ),
            InvertedResidualConfig(
                80, 3, 184, 80, False, "hardswish", 1, scale
            ),
            InvertedResidualConfig(
                80, 3, 184, 80, False, "hardswish", 1, scale
            ),
            InvertedResidualConfig(
                80, 3, 480, 112, True, "hardswish", 1, scale
            ),
            InvertedResidualConfig(
                112, 3, 672, 112, True, "hardswish", 1, scale
            ),
            InvertedResidualConfig(
                112, 5, 672, 160, True, "hardswish", 2, scale
            ),
            InvertedResidualConfig(
                160, 5, 960, 160, True, "hardswish", 1, scale
            ),
            InvertedResidualConfig(
                160, 5, 960, 160, True, "hardswish", 1, scale
            ),
        ]
        last_channel = _make_divisible(1280 * scale, 8)
        super().__init__(
            config,
            last_channel=last_channel,
            scale=scale,
            with_pool=with_pool,
            num_classes=num_classes,
        )


def _mobilenet_v3(
    arch: str,
    pretrained: bool = False,
    scale: float = 1.0,
    **kwargs: Unpack[_MobileNetV3Options],
) -> MobileNetV3:
    if arch == "mobilenet_v3_large":
        model = MobileNetV3Large(scale=scale, **kwargs)
    else:
        model = MobileNetV3Small(scale=scale, **kwargs)
    if pretrained:
        arch = f"{arch}_x{scale}"
        assert (
            arch in model_urls
        ), f"{arch} model do not have a pretrained model now, you should set pretrained=False"
        weight_path = get_weights_path_from_url(
            model_urls[arch][0], model_urls[arch][1]
        )

        param = paddle.load(weight_path)
        model.set_dict(param)
    return model


def mobilenet_v3_small(
    pretrained: bool = False,
    scale: float = 1.0,
    **kwargs: Unpack[_MobileNetV3Options],
) -> MobileNetV3Small:
    """MobileNetV3 Small architecture model from
    `"Searching for MobileNetV3" <https://arxiv.org/abs/1905.02244>`_.

    Args:
        pretrained (bool, optional): Whether to load pre-trained weights. If True, returns a model pre-trained on ImageNet. Default: False.
        scale (float, optional): Scale of channels in each layer. Default: 1.0.
        **kwargs (optional): Additional keyword arguments. For details, please refer to :ref:`MobileNetV3Small <api_paddle_vision_models_MobileNetV3Small>`.

    Returns:
        :ref:`api_paddle_nn_Layer`. An instance of MobileNetV3 Small architecture model.

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> from paddle.vision.models import mobilenet_v3_small

            >>> # Build model
            >>> model = mobilenet_v3_small()

            >>> # Build model and load imagenet pretrained weight
            >>> # model = mobilenet_v3_small(pretrained=True)

            >>> # Build mobilenet v3 small model with scale=0.5
            >>> model = mobilenet_v3_small(scale=0.5)

            >>> x = paddle.rand([1, 3, 224, 224])
            >>> out = model(x)

            >>> print(out.shape)
            [1, 1000]
    """
    model = _mobilenet_v3(
        "mobilenet_v3_small", scale=scale, pretrained=pretrained, **kwargs
    )
    return model


def mobilenet_v3_large(
    pretrained: bool = False,
    scale: float = 1.0,
    **kwargs: Unpack[_MobileNetV3Options],
) -> MobileNetV3Large:
    """MobileNetV3 Large architecture model from
    `"Searching for MobileNetV3" <https://arxiv.org/abs/1905.02244>`_.

    Args:
        pretrained (bool, optional): Whether to load pre-trained weights. If True, returns a model pre-trained on ImageNet. Default: False.
        scale (float, optional): Scale of channels in each layer. Default: 1.0.
        **kwargs (optional): Additional keyword arguments. For details, please refer to :ref:`MobileNetV3Large <api_paddle_vision_models_MobileNetV3Large>`.

    Returns:
        :ref:`api_paddle_nn_Layer`. An instance of MobileNetV3 Large architecture model.

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> from paddle.vision.models import mobilenet_v3_large

            >>> # Build model
            >>> model = mobilenet_v3_large()

            >>> # Build model and load imagenet pretrained weight
            >>> # model = mobilenet_v3_large(pretrained=True)

            >>> # Build mobilenet v3 large model with scale=0.5
            >>> model = mobilenet_v3_large(scale=0.5)

            >>> x = paddle.rand([1, 3, 224, 224])
            >>> out = model(x)

            >>> print(out.shape)
            [1, 1000]
    """
    model = _mobilenet_v3(
        "mobilenet_v3_large", scale=scale, pretrained=pretrained, **kwargs
    )
    return model
