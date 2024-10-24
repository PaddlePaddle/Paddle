# copyright (c) 2021 PaddlePaddle Authors. All Rights Reserve.
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
from __future__ import annotations

import math
from typing import TYPE_CHECKING

from typing_extensions import Unpack

import paddle
from paddle import nn
from paddle.base.param_attr import ParamAttr
from paddle.nn import (
    AdaptiveAvgPool2D,
    AvgPool2D,
    BatchNorm,
    Conv2D,
    Dropout,
    Linear,
    MaxPool2D,
)
from paddle.nn.initializer import Uniform
from paddle.utils.download import get_weights_path_from_url

if TYPE_CHECKING:
    from typing import Literal, TypedDict

    from typing_extensions import NotRequired

    from paddle import Tensor
    from paddle._typing import Size2

    _DenseNetArch = Literal[
        "densenet121",
        "densenet161",
        "densenet169",
        "densenet201",
        "densenet264",
    ]

    class _DenseNetOptions(TypedDict):
        bn_size: NotRequired[int]
        dropout: NotRequired[float]
        num_classes: NotRequired[int]
        with_pool: NotRequired[bool]


__all__ = []

model_urls: dict[str, tuple[str, str]] = {
    'densenet121': (
        'https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/DenseNet121_pretrained.pdparams',
        'db1b239ed80a905290fd8b01d3af08e4',
    ),
    'densenet161': (
        'https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/DenseNet161_pretrained.pdparams',
        '62158869cb315098bd25ddbfd308a853',
    ),
    'densenet169': (
        'https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/DenseNet169_pretrained.pdparams',
        '82cc7c635c3f19098c748850efb2d796',
    ),
    'densenet201': (
        'https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/DenseNet201_pretrained.pdparams',
        '16ca29565a7712329cf9e36e02caaf58',
    ),
    'densenet264': (
        'https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/DenseNet264_pretrained.pdparams',
        '3270ce516b85370bba88cfdd9f60bff4',
    ),
}


class BNACConvLayer(nn.Layer):
    def __init__(
        self,
        num_channels: int,
        num_filters: int,
        filter_size: Size2,
        stride: Size2 = 1,
        pad: Size2 = 0,
        groups: int = 1,
        act: str = "relu",
    ) -> None:
        super().__init__()
        self._batch_norm = BatchNorm(num_channels, act=act)

        self._conv = Conv2D(
            in_channels=num_channels,
            out_channels=num_filters,
            kernel_size=filter_size,
            stride=stride,
            padding=pad,
            groups=groups,
            weight_attr=ParamAttr(),
            bias_attr=False,
        )

    def forward(self, input: Tensor) -> Tensor:
        y = self._batch_norm(input)
        y = self._conv(y)
        return y


class DenseLayer(nn.Layer):
    dropout: float

    def __init__(
        self, num_channels: int, growth_rate: int, bn_size: int, dropout: float
    ) -> None:
        super().__init__()
        self.dropout = dropout

        self.bn_ac_func1 = BNACConvLayer(
            num_channels=num_channels,
            num_filters=bn_size * growth_rate,
            filter_size=1,
            pad=0,
            stride=1,
        )

        self.bn_ac_func2 = BNACConvLayer(
            num_channels=bn_size * growth_rate,
            num_filters=growth_rate,
            filter_size=3,
            pad=1,
            stride=1,
        )

        if dropout:
            self.dropout_func = Dropout(p=dropout, mode="downscale_in_infer")

    def forward(self, input: Tensor) -> Tensor:
        conv = self.bn_ac_func1(input)
        conv = self.bn_ac_func2(conv)
        if self.dropout:
            conv = self.dropout_func(conv)
        conv = paddle.concat([input, conv], axis=1)
        return conv


class DenseBlock(nn.Layer):
    dropout: float

    def __init__(
        self,
        num_channels: int,
        num_layers: int,
        bn_size: int,
        growth_rate: int,
        dropout: float,
        name: str | None = None,
    ) -> None:
        super().__init__()
        self.dropout = dropout
        self.dense_layer_func = []

        pre_channel = num_channels
        for layer in range(num_layers):
            self.dense_layer_func.append(
                self.add_sublayer(
                    f"{name}_{layer + 1}",
                    DenseLayer(
                        num_channels=pre_channel,
                        growth_rate=growth_rate,
                        bn_size=bn_size,
                        dropout=dropout,
                    ),
                )
            )
            pre_channel = pre_channel + growth_rate

    def forward(self, input: Tensor) -> Tensor:
        conv = input
        for func in self.dense_layer_func:
            conv = func(conv)
        return conv


class TransitionLayer(nn.Layer):
    def __init__(self, num_channels: int, num_output_features: int) -> None:
        super().__init__()

        self.conv_ac_func = BNACConvLayer(
            num_channels=num_channels,
            num_filters=num_output_features,
            filter_size=1,
            pad=0,
            stride=1,
        )

        self.pool2d_avg = AvgPool2D(kernel_size=2, stride=2, padding=0)

    def forward(self, input: Tensor) -> Tensor:
        y = self.conv_ac_func(input)
        y = self.pool2d_avg(y)
        return y


class ConvBNLayer(nn.Layer):
    def __init__(
        self,
        num_channels: int,
        num_filters: int,
        filter_size: Size2,
        stride: Size2 = 1,
        pad: Size2 = 0,
        groups: int = 1,
        act: str = "relu",
    ) -> None:
        super().__init__()

        self._conv = Conv2D(
            in_channels=num_channels,
            out_channels=num_filters,
            kernel_size=filter_size,
            stride=stride,
            padding=pad,
            groups=groups,
            weight_attr=ParamAttr(),
            bias_attr=False,
        )
        self._batch_norm = BatchNorm(num_filters, act=act)

    def forward(self, input: Tensor) -> Tensor:
        y = self._conv(input)
        y = self._batch_norm(y)
        return y


class DenseNet(nn.Layer):
    """DenseNet model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_.

    Args:
        layers (int, optional): Layers of DenseNet. Default: 121.
        bn_size (int, optional): Expansion of growth rate in the middle layer. Default: 4.
        dropout (float, optional): Dropout rate. Default: :math:`0.0`.
        num_classes (int, optional): Output dim of last fc layer. If num_classes <= 0, last fc layer
            will not be defined. Default: 1000.
        with_pool (bool, optional): Use pool before the last fc layer or not. Default: True.

    Returns:
        :ref:`api_paddle_nn_Layer`. An instance of DenseNet model.

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> from paddle.vision.models import DenseNet

            >>> # Build model
            >>> densenet = DenseNet()

            >>> x = paddle.rand([1, 3, 224, 224])
            >>> out = densenet(x)

            >>> print(out.shape)
            [1, 1000]
    """

    num_classes: int
    with_pool: bool

    def __init__(
        self,
        layers: int = 121,
        bn_size: int = 4,
        dropout: float = 0.0,
        num_classes: int = 1000,
        with_pool: bool = True,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.with_pool = with_pool
        supported_layers = [121, 161, 169, 201, 264]
        assert (
            layers in supported_layers
        ), f"supported layers are {supported_layers} but input layer is {layers}"
        densenet_spec = {
            121: (64, 32, [6, 12, 24, 16]),
            161: (96, 48, [6, 12, 36, 24]),
            169: (64, 32, [6, 12, 32, 32]),
            201: (64, 32, [6, 12, 48, 32]),
            264: (64, 32, [6, 12, 64, 48]),
        }
        num_init_features, growth_rate, block_config = densenet_spec[layers]

        self.conv1_func = ConvBNLayer(
            num_channels=3,
            num_filters=num_init_features,
            filter_size=7,
            stride=2,
            pad=3,
            act='relu',
        )
        self.pool2d_max = MaxPool2D(kernel_size=3, stride=2, padding=1)
        self.block_config = block_config
        self.dense_block_func_list = []
        self.transition_func_list = []
        pre_num_channels = num_init_features
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            self.dense_block_func_list.append(
                self.add_sublayer(
                    f"db_conv_{i + 2}",
                    DenseBlock(
                        num_channels=pre_num_channels,
                        num_layers=num_layers,
                        bn_size=bn_size,
                        growth_rate=growth_rate,
                        dropout=dropout,
                        name='conv' + str(i + 2),
                    ),
                )
            )

            num_features = num_features + num_layers * growth_rate
            pre_num_channels = num_features

            if i != len(block_config) - 1:
                self.transition_func_list.append(
                    self.add_sublayer(
                        f"tr_conv{i + 2}_blk",
                        TransitionLayer(
                            num_channels=pre_num_channels,
                            num_output_features=num_features // 2,
                        ),
                    )
                )
                pre_num_channels = num_features // 2
                num_features = num_features // 2

        self.batch_norm = BatchNorm(num_features, act="relu")
        if self.with_pool:
            self.pool2d_avg = AdaptiveAvgPool2D(1)

        if self.num_classes > 0:
            stdv = 1.0 / math.sqrt(num_features * 1.0)
            self.out = Linear(
                num_features,
                num_classes,
                weight_attr=ParamAttr(initializer=Uniform(-stdv, stdv)),
                bias_attr=ParamAttr(),
            )

    def forward(self, input: Tensor) -> Tensor:
        conv = self.conv1_func(input)
        conv = self.pool2d_max(conv)

        for i, num_layers in enumerate(self.block_config):
            conv = self.dense_block_func_list[i](conv)
            if i != len(self.block_config) - 1:
                conv = self.transition_func_list[i](conv)

        conv = self.batch_norm(conv)

        if self.with_pool:
            y = self.pool2d_avg(conv)

        if self.num_classes > 0:
            y = paddle.flatten(y, start_axis=1, stop_axis=-1)
            y = self.out(y)

        return y


def _densenet(
    arch: _DenseNetArch,
    layers: int,
    pretrained: bool,
    **kwargs: Unpack[_DenseNetOptions],
) -> DenseNet:
    model = DenseNet(layers=layers, **kwargs)
    if pretrained:
        assert (
            arch in model_urls
        ), f"{arch} model do not have a pretrained model now, you should set pretrained=False"
        weight_path = get_weights_path_from_url(
            model_urls[arch][0], model_urls[arch][1]
        )

        param = paddle.load(weight_path)
        model.set_dict(param)

    return model


def densenet121(
    pretrained: bool = False, **kwargs: Unpack[_DenseNetOptions]
) -> DenseNet:
    """DenseNet 121-layer model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_.

    Args:
        pretrained (bool, optional): Whether to load pre-trained weights. If True, returns a model pre-trained
            on ImageNet. Default: False.
        **kwargs (optional): Additional keyword arguments. For details, please refer to :ref:`DenseNet <api_paddle_vision_models_DenseNet>`.

    Returns:
        :ref:`api_paddle_nn_Layer`. An instance of DenseNet 121-layer model.

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> from paddle.vision.models import densenet121

            >>> # Build model
            >>> model = densenet121()

            >>> # Build model and load imagenet pretrained weight
            >>> # model = densenet121(pretrained=True)

            >>> x = paddle.rand([1, 3, 224, 224])
            >>> out = model(x)

            >>> print(out.shape)
            [1, 1000]
    """
    return _densenet('densenet121', 121, pretrained, **kwargs)


def densenet161(
    pretrained: bool = False, **kwargs: Unpack[_DenseNetOptions]
) -> DenseNet:
    """DenseNet 161-layer model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_.

    Args:
        pretrained (bool, optional): Whether to load pre-trained weights. If True, returns a model pre-trained
            on ImageNet. Default: False.
        **kwargs (optional): Additional keyword arguments. For details, please refer to :ref:`DenseNet <api_paddle_vision_models_DenseNet>`.

    Returns:
        :ref:`api_paddle_nn_Layer`. An instance of DenseNet 161-layer model.

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> from paddle.vision.models import densenet161

            >>> # Build model
            >>> model = densenet161()

            >>> # Build model and load imagenet pretrained weight
            >>> # model = densenet161(pretrained=True)

            >>> x = paddle.rand([1, 3, 224, 224])
            >>> out = model(x)

            >>> print(out.shape)
            [1, 1000]
    """
    return _densenet('densenet161', 161, pretrained, **kwargs)


def densenet169(
    pretrained: bool = False, **kwargs: Unpack[_DenseNetOptions]
) -> DenseNet:
    """DenseNet 169-layer model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_.

    Args:
        pretrained (bool, optional): Whether to load pre-trained weights. If True, returns a model pre-trained
            on ImageNet. Default: False.
        **kwargs (optional): Additional keyword arguments. For details, please refer to :ref:`DenseNet <api_paddle_vision_models_DenseNet>`.

    Returns:
        :ref:`api_paddle_nn_Layer`. An instance of DenseNet 169-layer model.

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> from paddle.vision.models import densenet169

            >>> # Build model
            >>> model = densenet169()

            >>> # Build model and load imagenet pretrained weight
            >>> # model = densenet169(pretrained=True)

            >>> x = paddle.rand([1, 3, 224, 224])
            >>> out = model(x)

            >>> print(out.shape)
            [1, 1000]
    """
    return _densenet('densenet169', 169, pretrained, **kwargs)


def densenet201(
    pretrained: bool = False, **kwargs: Unpack[_DenseNetOptions]
) -> DenseNet:
    """DenseNet 201-layer model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_.

    Args:
        pretrained (bool, optional): Whether to load pre-trained weights. If True, returns a model pre-trained
            on ImageNet. Default: False.
        **kwargs (optional): Additional keyword arguments. For details, please refer to :ref:`DenseNet <api_paddle_vision_models_DenseNet>`.

    Returns:
        :ref:`api_paddle_nn_Layer`. An instance of DenseNet 201-layer model.

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> from paddle.vision.models import densenet201

            >>> # Build model
            >>> model = densenet201()

            >>> # Build model and load imagenet pretrained weight
            >>> # model = densenet201(pretrained=True)
            >>> x = paddle.rand([1, 3, 224, 224])
            >>> out = model(x)

            >>> print(out.shape)
            [1, 1000]
    """
    return _densenet('densenet201', 201, pretrained, **kwargs)


def densenet264(
    pretrained: bool = False, **kwargs: Unpack[_DenseNetOptions]
) -> DenseNet:
    """DenseNet 264-layer model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_.

    Args:
        pretrained (bool, optional): Whether to load pre-trained weights. If True, returns a model pre-trained
            on ImageNet. Default: False.
        **kwargs (optional): Additional keyword arguments. For details, please refer to :ref:`DenseNet <api_paddle_vision_models_DenseNet>`.

    Returns:
        :ref:`api_paddle_nn_Layer`. An instance of DenseNet 264-layer model.

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> from paddle.vision.models import densenet264

            >>> # Build model
            >>> model = densenet264()

            >>> # Build model and load imagenet pretrained weight
            >>> # model = densenet264(pretrained=True)

            >>> x = paddle.rand([1, 3, 224, 224])
            >>> out = model(x)

            >>> print(out.shape)
            [1, 1000]
    """
    return _densenet('densenet264', 264, pretrained, **kwargs)
