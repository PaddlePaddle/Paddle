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

from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Literal,
    TypedDict,
)

from typing_extensions import NotRequired, Unpack

import paddle
from paddle import nn
from paddle.utils.download import get_weights_path_from_url

if TYPE_CHECKING:
    from paddle import Tensor
    from paddle.nn import Layer, Sequential

    class _VGGOptions(TypedDict):
        num_classes: NotRequired[int]
        with_pool: NotRequired[bool]


__all__ = []

model_urls = {
    'vgg16': (
        'https://paddle-hapi.bj.bcebos.com/models/vgg16.pdparams',
        '89bbffc0f87d260be9b8cdc169c991c4',
    ),
    'vgg19': (
        'https://paddle-hapi.bj.bcebos.com/models/vgg19.pdparams',
        '23b18bb13d8894f60f54e642be79a0dd',
    ),
}


class VGG(nn.Layer):
    """VGG model from
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_.

    Args:
        features (nn.Layer): Vgg features create by function make_layers.
        num_classes (int, optional): Output dim of last fc layer. If num_classes <= 0, last fc layer
                            will not be defined. Default: 1000.
        with_pool (bool, optional): Use pool before the last three fc layer or not. Default: True.

    Returns:
        :ref:`api_paddle_nn_Layer`. An instance of VGG model.

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> from paddle.vision.models import VGG
            >>> from paddle.vision.models.vgg import make_layers

            >>> vgg11_cfg = [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']

            >>> features = make_layers(vgg11_cfg)  # type: ignore

            >>> vgg11 = VGG(features)

            >>> x = paddle.rand([1, 3, 224, 224])
            >>> out = vgg11(x)

            >>> print(out.shape)
            [1, 1000]
    """

    num_classes: int
    with_pool: bool

    def __init__(
        self, features: Layer, num_classes: int = 1000, with_pool: bool = True
    ) -> None:
        super().__init__()
        self.features = features
        self.num_classes = num_classes
        self.with_pool = with_pool

        if with_pool:
            self.avgpool = nn.AdaptiveAvgPool2D((7, 7))

        if num_classes > 0:
            self.classifier = nn.Sequential(
                nn.Linear(512 * 7 * 7, 4096),
                nn.ReLU(),
                nn.Dropout(),
                nn.Linear(4096, 4096),
                nn.ReLU(),
                nn.Dropout(),
                nn.Linear(4096, num_classes),
            )

    def forward(self, x: Tensor) -> Tensor:
        x = self.features(x)

        if self.with_pool:
            x = self.avgpool(x)

        if self.num_classes > 0:
            x = paddle.flatten(x, 1)
            x = self.classifier(x)

        return x


def make_layers(
    cfg: list[int | Literal['M']], batch_norm: bool = False
) -> Sequential:
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2D(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2D(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2D(v), nn.ReLU()]
            else:
                layers += [conv2d, nn.ReLU()]
            in_channels = v
    return nn.Sequential(*layers)


cfgs = {
    'A': [
        64, 'M',
        128, 'M',
        256, 256, 'M',
        512, 512, 'M',
        512, 512, 'M',
    ],
    'B': [
        64, 64, 'M',
        128, 128, 'M',
        256, 256, 'M',
        512, 512, 'M',
        512, 512, 'M',
    ],
    'D': [
        64, 64, 'M',
        128, 128, 'M',
        256, 256, 256, 'M',
        512, 512, 512, 'M',
        512, 512, 512, 'M',
    ],
    'E': [
        64, 64, 'M',
        128, 128, 'M',
        256, 256, 256, 256, 'M',
        512, 512, 512, 512, 'M',
        512, 512, 512, 512, 'M',
    ],
}  # fmt: skip


def _vgg(
    arch: str,
    cfg: Literal["A", "B", "D", "E"],
    batch_norm: bool,
    pretrained: bool,
    **kwargs: Unpack[_VGGOptions],
) -> VGG:
    model = VGG(make_layers(cfgs[cfg], batch_norm=batch_norm), **kwargs)

    if pretrained:
        assert (
            arch in model_urls
        ), f"{arch} model do not have a pretrained model now, you should set pretrained=False"
        weight_path = get_weights_path_from_url(
            model_urls[arch][0], model_urls[arch][1]
        )

        param = paddle.load(weight_path)
        model.load_dict(param)

    return model


def vgg11(
    pretrained: bool = False,
    batch_norm: bool = False,
    **kwargs: Unpack[_VGGOptions],
) -> VGG:
    """VGG 11-layer model from
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_.

    Args:
        pretrained (bool, optional): Whether to load pre-trained weights. If True, returns a model pre-trained
                            on ImageNet. Default: False.
        batch_norm (bool, optional): If True, returns a model with batch_norm layer. Default: False.
        **kwargs (optional): Additional keyword arguments. For details, please refer to :ref:`VGG <api_paddle_vision_models_VGG>`.

    Returns:
        :ref:`api_paddle_nn_Layer`. An instance of VGG 11-layer model.

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> from paddle.vision.models import vgg11

            >>> # build model
            >>> model = vgg11()

            >>> # build vgg11 model with batch_norm
            >>> model = vgg11(batch_norm=True)

            >>> x = paddle.rand([1, 3, 224, 224])
            >>> out = model(x)

            >>> print(out.shape)
            [1, 1000]
    """
    model_name = 'vgg11'
    if batch_norm:
        model_name += '_bn'
    return _vgg(model_name, 'A', batch_norm, pretrained, **kwargs)


def vgg13(
    pretrained: bool = False,
    batch_norm: bool = False,
    **kwargs: Unpack[_VGGOptions],
) -> VGG:
    """VGG 13-layer model from
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_.

    Args:
        pretrained (bool, optional): Whether to load pre-trained weights. If True, returns a model pre-trained
                            on ImageNet. Default: False.
        batch_norm (bool): If True, returns a model with batch_norm layer. Default: False.
        **kwargs (optional): Additional keyword arguments. For details, please refer to :ref:`VGG <api_paddle_vision_models_VGG>`.

    Returns:
        :ref:`api_paddle_nn_Layer`. An instance of VGG 13-layer model.

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> from paddle.vision.models import vgg13

            >>> # build model
            >>> model = vgg13()

            >>> # build vgg13 model with batch_norm
            >>> model = vgg13(batch_norm=True)

            >>> x = paddle.rand([1, 3, 224, 224])
            >>> out = model(x)

            >>> print(out.shape)
            [1, 1000]
    """
    model_name = 'vgg13'
    if batch_norm:
        model_name += '_bn'
    return _vgg(model_name, 'B', batch_norm, pretrained, **kwargs)


def vgg16(
    pretrained: bool = False,
    batch_norm: bool = False,
    **kwargs: Unpack[_VGGOptions],
) -> VGG:
    """VGG 16-layer model from
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_.

    Args:
        pretrained (bool, optional): Whether to load pre-trained weights. If True, returns a model pre-trained
                            on ImageNet. Default: False.
        batch_norm (bool, optional): If True, returns a model with batch_norm layer. Default: False.
        **kwargs (optional): Additional keyword arguments. For details, please refer to :ref:`VGG <api_paddle_vision_models_VGG>`.

    Returns:
        :ref:`api_paddle_nn_Layer`. An instance of VGG 16-layer model.

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> from paddle.vision.models import vgg16

            >>> # build model
            >>> model = vgg16()

            >>> # build vgg16 model with batch_norm
            >>> model = vgg16(batch_norm=True)

            >>> x = paddle.rand([1, 3, 224, 224])
            >>> out = model(x)

            >>> print(out.shape)
            [1, 1000]
    """
    model_name = 'vgg16'
    if batch_norm:
        model_name += '_bn'
    return _vgg(model_name, 'D', batch_norm, pretrained, **kwargs)


def vgg19(
    pretrained: bool = False,
    batch_norm: bool = False,
    **kwargs: Unpack[_VGGOptions],
) -> VGG:
    """VGG 19-layer model from
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_.

    Args:
        pretrained (bool, optional): Whether to load pre-trained weights. If True, returns a model pre-trained
                            on ImageNet. Default: False.
        batch_norm (bool, optional): If True, returns a model with batch_norm layer. Default: False.
        **kwargs (optional): Additional keyword arguments. For details, please refer to :ref:`VGG <api_paddle_vision_models_VGG>`.

    Returns:
        :ref:`api_paddle_nn_Layer`. An instance of VGG 19-layer model.

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> from paddle.vision.models import vgg19

            >>> # build model
            >>> model = vgg19()

            >>> # build vgg19 model with batch_norm
            >>> model = vgg19(batch_norm=True)

            >>> x = paddle.rand([1, 3, 224, 224])
            >>> out = model(x)

            >>> print(out.shape)
            [1, 1000]
    """
    model_name = 'vgg19'
    if batch_norm:
        model_name += '_bn'
    return _vgg(model_name, 'E', batch_norm, pretrained, **kwargs)
