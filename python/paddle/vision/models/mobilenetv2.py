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

import paddle
import paddle.nn as nn
from paddle.utils.download import get_weights_path_from_url

from .utils import _make_divisible
from ..ops import ConvNormActivation

__all__ = []

model_urls = {
    'mobilenetv2_1.0':
    ('https://paddle-hapi.bj.bcebos.com/models/mobilenet_v2_x1.0.pdparams',
     '0340af0a901346c8d46f4529882fb63d')
}


class InvertedResidual(nn.Layer):

    def __init__(self,
                 inp,
                 oup,
                 stride,
                 expand_ratio,
                 norm_layer=nn.BatchNorm2D):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            layers.append(
                ConvNormActivation(inp,
                                   hidden_dim,
                                   kernel_size=1,
                                   norm_layer=norm_layer,
                                   activation_layer=nn.ReLU6))
        layers.extend([
            ConvNormActivation(hidden_dim,
                               hidden_dim,
                               stride=stride,
                               groups=hidden_dim,
                               norm_layer=norm_layer,
                               activation_layer=nn.ReLU6),
            nn.Conv2D(hidden_dim, oup, 1, 1, 0, bias_attr=False),
            norm_layer(oup),
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(nn.Layer):
    """MobileNetV2 model from
    `"MobileNetV2: Inverted Residuals and Linear Bottlenecks" <https://arxiv.org/abs/1801.04381>`_.

    Args:
        scale (float, optional): Scale of channels in each layer. Default: 1.0.
        num_classes (int, optional): Output dim of last fc layer. If num_classes <= 0, last fc layer 
                            will not be defined. Default: 1000.
        with_pool (bool, optional): Use pool before the last fc layer or not. Default: True.

    Returns:
        :ref:`api_paddle_nn_Layer`. An instance of MobileNetV2 model.

    Examples:
        .. code-block:: python

            import paddle
            from paddle.vision.models import MobileNetV2

            model = MobileNetV2()

            x = paddle.rand([1, 3, 224, 224])
            out = model(x)

            print(out.shape)
            # [1, 1000]
    """

    def __init__(self, scale=1.0, num_classes=1000, with_pool=True):
        super(MobileNetV2, self).__init__()
        self.num_classes = num_classes
        self.with_pool = with_pool
        input_channel = 32
        last_channel = 1280

        block = InvertedResidual
        round_nearest = 8
        norm_layer = nn.BatchNorm2D
        inverted_residual_setting = [
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        input_channel = _make_divisible(input_channel * scale, round_nearest)
        self.last_channel = _make_divisible(last_channel * max(1.0, scale),
                                            round_nearest)
        features = [
            ConvNormActivation(3,
                               input_channel,
                               stride=2,
                               norm_layer=norm_layer,
                               activation_layer=nn.ReLU6)
        ]

        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c * scale, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(
                    block(input_channel,
                          output_channel,
                          stride,
                          expand_ratio=t,
                          norm_layer=norm_layer))
                input_channel = output_channel

        features.append(
            ConvNormActivation(input_channel,
                               self.last_channel,
                               kernel_size=1,
                               norm_layer=norm_layer,
                               activation_layer=nn.ReLU6))

        self.features = nn.Sequential(*features)

        if with_pool:
            self.pool2d_avg = nn.AdaptiveAvgPool2D(1)

        if self.num_classes > 0:
            self.classifier = nn.Sequential(
                nn.Dropout(0.2), nn.Linear(self.last_channel, num_classes))

    def forward(self, x):
        x = self.features(x)

        if self.with_pool:
            x = self.pool2d_avg(x)

        if self.num_classes > 0:
            x = paddle.flatten(x, 1)
            x = self.classifier(x)
        return x


def _mobilenet(arch, pretrained=False, **kwargs):
    model = MobileNetV2(**kwargs)
    if pretrained:
        assert arch in model_urls, "{} model do not have a pretrained model now, you should set pretrained=False".format(
            arch)
        weight_path = get_weights_path_from_url(model_urls[arch][0],
                                                model_urls[arch][1])

        param = paddle.load(weight_path)
        model.load_dict(param)

    return model


def mobilenet_v2(pretrained=False, scale=1.0, **kwargs):
    """MobileNetV2 from
    `"MobileNetV2: Inverted Residuals and Linear Bottlenecks" <https://arxiv.org/abs/1801.04381>`_.
    
    Args:
        pretrained (bool, optional): Whether to load pre-trained weights. If True, returns a model pre-trained
                            on ImageNet. Default: False.
        scale (float, optional): Scale of channels in each layer. Default: 1.0.
        **kwargs (optional): Additional keyword arguments. For details, please refer to :ref:`MobileNetV2 <api_paddle_vision_MobileNetV2>`.

    Returns:
        :ref:`api_paddle_nn_Layer`. An instance of MobileNetV2 model.

    Examples:
        .. code-block:: python

            import paddle
            from paddle.vision.models import mobilenet_v2

            # build model
            model = mobilenet_v2()

            # build model and load imagenet pretrained weight
            # model = mobilenet_v2(pretrained=True)

            # build mobilenet v2 with scale=0.5
            model = mobilenet_v2(scale=0.5)

            x = paddle.rand([1, 3, 224, 224])
            out = model(x)

            print(out.shape)
            # [1, 1000]
    """
    model = _mobilenet('mobilenetv2_' + str(scale),
                       pretrained,
                       scale=scale,
                       **kwargs)
    return model
