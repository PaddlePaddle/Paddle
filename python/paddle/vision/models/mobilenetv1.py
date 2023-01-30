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

import paddle
import paddle.nn as nn
<<<<<<< HEAD
from paddle.utils.download import get_weights_path_from_url

=======

from paddle.utils.download import get_weights_path_from_url
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
from ..ops import ConvNormActivation

__all__ = []

model_urls = {
<<<<<<< HEAD
    'mobilenetv1_1.0': (
        'https://paddle-hapi.bj.bcebos.com/models/mobilenetv1_1.0.pdparams',
        '3033ab1975b1670bef51545feb65fc45',
    )
=======
    'mobilenetv1_1.0':
    ('https://paddle-hapi.bj.bcebos.com/models/mobilenetv1_1.0.pdparams',
     '3033ab1975b1670bef51545feb65fc45')
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
}


class DepthwiseSeparable(nn.Layer):
<<<<<<< HEAD
    def __init__(
        self,
        in_channels,
        out_channels1,
        out_channels2,
        num_groups,
        stride,
        scale,
    ):
        super().__init__()

        self._depthwise_conv = ConvNormActivation(
            in_channels,
            int(out_channels1 * scale),
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=int(num_groups * scale),
        )

        self._pointwise_conv = ConvNormActivation(
            int(out_channels1 * scale),
            int(out_channels2 * scale),
            kernel_size=1,
            stride=1,
            padding=0,
        )
=======

    def __init__(self, in_channels, out_channels1, out_channels2, num_groups,
                 stride, scale):
        super(DepthwiseSeparable, self).__init__()

        self._depthwise_conv = ConvNormActivation(in_channels,
                                                  int(out_channels1 * scale),
                                                  kernel_size=3,
                                                  stride=stride,
                                                  padding=1,
                                                  groups=int(num_groups *
                                                             scale))

        self._pointwise_conv = ConvNormActivation(int(out_channels1 * scale),
                                                  int(out_channels2 * scale),
                                                  kernel_size=1,
                                                  stride=1,
                                                  padding=0)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

    def forward(self, x):
        x = self._depthwise_conv(x)
        x = self._pointwise_conv(x)
        return x


class MobileNetV1(nn.Layer):
    """MobileNetV1 model from
    `"MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications" <https://arxiv.org/abs/1704.04861>`_.

    Args:
        scale (float, optional): Scale of channels in each layer. Default: 1.0.
<<<<<<< HEAD
        num_classes (int, optional): Output dim of last fc layer. If num_classes <= 0, last fc layer
=======
        num_classes (int, optional): Output dim of last fc layer. If num_classes <= 0, last fc layer 
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
                            will not be defined. Default: 1000.
        with_pool (bool, optional): Use pool before the last fc layer or not. Default: True.

    Returns:
        :ref:`api_paddle_nn_Layer`. An instance of MobileNetV1 model.

    Examples:
        .. code-block:: python

            import paddle
            from paddle.vision.models import MobileNetV1

            model = MobileNetV1()

            x = paddle.rand([1, 3, 224, 224])
            out = model(x)

            print(out.shape)
            # [1, 1000]
    """

    def __init__(self, scale=1.0, num_classes=1000, with_pool=True):
<<<<<<< HEAD
        super().__init__()
=======
        super(MobileNetV1, self).__init__()
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        self.scale = scale
        self.dwsl = []
        self.num_classes = num_classes
        self.with_pool = with_pool

<<<<<<< HEAD
        self.conv1 = ConvNormActivation(
            in_channels=3,
            out_channels=int(32 * scale),
            kernel_size=3,
            stride=2,
            padding=1,
        )

        dws21 = self.add_sublayer(
            sublayer=DepthwiseSeparable(
                in_channels=int(32 * scale),
                out_channels1=32,
                out_channels2=64,
                num_groups=32,
                stride=1,
                scale=scale,
            ),
            name="conv2_1",
        )
        self.dwsl.append(dws21)

        dws22 = self.add_sublayer(
            sublayer=DepthwiseSeparable(
                in_channels=int(64 * scale),
                out_channels1=64,
                out_channels2=128,
                num_groups=64,
                stride=2,
                scale=scale,
            ),
            name="conv2_2",
        )
        self.dwsl.append(dws22)

        dws31 = self.add_sublayer(
            sublayer=DepthwiseSeparable(
                in_channels=int(128 * scale),
                out_channels1=128,
                out_channels2=128,
                num_groups=128,
                stride=1,
                scale=scale,
            ),
            name="conv3_1",
        )
        self.dwsl.append(dws31)

        dws32 = self.add_sublayer(
            sublayer=DepthwiseSeparable(
                in_channels=int(128 * scale),
                out_channels1=128,
                out_channels2=256,
                num_groups=128,
                stride=2,
                scale=scale,
            ),
            name="conv3_2",
        )
        self.dwsl.append(dws32)

        dws41 = self.add_sublayer(
            sublayer=DepthwiseSeparable(
                in_channels=int(256 * scale),
                out_channels1=256,
                out_channels2=256,
                num_groups=256,
                stride=1,
                scale=scale,
            ),
            name="conv4_1",
        )
        self.dwsl.append(dws41)

        dws42 = self.add_sublayer(
            sublayer=DepthwiseSeparable(
                in_channels=int(256 * scale),
                out_channels1=256,
                out_channels2=512,
                num_groups=256,
                stride=2,
                scale=scale,
            ),
            name="conv4_2",
        )
        self.dwsl.append(dws42)

        for i in range(5):
            tmp = self.add_sublayer(
                sublayer=DepthwiseSeparable(
                    in_channels=int(512 * scale),
                    out_channels1=512,
                    out_channels2=512,
                    num_groups=512,
                    stride=1,
                    scale=scale,
                ),
                name="conv5_" + str(i + 1),
            )
            self.dwsl.append(tmp)

        dws56 = self.add_sublayer(
            sublayer=DepthwiseSeparable(
                in_channels=int(512 * scale),
                out_channels1=512,
                out_channels2=1024,
                num_groups=512,
                stride=2,
                scale=scale,
            ),
            name="conv5_6",
        )
        self.dwsl.append(dws56)

        dws6 = self.add_sublayer(
            sublayer=DepthwiseSeparable(
                in_channels=int(1024 * scale),
                out_channels1=1024,
                out_channels2=1024,
                num_groups=1024,
                stride=1,
                scale=scale,
            ),
            name="conv6",
        )
=======
        self.conv1 = ConvNormActivation(in_channels=3,
                                        out_channels=int(32 * scale),
                                        kernel_size=3,
                                        stride=2,
                                        padding=1)

        dws21 = self.add_sublayer(sublayer=DepthwiseSeparable(in_channels=int(
            32 * scale),
                                                              out_channels1=32,
                                                              out_channels2=64,
                                                              num_groups=32,
                                                              stride=1,
                                                              scale=scale),
                                  name="conv2_1")
        self.dwsl.append(dws21)

        dws22 = self.add_sublayer(sublayer=DepthwiseSeparable(in_channels=int(
            64 * scale),
                                                              out_channels1=64,
                                                              out_channels2=128,
                                                              num_groups=64,
                                                              stride=2,
                                                              scale=scale),
                                  name="conv2_2")
        self.dwsl.append(dws22)

        dws31 = self.add_sublayer(sublayer=DepthwiseSeparable(in_channels=int(
            128 * scale),
                                                              out_channels1=128,
                                                              out_channels2=128,
                                                              num_groups=128,
                                                              stride=1,
                                                              scale=scale),
                                  name="conv3_1")
        self.dwsl.append(dws31)

        dws32 = self.add_sublayer(sublayer=DepthwiseSeparable(in_channels=int(
            128 * scale),
                                                              out_channels1=128,
                                                              out_channels2=256,
                                                              num_groups=128,
                                                              stride=2,
                                                              scale=scale),
                                  name="conv3_2")
        self.dwsl.append(dws32)

        dws41 = self.add_sublayer(sublayer=DepthwiseSeparable(in_channels=int(
            256 * scale),
                                                              out_channels1=256,
                                                              out_channels2=256,
                                                              num_groups=256,
                                                              stride=1,
                                                              scale=scale),
                                  name="conv4_1")
        self.dwsl.append(dws41)

        dws42 = self.add_sublayer(sublayer=DepthwiseSeparable(in_channels=int(
            256 * scale),
                                                              out_channels1=256,
                                                              out_channels2=512,
                                                              num_groups=256,
                                                              stride=2,
                                                              scale=scale),
                                  name="conv4_2")
        self.dwsl.append(dws42)

        for i in range(5):
            tmp = self.add_sublayer(sublayer=DepthwiseSeparable(
                in_channels=int(512 * scale),
                out_channels1=512,
                out_channels2=512,
                num_groups=512,
                stride=1,
                scale=scale),
                                    name="conv5_" + str(i + 1))
            self.dwsl.append(tmp)

        dws56 = self.add_sublayer(sublayer=DepthwiseSeparable(
            in_channels=int(512 * scale),
            out_channels1=512,
            out_channels2=1024,
            num_groups=512,
            stride=2,
            scale=scale),
                                  name="conv5_6")
        self.dwsl.append(dws56)

        dws6 = self.add_sublayer(sublayer=DepthwiseSeparable(in_channels=int(
            1024 * scale),
                                                             out_channels1=1024,
                                                             out_channels2=1024,
                                                             num_groups=1024,
                                                             stride=1,
                                                             scale=scale),
                                 name="conv6")
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        self.dwsl.append(dws6)

        if with_pool:
            self.pool2d_avg = nn.AdaptiveAvgPool2D(1)

        if num_classes > 0:
            self.fc = nn.Linear(int(1024 * scale), num_classes)

    def forward(self, x):
        x = self.conv1(x)
        for dws in self.dwsl:
            x = dws(x)

        if self.with_pool:
            x = self.pool2d_avg(x)

        if self.num_classes > 0:
            x = paddle.flatten(x, 1)
            x = self.fc(x)
        return x


def _mobilenet(arch, pretrained=False, **kwargs):
    model = MobileNetV1(**kwargs)
    if pretrained:
<<<<<<< HEAD
        assert (
            arch in model_urls
        ), "{} model do not have a pretrained model now, you should set pretrained=False".format(
            arch
        )
        weight_path = get_weights_path_from_url(
            model_urls[arch][0], model_urls[arch][1]
        )
=======
        assert arch in model_urls, "{} model do not have a pretrained model now, you should set pretrained=False".format(
            arch)
        weight_path = get_weights_path_from_url(model_urls[arch][0],
                                                model_urls[arch][1])
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

        param = paddle.load(weight_path)
        model.load_dict(param)

    return model


def mobilenet_v1(pretrained=False, scale=1.0, **kwargs):
    """MobileNetV1 from
    `"MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications" <https://arxiv.org/abs/1704.04861>`_.
<<<<<<< HEAD

=======
    
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    Args:
        pretrained (bool, optional): Whether to load pre-trained weights. If True, returns a model pre-trained
                            on ImageNet. Default: False.
        scale (float, optional): Scale of channels in each layer. Default: 1.0.
        **kwargs (optional): Additional keyword arguments. For details, please refer to :ref:`MobileNetV1 <api_paddle_vision_MobileNetV1>`.

    Returns:
        :ref:`api_paddle_nn_Layer`. An instance of MobileNetV1 model.

    Examples:
        .. code-block:: python

            import paddle
            from paddle.vision.models import mobilenet_v1

            # build model
            model = mobilenet_v1()

            # build model and load imagenet pretrained weight
            # model = mobilenet_v1(pretrained=True)

            # build mobilenet v1 with scale=0.5
            model_scale = mobilenet_v1(scale=0.5)

            x = paddle.rand([1, 3, 224, 224])
            out = model(x)

            print(out.shape)
            # [1, 1000]
    """
<<<<<<< HEAD
    model = _mobilenet(
        'mobilenetv1_' + str(scale), pretrained, scale=scale, **kwargs
    )
=======
    model = _mobilenet('mobilenetv1_' + str(scale),
                       pretrained,
                       scale=scale,
                       **kwargs)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    return model
