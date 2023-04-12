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

import paddle
from paddle import nn
from paddle.nn import AdaptiveAvgPool2D, Linear, MaxPool2D
from paddle.utils.download import get_weights_path_from_url

from ..ops import ConvNormActivation

__all__ = []

model_urls = {
    "shufflenet_v2_x0_25": (
        "https://paddle-hapi.bj.bcebos.com/models/shufflenet_v2_x0_25.pdparams",
        "1e509b4c140eeb096bb16e214796d03b",
    ),
    "shufflenet_v2_x0_33": (
        "https://paddle-hapi.bj.bcebos.com/models/shufflenet_v2_x0_33.pdparams",
        "3d7b3ab0eaa5c0927ff1026d31b729bd",
    ),
    "shufflenet_v2_x0_5": (
        "https://paddle-hapi.bj.bcebos.com/models/shufflenet_v2_x0_5.pdparams",
        "5e5cee182a7793c4e4c73949b1a71bd4",
    ),
    "shufflenet_v2_x1_0": (
        "https://paddle-hapi.bj.bcebos.com/models/shufflenet_v2_x1_0.pdparams",
        "122d42478b9e81eb49f8a9ede327b1a4",
    ),
    "shufflenet_v2_x1_5": (
        "https://paddle-hapi.bj.bcebos.com/models/shufflenet_v2_x1_5.pdparams",
        "faced5827380d73531d0ee027c67826d",
    ),
    "shufflenet_v2_x2_0": (
        "https://paddle-hapi.bj.bcebos.com/models/shufflenet_v2_x2_0.pdparams",
        "cd3dddcd8305e7bcd8ad14d1c69a5784",
    ),
    "shufflenet_v2_swish": (
        "https://paddle-hapi.bj.bcebos.com/models/shufflenet_v2_swish.pdparams",
        "adde0aa3b023e5b0c94a68be1c394b84",
    ),
}


def create_activation_layer(act):
    if act == "swish":
        return nn.Swish
    elif act == "relu":
        return nn.ReLU
    elif act is None:
        return None
    else:
        raise RuntimeError(f"The activation function is not supported: {act}")


def channel_shuffle(x, groups):
    batch_size, num_channels, height, width = x.shape[0:4]
    channels_per_group = num_channels // groups

    # reshape
    x = paddle.reshape(
        x, shape=[batch_size, groups, channels_per_group, height, width]
    )

    # transpose
    x = paddle.transpose(x, perm=[0, 2, 1, 3, 4])

    # flatten
    x = paddle.reshape(x, shape=[batch_size, num_channels, height, width])
    return x


class InvertedResidual(nn.Layer):
    def __init__(
        self, in_channels, out_channels, stride, activation_layer=nn.ReLU
    ):
        super().__init__()
        self._conv_pw = ConvNormActivation(
            in_channels=in_channels // 2,
            out_channels=out_channels // 2,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1,
            activation_layer=activation_layer,
        )
        self._conv_dw = ConvNormActivation(
            in_channels=out_channels // 2,
            out_channels=out_channels // 2,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=out_channels // 2,
            activation_layer=None,
        )
        self._conv_linear = ConvNormActivation(
            in_channels=out_channels // 2,
            out_channels=out_channels // 2,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1,
            activation_layer=activation_layer,
        )

    def forward(self, inputs):
        x1, x2 = paddle.split(
            inputs,
            num_or_sections=[inputs.shape[1] // 2, inputs.shape[1] // 2],
            axis=1,
        )
        x2 = self._conv_pw(x2)
        x2 = self._conv_dw(x2)
        x2 = self._conv_linear(x2)
        out = paddle.concat([x1, x2], axis=1)
        return channel_shuffle(out, 2)


class InvertedResidualDS(nn.Layer):
    def __init__(
        self, in_channels, out_channels, stride, activation_layer=nn.ReLU
    ):
        super().__init__()

        # branch1
        self._conv_dw_1 = ConvNormActivation(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=in_channels,
            activation_layer=None,
        )
        self._conv_linear_1 = ConvNormActivation(
            in_channels=in_channels,
            out_channels=out_channels // 2,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1,
            activation_layer=activation_layer,
        )
        # branch2
        self._conv_pw_2 = ConvNormActivation(
            in_channels=in_channels,
            out_channels=out_channels // 2,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1,
            activation_layer=activation_layer,
        )
        self._conv_dw_2 = ConvNormActivation(
            in_channels=out_channels // 2,
            out_channels=out_channels // 2,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=out_channels // 2,
            activation_layer=None,
        )
        self._conv_linear_2 = ConvNormActivation(
            in_channels=out_channels // 2,
            out_channels=out_channels // 2,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1,
            activation_layer=activation_layer,
        )

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
    `"ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design" <https://arxiv.org/pdf/1807.11164.pdf>`_.

    Args:
        scale (float, optional): Scale of output channels. Default: True.
        act (str, optional): Activation function of neural network. Default: "relu".
        num_classes (int, optional): Output dim of last fc layer. If num_classes <= 0, last fc layer
                            will not be defined. Default: 1000.
        with_pool (bool, optional): Use pool before the last fc layer or not. Default: True.

    Returns:
        :ref:`api_paddle_nn_Layer`. An instance of ShuffleNetV2 model.

    Examples:
        .. code-block:: python

            import paddle
            from paddle.vision.models import ShuffleNetV2

            shufflenet_v2_swish = ShuffleNetV2(scale=1.0, act="swish")
            x = paddle.rand([1, 3, 224, 224])
            out = shufflenet_v2_swish(x)
            print(out.shape)
            # [1, 1000]
    """

    def __init__(self, scale=1.0, act="relu", num_classes=1000, with_pool=True):
        super().__init__()
        self.scale = scale
        self.num_classes = num_classes
        self.with_pool = with_pool
        stage_repeats = [4, 8, 4]
        activation_layer = create_activation_layer(act)

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
            raise NotImplementedError(
                "This scale size:[" + str(scale) + "] is not implemented!"
            )
        # 1. conv1
        self._conv1 = ConvNormActivation(
            in_channels=3,
            out_channels=stage_out_channels[1],
            kernel_size=3,
            stride=2,
            padding=1,
            activation_layer=activation_layer,
        )
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
                            activation_layer=activation_layer,
                        ),
                        name=str(stage_id + 2) + "_" + str(i + 1),
                    )
                else:
                    block = self.add_sublayer(
                        sublayer=InvertedResidual(
                            in_channels=stage_out_channels[stage_id + 2],
                            out_channels=stage_out_channels[stage_id + 2],
                            stride=1,
                            activation_layer=activation_layer,
                        ),
                        name=str(stage_id + 2) + "_" + str(i + 1),
                    )
                self._block_list.append(block)
        # 3. last_conv
        self._last_conv = ConvNormActivation(
            in_channels=stage_out_channels[-2],
            out_channels=stage_out_channels[-1],
            kernel_size=1,
            stride=1,
            padding=0,
            activation_layer=activation_layer,
        )
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
            arch
        )
        weight_path = get_weights_path_from_url(
            model_urls[arch][0], model_urls[arch][1]
        )

        param = paddle.load(weight_path)
        model.set_dict(param)
    return model


def shufflenet_v2_x0_25(pretrained=False, **kwargs):
    """ShuffleNetV2 with 0.25x output channels, as described in
    `"ShuffleNet V2: Practical Guidelines for Ecient CNN Architecture Design" <https://arxiv.org/pdf/1807.11164.pdf>`_.

    Args:
        pretrained (bool, optional): Whether to load pre-trained weights. If True, returns a model pre-trained
                            on ImageNet. Default: False.
        **kwargs (optional): Additional keyword arguments. For details, please refer to :ref:`ShuffleNetV2 <api_paddle_vision_ShuffleNetV2>`.

    Returns:
        :ref:`api_paddle_nn_Layer`. An instance of ShuffleNetV2 with 0.25x output channels.

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
            # [1, 1000]
    """
    return _shufflenet_v2(
        "shufflenet_v2_x0_25", scale=0.25, pretrained=pretrained, **kwargs
    )


def shufflenet_v2_x0_33(pretrained=False, **kwargs):
    """ShuffleNetV2 with 0.33x output channels, as described in
    `"ShuffleNet V2: Practical Guidelines for Ecient CNN Architecture Design" <https://arxiv.org/pdf/1807.11164.pdf>`_.

    Args:
        pretrained (bool, optional): Whether to load pre-trained weights. If True, returns a model pre-trained
                            on ImageNet. Default: False.
        **kwargs (optional): Additional keyword arguments. For details, please refer to :ref:`ShuffleNetV2 <api_paddle_vision_ShuffleNetV2>`.

    Returns:
        :ref:`api_paddle_nn_Layer`. An instance of ShuffleNetV2 with 0.33x output channels.

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
            # [1, 1000]
    """
    return _shufflenet_v2(
        "shufflenet_v2_x0_33", scale=0.33, pretrained=pretrained, **kwargs
    )


def shufflenet_v2_x0_5(pretrained=False, **kwargs):
    """ShuffleNetV2 with 0.5x output channels, as described in
    `"ShuffleNet V2: Practical Guidelines for Ecient CNN Architecture Design" <https://arxiv.org/pdf/1807.11164.pdf>`_.

    Args:
        pretrained (bool, optional): Whether to load pre-trained weights. If True, returns a model pre-trained
                            on ImageNet. Default: False.
        **kwargs (optional): Additional keyword arguments. For details, please refer to :ref:`ShuffleNetV2 <api_paddle_vision_ShuffleNetV2>`.

    Returns:
        :ref:`api_paddle_nn_Layer`. An instance of ShuffleNetV2 with 0.5x output channels.

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
            # [1, 1000]
    """
    return _shufflenet_v2(
        "shufflenet_v2_x0_5", scale=0.5, pretrained=pretrained, **kwargs
    )


def shufflenet_v2_x1_0(pretrained=False, **kwargs):
    """ShuffleNetV2 with 1.0x output channels, as described in
    `"ShuffleNet V2: Practical Guidelines for Ecient CNN Architecture Design" <https://arxiv.org/pdf/1807.11164.pdf>`_.

    Args:
        pretrained (bool, optional): Whether to load pre-trained weights. If True, returns a model pre-trained
                            on ImageNet. Default: False.
        **kwargs (optional): Additional keyword arguments. For details, please refer to :ref:`ShuffleNetV2 <api_paddle_vision_ShuffleNetV2>`.

    Returns:
        :ref:`api_paddle_nn_Layer`. An instance of ShuffleNetV2 with 1.0x output channels.

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
            # [1, 1000]
    """
    return _shufflenet_v2(
        "shufflenet_v2_x1_0", scale=1.0, pretrained=pretrained, **kwargs
    )


def shufflenet_v2_x1_5(pretrained=False, **kwargs):
    """ShuffleNetV2 with 1.5x output channels, as described in
    `"ShuffleNet V2: Practical Guidelines for Ecient CNN Architecture Design" <https://arxiv.org/pdf/1807.11164.pdf>`_.

    Args:
        pretrained (bool, optional): Whether to load pre-trained weights. If True, returns a model pre-trained
                            on ImageNet. Default: False.
        **kwargs (optional): Additional keyword arguments. For details, please refer to :ref:`ShuffleNetV2 <api_paddle_vision_ShuffleNetV2>`.

    Returns:
        :ref:`api_paddle_nn_Layer`. An instance of ShuffleNetV2 with 1.5x output channels.

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
            # [1, 1000]
    """
    return _shufflenet_v2(
        "shufflenet_v2_x1_5", scale=1.5, pretrained=pretrained, **kwargs
    )


def shufflenet_v2_x2_0(pretrained=False, **kwargs):
    """ShuffleNetV2 with 2.0x output channels, as described in
    `"ShuffleNet V2: Practical Guidelines for Ecient CNN Architecture Design" <https://arxiv.org/pdf/1807.11164.pdf>`_.

    Args:
        pretrained (bool, optional): Whether to load pre-trained weights. If True, returns a model pre-trained
                            on ImageNet. Default: False.
        **kwargs (optional): Additional keyword arguments. For details, please refer to :ref:`ShuffleNetV2 <api_paddle_vision_ShuffleNetV2>`.

    Returns:
        :ref:`api_paddle_nn_Layer`. An instance of ShuffleNetV2 with 2.0x output channels.

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
            # [1, 1000]
    """
    return _shufflenet_v2(
        "shufflenet_v2_x2_0", scale=2.0, pretrained=pretrained, **kwargs
    )


def shufflenet_v2_swish(pretrained=False, **kwargs):
    """ShuffleNetV2 with swish activation function, as described in
    `"ShuffleNet V2: Practical Guidelines for Ecient CNN Architecture Design" <https://arxiv.org/pdf/1807.11164.pdf>`_.

    Args:
        pretrained (bool, optional): Whether to load pre-trained weights. If True, returns a model pre-trained
                            on ImageNet. Default: False.
        **kwargs (optional): Additional keyword arguments. For details, please refer to :ref:`ShuffleNetV2 <api_paddle_vision_ShuffleNetV2>`.

    Returns:
        :ref:`api_paddle_nn_Layer`. An instance of ShuffleNetV2 with swish activation function.

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
            # [1, 1000]
    """
    return _shufflenet_v2(
        "shufflenet_v2_swish",
        scale=1.0,
        act="swish",
        pretrained=pretrained,
        **kwargs,
    )
