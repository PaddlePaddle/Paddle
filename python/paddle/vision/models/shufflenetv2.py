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
import paddle.nn as nn

from paddle.nn import Conv2D, Linear, BatchNorm
from paddle.nn import MaxPool2D, AdaptiveAvgPool2D
from paddle.utils.download import get_weights_path_from_url


model_urls = {
    'shufflenetv2_x0.25': (
        'https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/ShuffleNetV2_x0_25_pretrained.pdparams',
        'e753404cbd95027759c5f56ecd6c9c4b'),
    'shufflenetv2_x0.33': (
        'https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/ShuffleNetV2_x0_33_pretrained.pdparams',
        '776e3cf9a4923abdfce789c45b8fe1f2'),
    'shufflenetv2_x0.5': (
        'https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/ShuffleNetV2_x0_5_pretrained.pdparams',
        'e3649cf531566917e2969487d2bc6b60'),
    'shufflenetv2_x1.0': (
        'https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/ShuffleNetV2_x1_0_pretrained.pdparams',
        '7821c348ea34e58847c43a08a4ac0bdf'),
    'shufflenetv2_x1.5': (
        'https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/ShuffleNetV2_x1_5_pretrained.pdparams',
        '93a07fa557ab2d8803550f39e5b6c391'),
    'shufflenetv2_x2.0': (
        'https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/ShuffleNetV2_x2_0_pretrained.pdparams',
        '4ab1f622fd0d341e0f84b4e057797563'),
    "shufflenetv2_swish": (
        "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/ShuffleNetV2_swish_pretrained.pdparams",
        "daff38b3df1b3748fccbb13cfdf02519", ),
}


class conv_bn(nn.Layer):
    def __init__(self, inp, oup, kernel_size=1, stride=1, padding=0, act=None):
        super(conv_bn, self).__init__()

        self._conv = Conv2D(inp, oup, kernel_size, stride, padding, bias_attr=False)
        self._batch_norm = BatchNorm(oup, act=act)

    def forward(self, inputs):
        x = self._conv(inputs)
        x = self._batch_norm(x)
        return x


class depthwise_conv(nn.Layer):

    def __init__(self, inp, oup, kernel_size=3, stride=1, padding=1, act=None):
        super(depthwise_conv, self).__init__()

        self._conv = Conv2D(inp, oup, kernel_size, stride, padding, groups=inp, bias_attr=False)
        self._batch_norm = BatchNorm(oup, act=act)

    def forward(self, inputs):
        x = self._conv(inputs)
        x = self._batch_norm(x)
        return x


def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.shape[:]
    channels_per_group = num_channels // groups

    # reshape
    x = x.reshape((batchsize, groups,
               channels_per_group, height, width))

    x = paddle.transpose(x, perm=[0, 2, 1, 3, 4])

    # flatten
    x = x.reshape((batchsize, -1, height, width))

    return x


class ShuffleNetUnit1(nn.Layer):
    def __init__(self, out_channel, stride=1, act="relu"):
        """The unit of shuffleNetv2 for stride=1
        Args:
            out_channel: int, number of channels
        """
        super(ShuffleNetUnit1, self).__init__()

        assert out_channel % 2 == 0
        self.oup_inc = out_channel // 2

        # pw
        self._conv_pw = conv_bn(self.oup_inc, self.oup_inc, 1, 1, 0, act=act)
        # dw
        self._conv_dw = depthwise_conv(self.oup_inc, self.oup_inc, 3, stride, 1, act=None)
        # pw-linear
        self._conv_linear = conv_bn(self.oup_inc, self.oup_inc, 1, 1, 0, act=act)

    def forward(self, inputs):
        # split the channel
        shortcut, x = paddle.split(inputs, 2, axis=1)

        x = self._conv_pw(x)
        x = self._conv_dw(x)
        x = self._conv_linear(x)
        x = paddle.concat([shortcut, x], axis=1)
        x = channel_shuffle(x, 2)
        return x


class ShuffleNetUnit2(nn.Layer):
    """The unit of shuffleNetv2 for stride=2
    """
    def __init__(self, inp_channel, out_channel, stride=2, act="relu"):
        super(ShuffleNetUnit2, self).__init__()

        assert out_channel % 2 == 0
        self.oup_inc = out_channel // 2

        # dw
        self._conv_dw_1 = depthwise_conv(inp_channel, inp_channel, 3, stride, 1, act=None)
        # pw-linear
        self._conv_linear_1 = conv_bn(inp_channel, self.oup_inc, 1, 1, 0, act=act)

        # pw
        self._conv_pw_2 = conv_bn(inp_channel, self.oup_inc, 1, 1, 0, act=act)
        # dw
        self._conv_dw_2 = depthwise_conv(self.oup_inc, self.oup_inc, 3, stride, 1, act=None)
        # pw-linear
        self._conv_linear_2 = conv_bn(self.oup_inc, self.oup_inc, 1, 1, 0, act=act)

    def forward(self, inputs):
        shortcut, x = inputs, inputs

        shortcut = self._conv_dw_1(shortcut)
        shortcut = self._conv_linear_1(shortcut)
        x = self._conv_pw_2(x)
        x = self._conv_dw_2(x)
        x = self._conv_linear_2(x)

        x = paddle.concat([shortcut, x], axis=1)
        x = channel_shuffle(x, 2)
        return x


class ShuffleNetV2(nn.Layer):
    """ShuffleNetV2 model architecture from
    `"Going Deeper with Convolutions"
    <https://arxiv.org/pdf/1807.11164.pdf>`_

    Args:
       scale (float): scale of shufflenetv2. Default: 1.0.
       num_classes (int): output dim of last fc layer. If num_classes <=0, last fc layer
                           will not be defined. Default: 1000.
    Examples:
       .. code-block:: python

           from paddle.vision.models import ShuffleNetV2

           # build model
           model = ShuffleNetV2(scale=1.0)
    """

    def __init__(self, scale=1.0, act="relu", num_classes=1000, with_pool=True):
        super(ShuffleNetV2, self).__init__()
        stages_repeats = [4, 8, 4]

        if scale == 0.25:
            stages_out_channels = [24, 24, 48, 96, 512]
        elif scale == 0.33:
            stages_out_channels = [24, 32, 64, 128, 512]
        elif scale == 0.5:
            stages_out_channels = [24, 48, 96, 192, 1024]
        elif scale == 1.0:
            stages_out_channels = [24, 116, 232, 464, 1024]
        elif scale == 1.5:
            stages_out_channels = [24, 176, 352, 704, 1024]
        elif scale == 2.0:
            stages_out_channels = [24, 224, 488, 976, 2048]
        else:
            raise NotImplementedError("This scale " + str(scale) +
                                      " size is not implemented!")

        self.stage_out_channels = stages_out_channels
        self.stages_repeats = stages_repeats

        # building first layer
        first_channel = self.stage_out_channels[0]
        self._conv1 = conv_bn(3, first_channel, 3, 2, 1, act)
        self.maxpool = MaxPool2D(kernel_size=3, stride=2, padding=1)

        self.stages = [2, 3, 4]
        self.shuffle_block_list = []
        for i, stage in enumerate(self.stages):
            self.stage_block_list = []
            for block in range(self.stages_repeats[i]):
                if block == 0:
                    stage_block = self.add_sublayer(
                        "{}_{}".format(stage, block + 1),
                        ShuffleNetUnit2(self.stage_out_channels[i],
                                        self.stage_out_channels[i+1],
                                        act=act))
                else:
                    stage_block = self.add_sublayer(
                        "{}_{}".format(stage, block + 1),
                        ShuffleNetUnit1(self.stage_out_channels[i+1],
                                        act=act))
                self.stage_block_list.append(stage_block)
            self.shuffle_block_list.append(self.stage_block_list)

        outout_channel = self.stage_out_channels[-1]

        # building last several layers
        self._last_conv = conv_bn(self.stage_out_channels[3],
                                  outout_channel, 1, 1, 0, act)
        if with_pool:
            self.globalpool = AdaptiveAvgPool2D(1)

        # building classifier
        self._fc = Linear(outout_channel, num_classes)

    def forward(self, x):
        x = self._conv1(x)
        x = self.maxpool(x)

        for i in range(len(self.stages)):
            for block in range(self.stages_repeats[i]):
                x = self.shuffle_block_list[i][block](x)

        x = self._last_conv(x)
        x = self.globalpool(x)
        x = paddle.flatten(x, start_axis=1, stop_axis=-1)
        x = self._fc(x)
        return x


def _shufflenetv2(arch, scale, pretrained=False, act="relu", **kwargs):
    model = ShuffleNetV2(scale, act, **kwargs)
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


def shufflenetv2_x0_25(pretrained=False, **kwargs):
    """ShuffleNetV2 x0.25 model

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet. Default: False.

    Examples:
        .. code-block:: python
            from paddle.vision.models import shufflenetv2_x0_25

            # build model
            model = shufflenetv2_x0_25()

            # build model and load imagenet pretrained weight
            # model = shufflenetv2_x0_25(pretrained=True)
    """
    return _shufflenetv2('shufflenetv2_x0.25', 0.25, pretrained, **kwargs)


def shufflenetv2_x0_33(pretrained=False, **kwargs):
    """ShuffleNetV2 x0.33 model

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet. Default: False.

    Examples:
        .. code-block:: python
            from paddle.vision.models import shufflenetv2_x0_33

            # build model
            model = shufflenetv2_x0_33()

            # build model and load imagenet pretrained weight
            # model = shufflenetv2_x0_33(pretrained=True)
    """
    return _shufflenetv2('shufflenetv2_x0.33', 0.33, pretrained, **kwargs)


def shufflenetv2_x0_5(pretrained=False, **kwargs):
    """ShuffleNetV2 x0.5 model

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet. Default: False.

    Examples:
        .. code-block:: python
            from paddle.vision.models import shufflenetv2_x0_5

            # build model
            model = shufflenetv2_x0_5()

            # build model and load imagenet pretrained weight
            # model = shufflenetv2_x0_5(pretrained=True)
    """
    return _shufflenetv2('shufflenetv2_x0.5', 0.5, pretrained, **kwargs)


def shufflenetv2_x1_0(pretrained=False, **kwargs):
    """ShuffleNetV2 x1.0 model

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet. Default: False.

    Examples:
        .. code-block:: python
            from paddle.vision.models import shufflenetv2_x1_0

            # build model
            model = shufflenetv2_x1_0()

            # build model and load imagenet pretrained weight
            # model = shufflenetv2_x1_0(pretrained=True)
    """
    return _shufflenetv2('shufflenetv2_x1.0', 1.0, pretrained, **kwargs)


def shufflenetv2_x1_5(pretrained=False, **kwargs):
    """ShuffleNetV2 x1.5 model

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet. Default: False.

    Examples:
        .. code-block:: python
            from paddle.vision.models import shufflenetv2_x1_5

            # build model
            model = shufflenetv2_x1_5()

            # build model and load imagenet pretrained weight
            # model = shufflenetv2_x1_5(pretrained=True)
    """
    return _shufflenetv2('shufflenetv2_x1.5', 1.5, pretrained, **kwargs)


def shufflenetv2_x2_0(pretrained=False, **kwargs):
    """ShuffleNetV2 x2.0 model

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet. Default: False.

    Examples:
        .. code-block:: python
            from paddle.vision.models import shufflenetv2_x2_0

            # build model
            model = shufflenetv2_x2_0()

            # build model and load imagenet pretrained weight
            # model = shufflenetv2_x2_0(pretrained=True)
    """
    return _shufflenetv2('shufflenetv2_x2.0', 2.0, pretrained, **kwargs)


def shufflenetv2_swish(pretrained=False, **kwargs):
    """ShuffleNetV2 with 1.0x output channels and swish activation function, as described in
    `"ShuffleNetV2: Practical Guidelines for Ecient CNN Architecture Design" <https://arxiv.org/pdf/1807.11164.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet. Default: False.

    Examples:
        .. code-block:: python

            from paddle.vision.models import shufflenet_v2_swish

            # build model
            model = shufflenetv2_swish()

            # build model and load imagenet pretrained weight
            # model = shufflenetv2_swish(pretrained=True)

    """
    model = _shufflenetv2("shufflenetv2_swish",
                          scale=1.0,
                          pretrained=pretrained,
                          act="swish",
                          **kwargs)
    return model