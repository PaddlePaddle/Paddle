# copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import paddle
from paddle import ParamAttr, reshape, transpose, concat, split
from paddle.nn import Layer, Conv2D, MaxPool2D, AdaptiveAvgPool2D, BatchNorm, Linear
from paddle.nn.initializer import KaimingNormal
from paddle.nn.functional import swish

from paddle.utils.download import get_weights_path_from_url


MODEL_URLS = {
    "ShuffleNetV2_x0_25":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/ShuffleNetV2_x0_25_pretrained.pdparams",
    "ShuffleNetV2_x0_33":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/ShuffleNetV2_x0_33_pretrained.pdparams",
    "ShuffleNetV2_x0_5":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/ShuffleNetV2_x0_5_pretrained.pdparams",
    "ShuffleNetV2_x1_0":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/ShuffleNetV2_x1_0_pretrained.pdparams",
    "ShuffleNetV2_x1_5":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/ShuffleNetV2_x1_5_pretrained.pdparams",
    "ShuffleNetV2_x2_0":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/ShuffleNetV2_x2_0_pretrained.pdparams",
    "ShuffleNetV2_swish":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/ShuffleNetV2_swish_pretrained.pdparams"
}

__all__ = list(MODEL_URLS.keys())


def channel_shuffle(x, groups):
    batch_size, num_channels, height, width = x.shape[0:4]
    channels_per_group = num_channels // groups

    # reshape
    x = reshape(
        x=x, shape=[batch_size, groups, channels_per_group, height, width])

    # transpose
    x = transpose(x=x, perm=[0, 2, 1, 3, 4])

    # flatten
    x = reshape(x=x, shape=[batch_size, num_channels, height, width])
    return x


class ConvBNLayer(Layer):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            groups=1,
            act=None,
            name=None, ):
        super(ConvBNLayer, self).__init__()
        self._conv = Conv2D(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            weight_attr=ParamAttr(
                initializer=KaimingNormal(), name=name + "_weights"),
            bias_attr=False)

        self._batch_norm = BatchNorm(
            out_channels,
            param_attr=ParamAttr(name=name + "_bn_scale"),
            bias_attr=ParamAttr(name=name + "_bn_offset"),
            act=act,
            moving_mean_name=name + "_bn_mean",
            moving_variance_name=name + "_bn_variance")

    def forward(self, inputs):
        y = self._conv(inputs)
        y = self._batch_norm(y)
        return y


class InvertedResidual(Layer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 stride,
                 act="relu",
                 name=None):
        super(InvertedResidual, self).__init__()
        self._conv_pw = ConvBNLayer(
            in_channels=in_channels // 2,
            out_channels=out_channels // 2,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1,
            act=act,
            name='stage_' + name + '_conv1')
        self._conv_dw = ConvBNLayer(
            in_channels=out_channels // 2,
            out_channels=out_channels // 2,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=out_channels // 2,
            act=None,
            name='stage_' + name + '_conv2')
        self._conv_linear = ConvBNLayer(
            in_channels=out_channels // 2,
            out_channels=out_channels // 2,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1,
            act=act,
            name='stage_' + name + '_conv3')

    def forward(self, inputs):
        x1, x2 = split(
            inputs,
            num_or_sections=[inputs.shape[1] // 2, inputs.shape[1] // 2],
            axis=1)
        x2 = self._conv_pw(x2)
        x2 = self._conv_dw(x2)
        x2 = self._conv_linear(x2)
        out = concat([x1, x2], axis=1)
        return channel_shuffle(out, 2)


class InvertedResidualDS(Layer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 stride,
                 act="relu",
                 name=None):
        super(InvertedResidualDS, self).__init__()

        # branch1
        self._conv_dw_1 = ConvBNLayer(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=in_channels,
            act=None,
            name='stage_' + name + '_conv4')
        self._conv_linear_1 = ConvBNLayer(
            in_channels=in_channels,
            out_channels=out_channels // 2,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1,
            act=act,
            name='stage_' + name + '_conv5')
        # branch2
        self._conv_pw_2 = ConvBNLayer(
            in_channels=in_channels,
            out_channels=out_channels // 2,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1,
            act=act,
            name='stage_' + name + '_conv1')
        self._conv_dw_2 = ConvBNLayer(
            in_channels=out_channels // 2,
            out_channels=out_channels // 2,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=out_channels // 2,
            act=None,
            name='stage_' + name + '_conv2')
        self._conv_linear_2 = ConvBNLayer(
            in_channels=out_channels // 2,
            out_channels=out_channels // 2,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1,
            act=act,
            name='stage_' + name + '_conv3')

    def forward(self, inputs):
        x1 = self._conv_dw_1(inputs)
        x1 = self._conv_linear_1(x1)
        x2 = self._conv_pw_2(inputs)
        x2 = self._conv_dw_2(x2)
        x2 = self._conv_linear_2(x2)
        out = concat([x1, x2], axis=1)

        return channel_shuffle(out, 2)


class ShuffleNetV2(Layer):
    """ShuffleNetV2 model from
    `"ShuffleNet V2: Practical Guidelines for Ecient CNN Architecture Design" <https://arxiv.org/pdf/1807.11164.pdf>`_
    Args:
        num_classes (int): output dim of last fc layer. If num_classes <=0, last fc layer 
                            will not be defined. Default: 1000.
        scale (float): network architecture. Default: 1.0.
        act (str, optional): Activation to be applied to the output of batch normalization. Default: "relu".
    Examples:
        .. code-block:: python
            from paddle.vision.models import ShuffleNetV2

            shufflenetv2_x0_25 = ShuffleNetV2(num_classes=1000, scale=0.25, act="relu")
            shufflenetv2_swish = ShuffleNetV2(num_classes=1000, scale=1.0, act="swish")
    """    
    def __init__(self, num_classes=1000, scale=1.0, act="relu"):
        super(ShuffleNetV2, self).__init__()
        self.scale = scale
        self.num_classes = num_classes
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
            act=act,
            name='stage1_conv')
        self._max_pool = MaxPool2D(kernel_size=3, stride=2, padding=1)

        # 2. bottleneck sequences
        self._block_list = []
        for stage_id, num_repeat in enumerate(stage_repeats):
            for i in range(num_repeat):
                if i == 0:
                    block = self.add_sublayer(
                        name=str(stage_id + 2) + '_' + str(i + 1),
                        sublayer=InvertedResidualDS(
                            in_channels=stage_out_channels[stage_id + 1],
                            out_channels=stage_out_channels[stage_id + 2],
                            stride=2,
                            act=act,
                            name=str(stage_id + 2) + '_' + str(i + 1)))
                else:
                    block = self.add_sublayer(
                        name=str(stage_id + 2) + '_' + str(i + 1),
                        sublayer=InvertedResidual(
                            in_channels=stage_out_channels[stage_id + 2],
                            out_channels=stage_out_channels[stage_id + 2],
                            stride=1,
                            act=act,
                            name=str(stage_id + 2) + '_' + str(i + 1)))
                self._block_list.append(block)
        # 3. last_conv
        self._last_conv = ConvBNLayer(
            in_channels=stage_out_channels[-2],
            out_channels=stage_out_channels[-1],
            kernel_size=1,
            stride=1,
            padding=0,
            act=act,
            name='conv5')
        # 4. pool
        self._pool2d_avg = AdaptiveAvgPool2D(1)
        self._out_c = stage_out_channels[-1]
        # 5. fc
        self._fc = Linear(
            stage_out_channels[-1],
            num_classes,
            weight_attr=ParamAttr(name='fc6_weights'),
            bias_attr=ParamAttr(name='fc6_offset'))

    def forward(self, inputs):
        y = self._conv1(inputs)
        y = self._max_pool(y)
        for inv in self._block_list:
            y = inv(y)
        y = self._last_conv(y)
        y = self._pool2d_avg(y)
        y = paddle.flatten(y, start_axis=1, stop_axis=-1)
        y = self._fc(y)
        return y


def shufflenetv2_x0_25(pretrained=False, **kwargs):
    """ShuffleNetV2_x0_25 model
    
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    
    Examples:
        .. code-block:: python
            from paddle.vision.models import shufflenetv2_x0_25
            # build model
            model = shufflenetv2_x0_25()
            # build model and load imagenet pretrained weight
            # model = shufflenetv2_x0_25(pretrained=True)
    """     
    model = ShuffleNetV2(scale=0.25, **kwargs)
    if pretrained:
        weight_path = get_weights_path_from_url(MODEL_URLS["ShuffleNetV2_x0_25"])
        param = paddle.load(weight_path)
        model.set_dict(param)              
    return model


def shufflenetv2_x0_33(pretrained=False, **kwargs):
    """ShuffleNetV2_x0_33 model
    
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    
    Examples:
        .. code-block:: python
            from paddle.vision.models import shufflenetv2_x0_33
            # build model
            model = shufflenetv2_x0_33()
            # build model and load imagenet pretrained weight
            # model = shufflenetv2_x0_33(pretrained=True)
    """      
    model = ShuffleNetV2(scale=0.33, **kwargs)
    if pretrained:
        weight_path = get_weights_path_from_url(MODEL_URLS["ShuffleNetV2_x0_33"])
        param = paddle.load(weight_path)
        model.set_dict(param)            
    return model


def shufflenetv2_x0_5(pretrained=False, **kwargs):
    """ShuffleNetV2_x0_5 model
    
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    
    Examples:
        .. code-block:: python
            from paddle.vision.models import shufflenetv2_x0_5
            # build model
            model = shufflenetv2_x0_5()
            # build model and load imagenet pretrained weight
            # model = shufflenetv2_x0_5(pretrained=True)
    """        
    model = ShuffleNetV2(scale=0.5, **kwargs)
    if pretrained:
        weight_path = get_weights_path_from_url(MODEL_URLS["ShuffleNetV2_x0_5"])
        param = paddle.load(weight_path)
        model.set_dict(param)    
    return model


def shufflenetv2_x1_0(pretrained=False, **kwargs):
    """ShuffleNetV2_x1_0 model
    
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    
    Examples:
        .. code-block:: python
            from paddle.vision.models import shufflenetv2_x1_0
            # build model
            model = shufflenetv2_x1_0()
            # build model and load imagenet pretrained weight
            # model = shufflenetv2_x1_0(pretrained=True)
    """     
    model = ShuffleNetV2(scale=1.0, **kwargs)
    if pretrained:
        weight_path = get_weights_path_from_url(MODEL_URLS["ShuffleNetV2_x1_0"])
        param = paddle.load(weight_path)
        model.set_dict(param)    

    return model


def shufflenetv2_x1_5(pretrained=False, **kwargs):
    """ShuffleNetV2_x1_5 model
    
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    
    Examples:
        .. code-block:: python
            from paddle.vision.models import shufflenetv2_x1_5
            # build model
            model = shufflenetv2_x1_5()
            # build model and load imagenet pretrained weight
            # model = shufflenetv2_x1_5(pretrained=True)
    """       
    model = ShuffleNetV2(scale=1.5, **kwargs)
    if pretrained:
        weight_path = get_weights_path_from_url(MODEL_URLS["ShuffleNetV2_x1_5"])
        param = paddle.load(weight_path)
        model.set_dict(param)

    return model


def shufflenetv2_x2_0(pretrained=False, **kwargs):
    """ShuffleNetV2_x2_0 model
    
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    
    Examples:
        .. code-block:: python
            from paddle.vision.models import shufflenetv2_x2_0
            # build model
            model = shufflenetv2_x2_0()
            # build model and load imagenet pretrained weight
            # model = shufflenetv2_x2_0(pretrained=True)
    """    
    model = ShuffleNetV2(scale=2.0, **kwargs)
    if pretrained:
        weight_path = get_weights_path_from_url(MODEL_URLS["ShuffleNetV2_x2_0"])
        param = paddle.load(weight_path)
        model.set_dict(param)

    return model


def shufflenetv2_swish(pretrained=False, **kwargs):
    """ShuffleNetV2_swish model
    
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    
    Examples:
        .. code-block:: python
            from paddle.vision.models import shufflenetv2_swish
            # build model
            model = shufflenetv2_swish()
            # build model and load imagenet pretrained weight
            # model = shufflenetv2_swish(pretrained=True)
    """
    model = ShuffleNetV2(scale=1.0, act="swish", **kwargs)
    if pretrained:
        weight_path = get_weights_path_from_url(MODEL_URLS["ShuffleNetV2_swish"])
        param = paddle.load(weight_path)
        model.set_dict(param)

    return model
