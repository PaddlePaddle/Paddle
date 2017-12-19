# Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved
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
import math

from activations import LinearActivation, ReluActivation, SoftmaxActivation, \
    IdentityActivation, TanhActivation, SequenceSoftmaxActivation
from attrs import ExtraAttr
from default_decorators import wrap_name_default, wrap_act_default, \
    wrap_param_default, wrap_bias_attr_default, wrap_param_attr_default
from layers import *  # There are too many layers used in network, so import *
from poolings import MaxPooling, SumPooling
from paddle.trainer.config_parser import *

__all__ = [
    'sequence_conv_pool', 'simple_lstm', "simple_img_conv_pool",
    "img_conv_bn_pool", 'lstmemory_group', 'lstmemory_unit', 'small_vgg',
    'img_conv_group', 'img_separable_conv', 'vgg_16_network', 'gru_unit',
    'gru_group', 'simple_gru', 'simple_attention', 'dot_product_attention',
    'multi_head_attention', 'simple_gru2', 'bidirectional_gru',
    'text_conv_pool', 'bidirectional_lstm', 'inputs', 'outputs'
]

######################################################
#                     Text CNN                       #
######################################################


@wrap_name_default("sequence_conv_pooling")
def sequence_conv_pool(input,
                       context_len,
                       hidden_size,
                       name=None,
                       context_start=None,
                       pool_type=None,
                       context_proj_layer_name=None,
                       context_proj_param_attr=False,
                       fc_layer_name=None,
                       fc_param_attr=None,
                       fc_bias_attr=None,
                       fc_act=None,
                       pool_bias_attr=None,
                       fc_attr=None,
                       context_attr=None,
                       pool_attr=None):
    """
    Text convolution pooling group.

    Text input => Context Projection => FC Layer => Pooling => Output.

    :param name: group name.
    :type name: basestring
    :param input: input layer.
    :type input: LayerOutput
    :param context_len: context projection length. See
                        context_projection's document.
    :type context_len: int
    :param hidden_size: FC Layer size.
    :type hidden_size: int
    :param context_start: context start position. See
                          context_projection's context_start.
    :type context_start: int|None
    :param pool_type: pooling layer type. See pooling_layer's document.
    :type pool_type: BasePoolingType
    :param context_proj_layer_name: context projection layer name.
                                    None if user don't care.
    :type context_proj_layer_name: basestring
    :param context_proj_param_attr: padding parameter attribute of context projection layer.
                                    If false, it means padding always be zero.
    :type context_proj_param_attr: ParameterAttribute|None
    :param fc_layer_name: fc layer name. None if user don't care.
    :type fc_layer_name: basestring
    :param fc_param_attr: fc layer parameter attribute. None if user don't care.
    :type fc_param_attr: ParameterAttribute|None
    :param fc_bias_attr: fc bias parameter attribute. False if no bias,
                         None if user don't care.
    :type fc_bias_attr: ParameterAttribute|False|None
    :param fc_act: fc layer activation type. None means tanh.
    :type fc_act: BaseActivation
    :param pool_bias_attr: pooling layer bias attr. False if no bias.
                           None if user don't care.
    :type pool_bias_attr: ParameterAttribute|False|None
    :param fc_attr: fc layer extra attribute.
    :type fc_attr: ExtraLayerAttribute
    :param context_attr: context projection layer extra attribute.
    :type context_attr: ExtraLayerAttribute
    :param pool_attr: pooling layer extra attribute.
    :type pool_attr: ExtraLayerAttribute
    :return: layer's output.
    :rtype: LayerOutput
    """
    # Set Default Value to param
    context_proj_layer_name = "%s_conv_proj" % name \
        if context_proj_layer_name is None else context_proj_layer_name

    with mixed_layer(
            name=context_proj_layer_name,
            size=input.size * context_len,
            act=LinearActivation(),
            layer_attr=context_attr) as m:
        m += context_projection(
            input,
            context_len=context_len,
            context_start=context_start,
            padding_attr=context_proj_param_attr)

    fc_layer_name = "%s_conv_fc" % name \
        if fc_layer_name is None else fc_layer_name
    fl = fc_layer(
        name=fc_layer_name,
        input=m,
        size=hidden_size,
        act=fc_act,
        layer_attr=fc_attr,
        param_attr=fc_param_attr,
        bias_attr=fc_bias_attr)

    return pooling_layer(
        name=name,
        input=fl,
        pooling_type=pool_type,
        bias_attr=pool_bias_attr,
        layer_attr=pool_attr)


text_conv_pool = sequence_conv_pool

############################################################################
#                       Images                                             #
############################################################################


@wrap_name_default("conv_pool")
def simple_img_conv_pool(input,
                         filter_size,
                         num_filters,
                         pool_size,
                         name=None,
                         pool_type=None,
                         act=None,
                         groups=1,
                         conv_stride=1,
                         conv_padding=0,
                         bias_attr=None,
                         num_channel=None,
                         param_attr=None,
                         shared_bias=True,
                         conv_layer_attr=None,
                         pool_stride=1,
                         pool_padding=0,
                         pool_layer_attr=None):
    """
    Simple image convolution and pooling group.

    Img input => Conv => Pooling => Output.

    :param name: group name.
    :type name: basestring
    :param input: input layer.
    :type input: LayerOutput
    :param filter_size: see img_conv_layer for details.
    :type filter_size: int
    :param num_filters: see img_conv_layer for details.
    :type num_filters: int
    :param pool_size: see img_pool_layer for details.
    :type pool_size: int
    :param pool_type: see img_pool_layer for details.
    :type pool_type: BasePoolingType
    :param act: see img_conv_layer for details.
    :type act: BaseActivation
    :param groups: see img_conv_layer for details.
    :type groups: int
    :param conv_stride: see img_conv_layer for details.
    :type conv_stride: int
    :param conv_padding: see img_conv_layer for details.
    :type conv_padding: int
    :param bias_attr: see img_conv_layer for details.
    :type bias_attr: ParameterAttribute
    :param num_channel: see img_conv_layer for details.
    :type num_channel: int
    :param param_attr: see img_conv_layer for details.
    :type param_attr: ParameterAttribute
    :param shared_bias: see img_conv_layer for details.
    :type shared_bias: bool
    :param conv_layer_attr: see img_conv_layer for details.
    :type conv_layer_attr: ExtraLayerAttribute
    :param pool_stride: see img_pool_layer for details.
    :type pool_stride: int
    :param pool_padding: see img_pool_layer for details.
    :type pool_padding: int
    :param pool_layer_attr: see img_pool_layer for details.
    :type pool_layer_attr: ExtraLayerAttribute
    :return: layer's output
    :rtype: LayerOutput
    """
    _conv_ = img_conv_layer(
        name="%s_conv" % name,
        input=input,
        filter_size=filter_size,
        num_filters=num_filters,
        num_channels=num_channel,
        act=act,
        groups=groups,
        stride=conv_stride,
        padding=conv_padding,
        bias_attr=bias_attr,
        param_attr=param_attr,
        shared_biases=shared_bias,
        layer_attr=conv_layer_attr)
    return img_pool_layer(
        name="%s_pool" % name,
        input=_conv_,
        pool_size=pool_size,
        pool_type=pool_type,
        stride=pool_stride,
        padding=pool_padding,
        layer_attr=pool_layer_attr)


@wrap_name_default("conv_bn_pool")
def img_conv_bn_pool(input,
                     filter_size,
                     num_filters,
                     pool_size,
                     name=None,
                     pool_type=None,
                     act=None,
                     groups=1,
                     conv_stride=1,
                     conv_padding=0,
                     conv_bias_attr=None,
                     num_channel=None,
                     conv_param_attr=None,
                     shared_bias=True,
                     conv_layer_attr=None,
                     bn_param_attr=None,
                     bn_bias_attr=None,
                     bn_layer_attr=None,
                     pool_stride=1,
                     pool_padding=0,
                     pool_layer_attr=None):
    """
    Convolution, batch normalization, pooling group.

    Img input => Conv => BN => Pooling => Output.

    :param name: group name.
    :type name: basestring
    :param input: input layer.
    :type input: LayerOutput
    :param filter_size: see img_conv_layer for details.
    :type filter_size: int
    :param num_filters: see img_conv_layer for details.
    :type num_filters: int
    :param pool_size: see img_pool_layer for details.
    :type pool_size: int
    :param pool_type: see img_pool_layer for details.
    :type pool_type: BasePoolingType
    :param act: see batch_norm_layer for details.
    :type act: BaseActivation
    :param groups: see img_conv_layer for details.
    :type groups: int
    :param conv_stride: see img_conv_layer for details.
    :type conv_stride: int
    :param conv_padding: see img_conv_layer for details.
    :type conv_padding: int
    :param conv_bias_attr: see img_conv_layer for details.
    :type conv_bias_attr: ParameterAttribute
    :param num_channel: see img_conv_layer for details.
    :type num_channel: int
    :param conv_param_attr: see img_conv_layer for details.
    :type conv_param_attr: ParameterAttribute
    :param shared_bias: see img_conv_layer for details.
    :type shared_bias: bool
    :param conv_layer_attr: see img_conv_layer for details.
    :type conv_layer_attr: ExtraLayerOutput
    :param bn_param_attr: see batch_norm_layer for details.
    :type bn_param_attr: ParameterAttribute
    :param bn_bias_attr: see batch_norm_layer for details.
    :type bn_bias_attr: ParameterAttribute
    :param bn_layer_attr: see batch_norm_layer for details.
    :type bn_layer_attr: ExtraLayerAttribute
    :param pool_stride: see img_pool_layer for details.
    :type pool_stride: int
    :param pool_padding: see img_pool_layer for details.
    :type pool_padding: int
    :param pool_layer_attr: see img_pool_layer for details.
    :type pool_layer_attr: ExtraLayerAttribute
    :return: layer's output
    :rtype: LayerOutput
    """
    __conv__ = img_conv_layer(
        name="%s_conv" % name,
        input=input,
        filter_size=filter_size,
        num_filters=num_filters,
        num_channels=num_channel,
        act=LinearActivation(),
        groups=groups,
        stride=conv_stride,
        padding=conv_padding,
        bias_attr=conv_bias_attr,
        param_attr=conv_param_attr,
        shared_biases=shared_bias,
        layer_attr=conv_layer_attr)
    __bn__ = batch_norm_layer(
        name="%s_bn" % name,
        input=__conv__,
        act=act,
        bias_attr=bn_bias_attr,
        param_attr=bn_param_attr,
        layer_attr=bn_layer_attr)
    return img_pool_layer(
        name="%s_pool" % name,
        input=__bn__,
        pool_type=pool_type,
        pool_size=pool_size,
        stride=pool_stride,
        padding=pool_padding,
        layer_attr=pool_layer_attr)


@wrap_act_default(param_names=['conv_act'], act=ReluActivation())
@wrap_param_default(
    param_names=['pool_type'], default_factory=lambda _: MaxPooling())
def img_conv_group(input,
                   conv_num_filter,
                   pool_size,
                   num_channels=None,
                   conv_padding=1,
                   conv_filter_size=3,
                   conv_act=None,
                   conv_with_batchnorm=False,
                   conv_batchnorm_drop_rate=0,
                   pool_stride=1,
                   pool_type=None,
                   param_attr=None):
    """
    Image Convolution Group, Used for vgg net.

    :param conv_batchnorm_drop_rate: if conv_with_batchnorm[i] is true,
        conv_batchnorm_drop_rate[i] represents the drop rate of each batch norm.
    :type conv_batchnorm_drop_rate: list
    :param input: input layer.
    :type input: LayerOutput
    :param conv_num_filter: list of output channels num.
    :type conv_num_filter: list|tuple
    :param pool_size: pooling filter size.
    :type pool_size: int
    :param num_channels: input channels num.
    :type num_channels: int
    :param conv_padding: convolution padding size.
    :type conv_padding: int
    :param conv_filter_size: convolution filter size.
    :type conv_filter_size: int
    :param conv_act: activation funciton after convolution.
    :type conv_act: BaseActivation
    :param conv_with_batchnorm: if conv_with_batchnorm[i] is true,
        there is a batch normalization operation after each convolution.
    :type conv_with_batchnorm: list
    :param pool_stride: pooling stride size.
    :type pool_stride: int
    :param pool_type: pooling type.
    :type pool_type: BasePoolingType
    :param param_attr: param attribute of convolution layer,
                       None means default attribute.
    :type param_attr: ParameterAttribute
    :return: layer's output
    :rtype: LayerOutput
    """
    tmp = input

    # Type checks
    assert isinstance(tmp, LayerOutput)
    assert isinstance(conv_num_filter, list) or isinstance(conv_num_filter,
                                                           tuple)
    for each_num_filter in conv_num_filter:
        assert isinstance(each_num_filter, int)

    assert isinstance(pool_size, int)

    def __extend_list__(obj):
        if not hasattr(obj, '__len__'):
            return [obj] * len(conv_num_filter)
        else:
            return obj

    conv_padding = __extend_list__(conv_padding)
    conv_filter_size = __extend_list__(conv_filter_size)
    conv_act = __extend_list__(conv_act)
    conv_with_batchnorm = __extend_list__(conv_with_batchnorm)
    conv_batchnorm_drop_rate = __extend_list__(conv_batchnorm_drop_rate)

    for i in xrange(len(conv_num_filter)):
        extra_kwargs = dict()
        if num_channels is not None:
            extra_kwargs['num_channels'] = num_channels
            num_channels = None
        if conv_with_batchnorm[i]:
            extra_kwargs['act'] = LinearActivation()
        else:
            extra_kwargs['act'] = conv_act[i]

        tmp = img_conv_layer(
            input=tmp,
            padding=conv_padding[i],
            filter_size=conv_filter_size[i],
            num_filters=conv_num_filter[i],
            param_attr=param_attr,
            **extra_kwargs)

        # logger.debug("tmp.num_filters = %d" % tmp.num_filters)

        if conv_with_batchnorm[i]:
            dropout = conv_batchnorm_drop_rate[i]
            if dropout == 0 or abs(dropout) < 1e-5:  # dropout not set
                tmp = batch_norm_layer(input=tmp, act=conv_act[i])
            else:
                tmp = batch_norm_layer(
                    input=tmp,
                    act=conv_act[i],
                    layer_attr=ExtraAttr(drop_rate=dropout))

    return img_pool_layer(
        input=tmp, stride=pool_stride, pool_size=pool_size, pool_type=pool_type)


@wrap_name_default("separable_conv")
def img_separable_conv(input,
                       num_channels,
                       num_out_channels,
                       filter_size,
                       stride=1,
                       padding=0,
                       depth_multiplier=1,
                       act=None,
                       bias_attr=None,
                       param_attr=None,
                       shared_bias=True,
                       layer_type='exconv',
                       name=None):
    """
    Separable Convolution.

    The separable convolution module is consisted of a depthwise convolution
    that acts separately on input channels, followed by a pointwise convolution
    with 1*1 kernels that mixes channels. It is used for Xception:
    https://arxiv.org/pdf/1610.02357.pdf

    :param input: input layer.
    :type input: LayerOutput
    :param num_channels: the number of input channels.
    :type num_channels: int
    :param num_out_channels: the number of output channels.
    :type num_out_channels: int
    :param filter_size: the filter size for the depthwise convolution.
    :type filter_size: int|tuple
    :param stride: the stride size for the depthwise convolution.
    :type stride: int|tuple
    :param padding: the padding size for the depthwise convolution.
    :type padding: int|tuple
    :param depth_multiplier: the number of filter for one channel in the
                             depthwize convolution.
    :type depth_multiplier: int
    :param act: the activation function for the output.
    :type act: BaseActivation
    :param bias_attr: see img_conv_layer for details.
    :type bias_attr: ParameterAttribute
    :param param_attr: see img_conv_layer for details.
    :type param_attr: ParameterAttribute
    :param shared_bias: see img_conv_layer for details.
    :type shared_bias: bool
    :param layer_type: see img_conv_layer for details.
    :type layer_type: bool
    :return: layer's output
    :rtype: LayerOutput
    """
    __depthwise_conv__ = img_conv_layer(
        name="%s_depthwise_conv" % name,
        input=input,
        num_channels=num_channels,
        num_filters=num_channels * depth_multiplier,
        groups=num_channels,
        filter_size=filter_size,
        stride=stride,
        padding=padding,
        act=LinearActivation(),
        bias_attr=bias_attr,
        param_attr=param_attr,
        shared_biases=shared_bias,
        layer_type=layer_type)
    __pointwise_conv__ = img_conv_layer(
        name="%s_pointwise_conv" % name,
        input=__depthwise_conv__,
        num_channels=num_channels * depth_multiplier,
        num_filters=num_out_channels,
        filter_size=1,
        stride=1,
        padding=0,
        act=act,
        bias_attr=bias_attr,
        param_attr=param_attr,
        shared_biases=shared_bias)
    return __pointwise_conv__


def small_vgg(input_image, num_channels, num_classes):
    def __vgg__(ipt, num_filter, times, dropouts, num_channels_=None):
        return img_conv_group(
            input=ipt,
            num_channels=num_channels_,
            pool_size=2,
            pool_stride=2,
            conv_num_filter=[num_filter] * times,
            conv_filter_size=3,
            conv_act=ReluActivation(),
            conv_with_batchnorm=True,
            conv_batchnorm_drop_rate=dropouts,
            pool_type=MaxPooling())

    tmp = __vgg__(input_image, 64, 2, [0.3, 0], num_channels)
    tmp = __vgg__(tmp, 128, 2, [0.4, 0])
    tmp = __vgg__(tmp, 256, 3, [0.4, 0.4, 0])
    tmp = __vgg__(tmp, 512, 3, [0.4, 0.4, 0])
    tmp = img_pool_layer(
        input=tmp, stride=2, pool_size=2, pool_type=MaxPooling())
    tmp = dropout_layer(input=tmp, dropout_rate=0.5)
    tmp = fc_layer(
        input=tmp,
        size=512,
        layer_attr=ExtraAttr(drop_rate=0.5),
        act=LinearActivation())
    tmp = batch_norm_layer(input=tmp, act=ReluActivation())
    return fc_layer(input=tmp, size=num_classes, act=SoftmaxActivation())


def vgg_16_network(input_image, num_channels, num_classes=1000):
    """
    Same model from https://gist.github.com/ksimonyan/211839e770f7b538e2d8

    :param num_classes: number of class.
    :type num_classes: int
    :param input_image: input layer.
    :type input_image: LayerOutput
    :param num_channels: input channels num.
    :type num_channels: int
    :return: layer's output
    :rtype: LayerOutput
    """

    tmp = img_conv_group(
        input=input_image,
        num_channels=num_channels,
        conv_padding=1,
        conv_num_filter=[64, 64],
        conv_filter_size=3,
        conv_act=ReluActivation(),
        pool_size=2,
        pool_stride=2,
        pool_type=MaxPooling())

    tmp = img_conv_group(
        input=tmp,
        conv_num_filter=[128, 128],
        conv_padding=1,
        conv_filter_size=3,
        conv_act=ReluActivation(),
        pool_stride=2,
        pool_type=MaxPooling(),
        pool_size=2)

    tmp = img_conv_group(
        input=tmp,
        conv_num_filter=[256, 256, 256],
        conv_padding=1,
        conv_filter_size=3,
        conv_act=ReluActivation(),
        pool_stride=2,
        pool_type=MaxPooling(),
        pool_size=2)

    tmp = img_conv_group(
        input=tmp,
        conv_num_filter=[512, 512, 512],
        conv_padding=1,
        conv_filter_size=3,
        conv_act=ReluActivation(),
        pool_stride=2,
        pool_type=MaxPooling(),
        pool_size=2)
    tmp = img_conv_group(
        input=tmp,
        conv_num_filter=[512, 512, 512],
        conv_padding=1,
        conv_filter_size=3,
        conv_act=ReluActivation(),
        pool_stride=2,
        pool_type=MaxPooling(),
        pool_size=2)

    tmp = fc_layer(
        input=tmp,
        size=4096,
        act=ReluActivation(),
        layer_attr=ExtraAttr(drop_rate=0.5))

    tmp = fc_layer(
        input=tmp,
        size=4096,
        act=ReluActivation(),
        layer_attr=ExtraAttr(drop_rate=0.5))

    return fc_layer(input=tmp, size=num_classes, act=SoftmaxActivation())


############################################################################
#                       Recurrent                                          #
############################################################################


@wrap_name_default("lstm")
def simple_lstm(input,
                size,
                name=None,
                reverse=False,
                mat_param_attr=None,
                bias_param_attr=None,
                inner_param_attr=None,
                act=None,
                gate_act=None,
                state_act=None,
                mixed_layer_attr=None,
                lstm_cell_attr=None):
    """
    Simple LSTM Cell.

    It just combines a mixed layer with fully_matrix_projection and a lstmemory
    layer. The simple lstm cell was implemented with follow equations.

    ..  math::

        i_t & = \\sigma(W_{xi}x_{t} + W_{hi}h_{t-1} + W_{ci}c_{t-1} + b_i)

        f_t & = \\sigma(W_{xf}x_{t} + W_{hf}h_{t-1} + W_{cf}c_{t-1} + b_f)

        c_t & = f_tc_{t-1} + i_t tanh (W_{xc}x_t+W_{hc}h_{t-1} + b_c)

        o_t & = \\sigma(W_{xo}x_{t} + W_{ho}h_{t-1} + W_{co}c_t + b_o)

        h_t & = o_t tanh(c_t)

    Please refer to **Generating Sequences With Recurrent Neural Networks** for more
    details about lstm. Link_ is here.

    .. _Link: http://arxiv.org/abs/1308.0850

    :param name: lstm layer name.
    :type name: basestring
    :param input: layer's input.
    :type input: LayerOutput
    :param size: lstm layer size.
    :type size: int
    :param reverse: process the input in a reverse order or not.
    :type reverse: bool
    :param mat_param_attr: parameter attribute of matrix projection in mixed layer.
    :type mat_param_attr: ParameterAttribute
    :param bias_param_attr: bias parameter attribute. False means no bias, None
                            means default bias.
    :type bias_param_attr: ParameterAttribute|False
    :param inner_param_attr: parameter attribute of lstm cell.
    :type inner_param_attr: ParameterAttribute
    :param act: last activiation type of lstm.
    :type act: BaseActivation
    :param gate_act: gate activiation type of lstm.
    :type gate_act: BaseActivation
    :param state_act: state activiation type of lstm.
    :type state_act: BaseActivation
    :param mixed_layer_attr: extra attribute of mixed layer.
    :type mixed_layer_attr: ExtraLayerAttribute
    :param lstm_cell_attr: extra attribute of lstm.
    :type lstm_cell_attr: ExtraLayerAttribute
    :return: layer's output.
    :rtype: LayerOutput
    """
    fc_name = 'lstm_transform_%s' % name
    with mixed_layer(
            name=fc_name,
            size=size * 4,
            act=IdentityActivation(),
            layer_attr=mixed_layer_attr,
            bias_attr=False) as m:
        m += full_matrix_projection(input, param_attr=mat_param_attr)

    return lstmemory(
        name=name,
        input=m,
        reverse=reverse,
        bias_attr=bias_param_attr,
        param_attr=inner_param_attr,
        act=act,
        gate_act=gate_act,
        state_act=state_act,
        layer_attr=lstm_cell_attr)


@wrap_name_default('lstm_unit')
def lstmemory_unit(input,
                   out_memory=None,
                   name=None,
                   size=None,
                   param_attr=None,
                   act=None,
                   gate_act=None,
                   state_act=None,
                   input_proj_bias_attr=None,
                   input_proj_layer_attr=None,
                   lstm_bias_attr=None,
                   lstm_layer_attr=None):
    """
    lstmemory_unit defines the caculation process of a LSTM unit during a
    single time step. This function is not a recurrent layer, so it can not be
    directly used to process sequence input. This function is always used in
    recurrent_group (see layers.py for more details) to implement attention
    mechanism.

    Please refer to  **Generating Sequences With Recurrent Neural Networks**
    for more details about LSTM. The link goes as follows:
    .. _Link: https://arxiv.org/abs/1308.0850

    ..  math::

        i_t & = \\sigma(W_{x_i}x_{t} + W_{h_i}h_{t-1} + W_{c_i}c_{t-1} + b_i)

        f_t & = \\sigma(W_{x_f}x_{t} + W_{h_f}h_{t-1} + W_{c_f}c_{t-1} + b_f)

        c_t & = f_tc_{t-1} + i_t tanh (W_{x_c}x_t+W_{h_c}h_{t-1} + b_c)

        o_t & = \\sigma(W_{x_o}x_{t} + W_{h_o}h_{t-1} + W_{c_o}c_t + b_o)

        h_t & = o_t tanh(c_t)

    The example usage is:

    ..  code-block:: python

        lstm_step = lstmemory_unit(input=[layer1],
                                   size=256,
                                   act=TanhActivation(),
                                   gate_act=SigmoidActivation(),
                                   state_act=TanhActivation())


    :param input: Input layer.
    :type input: LayerOutput
    :param out_memory: The output of previous time step.
    :type out_memory: LayerOutput | None
    :param name: The lstmemory unit name.
    :type name: basestring
    :param size: The lstmemory unit size.
    :type size: int
    :param param_attr: The parameter attribute for the weights in
                     input to hidden projection.
                     None means default attribute.
    :type param_attr: ParameterAttribute
    :param act: The last activiation type of lstm.
    :type act: BaseActivation
    :param gate_act: The gate activiation type of lstm.
    :type gate_act: BaseActivation
    :param state_act: The state activiation type of lstm.
    :type state_act: BaseActivation
    :param input_proj_bias_attr: The parameter attribute for the bias in
                      input to hidden projection.
                      False or None means no bias.
                      If this parameter is set to True,
                      the bias is initialized to zero.
    :type input_proj_bias_attr: ParameterAttribute|bool|None
    :param input_proj_layer_attr: The extra layer attribute for
                     input to hidden projection of the LSTM unit,
                     such as dropout, error clipping.
    :type input_proj_layer_attr: ExtraLayerAttribute
    :param lstm_bias_attr: The parameter attribute for the bias in lstm layer.
                      False or None means no bias.
                      If this parameter is set to True,
                      the bias is initialized to zero.
    :type lstm_bias_attr: ParameterAttribute|True|None
    :param lstm_layer_attr: The extra attribute of lstm layer.
    :type lstm_layer_attr: ExtraLayerAttribute
    :return: The lstmemory unit name.
    :rtype: LayerOutput
    """
    if size is None:
        assert input.size % 4 == 0
        size = input.size / 4
    if out_memory is None:
        out_mem = memory(name=name, size=size)
    else:
        out_mem = out_memory

    state_mem = memory(name="%s_state" % name, size=size)

    with mixed_layer(
            name="%s_input_recurrent" % name,
            size=size * 4,
            bias_attr=input_proj_bias_attr,
            layer_attr=input_proj_layer_attr,
            act=IdentityActivation()) as m:
        m += identity_projection(input=input)
        m += full_matrix_projection(input=out_mem, param_attr=param_attr)

    lstm_out = lstm_step_layer(
        name=name,
        input=m,
        state=state_mem,
        size=size,
        bias_attr=lstm_bias_attr,
        act=act,
        gate_act=gate_act,
        state_act=state_act,
        layer_attr=lstm_layer_attr)
    get_output_layer(name='%s_state' % name, input=lstm_out, arg_name='state')

    return lstm_out


@wrap_name_default('lstm_group')
def lstmemory_group(input,
                    size=None,
                    name=None,
                    out_memory=None,
                    reverse=False,
                    param_attr=None,
                    act=None,
                    gate_act=None,
                    state_act=None,
                    input_proj_bias_attr=None,
                    input_proj_layer_attr=None,
                    lstm_bias_attr=None,
                    lstm_layer_attr=None):
    """
    lstm_group is a recurrent_group version of Long Short Term Memory. It
    does exactly the same calculation as the lstmemory layer (see lstmemory in
    layers.py for the maths) does. A promising benefit is that LSTM memory
    cell states(or hidden states) in every time step are accessible to the
    user. This is especially useful in attention model. If you do not need to
    access the internal states of the lstm and merely use its outputs,
    it is recommended to use the lstmemory, which is relatively faster than
    lstmemory_group.

    NOTE: In PaddlePaddle's implementation, the following input-to-hidden
    multiplications:
    :math:`W_{x_i}x_{t}` , :math:`W_{x_f}x_{t}`,
    :math:`W_{x_c}x_t`, :math:`W_{x_o}x_{t}` are not done in lstmemory_unit to
    speed up the calculations. Consequently, an additional mixed_layer with
    full_matrix_projection must be included before lstmemory_unit is called.

    The example usage is:

    ..  code-block:: python

        lstm_step = lstmemory_group(input=[layer1],
                                    size=256,
                                    act=TanhActivation(),
                                    gate_act=SigmoidActivation(),
                                    state_act=TanhActivation())

    :param input: Input layer.
    :type input: LayerOutput
    :param size: The lstmemory group size.
    :type size: int
    :param name: The name of lstmemory group.
    :type name: basestring
    :param out_memory: The output of previous time step.
    :type out_memory: LayerOutput | None
    :param reverse: Process the input in a reverse order or not.
    :type reverse: bool
    :param param_attr: The parameter attribute for the weights in
                     input to hidden projection.
                     None means default attribute.
    :type param_attr: ParameterAttribute
    :param act: The last activiation type of lstm.
    :type act: BaseActivation
    :param gate_act: The gate activiation type of lstm.
    :type gate_act: BaseActivation
    :param state_act: The state activiation type of lstm.
    :type state_act: BaseActivation
    :param input_proj_bias_attr: The parameter attribute for the bias in
                      input to hidden projection.
                      False or None means no bias.
                      If this parameter is set to True,
                      the bias is initialized to zero.
    :type input_proj_bias_attr: ParameterAttribute|bool|None
    :param input_proj_layer_attr: The extra layer attribute for
                     input to hidden projection of the LSTM unit,
                     such as dropout, error clipping.
    :type input_proj_layer_attr: ExtraLayerAttribute
    :param lstm_bias_attr: The parameter attribute for the bias in lstm layer.
                      False or None means no bias.
                      If this parameter is set to True,
                      the bias is initialized to zero.
    :type lstm_bias_attr: ParameterAttribute|True|None
    :param lstm_layer_attr: The extra attribute of lstm layer.
    :type lstm_layer_attr: ExtraLayerAttribute
    :return: the lstmemory group.
    :rtype: LayerOutput
    """

    def __lstm_step__(ipt):
        return lstmemory_unit(
            input=ipt,
            name=name,
            size=size,
            act=act,
            gate_act=gate_act,
            state_act=state_act,
            out_memory=out_memory,
            input_proj_bias_attr=input_proj_bias_attr,
            input_proj_layer_attr=input_proj_layer_attr,
            param_attr=param_attr,
            lstm_layer_attr=lstm_layer_attr,
            lstm_bias_attr=lstm_bias_attr)

    return recurrent_group(
        name='%s_recurrent_group' % name,
        step=__lstm_step__,
        reverse=reverse,
        input=input)


@wrap_name_default('gru_unit')
def gru_unit(input,
             memory_boot=None,
             size=None,
             name=None,
             gru_bias_attr=None,
             gru_param_attr=None,
             act=None,
             gate_act=None,
             gru_layer_attr=None,
             naive=False):
    """
    gru_unit defines the calculation process of a gated recurrent unit during a single
    time step. This function is not a recurrent layer, so it can not be
    directly used to process sequence input. This function is always used in
    the recurrent_group (see layers.py for more details) to implement attention
    mechanism.

    Please see grumemory in layers.py for the details about the maths.

    :param input: input layer.
    :type input: LayerOutput
    :param memory_boot: the initialization state of the LSTM cell.
    :type memory_boot: LayerOutput | None
    :param name: name of the gru group.
    :type name: basestring
    :param size: hidden size of the gru.
    :type size: int
    :param act: activation type of gru
    :type act: BaseActivation
    :param gate_act: gate activation type or gru
    :type gate_act: BaseActivation
    :param gru_layer_attr: Extra attribute of the gru layer.
    :type gru_layer_attr: ExtraLayerAttribute
    :return: the gru output layer.
    :rtype: LayerOutput
    """

    assert input.size % 3 == 0
    if size is None:
        size = input.size / 3

    out_mem = memory(name=name, size=size, boot_layer=memory_boot)

    if naive:
        __step__ = gru_step_naive_layer
    else:
        __step__ = gru_step_layer

    gru_out = __step__(
        name=name,
        input=input,
        output_mem=out_mem,
        size=size,
        bias_attr=gru_bias_attr,
        param_attr=gru_param_attr,
        act=act,
        gate_act=gate_act,
        layer_attr=gru_layer_attr)
    return gru_out


@wrap_name_default('gru_group')
def gru_group(input,
              memory_boot=None,
              size=None,
              name=None,
              reverse=False,
              gru_bias_attr=None,
              gru_param_attr=None,
              act=None,
              gate_act=None,
              gru_layer_attr=None,
              naive=False):
    """
    gru_group is a recurrent_group version of Gated Recurrent Unit. It
    does exactly the same calculation as the grumemory layer does. A promising
    benefit is that gru hidden states are accessible to the user. This is
    especially useful in attention model. If you do not need to access
    any internal state and merely use the outputs of a GRU, it is recommended
    to use the grumemory, which is relatively faster.

    Please see grumemory in layers.py for more detail about the maths.

    The example usage is:

    ..  code-block:: python

        gru = gru_group(input=[layer1],
                        size=256,
                        act=TanhActivation(),
                        gate_act=SigmoidActivation())

    :param input: input layer.
    :type input: LayerOutput
    :param memory_boot: the initialization state of the LSTM cell.
    :type memory_boot: LayerOutput | None
    :param name: name of the gru group.
    :type name: basestring
    :param size: hidden size of the gru.
    :type size: int
    :param reverse: process the input in a reverse order or not.
    :type reverse: bool
    :param act: activiation type of gru
    :type act: BaseActivation
    :param gate_act: gate activiation type of gru
    :type gate_act: BaseActivation
    :param gru_bias_attr: bias parameter attribute of gru layer,
                          False means no bias, None means default bias.
    :type gru_bias_attr: ParameterAttribute|False|None
    :param gru_layer_attr: Extra attribute of the gru layer.
    :type gru_layer_attr: ExtraLayerAttribute
    :return: the gru group.
    :rtype: LayerOutput
    """

    def __gru_step__(ipt):
        return gru_unit(
            input=ipt,
            memory_boot=memory_boot,
            name=name,
            size=size,
            gru_bias_attr=gru_bias_attr,
            gru_param_attr=gru_param_attr,
            act=act,
            gate_act=gate_act,
            gru_layer_attr=gru_layer_attr,
            naive=naive)

    return recurrent_group(
        name='%s_recurrent_group' % name,
        step=__gru_step__,
        reverse=reverse,
        input=input)


@wrap_name_default('simple_gru')
def simple_gru(input,
               size,
               name=None,
               reverse=False,
               mixed_param_attr=None,
               mixed_bias_param_attr=None,
               mixed_layer_attr=None,
               gru_bias_attr=None,
               gru_param_attr=None,
               act=None,
               gate_act=None,
               gru_layer_attr=None,
               naive=False):
    """
    You may see gru_step_layer, grumemory in layers.py, gru_unit, gru_group,
    simple_gru in network.py. The reason why there are so many interfaces is
    that we have two ways to implement recurrent neural network. One way is to
    use one complete layer to implement rnn (including simple rnn, gru and lstm)
    with multiple time steps, such as recurrent_layer, lstmemory, grumemory. But
    the multiplication operation :math:`W x_t` is not computed in these layers.
    See details in their interfaces in layers.py.
    The other implementation is to use an recurrent group which can ensemble a
    series of layers to compute rnn step by step. This way is flexible for
    attenion mechanism or other complex connections.

    - gru_step_layer: only compute rnn by one step. It needs an memory as input
      and can be used in recurrent group.
    - gru_unit: a wrapper of gru_step_layer with memory.
    - gru_group: a GRU cell implemented by a combination of multiple layers in
      recurrent group.
      But :math:`W x_t` is not done in group.
    - gru_memory: a GRU cell implemented by one layer, which does same calculation
      with gru_group and is faster than gru_group.
    - simple_gru: a complete GRU implementation inlcuding :math:`W x_t` and
      gru_group. :math:`W` contains :math:`W_r`, :math:`W_z` and :math:`W`, see
      formula in grumemory.

    The computational speed is that, grumemory is relatively better than
    gru_group, and gru_group is relatively better than simple_gru.

    The example usage is:

    ..  code-block:: python

        gru = simple_gru(input=[layer1], size=256)

    :param input: input layer.
    :type input: LayerOutput
    :param name: name of the gru group.
    :type name: basestring
    :param size: hidden size of the gru.
    :type size: int
    :param reverse: process the input in a reverse order or not.
    :type reverse: bool
    :param act: activiation type of gru
    :type act: BaseActivation
    :param gate_act: gate activiation type of gru
    :type gate_act: BaseActivation
    :param gru_bias_attr: bias parameter attribute of gru layer,
                          False means no bias, None means default bias.
    :type gru_bias_attr: ParameterAttribute|False|None
    :param gru_layer_attr: Extra attribute of the gru layer.
    :type gru_layer_attr: ExtraLayerAttribute
    :return: the gru group.
    :rtype: LayerOutput
    """
    with mixed_layer(
            name='%s_transform' % name,
            size=size * 3,
            bias_attr=mixed_bias_param_attr,
            layer_attr=mixed_layer_attr) as m:
        m += full_matrix_projection(input=input, param_attr=mixed_param_attr)

    return gru_group(
        name=name,
        size=size,
        input=m,
        reverse=reverse,
        gru_bias_attr=gru_bias_attr,
        gru_param_attr=gru_param_attr,
        act=act,
        gate_act=gate_act,
        gru_layer_attr=gru_layer_attr,
        naive=naive)


@wrap_name_default('simple_gru2')
def simple_gru2(input,
                size,
                name=None,
                reverse=False,
                mixed_param_attr=None,
                mixed_bias_attr=None,
                gru_param_attr=None,
                gru_bias_attr=None,
                act=None,
                gate_act=None,
                mixed_layer_attr=None,
                gru_cell_attr=None):
    """
    simple_gru2 is the same with simple_gru, but using grumemory instead.
    Please refer to grumemory in layers.py for more detail about the math.
    simple_gru2 is faster than simple_gru.

    The example usage is:

    ..  code-block:: python

        gru = simple_gru2(input=[layer1], size=256)

    :param input: input layer.
    :type input: LayerOutput
    :param name: name of the gru group.
    :type name: basestring
    :param size: hidden size of the gru.
    :type size: int
    :param reverse: process the input in a reverse order or not.
    :type reverse: bool
    :param act: activiation type of gru
    :type act: BaseActivation
    :param gate_act: gate activiation type of gru
    :type gate_act: BaseActivation
    :param gru_bias_attr: bias parameter attribute of gru layer,
                          False means no bias, None means default bias.
    :type gru_bias_attr: ParameterAttribute|False|None
    :param gru_param_attr: param parameter attribute of gru layer,
                          None means default param.
    :type gru_param_attr: ParameterAttribute|None
    :return: the gru group.
    :rtype: LayerOutput
    """
    with mixed_layer(
            name='%s_transform' % name,
            size=size * 3,
            bias_attr=mixed_bias_attr,
            layer_attr=mixed_layer_attr) as m:
        m += full_matrix_projection(input=input, param_attr=mixed_param_attr)

    return grumemory(
        name=name,
        input=m,
        reverse=reverse,
        bias_attr=gru_bias_attr,
        param_attr=gru_param_attr,
        act=act,
        gate_act=gate_act,
        layer_attr=gru_cell_attr)


@wrap_name_default("bidirectional_gru")
def bidirectional_gru(input,
                      size,
                      name=None,
                      return_seq=False,
                      fwd_mixed_param_attr=None,
                      fwd_mixed_bias_attr=None,
                      fwd_gru_param_attr=None,
                      fwd_gru_bias_attr=None,
                      fwd_act=None,
                      fwd_gate_act=None,
                      fwd_mixed_layer_attr=None,
                      fwd_gru_cell_attr=None,
                      bwd_mixed_param_attr=None,
                      bwd_mixed_bias_attr=None,
                      bwd_gru_param_attr=None,
                      bwd_gru_bias_attr=None,
                      bwd_act=None,
                      bwd_gate_act=None,
                      bwd_mixed_layer_attr=None,
                      bwd_gru_cell_attr=None,
                      last_seq_attr=None,
                      first_seq_attr=None,
                      concat_attr=None,
                      concat_act=None):
    """
    A bidirectional_gru is a recurrent unit that iterates over the input
    sequence both in forward and backward orders, and then concatenate two
    outputs to form a final output. However, concatenation of two outputs
    is not the only way to form the final output, you can also, for example,
    just add them together.

    The example usage is:

    ..  code-block:: python

        bi_gru = bidirectional_gru(input=[input1], size=512)

    :param name: bidirectional gru layer name.
    :type name: basestring
    :param input: input layer.
    :type input: LayerOutput
    :param size: gru layer size.
    :type size: int
    :param return_seq: If set False, the last time step of output are
                       concatenated and returned.
                       If set True, the entire output sequences in forward
                       and backward directions are concatenated and returned.
    :type return_seq: bool
    :return: LayerOutput object.
    :rtype: LayerOutput
    """
    args = locals()

    fw = simple_gru2(
        name='%s_fw' % name,
        input=input,
        size=size,
        **dict((k[len('fwd_'):], v) for k, v in args.iteritems()
               if k.startswith('fwd_')))

    bw = simple_gru2(
        name="%s_bw" % name,
        input=input,
        size=size,
        reverse=True,
        **dict((k[len('bwd_'):], v) for k, v in args.iteritems()
               if k.startswith('bwd_')))

    if return_seq:
        return concat_layer(
            name=name, input=[fw, bw], layer_attr=concat_attr, act=concat_act)
    else:
        fw_seq = last_seq(
            name="%s_fw_last" % name, input=fw, layer_attr=last_seq_attr)
        bw_seq = first_seq(
            name="%s_bw_last" % name, input=bw, layer_attr=first_seq_attr)
        return concat_layer(
            name=name,
            input=[fw_seq, bw_seq],
            layer_attr=concat_attr,
            act=concat_act)


@wrap_name_default("bidirectional_lstm")
def bidirectional_lstm(input,
                       size,
                       name=None,
                       return_seq=False,
                       fwd_mat_param_attr=None,
                       fwd_bias_param_attr=None,
                       fwd_inner_param_attr=None,
                       fwd_act=None,
                       fwd_gate_act=None,
                       fwd_state_act=None,
                       fwd_mixed_layer_attr=None,
                       fwd_lstm_cell_attr=None,
                       bwd_mat_param_attr=None,
                       bwd_bias_param_attr=None,
                       bwd_inner_param_attr=None,
                       bwd_act=None,
                       bwd_gate_act=None,
                       bwd_state_act=None,
                       bwd_mixed_layer_attr=None,
                       bwd_lstm_cell_attr=None,
                       last_seq_attr=None,
                       first_seq_attr=None,
                       concat_attr=None,
                       concat_act=None):
    """
    A bidirectional_lstm is a recurrent unit that iterates over the input
    sequence both in forward and backward orders, and then concatenate two
    outputs to form a final output. However, concatenation of two outputs
    is not the only way to form the final output, you can also, for example,
    just add them together.

    Please refer to  **Neural Machine Translation by Jointly Learning to Align
    and Translate** for more details about the bidirectional lstm.
    The link goes as follows:
    .. _Link: https://arxiv.org/pdf/1409.0473v3.pdf

    The example usage is:

    ..  code-block:: python

        bi_lstm = bidirectional_lstm(input=[input1], size=512)

    :param name: bidirectional lstm layer name.
    :type name: basestring
    :param input: input layer.
    :type input: LayerOutput
    :param size: lstm layer size.
    :type size: int
    :param return_seq: If set False, the last time step of output are
                       concatenated and returned.
                       If set True, the entire output sequences in forward
                       and backward directions are concatenated and returned.
    :type return_seq: bool
    :return: LayerOutput object.
    :rtype: LayerOutput
    """
    args = locals()

    fw = simple_lstm(
        name='%s_fw' % name,
        input=input,
        size=size,
        **dict((k[len('fwd_'):], v) for k, v in args.iteritems()
               if k.startswith('fwd_')))

    bw = simple_lstm(
        name="%s_bw" % name,
        input=input,
        size=size,
        reverse=True,
        **dict((k[len('bwd_'):], v) for k, v in args.iteritems()
               if k.startswith('bwd_')))

    if return_seq:
        return concat_layer(
            name=name, input=[fw, bw], layer_attr=concat_attr, act=concat_act)
    else:
        fw_seq = last_seq(
            name="%s_fw_last" % name, input=fw, layer_attr=last_seq_attr)
        bw_seq = first_seq(
            name="%s_bw_last" % name, input=bw, layer_attr=first_seq_attr)
        return concat_layer(
            name=name,
            input=[fw_seq, bw_seq],
            layer_attr=concat_attr,
            act=concat_act)


@wrap_name_default()
@wrap_act_default(param_names=['weight_act'], act=TanhActivation())
def simple_attention(encoded_sequence,
                     encoded_proj,
                     decoder_state,
                     transform_param_attr=None,
                     softmax_param_attr=None,
                     weight_act=None,
                     name=None):
    """
    Calculate and return a context vector with attention mechanism.
    Size of the context vector equals to size of the encoded_sequence.

    ..  math::

        a(s_{i-1},h_{j}) & = v_{a}f(W_{a}s_{t-1} + U_{a}h_{j})

        e_{i,j} & = a(s_{i-1}, h_{j})

        a_{i,j} & = \\frac{exp(e_{i,j})}{\\sum_{k=1}^{T_x}{exp(e_{i,k})}}

        c_{i} & = \\sum_{j=1}^{T_{x}}a_{i,j}h_{j}

    where :math:`h_{j}` is the jth element of encoded_sequence,
    :math:`U_{a}h_{j}` is the jth element of encoded_proj
    :math:`s_{i-1}` is decoder_state
    :math:`f` is weight_act, and is set to tanh by default.

    Please refer to **Neural Machine Translation by Jointly Learning to
    Align and Translate** for more details. The link is as follows:
    https://arxiv.org/abs/1409.0473.

    The example usage is:

    ..  code-block:: python

        context = simple_attention(encoded_sequence=enc_seq,
                                   encoded_proj=enc_proj,
                                   decoder_state=decoder_prev,)

    :param name: name of the attention model.
    :type name: basestring
    :param softmax_param_attr: parameter attribute of sequence softmax
                               that is used to produce attention weight.
    :type softmax_param_attr: ParameterAttribute
    :param weight_act: activation of the attention model.
    :type weight_act: BaseActivation
    :param encoded_sequence: output of the encoder
    :type encoded_sequence: LayerOutput
    :param encoded_proj: attention weight is computed by a feed forward neural
                         network which has two inputs : decoder's hidden state
                         of previous time step and encoder's output.
                         encoded_proj is output of the feed-forward network for
                         encoder's output. Here we pre-compute it outside
                         simple_attention for speed consideration.
    :type encoded_proj: LayerOutput
    :param decoder_state: hidden state of decoder in previous time step
    :type decoder_state: LayerOutput
    :param transform_param_attr: parameter attribute of the feed-forward
                                network that takes decoder_state as inputs to
                                compute attention weight.
    :type transform_param_attr: ParameterAttribute
    :return: a context vector
    :rtype: LayerOutput
    """
    assert encoded_proj.size == decoder_state.size
    proj_size = encoded_proj.size

    with mixed_layer(size=proj_size, name="%s_transform" % name) as m:
        m += full_matrix_projection(
            decoder_state, param_attr=transform_param_attr)

    expanded = expand_layer(
        input=m, expand_as=encoded_sequence, name='%s_expand' % name)

    with mixed_layer(
            size=proj_size, act=weight_act, name="%s_combine" % name) as m:
        m += identity_projection(expanded)
        m += identity_projection(encoded_proj)

    # sequence softmax is used to normalize similarities between decoder state
    # and encoder outputs into a distribution
    attention_weight = fc_layer(
        input=m,
        size=1,
        act=SequenceSoftmaxActivation(),
        param_attr=softmax_param_attr,
        name="%s_softmax" % name,
        bias_attr=False)

    scaled = scaling_layer(
        weight=attention_weight,
        input=encoded_sequence,
        name='%s_scaling' % name)

    return pooling_layer(
        input=scaled, pooling_type=SumPooling(), name="%s_pooling" % name)


@wrap_name_default()
def dot_product_attention(encoded_sequence,
                          attended_sequence,
                          transformed_state,
                          softmax_param_attr=None,
                          name=None):
    """
    Calculate and return a context vector with dot-product attention mechanism.
    The dimension of the context vector equals to that of the attended_sequence.

    ..  math::

        a(s_{i-1},h_{j}) & = s_{i-1}^\mathrm{T} h_{j}

        e_{i,j} & = a(s_{i-1}, h_{j})

        a_{i,j} & = \\frac{exp(e_{i,j})}{\\sum_{k=1}^{T_x}{exp(e_{i,k})}}

        c_{i} & = \\sum_{j=1}^{T_{x}}a_{i,j}z_{j}

    where :math:`h_{j}` is the jth element of encoded_sequence,
    :math:`z_{j}` is the jth element of attended_sequence,
    :math:`s_{i-1}` is transformed_state.

    The example usage is:

    ..  code-block:: python

        context = dot_product_attention(encoded_sequence=enc_seq,
                                        attended_sequence=att_seq,
                                        transformed_state=state,)

    :param name: A prefix attached to the name of each layer that defined inside
                 the dot_product_attention.
    :type name: basestring
    :param softmax_param_attr: The parameter attribute of sequence softmax
                               that is used to produce attention weight.
    :type softmax_param_attr: ParameterAttribute
    :param encoded_sequence: The output hidden vectors of the encoder.
    :type encoded_sequence: LayerOutput
    :param attended_sequence: The attention weight is computed by a feed forward neural
                              network which has two inputs : decoder's transformed hidden
                              state of previous time step and encoder's output.
                              attended_sequence is the sequence to be attended.
    :type attended_sequence: LayerOutput
    :param transformed_state: The transformed hidden state of decoder in previous time step.
                              Since the dot-product operation will be performed on it and the
                              encoded_sequence, their dimensions must be equal. For flexibility,
                              we suppose transformations of the decoder's hidden state have been
                              done outside dot_product_attention and no more will be performed
                              inside. Then users can use either the original or transformed one.
    :type transformed_state: LayerOutput
    :return: The context vector.
    :rtype: LayerOutput
    """
    assert transformed_state.size == encoded_sequence.size

    expanded = expand_layer(
        input=transformed_state,
        expand_as=encoded_sequence,
        name='%s_expand' % name)

    m = dot_prod_layer(
        input1=expanded, input2=encoded_sequence, name='%s_dot-product' % name)

    attention_weight = fc_layer(
        input=m,
        size=1,
        act=SequenceSoftmaxActivation(),
        param_attr=softmax_param_attr,
        name="%s_softmax" % name,
        bias_attr=False)

    scaled = scaling_layer(
        weight=attention_weight,
        input=attended_sequence,
        name='%s_scaling' % name)

    return pooling_layer(
        input=scaled, pooling_type=SumPooling(), name="%s_pooling" % name)


@wrap_name_default()
def multi_head_attention(query,
                         key,
                         value,
                         key_proj_size,
                         value_proj_size,
                         head_num,
                         attention_type,
                         softmax_param_attr=None,
                         name=None):
    """
    Calculate and return a context vector with dot-product attention mechanism.
    The dimension of the context vector equals to value_proj_size * head_num.

    Please refer to **Attention Is All You Need** for more details. The link is
    as follows:
    https://arxiv.org/abs/1706.03762.

    The example usage is:

    ..  code-block:: python

        context = multi_head_attention(query=decoder_state,
                                       key=enc_seq,
                                       value=enc_seq,
                                       key_proj_size=64,
                                       value_pro_size=64,
                                       head_num=8,
                                       attention_type='dot-product attention')

    :param name: A prefix attached to the name of each layer that defined inside
                 the multi_head_attention.
    :type name: basestring
    :param softmax_param_attr: The parameter attribute of sequence softmax
                               that is used to produce attention weight.
    :type softmax_param_attr: ParameterAttribute
    :param query: query is used to calculate attention weights over values at current step.
    :type query: LayerOutput
    :param key: key is used to calculate the attention weight of the corresponding value.
    :type key: LayerOutput
    :param value: value is the sequence to be attended.
    :type value: LayerOutput
    :param key_proj_size: The dimension of the linear projection performed on key and query.
    :type key_proj_size: int
    :param value_proj_size: The dimension of the linear projection performed on value.
    :type value_proj_size: int
    :param head_num: The number of attention heads.
    :type head_num: int
    :param attention_type: The type of the attention mechanism used in each attention
                           heads. Now, we only support scaled dot-product attention and
                           additive attention.
    :type attention_type: basestring
    :return: The context vector.
    :rtype: LayerOutput
    """
    assert attention_type in ['dot-product attention', 'additive attention']

    with mixed_layer(
            size=key_proj_size * head_num,
            name='%s_query_proj' % name) as query_proj:
        query_proj += full_matrix_projection(query)
    query_proj = expand_layer(input=query_proj, expand_as=key)

    with mixed_layer(
            size=key_proj_size * head_num,
            name='%s_key_proj' % name) as key_proj:
        key_proj += full_matrix_projection(key)

    with mixed_layer(
            size=value_proj_size * head_num,
            name='%s_value_proj' % name) as value_proj:
        value_proj += full_matrix_projection(value)

    head_list = []
    for i in range(head_num):
        with mixed_layer(size=key_proj_size) as sub_query_proj:
            sub_query_proj += identity_projection(
                query_proj, offset=key_proj_size * i, size=key_proj_size)

        with mixed_layer(size=key_proj_size) as sub_key_proj:
            sub_key_proj += identity_projection(
                key_proj, offset=key_proj_size * i, size=key_proj_size)

        with mixed_layer(size=value_proj_size) as sub_value_proj:
            sub_value_proj += identity_projection(
                value_proj, offset=value_proj_size * i, size=value_proj_size)

        if attention_type == 'dot-product attention':
            m = dot_prod_layer(
                input1=sub_query_proj,
                input2=sub_key_proj,
                name='%s_dot-product_%d' % (name, i))
            m = slope_intercept_layer(
                input=m,
                slope=math.sqrt(1.0 / key_proj_size),
                name='%s_dot-product_scaling_%d' % (name, i))
        else:
            with mixed_layer(
                    size=key_proj_size,
                    act=TanhActivation(),
                    name='%s_combine_%d' % (name, i)) as m:
                m += identity_projection(sub_query_proj)
                m += identity_projection(sub_key_proj)

        attention_weight = fc_layer(
            input=m,
            size=1,
            act=SequenceSoftmaxActivation(),
            param_attr=softmax_param_attr,
            name="%s_softmax_%d" % (name, i),
            bias_attr=False)

        scaled = scaling_layer(
            weight=attention_weight,
            input=sub_value_proj,
            name='%s_scaling_%d' % (name, i))
        head = pooling_layer(
            input=scaled,
            pooling_type=SumPooling(),
            name="%s_pooling_%d" % (name, i))

        head_list.append(head)

    attended = concat_layer(head_list)

    return attended


def inputs(layers, *args):
    """
    Declare the inputs of network. The order of input should be as same as
    the data provider's return order.

    :param layers: Input Layers.
    :type layers: list|tuple|LayerOutput.
    :return:
    """

    if isinstance(layers, LayerOutput) or isinstance(layers, basestring):
        layers = [layers]
    if len(args) != 0:
        layers.extend(args)

    Inputs(*[l.name for l in layers])


def outputs(layers, *args):
    """
    Declare the outputs of network. If user has not defined the inputs of
    network, this method will calculate the input order by dfs travel.

    :param layers: Output layers.
    :type layers: list|tuple|LayerOutput
    :return:
    """

    traveled = set()

    def __dfs_travel__(layer,
                       predicate=lambda x: x.layer_type == LayerType.DATA):
        """
        DFS LRV Travel for output layer.

        The return order is define order for data_layer in this leaf node.

        :param layer:
        :type layer: LayerOutput
        :return:
        """
        if layer in traveled:
            return []
        else:
            traveled.add(layer)

        assert isinstance(layer, LayerOutput), "layer is %s" % (layer)
        retv = []
        if layer.parents is not None:
            for p in layer.parents:
                retv.extend(__dfs_travel__(p, predicate))

        if predicate(layer):
            retv.append(layer)
        return retv

    if isinstance(layers, LayerOutput):
        layers = [layers]

    if len(args) != 0:
        layers.extend(args)

    assert len(layers) > 0

    if HasInputsSet():  # input already set
        Outputs(*[l.name for l in layers])
        return  # just return outputs.

    if len(layers) != 1:
        logger.warning("`outputs` routine try to calculate network's"
                       " inputs and outputs order. It might not work well."
                       "Please see follow log carefully.")
    inputs = []
    outputs_ = []
    for each_layer in layers:
        assert isinstance(each_layer, LayerOutput)
        inputs.extend(__dfs_travel__(each_layer))
        outputs_.extend(
            __dfs_travel__(each_layer,
                           lambda x: x.layer_type == LayerType.COST))

    # Currently, we got each leaf node's inputs order, output order.
    # We merge them together.

    final_inputs = []
    final_outputs = []

    for each_input in inputs:
        assert isinstance(each_input, LayerOutput)
        if each_input.name not in final_inputs:
            final_inputs.append(each_input.name)

    for each_output in outputs_:
        assert isinstance(each_output, LayerOutput)
        if each_output.name not in final_outputs:
            final_outputs.append(each_output.name)

    logger.info("".join(["The input order is [", ", ".join(final_inputs), "]"]))

    if len(final_outputs) == 0:
        final_outputs = map(lambda x: x.name, layers)

    logger.info("".join(
        ["The output order is [", ", ".join(final_outputs), "]"]))

    Inputs(*final_inputs)
    Outputs(*final_outputs)
