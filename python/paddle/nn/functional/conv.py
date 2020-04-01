#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
from __future__ import print_function
__all__ = ['conv2d' 'conv2d_transpose', 'conv3d', 'conv3d_transpose']

import numpy as np
from ...fluid.framework import Variable, in_dygraph_mode
from ...fluid import core, dygraph_utils
from ...fluid.layers import nn, utils
from ...fluid.data_feeder import check_variable_and_dtype
from ...fluid.param_attr import ParamAttr
from ...fluid.layer_helper import LayerHelper


def _is_list_or_tuple(input):
    return isinstance(input, [list, tuple])


def _zero_padding_in_batch_and_channel(padding, channel_last):
    if channel_last:
        return list(padding[0]) == [0, 0] and list(padding[-1]) == [0, 0]
    else:
        return list(padding[0]) == [0, 0] and list(padding[1]) == [0, 0]


def _exclude_padding_in_batch_and_channel(padding, channel_last):
    padding_ = padding[1:-1] if channel_last else padding[2:]
    padding_ = [elem for pad_a_dim in padding_ for elem in pad_a_dim]
    return padding_


def _update_padding_nd(padding, channel_last, num_dims):
    if isinstance(padding, str):
        padding = padding.upper()
        if padding not in ["SAME", "VALID"]:
            raise ValueError(
                "Unknown padding: '{}'. It can only be 'SAME' or 'VALID'.".
                format(padding))
        if padding == "VALID":
            padding_algorithm = "VALID"
            padding = [0] * num_dims
        else:
            padding_algorithm = "SAME"
            padding = [0] * num_dims
    elif _is_list_or_tuple(padding):
        # for padding like
        # [(pad_before, pad_after), (pad_before, pad_after), ...]
        # padding for batch_dim and channel_dim included
        if len(padding) == 2 + num_dims and _is_list_or_tuple(padding[0]):
            if not _zero_padding_in_batch_and_channel(padding, channel_last):
                raise ValueError(
                    "Non-zero padding({}) in the batch or channel dimensions "
                    "is not supported.".formats(padding))
            padding_algorithm = "EXPLICIT"
            padding = _exclude_padding_in_batch_and_channel(padding,
                                                            channel_last)
            if utils._is_symmetric_padding(padding, num_dims):
                padding = padding[0::2]
        # for padding like [pad_before, pad_after, pad_before, pad_after, ...]
        elif len(padding) == 2 * num_dims:
            padding_algorithm = "EXPLICIT"
            padding = utils.convert_to_list(padding, 2 * num_dims, 'padding')
            if utils._is_symmetric_padding(padding, num_dims):
                padding = padding[0::2]
        # for padding like [pad_d1, pad_d2, ...]
        elif len(padding) == num_dims:
            padding_algorithm = "EXPLICIT"
            padding = utils.convert_to_list(padding, num_dims, 'padding')
    # for integer padding
    else:
        padding_algorithm = "EXPLICIT"
        padding = utils.convert_to_list(padding, num_dims, 'padding')
    return padding, padding_algorithm


def conv2d(input,
           weight,
           bias=None,
           padding=0,
           stride=1,
           dilation=1,
           groups=1,
           use_cudnn=True,
           act=None,
           data_format="NCHW"):
    # entry checks
    if not isinstance(use_cudnn, bool):
        raise ValueError("Attr(use_cudnn) should be True or False. "
                         "Received Attr(use_cudnn): {}.".format(use_cudnn))
    if data_format not in ["NCHW", "NHWC"]:
        raise ValueError("Attr(data_format) should be 'NCHW' or 'NHWC'. "
                         "Received Attr(data_format): {}.".format(data_format))

    channel_last = (data_format == "NHWC")
    channel_dim = -1 if channel_last else 1
    num_channels = input.shape[channel_dim]
    num_filters = weight.shape[0]
    if num_channels < 0:
        raise ValueError("The channel dimmention of the input({}) "
                         "should be defined. Received: {}.".format(
                             input.shape, num_channels))
    if num_channels % groups != 0:
        raise ValueError(
            "the channel of input must be divisible by groups,"
            "received: the channel of input is {}, the shape of input is {}"
            ", the groups is {}".format(num_channels, input.shape, groups))
    if num_filters % groups != 0:
        raise ValueError(
            "the number of filters must be divisible by groups,"
            "received: the number of filters is {}, the shape of weight is {}"
            ", the groups is {}".format(num_filters, weight.shape, groups))

    # update attrs
    padding, padding_algorithm = _update_padding_nd(padding, channel_last, 2)
    stride = utils.convert_to_list(stride, 2, 'stride')
    dilation = utils.convert_to_list(dilation, 2, 'dilation')

    l_type = "conv2d"
    if (num_channels == groups and num_filters % num_channels == 0 and
            not use_cudnn):
        l_type = 'depthwise_conv2d'

    inputs = {'Input': [input], 'Filter': [weight]}
    attrs = {
        'strides': stride,
        'paddings': padding,
        'dilations': dilation,
        'groups': groups,
        'use_cudnn': use_cudnn,
        'use_mkldnn': False,
        'fuse_relu_before_depthwise_conv': False,
        "padding_algorithm": padding_algorithm,
        "data_format": data_format
    }

    if in_dygraph_mode():
        pre_bias = getattr(core.ops, l_type)(inputs, attrs)["Output"][0]
        if bias is not None:
            pre_act = nn.elementwise_add(pre_bias, bias, axis=channel_dim)
        else:
            pre_act = pre_bias
        out = dygraph_utils._append_activation_in_dygraph(
            pre_act, act, use_cudnn=use_cudnn)
    else:
        check_variable_and_dtype(input, 'input',
                                 ['float16', 'float32', 'float64'], 'conv2d')
        helper = LayerHelper(l_type, **locals())
        dtype = helper.input_dtype()
        pre_bias = helper.create_variable_for_type_inference(dtype)
        outputs = {"Output": [pre_bias]}
        helper.append_op(
            type=l_type, inputs=inputs, outputs=outputs, attrs=attrs)
        if bias is not None:
            pre_act = nn.elementwise_add(pre_bias, bias, axis=channel_dim)
        else:
            pre_act = pre_bias
        out = helper.append_activation(pre_act)
    return out


def conv2d_transpose(input,
                     weight,
                     bias=None,
                     padding=0,
                     stride=1,
                     dilation=1,
                     groups=1,
                     use_cudnn=True,
                     act=None,
                     data_format='NCHW'):
    if not isinstance(use_cudnn, bool):
        raise ValueError("Attr(use_cudnn) should be True or False. "
                         "Received Attr(use_cudnn): {}.".format(use_cudnn))
    if data_format not in ['NCHW', 'NHWC']:
        raise ValueError(
            "Attr(data_format) of conv2d_transpose got wrong value: "
            "received {}, but only 'NCHW' or 'NHWC' are supported.".format(
                data_format))
    channel_last = (data_format == "NHWC")
    channel_dim = -1 if channel_last else 1
    num_channels = input.shape[channel_dim]
    num_filters = weight.shape[1]
    if num_channels < 0:
        raise ValueError("The channel dimmention of the input({}) "
                         "should be defined. Received: {}.".format(
                             input.shape, num_channels))
    if num_filters % groups != 0:
        raise ValueError(
            "the number of filters must be divisible by groups,"
            "received: the number of filters is {}, the groups is {}".format(
                num_channels, groups))
    if num_channels % groups != 0:
        raise ValueError(
            "the channel of input must be divisible by groups,"
            "received: the channel of input is {}, the shape of input is {}"
            ", the groups is {}".format(num_channels, input.shape, groups))

    # update attrs
    padding, padding_algorithm = _update_padding_nd(padding, channel_last, 2)
    stride = utils.convert_to_list(stride, 2, 'stride')
    dilation = utils.convert_to_list(dilation, 2, 'dilation')

    op_type = 'conv2d_transpose'
    if (num_channels == groups and num_filters == num_channels and
            not use_cudnn):
        op_type = 'depthwise_conv2d_transpose'

    inputs = {'Input': [input], 'Filter': [weight]}
    attrs = {
        'strides': stride,
        'paddings': padding,
        'padding_algorithm': padding_algorithm,
        'dilations': dilation,
        'groups': groups,
        'use_cudnn': use_cudnn,
        'data_format': data_format
    }

    if in_dygraph_mode():
        pre_bias = getattr(core.ops, op_type)(inputs, attrs)["Output"][0]
        if bias is not None:
            pre_act = nn.elementwise_add(pre_bias, bias, axis=channel_dim)
        else:
            pre_act = pre_bias
        out = dygraph_utils._append_activation_in_dygraph(
            pre_act, act, use_cudnn=use_cudnn)
    else:
        check_variable_and_dtype(input, 'input',
                                 ['float16', 'float32', 'float64'],
                                 'conv2d_transpose')
        helper = LayerHelper(op_type, **locals())
        dtype = helper.input_dtype()
        pre_bias = helper.create_variable_for_type_inference(dtype)
        outputs = {"Output": [pre_bias]}
        helper.append_op(
            type=op_type, inputs=inputs, outputs=outputs, attrs=attrs)
        if bias is not None:
            pre_act = nn.elementwise_add(pre_bias, bias, axis=channel_dim)
        else:
            pre_act = pre_bias
        out = helper.append_activation(pre_act)
    return out


def conv3d(input,
           weight,
           bias=None,
           padding=0,
           stride=1,
           dilation=1,
           groups=1,
           use_cudnn=True,
           act=None,
           data_format="NCDHW"):
    # entry check
    if not isinstance(use_cudnn, bool):
        raise ValueError("Attr(use_cudnn) should be True or False. Received "
                         "Attr(use_cudnn): {}. ".format(use_cudnn))

    if data_format not in ["NCDHW", "NDHWC"]:
        raise ValueError(
            "Attr(data_format) should be 'NCDHW' or 'NDHWC'. Received "
            "Attr(data_format): {}.".format(data_format))

    channel_last = (data_format == "NDHWC")
    channel_dim = -1 if channel_last else 1
    num_channels = input.shape[channel_dim]
    num_filters = weight.shape[0]
    if num_channels < 0:
        raise ValueError(
            "The channel dimmention of the input({}) should be defined. "
            "Received: {}.".format(input.shape, num_channels))
    if num_channels % groups != 0:
        raise ValueError(
            "The number of input channels must be divisible by Attr(groups). "
            "Received: number of channels({}), groups({}).".format(num_channels,
                                                                   groups))
    if num_filters % groups != 0:
        raise ValueError(
            "The number of filters must be divisible by Attr(groups). "
            "Received: number of filters({}), groups({}).".format(num_channels,
                                                                  groups))

    padding, padding_algorithm = _update_padding_nd(padding, channel_last, 3)
    stride = utils.convert_to_list(stride, 3, 'stride')
    dilation = utils.convert_to_list(dilation, 3, 'dilation')

    op_type = 'conv3d'
    inputs = {'Input': [input], 'Filter': [weight]}
    attrs = {
        'strides': stride,
        'paddings': padding,
        'dilations': dilation,
        'groups': groups,
        'use_cudnn': use_cudnn,
        'use_mkldnn': False,
        "padding_algorithm": padding_algorithm,
        "data_format": data_format
    }

    if in_dygraph_mode():
        pre_bias = getattr(core.ops, op_type)(inputs, attrs)["Output"][0]
        if bias is not None:
            pre_act = nn.elementwise_add(pre_bias, bias, axis=channel_dim)
        else:
            pre_act = pre_bias
        out = dygraph_utils._append_activation_in_dygraph(
            pre_act, act, use_cudnn=use_cudnn)
    else:
        helper = LayerHelper(op_type, **locals())
        dtype = helper.input_dtype()
        check_variable_and_dtype(input, 'input',
                                 ['float16', 'float32', 'float64'], 'conv3d')

        pre_bias = helper.create_variable_for_type_inference(dtype)
        outputs = {"Output": [pre_bias]}

        helper.append_op(
            type=op_type, inputs=inputs, outputs=outputs, attrs=attrs)
        if bias is not None:
            pre_act = nn.elementwise_add(pre_bias, bias, axis=channel_dim)
        else:
            pre_act = pre_bias
        out = helper.append_activation(pre_act)

    return out


def conv3d_transpose(input,
                     weight,
                     bias=None,
                     padding=0,
                     stride=1,
                     dilation=1,
                     groups=1,
                     use_cudnn=True,
                     act=None,
                     name=None,
                     data_format='NCDHW'):
    # entry checks
    if not isinstance(use_cudnn, bool):
        raise ValueError("Attr(use_cudnn) should be True or False. "
                         "Received Attr(use_cudnn): {}.".format(use_cudnn))
    if data_format not in ["NCDHW", "NDHWC"]:
        raise ValueError(
            "Attr(data_format) should be 'NCDHW' or 'NDHWC'. Received "
            "Attr(data_format): {}.".format(data_format))

    channel_last = (data_format == "NDHWC")
    channel_dim = -1 if channel_last else 1
    num_channels = input.shape[channel_dim]
    num_filters = weight.shape[1]
    if num_channels < 0:
        raise ValueError(
            "The channel dimmention of the input({}) should be defined. "
            "Received: {}.".format(input.shape, num_channels))
    if num_channels % groups != 0:
        raise ValueError(
            "The number of input channels must be divisible by Attr(groups). "
            "Received: number of channels({}), groups({}).".format(num_channels,
                                                                   groups))
    if num_filters % groups != 0:
        raise ValueError(
            "The number of filters must be divisible by Attr(groups). "
            "Received: number of filters({}), groups({}).".format(num_channels,
                                                                  groups))

    padding, padding_algorithm = _update_padding_nd(padding, channel_last, 3)
    stride = utils.convert_to_list(stride, 3, 'stride')
    dilation = utils.convert_to_list(dilation, 3, 'dilation')

    op_type = 'conv3d_transpose'
    data_format_ = "NHWC" if channel_last else "NHWC"
    inputs = {'Input': [input], 'Filter': [weight]}
    attrs = {
        'paddings': padding,
        "padding_algorithm": padding_algorithm,
        'strides': stride,
        'dilations': dilation,
        'groups': groups,
        'use_cudnn': use_cudnn,
        "data_format": data_format_
    }

    if in_dygraph_mode():
        pre_bias = getattr(core.ops, op_type)(inputs, attrs)["Output"][0]
        if bias is not None:
            pre_act = nn.elementwise_add(pre_bias, bias, axis=channel_dim)
        else:
            pre_act = pre_bias
        out = dygraph_utils._append_activation_in_dygraph(
            pre_act, act, use_cudnn=use_cudnn)
    else:
        helper = LayerHelper(op_type, **locals())
        dtype = helper.input_dtype()
        check_variable_and_dtype(input, 'input',
                                 ['float16', 'float32', 'float64'], 'conv3d')

        pre_bias = helper.create_variable_for_type_inference(dtype)
        outputs = {"Output": [pre_bias]}

        helper.append_op(
            type=op_type, inputs=inputs, outputs=outputs, attrs=attrs)
        if bias is not None:
            pre_act = nn.elementwise_add(pre_bias, bias, axis=channel_dim)
        else:
            pre_act = pre_bias
        out = helper.append_activation(pre_act)

    return out
