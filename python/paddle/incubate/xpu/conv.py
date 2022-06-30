#   Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

# TODO: define classes of convolutional neural network

import numpy as np

import paddle.fluid as fluid
from paddle.fluid import get_flags
from paddle.fluid import core
from paddle.device import get_cudnn_version
from paddle.nn import Layer
from paddle.nn.initializer import Normal
from paddle.nn import functional as F
from paddle.fluid.layers import utils
from paddle.nn.functional.conv import _update_padding_nd
from paddle.fluid.data_feeder import check_variable_and_dtype
from paddle.fluid.layer_helper import LayerHelper
from paddle import _C_ops

__all__ = []


def _get_default_param_initializer(num_channels, filter_size):
    filter_elem_num = num_channels * np.prod(filter_size)
    std = (2.0 / filter_elem_num)**0.5
    return Normal(0.0, std)


def _reverse_repeat_list(t, n):
    """Reverse the order of `t` and repeat each element for `n` times.
    This can be used to translate padding arg used by Conv and Pooling modules
    to the ones used by `F.pad`.
    """
    return list(x for x in reversed(t) for _ in range(n))


def _conv_nd_with_bias_act(x,
                           weight,
                           bias=None,
                           stride=1,
                           padding=0,
                           padding_algorithm=None,
                           dilation=1,
                           groups=1,
                           data_format="NCHW",
                           channel_dim=1,
                           op_type="fused_conv2d_bias_act",
                           name=None,
                           act_type="relu"):

    # Due to the poor performance of NHWC, we transpose the input to NCHW.
    if fluid.framework.in_dygraph_mode():
        attrs = ('activation', act_type, 'strides', stride, 'paddings', padding,
                 'dilations', dilation, 'groups', groups, "padding_algorithm",
                 padding_algorithm, 'data_format', "NCHW")
        out = getattr(_C_ops, op_type)(x, weight, bias, *attrs)
    else:
        inputs = {'Input': [x], 'Filter': [weight], 'Bias': [bias]}
        attrs = {
            'strides': stride,
            'paddings': padding,
            'dilations': dilation,
            'groups': groups,
            "padding_algorithm": padding_algorithm,
            "activation": act_type,
            'data_format': "NCHW"
        }
        check_variable_and_dtype(x, 'x', ['float16', 'float32', 'float64'],
                                 op_type)
        helper = LayerHelper(op_type, **locals())
        dtype = helper.input_dtype(input_param_name='x')
        out = helper.create_variable_for_type_inference(dtype)
        outputs = {"Output": [out]}
        helper.append_op(type=op_type,
                         inputs=inputs,
                         outputs=outputs,
                         attrs=attrs)
    return out


class _ConvNd(Layer):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 transposed,
                 dims,
                 stride=1,
                 padding=0,
                 padding_mode='zeros',
                 output_padding=0,
                 dilation=1,
                 groups=1,
                 weight_attr=None,
                 bias_attr=None,
                 data_format="NCHW"):
        super(_ConvNd, self).__init__()
        assert weight_attr is not False, "weight_attr should not be False in Conv."
        self._param_attr = weight_attr
        self._bias_attr = bias_attr
        self._groups = groups
        self._in_channels = in_channels
        self._out_channels = out_channels
        self._data_format = data_format

        valid_padding_modes = {'zeros', 'reflect', 'replicate', 'circular'}
        if padding_mode not in valid_padding_modes:
            raise ValueError(
                "padding_mode must be one of {}, but got padding_mode='{}'".
                format(valid_padding_modes, padding_mode))

        if padding_mode in {'reflect', 'replicate', 'circular'
                            } and not isinstance(padding, np.int):
            raise TypeError(
                "when padding_mode in ['reflect', 'replicate', 'circular'], type of padding must be int"
            )

        valid_format = {'NHWC', 'NCHW', 'NDHWC', 'NCDHW', 'NLC', 'NCL'}
        if data_format not in valid_format:
            raise ValueError(
                "data_format must be one of {}, but got data_format='{}'".
                format(valid_format, data_format))

        channel_last = (data_format == "NHWC") or (data_format
                                                   == "NDHWC") or (data_format
                                                                   == "NLC")
        if channel_last:
            self._channel_dim = len(data_format) - 1
        else:
            self._channel_dim = 1

        self._stride = utils.convert_to_list(stride, dims, 'stride')
        self._dilation = utils.convert_to_list(dilation, dims, 'dilation')
        self._kernel_size = utils.convert_to_list(kernel_size, dims,
                                                  'kernel_size')
        self._padding = padding
        self._padding_mode = padding_mode
        self.output_padding = output_padding
        if dims != 1:
            self._updated_padding, self._padding_algorithm = _update_padding_nd(
                padding, channel_last, dims)

        if transposed:
            filter_shape = [self._in_channels, out_channels // groups
                            ] + self._kernel_size
        else:
            if in_channels % groups != 0:
                raise ValueError("in_channels must be divisible by groups.")

            if padding_mode in {'reflect', 'replicate', 'circular'}:
                _paired_padding = utils.convert_to_list(padding, dims,
                                                        'padding')
                self._reversed_padding_repeated_twice = _reverse_repeat_list(
                    _paired_padding, 2)

                self._updated_padding, self._padding_algorithm = _update_padding_nd(
                    0, channel_last, dims)

            filter_shape = [out_channels, in_channels // groups
                            ] + self._kernel_size

        def _get_default_param_initializer():
            if transposed:
                return None
            filter_elem_num = np.prod(self._kernel_size) * self._in_channels
            std = (2.0 / filter_elem_num)**0.5
            return Normal(0.0, std)

        self.weight = self.create_parameter(
            shape=filter_shape,
            attr=self._param_attr,
            default_initializer=_get_default_param_initializer())
        self.bias = self.create_parameter(attr=self._bias_attr,
                                          shape=[self._out_channels],
                                          is_bias=True)

        cudnn_version = get_cudnn_version()

        self._use_cudnn = True if (core.is_compiled_with_cuda()
                                   and cudnn_version is not None) else False

        self._op_type = 'fused_conv2d_bias_act'

    def extra_repr(self):
        main_str = '{_in_channels}, {_out_channels}, kernel_size={_kernel_size}'
        if self._stride != [1] * len(self._stride):
            main_str += ', stride={_stride}'
        if self._padding != 0:
            main_str += ', padding={_padding}'
        if self._padding_mode is not 'zeros':
            main_str += ', padding_mode={_padding_mode}'
        if self.output_padding != 0:
            main_str += ', output_padding={output_padding}'
        if self._dilation != [1] * len(self._dilation):
            main_str += ', dilation={_dilation}'
        if self._groups != 1:
            main_str += ', groups={_groups}'
        main_str += ', data_format={_data_format}'
        return main_str.format(**self.__dict__)


class Conv2DWithBiasAct(_ConvNd):
    """
    The Conv2DWithBiasAct layer calculates the output of convolution2D based on
    the input, filter and strides, paddings, dilations, groups parameters,
    and if bias attribution and activation type are provided, bias is added to
    the output of the convolution2D, and the corresponding activation function
    is applied to the final result.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 padding_mode='zeros',
                 weight_attr=None,
                 bias_attr=None,
                 data_format="NCHW",
                 act_type="relu"):
        super(Conv2DWithBiasAct, self).__init__(in_channels,
                                                out_channels,
                                                kernel_size,
                                                False,
                                                2,
                                                stride=stride,
                                                padding=padding,
                                                padding_mode=padding_mode,
                                                dilation=dilation,
                                                groups=groups,
                                                weight_attr=weight_attr,
                                                bias_attr=bias_attr,
                                                data_format=data_format)
        self._act_type = act_type

    def forward(self, x):
        if self._padding_mode != 'zeros':
            x = F.pad(x,
                      self._reversed_padding_repeated_twice,
                      mode=self._padding_mode,
                      data_format=self._data_format)

        out = _conv_nd_with_bias_act(x,
                                     self.weight,
                                     bias=self.bias,
                                     stride=self._stride,
                                     padding=self._updated_padding,
                                     padding_algorithm=self._padding_algorithm,
                                     dilation=self._dilation,
                                     groups=self._groups,
                                     data_format=self._data_format,
                                     channel_dim=self._channel_dim,
                                     op_type=self._op_type,
                                     act_type=self._act_type)
        return out
