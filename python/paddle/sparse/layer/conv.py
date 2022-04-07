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

import numpy as np
from .. import functional as F
from paddle.nn import Layer
from paddle.nn.initializer import Normal
from ..functional.conv import _update_padding_nd
from ...fluid.layers import utils

__all__ = []


class _Conv3D(Layer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 subm=False,
                 padding_mode='zeros',
                 weight_attr=None,
                 bias_attr=None,
                 data_format="NCDHW"):
        super(_Conv3D, self).__init__()
        assert weight_attr is not False, "weight_attr should not be False in Conv."
        self._param_attr = weight_attr
        self._bias_attr = bias_attr
        self._groups = groups
        self._in_channels = in_channels
        self._out_channels = out_channels
        self._data_format = data_format
        self._subm = subm

        assert padding_mode == 'zeros', "Currently, only support padding_mode='zeros'"
        assert groups == 1, "Currently, only support groups=1"

        valid_format = {'NDHWC'}
        if data_format not in valid_format:
            raise ValueError(
                "data_format must be one of {}, but got data_format='{}'".
                format(valid_format, data_format))

        channel_last = data_format == "NDHWC"

        dims = 3
        self._stride = utils.convert_to_list(stride, dims, 'stride')
        self._dilation = utils.convert_to_list(dilation, dims, 'dilation')
        self._kernel_size = utils.convert_to_list(kernel_size, dims,
                                                  'kernel_size')
        self._padding = padding
        self._padding_mode = padding_mode
        self._updated_padding, self._padding_algorithm = _update_padding_nd(
            padding, channel_last, dims)

        # the sparse conv restricts the shape is [D, H, W, in_channels, out_channels]
        filter_shape = self._kernel_size + [
            self._in_channels, self._out_channels
        ]

        def _get_default_param_initializer():
            filter_elem_num = np.prod(self._kernel_size) * self._in_channels
            std = (2.0 / filter_elem_num)**0.5
            return Normal(0.0, std)

        self.weight = self.create_parameter(
            shape=filter_shape,
            attr=self._param_attr,
            default_initializer=_get_default_param_initializer())
        #Currently, sparse_conv3d does not support bias
        self.bias = None

    def forward(self, x):
        out = F.conv._conv3d(
            x,
            self.weight,
            bias=self.bias,
            stride=self._stride,
            padding=self._updated_padding,
            dilation=self._dilation,
            groups=self._groups,
            subm=self._subm,
            data_format=self._data_format)
        return out

    def extra_repr(self):
        main_str = '{_in_channels}, {_out_channels}, kernel_size={_kernel_size}'
        if self._stride != [1] * len(self._stride):
            main_str += ', stride={_stride}'
        if self._padding != 0:
            main_str += ', padding={_padding}'
        if self._padding_mode != 'zeros':
            main_str += ', padding_mode={_padding_mode}'
        if self._dilation != [1] * len(self._dilation):
            main_str += ', dilation={_dilation}'
        if self._groups != 1:
            main_str += ', groups={_groups}'
        main_str += ', data_format={_data_format}'
        return main_str.format(**self.__dict__)


class Conv3D(_Conv3D):
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
                 data_format="NCDHW"):
        super(Conv3D, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            subm=False,
            padding_mode=padding_mode,
            weight_attr=weight_attr,
            bias_attr=bias_attr,
            data_format=data_format)


class SubmConv3D(_Conv3D):
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
                 data_format="NCDHW"):
        super(SubmConv3D, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            subm=True,
            padding_mode=padding_mode,
            weight_attr=weight_attr,
            bias_attr=bias_attr,
            data_format=data_format)
