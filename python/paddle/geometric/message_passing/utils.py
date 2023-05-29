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

import paddle
from paddle.fluid.data_feeder import check_dtype, convert_dtype
from paddle.fluid.framework import Variable


def convert_out_size_to_list(out_size):
    """
    Convert out_size(int, np.int32, np.int64, Variable) to list
    in imperative mode.
    """
    if out_size is None:
        out_size = [0]
    elif isinstance(out_size, (int, np.int32, np.int64)):
        out_size = [out_size]
    else:
        out_size = [int(out_size)]
    return out_size


def get_out_size_tensor_inputs(inputs, attrs, out_size, op_type):
    """
    Convert out_size(int, np.int32, np.int64, Variable) to inputs
    and attrs in static graph mode.
    """
    if out_size is None:
        attrs['out_size'] = [0]
    elif isinstance(out_size, (int, np.int32, np.int64)):
        attrs['out_size'] = [out_size]
    elif isinstance(out_size, Variable):
        out_size.stop_gradient = True
        check_dtype(
            out_size.dtype,
            'out_size',
            ['int32', 'int64'],
            'op_type',
            '(When type of out_size in' + op_type + ' is Variable.)',
        )
        if convert_dtype(out_size.dtype) == 'int64':
            out_size = paddle.cast(out_size, 'int32')
        inputs["Out_size"] = out_size
    else:
        raise TypeError("Out_size only supports Variable or int.")


def reshape_lhs_rhs(x, y):
    """
    Expand dims to ensure there will be no broadcasting issues with different
    number of dimensions.
    """
    if len(x.shape) == 1:
        x = paddle.reshape(x, [-1, 1])
    if len(y.shape) == 1:
        y = paddle.reshape(y, [-1, 1])

    x_shape = paddle.shape(x)
    y_shape = paddle.shape(y)
    if len(x.shape) != len(y.shape):
        max_ndims = max(len(x.shape), len(y.shape))
        x_pad_ndims = max_ndims - len(x.shape)
        y_pad_ndims = max_ndims - len(y.shape)
        new_x_shape = (
            [
                x_shape[0],
            ]
            + [
                1,
            ]
            * x_pad_ndims
            + list(x_shape[1:])
        )
        new_y_shape = (
            [
                y_shape[0],
            ]
            + [
                1,
            ]
            * y_pad_ndims
            + list(y_shape[1:])
        )
        x = paddle.reshape(x, new_x_shape)
        y = paddle.reshape(y, new_y_shape)

    return x, y
