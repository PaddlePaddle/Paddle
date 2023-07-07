#   Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unlessf required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import paddle
import numpy
import warnings

from ..layer_helper import LayerHelper
from ..framework import (
    _current_expected_place,
    convert_np_dtype_to_dtype_,
    _create_tensor,
    in_dygraph_mode,
)
from ..framework import Variable
from ..core import VarDesc
from .. import core
from .layer_function_generator import templatedoc
from ..data_feeder import (
    check_variable_and_dtype,
    check_type,
    check_dtype,
    convert_dtype,
)
from paddle.utils import deprecated

from paddle import _C_ops, _legacy_C_ops

__all__ = [
    'fill_constant_batch_size_like',
]


@deprecated(since='1.8.0', update_to="paddle.fluid.layers.fill_constant")
@templatedoc()
def fill_constant_batch_size_like(
    input,
    shape,
    dtype,
    value,
    input_dim_idx=0,
    output_dim_idx=0,
    force_cpu=False,
):
    """
    This OP creates a Tesnor according the shape and dtype, and initializes the
    Tensor with the constants provided in ``value``. When the input is LoDTensor
    and the input_dim_idx is 0, the output_dim_idx dimension is set to the value
    of the batch_size input by the input, the Stop_gradient attribute of the created
    Tensor is False by default.

    Args:
        input(Variable): Tensor which data type is float32, float64, int32 and int64.
        shape(list): The shape of Tensor to be created, Tensor's shape may be changed
            according the input.
        dtype(np.dtype|core.VarDesc.VarType|str): The data type of created Tensor which
            can be float32, float64, int32, int64.
        value(float|int): The constant value used to initialize the Tensor to be created.
        input_dim_idx(int): When the value is 0 and the input is LoDTensor, the output_dim_idx
            dimension of the created Tensor is set to the batch_size value of input.
            The default value is 0.
        output_dim_idx(int): Used to specify which dimension of Tensor is created to be set
            the value of batch_size of input Tensor. The default value is 0.
        force_cpu(bool): data should be on CPU if it's true, default value is False.

    Returns:
        Variable: Tensor which will be created according to dtype.

    Examples:

        .. code-block:: python

             import paddle
             import paddle.fluid as fluid
             like = paddle.full(shape=[1,2], fill_value=10, dtype='int64') #like=[[10, 10]]
             data = fluid.layers.fill_constant_batch_size_like(
                    input=like, shape=[1], value=0, dtype='int64') #like=[[10, 10]] data=[0]

    """
    if in_dygraph_mode():
        if not isinstance(dtype, core.VarDesc.VarType):
            dtype = convert_np_dtype_to_dtype_(dtype)

        place = _current_expected_place()
        if force_cpu:
            place = core.CPUPlace()
        out = _C_ops.full_batch_size_like(
            input, shape, dtype, value, input_dim_idx, output_dim_idx, place
        )
        out.stop_gradient = True
        return out
    else:
        helper = LayerHelper("fill_constant_batch_size_like", **locals())
        out = helper.create_variable_for_type_inference(dtype=dtype)
        attrs = {
            'shape': shape,
            'dtype': out.dtype,
            'value': float(value),
            'input_dim_idx': input_dim_idx,
            'output_dim_idx': output_dim_idx,
            'force_cpu': force_cpu,
        }
        if convert_dtype(dtype) in ['int64', 'int32']:
            attrs['str_value'] = str(int(value))
        else:
            attrs['str_value'] = str(float(value))
        helper.append_op(
            type='fill_constant_batch_size_like',
            inputs={'Input': input},
            outputs={'Out': [out]},
            attrs=attrs,
        )
        out.stop_gradient = True
        return out
