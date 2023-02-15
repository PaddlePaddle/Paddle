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
    _varbase_creator,
    in_dygraph_mode,
)
from ..framework import Variable
from ..core import VarDesc
from .. import core
from .layer_function_generator import templatedoc
from . import utils
from ..data_feeder import (
    check_variable_and_dtype,
    check_type,
    check_dtype,
    convert_dtype,
)
from paddle.utils import deprecated

from .utils import check_shape
from paddle import _C_ops, _legacy_C_ops

__all__ = [
    'fill_constant_batch_size_like',
    'fill_constant',
    'zeros',
]


def fill_constant(shape, dtype, value, force_cpu=False, out=None, name=None):
    """

    This OP creates a Tensor with specified `shape` and `dtype`, and
    initializes it with a constant specified by `value`.

    The attribute `stop_gradient` of the created Tensor is set to True.

    Args:
        shape(list|tuple|Tensor): Shape of the output Tensor, the data type of ``shape`` is int32 or int64.
            If ``shape`` is a list or tuple, the elements of it should be integers or Tensors with shape [1].
            If ``shape`` is an Tensor, it should be an 1-D Tensor with date type int32 or int64.
        dtype(np.dtype|str): Data type of the output Tensor which can
            be float16, float32, float64, uint8, int16, int32, int64.
        value(bool|float|int|Tensor): The constant value used to initialize
            the Tensor to be created. If ``value`` is an Tensor, it should be an 1-D Tensor.
        force_cpu(bool, optional): data should be on CPU if it's true, default value is False.
        out(Tensor, optional): Optional output which can be any created
            Tensor that meets the requirements to store the result of operation.
            if ``out`` is None, a new Tensor will be create to store the result.
        name(str, optional): The default value is None.  Normally there is no need for user to set this
            property.  For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        Tensor: Tensor which is created according to shape and dtype.

    Examples:
        .. code-block:: python

          import paddle.fluid as fluid
          # attr shape is a list which doesn't contain  Tensor.
          data1 = fluid.layers.fill_constant(shape=[2,1], value=0, dtype='int64') # data1=[[0],[0]]
          data2 = fluid.layers.fill_constant(shape=[2,1], value=5, dtype='int64', out=data1)
          # data1=[[5], [5]] data2=[[5], [5]]

          # attr shape is a list which contains Tensor.
          positive_2 = fluid.layers.fill_constant([1], "int32", 2)
          data3 = fluid.layers.fill_constant(shape=[1, positive_2], dtype='float32', value=1.5) # data3=[[1.5, 1.5]]

          # attr shape is a Tensor.
          shape = fluid.layers.fill_constant([2], "int32", 2) # shape=[2,2]
          data4 = fluid.layers.fill_constant(shape=shape, dtype='bool', value=True) # data4=[[True,True],[True,True]]

          # attr value is a Tensor.
          val = fluid.layers.fill_constant([1], "float32", 2.0) # val=[2.0]
          data5 = fluid.layers.fill_constant(shape=[2,1], value=val, dtype='float32') #data5=[[2.0],[2.0]]
    """

    if in_dygraph_mode():
        place = _current_expected_place()
        if force_cpu:
            place = core.CPUPlace()
        if isinstance(shape, (list, tuple)):
            shape = utils.convert_shape_to_list(shape)

        if not isinstance(dtype, core.VarDesc.VarType):
            dtype = convert_np_dtype_to_dtype_(dtype)

        if out is None:
            out = _C_ops.full(shape, float(value), dtype, place)
            out.stop_gradient = True
            return out

        if out is not None:
            # final state mode is support out is not None.
            _C_ops.full_(out, shape, float(value), dtype, place)
            out.stop_gradient = True
            return out
    else:
        attrs = {'force_cpu': force_cpu}
        dtype = convert_dtype(dtype)
        if not isinstance(value, Variable):
            if dtype in ['uint8', 'int16', 'int32', 'int64']:
                attrs['str_value'] = str(int(value))
                attrs['value'] = int(value)
            else:
                attrs['str_value'] = str(float(value))
                attrs['value'] = float(value)

        helper = LayerHelper("fill_constant", **locals())
        inputs = {}
        if isinstance(value, Variable):
            if convert_dtype(value.dtype) != dtype:
                value = paddle.cast(value, dtype)
            inputs['ValueTensor'] = value

        check_shape(shape)
        check_dtype(
            dtype,
            'dtype',
            [
                'bool',
                'float16',
                'float32',
                'float64',
                'uint8',
                'int16',
                'int32',
                'int64',
                'complex64',
                'complex128',
            ],
            'fill_constant',
        )
        check_type(shape, 'shape', (Variable, list, tuple), 'fill_constant')

        if out is not None:
            check_variable_and_dtype(
                out, 'out', [convert_dtype(dtype)], 'fill_constant'
            )

        helper = LayerHelper("fill_constant", **locals())
        utils.get_shape_tensor_inputs(
            inputs=inputs, attrs=attrs, shape=shape, op_type='fill_constant'
        )

        if out is None:
            out = helper.create_variable_for_type_inference(dtype=dtype)
        attrs['dtype'] = out.dtype
        helper.append_op(
            type='fill_constant',
            inputs=inputs,
            outputs={'Out': [out]},
            attrs=attrs,
            stop_gradient=True,
        )
        out.stop_gradient = True
        return out


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

             import paddle.fluid as fluid
             like = fluid.layers.fill_constant(shape=[1,2], value=10, dtype='int64') #like=[[10, 10]]
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


def zeros(shape, dtype, force_cpu=False, name=None):
    """
    The OP creates a tensor of specified :attr:`shape` and :attr:`dtype`, and fills it with 0.
    Its :attr:`stop_gradient` will be set to True to stop gradient computation.

    Parameters:
        shape(tuple|list|Tensor): Shape of output Tensor, the data type of ``shape`` is int32 or int64.
        dtype (np.dtype|str): Data type of output Tensor, it supports
            bool, float16, float32, float64, int32 and int64.
        force_cpu (bool, optional): Whether force to store the output Tensor in CPU memory.
            If :attr:`force_cpu` is False, the output Tensor will be stored in running device memory.
            Default: False.
        name(str, optional): The default value is None.  Normally there is no need for user to set this
            property.  For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        Tensor: A tensor of data type :attr:`dtype` with shape :attr:`shape` and all elements set to 0.

    Examples:
        .. code-block:: python

          import paddle.fluid as fluid
          data = fluid.layers.zeros(shape=[3, 2], dtype='float32') # [[0., 0.], [0., 0.], [0., 0.]]

          # shape is a Tensor
          shape = fluid.layers.fill_constant(shape=[2], dtype='int32', value=2)
          data1 = fluid.layers.zeros(shape=shape, dtype='int32') #[[0, 0], [0, 0]]
    """
    return fill_constant(value=0.0, **locals())
