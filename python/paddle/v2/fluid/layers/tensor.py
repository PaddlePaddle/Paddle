#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserve.
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

from ..layer_helper import LayerHelper
from ..param_attr import ParamAttr
from ..framework import convert_np_dtype_to_dtype_
from ..framework import Variable
from ..initializer import Constant
from ..core import DataType
import numpy

__all__ = [
    'create_tensor',
    'create_parameter',
    'create_global_var',
    'cast',
    'concat',
    'sums',
    'assign',
    'fill_constant_batch_size_like',
    'fill_constant',
    'ones',
    'zeros',
]


def create_tensor(dtype, name=None):
    helper = LayerHelper("create_tensor", **locals())
    return helper.create_variable(name=helper.name, dtype=dtype)


def create_parameter(shape,
                     dtype,
                     attr=None,
                     is_bias=False,
                     default_initializer=None):
    """
    Create a parameter
    Args:
        shape(list[int]): shape of the parameter
        dtype(string): element type of the parameter
        attr(ParamAttr): attributes of the parameter
        is_bias(bool): This can affect which default initializer is chosen
                       when default_initializer is None. If is_bias,
                       initializer.Constant(0.0) will be used. Otherwise,
                       Xavier() will be used.
        default_initializer(Initializer): initializer for the parameter

    Returns:
        Parameter: the created parameter
    """
    helper = LayerHelper("create_parameter", **locals())
    if attr is None:
        attr = ParamAttr()
    return helper.create_parameter(attr, shape, dtype, is_bias,
                                   default_initializer)


def create_global_var(shape, value, dtype, persistable=False, name=None):
    helper = LayerHelper("global_var", **locals())
    var = helper.create_global_variable(
        dtype=dtype, shape=shape, persistable=persistable, name=name)
    helper.set_variable_initializer(
        var, initializer=Constant(value=float(value)))
    return var


def cast(x, dtype):
    """
    This function takes in the input with input_dtype
    and casts it to the output_dtype as the output.
    """
    helper = LayerHelper('cast', **locals())
    out = helper.create_tmp_variable(dtype=dtype)
    helper.append_op(
        type='cast',
        inputs={'X': [x]},
        outputs={'Out': [out]},
        attrs={'in_dtype': x.dtype,
               'out_dtype': out.dtype})
    return out


def concat(input, axis=0):
    """
    **Concat**

    This function concatenates the input along the axis mentioned
    and returns that as the output.

    Args:
        input(list): List of tensors to be concatenated
        axis(int): Integer axis along which the tensors will be concatenated

    Returns:
        Variable: Output variable of the concatenation

    Examples:
        .. code-block:: python
          out = fluid.layers.concat(input=[Efirst, Esecond, Ethird, Efourth])
    """
    helper = LayerHelper('concat', **locals())
    out = helper.create_tmp_variable(dtype=helper.input_dtype())
    helper.append_op(
        type='concat',
        inputs={'X': input},
        outputs={'Out': [out]},
        attrs={'axis': axis})
    return out


def sums(input, out=None):
    """This function performs the sum operation on the input and returns the
    result as the output.

    Args:
        input (Variable|list): The input tensor that has the elements
                               that need to be summed up.

    Returns:
        Variable: The tensor type variable that has the sum of input
                  written to it.

    Examples:
        .. code-block::python

          tmp = fluid.layers.zeros(shape=[10], dtype='int32')
          i = fluid.layers.fill_constant(shape=[1], dtype='int64', value=10)
          a0 = layers.array_read(array=tmp, i=i)
          i = layers.increment(x=i)
          a1 = layers.array_read(array=tmp, i=i)
          mean_a0 = layers.mean(x=a0)
          mean_a1 = layers.mean(x=a1)
          a_sum = layers.sums(input=[mean_a0, mean_a1])
    """
    helper = LayerHelper('sum', **locals())
    if out is None:
        out = helper.create_tmp_variable(dtype=helper.input_dtype())
    helper.append_op(type='sum', inputs={'X': input}, outputs={'Out': out})
    return out


def assign(input, output):
    """
    **Assign**

    This function copies the *input* Variable to the *output* Variable.

    Args:
        input(Variable|numpy.ndarray): The source variable
        output(Variable): The destination variable

    Returns:
        Variable: The destination variable that was supplied as the *output*.

    Examples:
        .. code-block:: python
          out = fluid.layers.create_tensor(dtype='float32')
          hidden = fluid.layers.fc(input=data, size=10)
          fluid.layers.assign(hidden, out)
    """
    helper = LayerHelper('assign', **locals())
    if isinstance(input, Variable):
        helper.append_op(
            type='scale',
            inputs={'X': [input]},
            outputs={'Out': [output]},
            attrs={'scale': 1.0})
    elif isinstance(input, numpy.ndarray):
        dtype = convert_np_dtype_to_dtype_(input.dtype)
        if dtype == DataType.FP32:
            value_name = "fp32_values"
            values = [float(v) for v in input.flat]
        elif dtype == DataType.INT32:
            value_name = "int32_values"
            values = [int(v) for v in input.flat]
        else:
            raise ValueError("Unsupported dtype %s", input.dtype)
        if input.size > 1024 * 1024:
            raise ValueError("The size of input is too big. Please consider "
                             "saving it to file and 'load_op' to load it")

        helper.append_op(
            type='assign_value',
            outputs={'Out': [output]},
            attrs={
                'dtype': dtype,
                'shape': list(input.shape),
                value_name: values
            })
    else:
        raise ValueError("Wrong type for assign input: %s" % type(input))

    return output


def fill_constant(shape, dtype, value, force_cpu=False, out=None):
    """
    **fill_constant**

    This function creates a tensor with specified `shape` and `dtype`, and
    initializes it with a constant specifed by `value`.

    The attribute `stop_gradient` of the created tensor is set to True.

    Args:
        shape(tuple|list|None): Shape of the output tensor.
        dtype(np.dtype|core.DataType|str): Data type of the output tensor.
        value(float): The constant value used to initialize the output tensor.
        out(Variable): The output tensor.

    Returns:
        Variable: The tensor variable storing the output.

    Examples:
        .. code-block:: python

          data = fluid.layers.fill_constant(shape=[1], value=0, dtype='int64')
    """

    helper = LayerHelper("fill_constant", **locals())
    if out is None:
        out = helper.create_tmp_variable(dtype=dtype)
    helper.append_op(
        type='fill_constant',
        inputs={},
        outputs={'Out': [out]},
        attrs={
            'shape': shape,
            'dtype': out.dtype,
            'value': float(value),
            'force_cpu': force_cpu
        })
    out.stop_gradient = True
    return out


def fill_constant_batch_size_like(input,
                                  shape,
                                  dtype,
                                  value,
                                  input_dim_idx=0,
                                  output_dim_idx=0):
    """
    **fill_constant_batch_size_like**

    This function creates a tensor of specified *shape*, *dtype* and batch size,
    and initializes this with a constant supplied in *value*. The batch size is
    obtained from the `input` tensor.

    It also sets *stop_gradient* to True.

    Args:
        input(Variable): Tensor whose dimensions will be used to get batch size
        shape(tuple|list|None): Shape of output tensor
        dtype(np.dtype|core.DataType|str): Data type of output tensor
        value(float): Constant value to initialize the output tensor
        input_dim_idx(int): Index of input's batch size dimension
        output_dim_idx(int): Index of output's batch size dimension

    Returns:
        Variable: The tensor variable storing the output

    Examples:
        .. code-block:: python

          data = fluid.layers.fill_constant_batch_size_like(
              input=like, shape=[1], value=0, dtype='int64')
    """
    helper = LayerHelper("fill_constant_batch_size_like", **locals())
    out = helper.create_tmp_variable(dtype=dtype)
    helper.append_op(
        type='fill_constant_batch_size_like',
        inputs={'Input': input},
        outputs={'Out': [out]},
        attrs={
            'shape': shape,
            'dtype': out.dtype,
            'value': float(value),
            'input_dim_idx': input_dim_idx,
            'output_dim_idx': output_dim_idx
        })
    out.stop_gradient = True
    return out


def ones(shape, dtype, force_cpu=False):
    """
    **ones**

    This function creates a tensor of specified *shape* and
    *dtype*, and initializes this with 1.

    It also sets *stop_gradient* to True.

    Args:
        shape(tuple|list|None): Shape of output tensor
        dtype(np.dtype|core.DataType|str): Data type of output tensor

    Returns:
        Variable: The tensor variable storing the output

    Examples:
        .. code-block:: python

          data = fluid.layers.ones(shape=[1], dtype='int64')
    """
    return fill_constant(value=1.0, **locals())


def zeros(shape, dtype, force_cpu=False):
    """
    **zeros**

    This function creates a tensor of specified *shape* and
    *dtype*, and initializes this with 0.

    It also sets *stop_gradient* to True.

    Args:
        shape(tuple|list|None): Shape of output tensor
        dtype(np.dtype|core.DataType|str): Data type of output tensor

    Returns:
        Variable: The tensor variable storing the output

    Examples:
        .. code-block:: python

          data = fluid.layers.zeros(shape=[1], dtype='int64')
    """
    return fill_constant(value=0.0, **locals())
