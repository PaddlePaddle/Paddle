#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import print_function
from six.moves import reduce
from ..layer_helper import LayerHelper
from ..param_attr import ParamAttr
from ..framework import convert_np_dtype_to_dtype_
from ..framework import Variable
from ..initializer import Constant, force_init_on_cpu
from ..core import VarDesc
from .layer_function_generator import templatedoc
from ..data_feeder import convert_dtype
import numpy

__all__ = [
    'create_tensor', 'create_parameter', 'create_global_var', 'cast',
    'tensor_array_to_tensor', 'concat', 'sums', 'assign',
    'fill_constant_batch_size_like', 'fill_constant', 'argmin', 'argmax',
    'argsort', 'ones', 'zeros', 'reverse', 'has_inf', 'has_nan', 'isfinite',
    'range', 'linspace', 'zeros_like', 'ones_like', 'diag', 'eye'
]


def create_tensor(dtype, name=None, persistable=False):
    """
    Create an variable, which will hold a LoDTensor with data type dtype.

    Args:
        dtype(string): 'float32'|'int32'|..., the data type of the
            created tensor.
        name(string): The name of the created tensor, if not set,
            the name will be a random unique one.
        persistable(bool): Set the persistable flag of the create tensor.

    Returns:
        Variable: The tensor variable storing the created tensor.

    Examples:
        .. code-block:: python

          import paddle.fluid as fluid
          tensor = fluid.layers.create_tensor(dtype='float32')
    """
    helper = LayerHelper("create_tensor", **locals())
    return helper.create_variable(
        name=helper.name, dtype=dtype, persistable=persistable)


def create_parameter(shape,
                     dtype,
                     name=None,
                     attr=None,
                     is_bias=False,
                     default_initializer=None):
    """
    Create a parameter. The parameter is a learnable variable, which can have
    gradient, and can be optimized.

    NOTE: this is a very low-level API. This API is useful when you create
    operator by your self. instead of using layers.

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
        the created parameter.

    Examples:
        .. code-block:: python

            import paddle.fluid as fluid
            import paddle.fluid.layers as layers
            W = layers.create_parameter(shape=[784, 200], dtype='float32')
    """
    helper = LayerHelper("create_parameter", **locals())
    if attr is None:
        attr = ParamAttr(name=name)
    return helper.create_parameter(attr, shape, dtype, is_bias,
                                   default_initializer)


def create_global_var(shape,
                      value,
                      dtype,
                      persistable=False,
                      force_cpu=False,
                      name=None):
    """
    Create a new tensor variable with value in the global block(block 0).

    Args:
        shape(list[int]): shape of the variable
        value(float): the value of the variable. The new created
                      variable will be filled with it.
        dtype(string): data type of the variable
        persistable(bool): if this variable is persistable.
                           Default: False
        force_cpu(bool): force this variable to be on CPU.
                         Default: False
        name(str|None): The name of the variable. If set to None the variable
                        name will be generated automatically.
                        Default: None

    Returns:
        Variable: the created Variable

    Examples:
        .. code-block:: python

            import paddle.fluid as fluid
            import paddle.fluid.layers as layers
            var = layers.create_global_var(shape=[2,3], value=1.0, dtype='float32',
                                          persistable=True, force_cpu=True, name='new_var')
    """
    helper = LayerHelper("global_var", **locals())
    var = helper.create_global_variable(
        dtype=dtype,
        shape=shape,
        persistable=persistable,
        name=name,
        stop_gradient=True)
    helper.set_variable_initializer(
        var, initializer=Constant(
            value=float(value), force_cpu=force_cpu))

    return var


def cast(x, dtype):
    """
    This layer takes in the Variable :attr:`x` with :attr:`x.dtype` and casts
    it to the output with :attr:`dtype`. It's meaningless if the output
    dtype equals the input dtype, but it's fine if you do so.

    Args:
        x (Variable): The input Variable for casting.
        dtype(np.dtype|core.VarDesc.VarType|str): Data type of the output Variable.

    Returns:
        Variable: The output Variable after casting.

    Examples:
        .. code-block:: python

            import paddle.fluid as fluid
            data = fluid.layers.data(name='x', shape=[13], dtype='float32')
            result = fluid.layers.cast(x=data, dtype='float64')
    """
    helper = LayerHelper('cast', **locals())
    out = helper.create_variable_for_type_inference(dtype=dtype)
    helper.append_op(
        type='cast',
        inputs={'X': [x]},
        outputs={'Out': [out]},
        attrs={'in_dtype': x.dtype,
               'out_dtype': out.dtype})
    return out


def concat(input, axis=0, name=None):
    """
    **Concat**

    This function concatenates the input along the axis mentioned
    and returns that as the output.

    Args:
        input(list): List of tensors to be concatenated
        axis(int): Integer axis along which the tensors will be concatenated
        name(str|None): A name for this layer(optional). If set None, the layer
                       will be named automatically.

    Returns:
        Variable: Output variable of the concatenation

    Examples:
        .. code-block:: python

            import paddle.fluid as fluid
            a = fluid.layers.data(name='a', shape=[2, 13], dtype='float32')
            b = fluid.layers.data(name='b', shape=[2, 3], dtype='float32')
            c = fluid.layers.data(name='c', shape=[2, 2], dtype='float32')
            d = fluid.layers.data(name='d', shape=[2, 5], dtype='float32')
            out = fluid.layers.concat(input=[a, b, c, d], axis=2)
    """
    helper = LayerHelper('concat', **locals())
    out = helper.create_variable_for_type_inference(dtype=helper.input_dtype())
    helper.append_op(
        type='concat',
        inputs={'X': input},
        outputs={'Out': [out]},
        attrs={'axis': axis})
    return out


def tensor_array_to_tensor(input, axis=1, name=None):
    """
    This function concatenates the input LodTensorArray along the axis mentioned
    and returns that as the output.

    A simple example as below:

    .. code-block:: text

        Given:

        input.data = {[[0.6, 0.1, 0.3],
                       [0.5, 0.3, 0.2]],
                      [[1.3],
                       [1.8]],
                      [[2.3, 2.1],
                       [2.5, 2.4]]}

        axis = 1

        Then:

        output.data = [[0.6, 0.1, 0.3, 1.3, 2.3, 2.1],
                       [0.5, 0.3, 0.2, 1.8, 2.5, 2.4]]

        output_index.data = [3, 1, 2]

    Args:
        input(list): Input LodTensorArray
        axis(int): Integer axis along which the tensors will be concatenated
        name(str|None): A name for this layer(optional). If set None, the layer
                       will be named automatically.

    Returns:
        Variable: Output variable of the concatenation
        Variable: The input LodTensorArray items' dims along the axis

    Examples:
        .. code-block:: python

            import paddle.fluid as fluid
            tensor_array = fluid.layers.create_parameter(shape=[784, 200], dtype='float32')
            output, output_index = fluid.layers.tensor_array_to_tensor(input=tensor_array)
    """
    helper = LayerHelper('tensor_array_to_tensor', **locals())
    out = helper.create_variable_for_type_inference(dtype=helper.input_dtype())
    out_index = helper.create_variable_for_type_inference(dtype="int32")
    helper.append_op(
        type='tensor_array_to_tensor',
        inputs={'X': input},
        outputs={'Out': [out],
                 'OutIndex': [out_index]},
        attrs={'axis': axis})
    return out, out_index


def sums(input, out=None):
    """
    This function performs the sum operation on the input and returns the
    result as the output.

    Args:
        input (Variable|list): The input tensor that has the elements
                               that need to be summed up.
        out (Variable|None): Output parameter. The sum result.
                             Default: None

    Returns:
        Variable: the sum of input. The same as the argument 'out'

    Examples:
        .. code-block:: python

          import paddle.fluid as fluid

          # sum of several tensors
          a0 = fluid.layers.fill_constant(shape=[1], dtype='int64', value=1)
          a1 = fluid.layers.fill_constant(shape=[1], dtype='int64', value=2)
          a2 = fluid.layers.fill_constant(shape=[1], dtype='int64', value=3)
          sums = fluid.layers.sums(input=[a0, a1, a2])

          # sum of a tensor array
          array = fluid.layers.create_array('int64')
          i = fluid.layers.zeros(shape=[1], dtype='int64', force_cpu=True)
          fluid.layers.array_write(a0, array=array, i=i)
          i = fluid.layers.increment(x=i)
          fluid.layers.array_write(a1, array=array, i=i)
          i = fluid.layers.increment(x=i)
          fluid.layers.array_write(a2, array=array, i=i)
          sums = fluid.layers.sums(input=array)
    """
    helper = LayerHelper('sum', **locals())
    if out is None:
        out = helper.create_variable_for_type_inference(
            dtype=helper.input_dtype())
    helper.append_op(
        type='sum',
        inputs={'X': input},
        outputs={'Out': out},
        attrs={'use_mkldnn': False})
    return out


def assign(input, output=None):
    """
    **Assign**

    This function copies the *input* Variable to the *output* Variable.

    Args:
        input(Variable|numpy.ndarray): The source variable
        output(Variable|None): The destination variable

    Returns:
        Variable: The destination variable that was supplied as the *output*.

    Examples:
        .. code-block:: python

          import paddle.fluid as fluid
          data = fluid.layers.data(name="data", shape=[3, 32, 32], dtype="float32")
          out = fluid.layers.create_tensor(dtype='float32')
          hidden = fluid.layers.fc(input=data, size=10)
          fluid.layers.assign(hidden, out)
    """
    helper = LayerHelper('assign', **locals())
    if output is None:
        output = helper.create_variable_for_type_inference(dtype=input.dtype)
    if isinstance(input, Variable):
        helper.append_op(
            type='assign', inputs={'X': [input]}, outputs={'Out': [output]})
    elif isinstance(input, numpy.ndarray):
        dtype = convert_np_dtype_to_dtype_(input.dtype)
        if dtype == VarDesc.VarType.FP32:
            value_name = "fp32_values"
            values = [float(v) for v in input.flat]
        elif dtype == VarDesc.VarType.INT32:
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
        dtype(np.dtype|core.VarDesc.VarType|str): Data type of the output tensor.
        value(float): The constant value used to initialize the output tensor.
        out(Variable): The output tensor.
        force_cpu(True|False): data should be on CPU if set true.

    Returns:
        Variable: The tensor variable storing the output.

    Examples:
        .. code-block:: python

          import paddle.fluid as fluid
          data = fluid.layers.fill_constant(shape=[1], value=0, dtype='int64')
    """

    helper = LayerHelper("fill_constant", **locals())
    if convert_dtype(dtype) not in [
            'bool', 'float16', 'float32', 'float64', 'int32', 'int64'
    ]:
        raise TypeError(
            "The create data type in fill_constant must be one of 'bool', float16, float32,"
            "float64, int32 or int64, but received %s." % convert_dtype(
                (dtype)))
    if out is None:
        out = helper.create_variable_for_type_inference(dtype=dtype)
    else:
        if not (convert_dtype(dtype) == convert_dtype(out.dtype)):
            raise TypeError(
                "The create data type in op must be same with out type"
                "but received %s and out dtype %s." % (convert_dtype(
                    (dtype), convert_dtype(out.dtype))))
    helper.append_op(
        type='fill_constant',
        inputs={},
        outputs={'Out': [out]},
        attrs={
            'shape': shape,
            'dtype': out.dtype,
            'value': float(value),
            'force_cpu': force_cpu or force_init_on_cpu()
        },
        stop_gradient=True)
    out.stop_gradient = True
    return out


@templatedoc()
def fill_constant_batch_size_like(input,
                                  shape,
                                  dtype,
                                  value,
                                  input_dim_idx=0,
                                  output_dim_idx=0):
    """
    ${comment}

    It also sets *stop_gradient* to True.

    Args:
        input(${input_type}): ${input_comment}.

        shape(${shape_type}): ${shape_comment}.

        dtype(${dtype_type}): ${dtype_comment}.

        value(${value_type}): ${value_comment}.

        input_dim_idx(${input_dim_idx_type}): ${input_dim_idx_comment}.

        output_dim_idx(${output_dim_idx_type}): ${output_dim_idx_comment}.

    Returns:
        ${out_comment}.

    Examples:

        .. code-block:: python

             import paddle.fluid as fluid
             like = fluid.layers.data(name='like', shape=[1], dtype='float32')
             data = fluid.layers.fill_constant_batch_size_like(
                         input=like, shape=[1], value=0, dtype='int64')

    """
    helper = LayerHelper("fill_constant_batch_size_like", **locals())
    out = helper.create_variable_for_type_inference(dtype=dtype)
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


def argmin(x, axis=0):
    """
    **argmin**

    This function computes the indices of the min elements
    of the input tensor's element along the provided axis.

    Args:
        x(Variable): The input to compute the indices of
                     the min elements.
        axis(int): Axis to compute indices along.

    Returns:
        Variable: The tensor variable storing the output

    Examples:
        .. code-block:: python

            import paddle.fluid as fluid
            x = fluid.layers.data(name="x", shape=[3, 4], dtype="float32")
            out = fluid.layers.argmin(x, axis=0)
            out = fluid.layers.argmin(x, axis=-1)
    """
    helper = LayerHelper("arg_min", **locals())
    out = helper.create_variable_for_type_inference(VarDesc.VarType.INT64)
    helper.append_op(
        type='arg_min',
        inputs={'X': x},
        outputs={'Out': [out]},
        attrs={'axis': axis})
    return out


def argmax(x, axis=0):
    """
    **argmax**

    This function computes the indices of the max elements
    of the input tensor's element along the provided axis.

    Args:
        x(Variable): The input to compute the indices of
                     the max elements.
        axis(int): Axis to compute indices along.

    Returns:
        Variable: The tensor variable storing the output

    Examples:
        .. code-block:: python

            import paddle.fluid as fluid
            x = fluid.layers.data(name="x", shape=[3, 4], dtype="float32")
            out = fluid.layers.argmax(x, axis=0)
            out = fluid.layers.argmax(x, axis=-1)
    """
    helper = LayerHelper("arg_max", **locals())
    out = helper.create_variable_for_type_inference(VarDesc.VarType.INT64)
    helper.append_op(
        type='arg_max',
        inputs={'X': x},
        outputs={'Out': [out]},
        attrs={'axis': axis})
    return out


def argsort(input, axis=-1, name=None):
    """
    Performs sorting on the input Variable along the given axis, and outputs
    sorted data Varibale and its corresponding index Variable with the same
    shape as :attr:`input`.

    .. code-block:: text

        For example, the given axis is -1 and the input Variable

            input = [[0.15849551, 0.45865775, 0.8563702 ],
                     [0.12070083, 0.28766365, 0.18776911]],

        after argsort, the sorted Vairable becomes

            out = [[0.15849551, 0.45865775, 0.8563702 ],
                   [0.12070083, 0.18776911, 0.28766365]],

        and the sorted indices along the given axis turn outs to be

            indices = [[0, 1, 2],
                       [0, 2, 1]]

    Args:
        input(Variable): The input Variable for sorting.
        axis(int): The axis along which to sort the input Variable. When
                   :attr:`axis` < 0, the actual axis will be :attr:`axis` +
                   rank(:attr:`input`). Default -1, the last dimension.
        name(str|None): (optional) A name for this layer. If set None, the
                   layer will be named automatically.

    Returns:
        tuple: A tuple of sorted data Variable and the sorted indices.

    Examples:
        .. code-block:: python

            import paddle.fluid as fluid
            x = fluid.layers.data(name="x", shape=[3, 4], dtype="float32")
            out, indices = fluid.layers.argsort(input=x, axis=0)
    """
    helper = LayerHelper("argsort", **locals())
    out = helper.create_variable_for_type_inference(
        dtype=input.dtype, stop_gradient=True)
    ids = helper.create_variable_for_type_inference(
        VarDesc.VarType.INT64, stop_gradient=True)
    helper.append_op(
        type='argsort',
        inputs={'X': input},
        outputs={'Out': out,
                 'Indices': ids},
        attrs={'axis': axis})
    return out, ids


def ones(shape, dtype, force_cpu=False):
    """
    **ones**

    This function creates a tensor of specified *shape* and
    *dtype*, and initializes this with 1.

    It also sets *stop_gradient* to True.

    Args:
        shape(tuple|list): Shape of output tensor
        dtype(np.dtype|core.VarDesc.VarType|str): Data type of output tensor

    Returns:
        Variable: The tensor variable storing the output

    Examples:
        .. code-block:: python

          import paddle.fluid as fluid
          data = fluid.layers.ones(shape=[1], dtype='int64')
    """
    assert isinstance(shape, list) or isinstance(
        shape, tuple), "The shape's type should be list or tuple."
    assert reduce(lambda x, y: x * y,
                  shape) > 0, "The shape is invalid: %s." % (str(shape))
    return fill_constant(value=1.0, **locals())


def zeros(shape, dtype, force_cpu=False):
    """
    **zeros**

    This function creates a tensor of specified *shape* and
    *dtype*, and initializes this with 0.

    It also sets *stop_gradient* to True.

    Args:
        shape(tuple|list|None): Shape of output tensor.
        dtype(np.dtype|core.VarDesc.VarType|str): Data type of output tensor.
        force_cpu(bool, default False): Whether to make output stay on CPU.

    Returns:
        Variable: The tensor variable storing the output.

    Examples:
        .. code-block:: python

          import paddle.fluid as fluid
          data = fluid.layers.zeros(shape=[1], dtype='int64')
    """
    return fill_constant(value=0.0, **locals())


def reverse(x, axis):
    """
    **reverse**

    This function reverse the input 'x' along given axises.

    Args:
        x(Vairbale): the input to be reversed.
        axis(int|tuple|list): Axis that along which order of elements
                    is reversed. If it is a tuple or a list, reversing
                    will be apply on each axis in the tuple or list.

    Returns:
        Variable: The reversed tensor.

    Examples:
        .. code-block:: python

          import paddle.fluid as fluid
          data = fluid.layers.data(name="data", shape=[4, 8], dtype="float32")
          out = fluid.layers.reverse(x=data, axis=0)
          # or:
          out = fluid.layers.reverse(x=data, axis=[0,1])
    """
    if isinstance(axis, int):
        axis = [axis]
    helper = LayerHelper("reverse", **locals())
    out = helper.create_variable_for_type_inference(dtype=x.dtype)
    helper.append_op(
        type='reverse',
        inputs={'X': x},
        outputs={'Out': [out]},
        attrs={'axis': axis})
    return out


def save(x, file_path, overwrite=True):
    """
    Saves a variable as a file.

    Args:
        x(variable): The Tensor/LoDTensor to be saved.
        file_path(str): The file path where the variable will be saved.
        overwrite(bool): Whether or not cover the given file when it has already
            existed. If it's set 'False' and the file is existed, a runtime
            error will be thrown.
    """
    helper = LayerHelper("save", **locals())
    helper.append_op(
        type="save",
        inputs={"input": x},
        outputs={},
        args={"file_path": file_path,
              "overwrite": overwrite})


def save_combine(x, file_path, overwrite=True):
    """
    Saves a list of variables into a single file.

    Args:
        x(list): A list of Tensor/LoDTensor variables to be saved together in
                 a single file.
        file_path(str): The file path where variables will be saved.
        overwrite(bool): Whether or not cover the given file when it has already
            existed. If it's set 'False' and the file is existed, a runtime
            error will be thrown.

    Returns:
        There is no return value.

    Examples:

        .. code-block:: python

            import paddle.fluid as fluid
            v1 = fluid.layers.data(name="data",
                                   shape=(4, 6),
                                   dtype="float32")
            v2 = fluid.layers.data(name="data",
                                   shape=(6, 8, 4),
                                   dtype="float32")
            normed = fluid.layers.save_combine([v1, v2], file_path="output")
    """
    helper = LayerHelper("save_combine", **locals())
    helper.append_op(
        type="save_combine",
        inputs={"input": x},
        outputs={},
        args={"file_path": file_path,
              "overwrite": overwrite})


def load_combine(out, file_path):
    """
    Loads a list of vairables from a single file.

    Args:
        out(list): The list of variables to be read from the disk file.
        file_path(str): The path of the disk file.
    """
    helper = LayerHelper("load_combine", **locals())
    helper.append_op(
        type="load_combine",
        inputs={},
        output={"Out": out},
        args={"file_path": file_path})


def has_inf(x):
    """
    Test if any of x contains an infinity number

    Args:
       x(variable): The Tensor/LoDTensor to be checked.

    Returns:
        Variable: The tensor variable storing the output, only a bool value.
    
    Examples:
        .. code-block:: python
          
          import paddle.fluid as fluid
          data = fluid.layers.data(name="input", shape=[4, 32, 32], dtype="float32")
          res = fluid.layers.has_inf(data)

    """
    helper = LayerHelper("isinf", **locals())
    out = helper.create_variable_for_type_inference(dtype=x.dtype)
    helper.append_op(type="isinf", inputs={"X": x}, outputs={"Out": out})
    return out


def has_nan(x):
    """
    Test if any of x contains a NAN

    Args:
       x(variable): The Tensor/LoDTensor to be checked.

    Returns:
        Variable: The tensor variable storing the output, only a bool value.
    
    Examples:
        .. code-block:: python
    
          import paddle.fluid as fluid
          data = fluid.layers.data(name="input", shape=[4, 32, 32], dtype="float32")
          res = fluid.layers.has_nan(data)

    """
    helper = LayerHelper("isnan", **locals())
    out = helper.create_variable_for_type_inference(dtype=x.dtype)
    helper.append_op(type="isnan", inputs={"X": x}, outputs={"Out": out})
    return out


def isfinite(x):
    """
    Test if any of x contains an infinity/NAN number. If all the elements are finite,
    returns true, else false.

    Args:
       x(variable): The Tensor/LoDTensor to be checked.

    Returns:
        Variable: The tensor variable storing the output, contains a bool value.

    Examples:

        .. code-block:: python

            import paddle.fluid as fluid
            var = fluid.layers.data(name="data",
                                    shape=(4, 6),
                                    dtype="float32")
            out = fluid.layers.isfinite(var)
    """
    helper = LayerHelper("isfinite", **locals())
    out = helper.create_variable_for_type_inference(dtype=x.dtype)
    helper.append_op(type="isfinite", inputs={"X": x}, outputs={"Out": out})
    return out


def range(start, end, step, dtype):
    """
    Return evenly spaced values within a given interval.

    Values are generated within the half-open interval [start, stop) (in other words,
    the interval including start but excluding stop).

    Parameters:
        start(float32 | float64 | int32 | int64 | Variable): Start of interval. The interval includes this value.
            when start is Variable, it is a 1-D Tensor with shape [1].
        end(float32 | float64 | int32 | int64 | Variable): End of interval. The interval does not include this
                                 value, except in some cases where step is not an integer
                                 and floating point round-off affects the length of out. When end is Variable,
                                 it is a 1-D Tensor with shape [1].
        step(float32 | float64 | int32 | int64 | Variable): Spacing between values. For any output out, this is the
                                  distance between two adjacent values, out[i+1] - out[i].
        dtype(str): the data type of the output tensor, can be float32, float64, int32, int64.

    Returns: a 1-D Tensor which is evenly spaced values within a given interval. Its data type is set by dtype.
    
    Return type: Variable

    examples:

        .. code-block:: python

             import paddle.fluid as fluid
             data = fluid.layers.range(0, 10, 2, 'int32')

    """
    helper = LayerHelper("range", **locals())

    if not isinstance(start, Variable):
        start = fill_constant([1], dtype, start)
    if not isinstance(end, Variable):
        end = fill_constant([1], dtype, end)
    if not isinstance(step, Variable):
        step = fill_constant([1], dtype, step)

    out = helper.create_variable_for_type_inference(dtype=start.dtype)

    helper.append_op(
        type='range',
        inputs={'Start': start,
                'End': end,
                'Step': step},
        outputs={'Out': [out]})
    out.stop_gradient = True
    return out


def linspace(start, stop, num, dtype):
    """
    This OP return fixed number of evenly spaced values within a given interval.

    Args:
        start(float|Variable): The input :attr:`start` is start variable of range. It is a float scalar, \
            or a tensor of shape [1] with input data type float32, float64.
        stop(float|Variable): The input :attr:`stop` is start variable of range. It is a float scalar, \
            or a tensor of shape [1] with input data type float32, float64.
        num(int|Variable): The input :attr:`num` is given num of the sequence. It is an int scalar, \
            or a tensor of shape [1] with type int32.
        dtype(string): The data type of output tensor, it could be 'float32' and 'float64'.

    Returns:
        Variable, the output data type will be float32, float64.: The 1-D tensor with fixed number of evenly spaced values, \
        the data shape of this tensor is :math:`[num]` . If the :attr:`num` is set 1, the output tensor just has \
        the value with input :attr:`start`. 

    Examples:
        .. code-block:: python

             import paddle.fluid as fluid
             data = fluid.layers.linspace(0, 10, 5, 'float32') # [0.0,  2.5,  5.0,  7.5, 10.0]
             data = fluid.layers.linspace(0, 10, 1, 'float32') # [0.0]

    """
    helper = LayerHelper("linspace", **locals())

    if not isinstance(start, Variable):
        start = fill_constant([1], dtype, start)
    if not isinstance(stop, Variable):
        stop = fill_constant([1], dtype, stop)
    if not isinstance(num, Variable):
        num = fill_constant([1], 'int32', num)

    out = helper.create_variable_for_type_inference(dtype=start.dtype)

    helper.append_op(
        type='linspace',
        inputs={'Start': start,
                'Stop': stop,
                'Num': num},
        outputs={'Out': [out]})
    return out


def zeros_like(x, out=None):
    """
    This OP creates a zeros tensor which has identical shape and dtype 
    with `x`.

    Args:
        x(Variable): The input tensor which specifies shape and dtype, the input data dtype could be bool, float32, float64, int32, int64.
        out(Variable, optional): If is :attr:`None` , the op will create the variable as output, the data type and shape of \
            this variable will be same as input :attr:`x`. If is a tensor, the data type and shape need to be same as input :attr:`x`. 
            The defalut value is :attr:`None` .

    Returns:
        Variable: The N-D tensor, the element in tensor is related to input data type, if the input data type is bool, \
            the output value is False, otherwise is zero. The output shape is the same as the input.

    Examples:
        .. code-block:: python

          import paddle.fluid as fluid
          x = fluid.data(name='x', dtype='float32', shape=[3])
          data = fluid.layers.zeros_like(x) # [0.0, 0.0, 0.0]

    """

    helper = LayerHelper("zeros_like", **locals())
    if out is None:
        out = helper.create_variable_for_type_inference(dtype=x.dtype)
    helper.append_op(
        type='fill_zeros_like', inputs={'X': [x]}, outputs={'Out': [out]})
    out.stop_gradient = True
    return out


def diag(diagonal):
    """
    This OP creates a square matrix which has diagonal values specified by input :attr:`diagonal`.

    Args:
        diagonal(Variable|numpy.ndarray): The input tensor should be 1D tensor, the input shape is :math:`[ N]` , \
            specifying diagonal values by this input tensor. The input data type should be float32, float64, int32, int64.

    Returns:
        Variable, the output data type is the same as input data type.: The tensor variable storing the square matrix, \
            the diagonal values specified by input :attr:`diagonal`. the output shape is :math:`[N, N]` with two dims.

    Examples:
        .. code-block:: python

          # [[3, 0, 0]
          #  [0, 4, 0]
          #  [0, 0, 5] 

          import paddle.fluid as fluid
          import numpy as np
          diagonal = np.arange(3, 6, dtype='int32')
          data = fluid.layers.diag(diagonal)
          # diagonal.shape=(3,) data.shape=(3, 3)

    """

    helper = LayerHelper("diag", **locals())

    if not isinstance(diagonal, Variable):
        diagonal = assign(diagonal)

    out = helper.create_variable_for_type_inference(dtype=diagonal.dtype)

    helper.append_op(
        type='diag', inputs={'Diagonal': [diagonal]}, outputs={'Out': [out]})

    out.stop_gradient = True
    return out


def eye(num_rows, num_columns=None, batch_shape=None, dtype='float32'):
    """
    **eye**

    This function constructs an identity tensor, or a batch of tensor.

    Args:
        num_rows(int): the number of rows in each batch tensor.
        num_columns(int): the number of columns in each batch tensor.
                          If None, default: num_rows.
        batch_shape(list(int)): If provided, the returned tensor will have a leading
                                batch size of this shape.
        dtype(string): The data type of the returned tensor.
                       It should be int32, int64, float16, float32, float64.

    Returns:
        Variable: An identity Tensor or LoDTensor of shape batch_shape + [num_rows, num_columns].

    Examples:
        .. code-block:: python

          import paddle.fluid as fluid
          data = fluid.layers.eye(3, dtype='int32')
          # [[1, 0, 0]
          #  [0, 1, 0]
          #  [0, 0, 1]]

          data = fluid.layers.eye(2, 3, dtype='int32')
          # [[1, 0, 0]
          #  [0, 1, 0]]

          data = fluid.layers.eye(2, batch_shape=[3])
          # Construct a batch of 3 identity tensors, each 2 x 2.
          # data[i, :, :] is a 2 x 2 identity tensor, i = 0, 1, 2.

    """

    helper = LayerHelper("eye", **locals())
    if not isinstance(num_rows, int) or num_rows < 0:
        raise TypeError("num_rows should be a non-negative int")
    if num_columns is not None:
        if not isinstance(num_columns, int) or num_columns < 0:
            raise TypeError("num_columns should be a non-negative int")
    else:
        num_columns = num_rows
    out = helper.create_variable_for_type_inference(dtype=dtype)
    c_dtype = convert_np_dtype_to_dtype_(dtype)
    helper.append_op(
        type='eye',
        inputs={},
        outputs={'Out': [out]},
        attrs={
            'num_rows': num_rows,
            'num_columns': num_columns,
            'dtype': c_dtype
        },
        stop_gradient=True)
    out.stop_gradient = True

    if batch_shape is not None:
        if not isinstance(batch_shape, list):
            raise TypeError("batch_shape should be a list")
        from .nn import stack
        for batch_val in reversed(batch_shape):
            if batch_val <= 0:
                raise TypeError("batch_shape should be a positive int list")
            else:
                stack_vars = [out for _ in numpy.arange(batch_val)]
                out = stack(stack_vars, axis=0)
    return out


def ones_like(x, out=None):
    """
    **ones_like**

    This function creates a ones tensor which has identical shape and dtype 
    with `x`.

    Args:
        x(Variable): The input tensor which specifies shape and dtype.
        out(Variable): The output tensor.

    Returns:
        out(Variable): The tensor variable storing the output.

    Examples:
        .. code-block:: python

          import paddle.fluid as fluid

          x = fluid.layers.data(name='x', dtype='float32', shape=[3], append_batch_size=False)
          data = fluid.layers.ones_like(x) # [1.0, 1.0, 1.0]

    """

    helper = LayerHelper("ones_like", **locals())
    if out is None:
        out = helper.create_variable_for_type_inference(dtype=x.dtype)
    helper.append_op(
        type='fill_any_like',
        inputs={'X': [x]},
        attrs={'value': 1.0},
        outputs={'Out': [out]})
    return out
