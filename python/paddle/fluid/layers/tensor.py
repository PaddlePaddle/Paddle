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
from ..framework import convert_np_dtype_to_dtype_, in_dygraph_mode, _varbase_creator
from ..framework import Variable
from ..initializer import Constant, force_init_on_cpu
from ..core import VarDesc
from .. import core
from .layer_function_generator import templatedoc
from . import utils
from ..data_feeder import check_variable_and_dtype, check_type, check_dtype, convert_dtype
import numpy
import warnings

__all__ = [
    'create_tensor', 'create_parameter', 'create_global_var', 'cast',
    'tensor_array_to_tensor', 'concat', 'sums', 'assign',
    'fill_constant_batch_size_like', 'fill_constant', 'argmin', 'argmax',
    'argsort', 'ones', 'zeros', 'reverse', 'has_inf', 'has_nan', 'isfinite',
    'range', 'linspace', 'zeros_like', 'ones_like', 'diag', 'eye'
]


def create_tensor(dtype, name=None, persistable=False):
    """
    Create a variable, which will hold a Tensor with data type dtype.

    Args:
        dtype(string|numpy.dtype): the data type of Tensor to be created, the
            data type is bool, float16, float32, float64, int8, int16, int32 and int64.
        name(string, optional): The default value is None.  Normally there is no need for 
            user to set this property.  For more information, please refer to :ref:`api_guide_Name`
        persistable(bool): Set the persistable flag of the create tensor.
            default value is False.

    Returns:
        Variable: The tensor to be created according to dtype.

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
    This function creates a parameter. The parameter is a learnable variable, which can have
    gradient, and can be optimized.

    NOTE: this is a very low-level API. This API is useful when you create
    operator by your self. instead of using layers.

    Parameters:
        shape (list of int): Shape of the parameter
        dtype (str): Data type of the parameter
        name (str, optional): For detailed information, please refer to
           :ref:`api_guide_Name` . Usually name is no need to set and None by default.
        attr (ParamAttr, optional): Attributes of the parameter
        is_bias (bool, optional): This can affect which default initializer is chosen
                       when default_initializer is None. If is_bias,
                       initializer.Constant(0.0) will be used. Otherwise,
                       Xavier() will be used.
        default_initializer (Initializer, optional): Initializer for the parameter

    Returns:
        The created parameter.

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
    This function creates a new tensor variable with value in the global block(block 0).

    Parameters:
        shape (list of int): Shape of the variable
        value (float): The value of the variable. The new created
                      variable will be filled with it.
        dtype (str): Data type of the variable
        persistable (bool, optional): If this variable is persistable.
                           Default: False
        force_cpu (bool, optional): Force this variable to be on CPU.
                         Default: False
        name (str, optional): For detailed information, please refer to
           :ref:`api_guide_Name` . Usually name is no need to set and None by default.

    Returns:
        Variable: The created Variable

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
    This OP takes in the Variable :attr:`x` with :attr:`x.dtype` and casts it
    to the output with :attr:`dtype`. It's meaningless if the output dtype
    equals the input dtype, but it's fine if you do so.

    Args:
        x(Variable): An input N-D Tensor with data type bool, float16,
            float32, float64, int32, int64, uint8.
        dtype(np.dtype|core.VarDesc.VarType|str): Data type of the output:
            bool, float15, float32, float64, int8, int32, int64, uint8.

    Returns:
        Variable: A Tensor with the same shape as input's.

    Examples:
        .. code-block:: python

            import paddle.fluid as fluid
            import numpy as np

            place = fluid.core.CPUPlace()

            x_lod = fluid.data(name="x", shape=[2,2], lod_level=0)
            cast_res1 = fluid.layers.cast(x=x_lod, dtype="uint8")
            cast_res2 = fluid.layers.cast(x=x_lod, dtype=np.int32)

            exe = fluid.Executor(place)
            exe.run(fluid.default_startup_program())

            x_i_lod = fluid.core.LoDTensor()
            x_i_lod.set(np.array([[1.3,-2.4],[0,4]]).astype("float32"), place)
            x_i_lod.set_recursive_sequence_lengths([[0,2]])
            res1 = exe.run(fluid.default_main_program(), feed={'x':x_i_lod}, fetch_list=[cast_res1], return_numpy=False)
            res2 = exe.run(fluid.default_main_program(), feed={'x':x_i_lod}, fetch_list=[cast_res2], return_numpy=False)
            print(np.array(res1[0]), np.array(res1[0]).dtype)
            # [[  1 254]
            #  [  0   4]] uint8
            print(np.array(res2[0]), np.array(res2[0]).dtype)
            # [[ 1 -2]
            #  [ 0  4]] int32
    """
    helper = LayerHelper('cast', **locals())
    check_variable_and_dtype(
        x, 'x',
        ['bool', 'float16', 'float32', 'float64', 'int32', 'int64', 'uint8'],
        'cast')
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

    This OP concatenates the input along the axis.

    Args:
        input(list): List of input Tensors with data type float32, float64, int32,
            int64.
        axis(int32|Variable, optional):  A scalar with type ``int32`` or a ``Tensor`` with shape [1] and type ``int32``. Axis to compute indices along. The effective range
            is [-R, R), where R is Rank(x). when axis<0, it works the same way
            as axis+R. Default is 0.
        name (str, optional): The default value is None. Normally there is no
            need for user to set this property. For more information, please
            refer to :ref:`api_guide_Name`.

    Returns:
        Variable: A Tensor with the same data type as input's.

    Examples:
        .. code-block:: python

            import paddle.fluid as fluid
            import numpy as np

            in1 = np.array([[1,2,3],
                            [4,5,6]])
            in2 = np.array([[11,12,13],
                            [14,15,16]])
            in3 = np.array([[21,22],
                            [23,24]])
            with fluid.dygraph.guard():
                x1 = fluid.dygraph.to_variable(in1)
                x2 = fluid.dygraph.to_variable(in2)
                x3 = fluid.dygraph.to_variable(in3)
                out1 = fluid.layers.concat(input=[x1,x2,x3], axis=-1)
                out2 = fluid.layers.concat(input=[x1,x2], axis=0)
                print(out1.numpy())
                # [[ 1  2  3 11 12 13 21 22]
                #  [ 4  5  6 14 15 16 23 24]]
                print(out2.numpy())
                # [[ 1  2  3]
                #  [ 4  5  6]
                #  [11 12 13]
                #  [14 15 16]]
    """

    if in_dygraph_mode():
        inputs = {'X': input}
        if not isinstance(axis, int):
            raise TypeError(
                "Input 'axis' in concat must be int in Dygraph mode.")
        attrs = {'axis': axis}
        outs = core.ops.concat(inputs, attrs)
        return outs['Out'][0]

    if not isinstance(input, list):
        warnings.warn(
            "The type of input in concat should be list, but received %s." %
            (type(input)))
        input = [input]
    for id, x in enumerate(input):
        check_variable_and_dtype(
            x, 'input[' + str(id) + ']',
            ['float16', 'float32', 'float64', 'int32', 'int64'], 'concat')
    check_type(axis, 'axis', (int, Variable), 'concat')
    inputs = {'X': input}
    attrs = {}
    if isinstance(axis, Variable):
        axis.stop_gradient = True
        inputs['AxisTensor'] = axis
    else:
        attrs['axis'] = axis

    helper = LayerHelper('concat', **locals())
    out = helper.create_variable_for_type_inference(dtype=helper.input_dtype())
    helper.append_op(
        type='concat', inputs=inputs, outputs={'Out': [out]}, attrs=attrs)
    return out


def tensor_array_to_tensor(input, axis=1, name=None, use_stack=False):
    """
    This function concatenates or stacks all tensors in the input LoDTensorArray
    along the axis mentioned and returns that as the output.

    For Example:

    .. code-block:: text

        Case 1:

            Given:

                input.data = {[[0.6, 0.1, 0.3],
                               [0.5, 0.3, 0.2]],
                              [[1.3],
                               [1.8]],
                              [[2.3, 2.1],
                               [2.5, 2.4]]}

                axis = 1, use_stack = False

            Then:

                output.data = [[0.6, 0.1, 0.3, 1.3, 2.3, 2.1],
                               [0.5, 0.3, 0.2, 1.8, 2.5, 2.4]]

                output_index.data = [3, 1, 2]

        Case 2:

            Given:

                input.data = {[[0.6, 0.1],
                               [0.5, 0.3]],
                              [[0.3, 1.3],
                               [0.2, 1.8]],
                              [[2.3, 2.1],
                               [2.5, 2.4]]}

                axis = 1, use_stack = True

            Then:

                output.data = [[[0.6, 0.1]
                                [0.3, 1.3]
                                [2.3, 2.1],
                               [[0.5, 0.3]
                                [0.2, 1.8]
                                [2.5, 2.4]]]

                output_index.data = [2, 2, 2]

    Args:
        input(Variable): A LodTensorArray variable.
        axis(int): The axis along which the tensors in attr::`input` will be
            concatenated or stacked.
        name(str|None): A name for this layer(optional). If set None, the layer
                       will be named automatically.
        use_stack(bool): Act as concat_op or stack_op. For stack mode, all
            tensors in the tensor array must have the same shape.

    Returns:
        Variable: The concatenated or stacked tensor variable.
        Variable: A 1-D tensor variable with int32 data type. The data in this \
            tensor contains all input including tensors' sizes along the axis.

    Examples:
        .. code-block:: python

            import paddle.fluid as fluid
            import numpy as np
            x0 = fluid.layers.assign(np.random.rand(2, 2).astype("float32"))
            x1 = fluid.layers.assign(np.random.rand(2, 2).astype("float32"))
            i = fluid.layers.fill_constant(shape=[1], dtype="int64", value=0)
            array = fluid.layers.create_array(dtype='float32')
            fluid.layers.array_write(x0, i, array)
            fluid.layers.array_write(x1, i + 1, array)
            output, output_index = fluid.layers.tensor_array_to_tensor(input=array)
    """
    helper = LayerHelper('tensor_array_to_tensor', **locals())
    out = helper.create_variable_for_type_inference(dtype=helper.input_dtype())
    out_index = helper.create_variable_for_type_inference(dtype="int32")
    helper.append_op(
        type='tensor_array_to_tensor',
        inputs={'X': input},
        outputs={'Out': [out],
                 'OutIndex': [out_index]},
        attrs={'axis': axis,
               'use_stack': use_stack})
    return out, out_index


def sums(input, out=None):
    """
    This function computes the sum of multiple input Tensors elementwisely.

    - Case 1, sum of 3 Tensors

    .. code-block:: text

        # Input Tensors
        x0.shape = [2, 3]
        x0.data = [[1., 2., 3.],
                   [4., 5., 6.]]
        x1.shape = [2, 3]
        x1.data = [[10., 20., 30.],
                   [40., 50., 60.]]
        x2.shape = [2, 3]
        x2.data = [[100., 200., 300.],
                   [400., 500., 600.]]

        # Output Tensor
        out.shape = [2, 3]
        out.data = [[111., 222., 333.],
                    [444., 555., 666.]]

    Args:
        input (list): A list of Variables which hold input Tensors with the same
            data type and shape. Optional data types are: float32, float64, int32, int64.
        out (Variable, optional): Output Tensor. It can be any existing Variable.
            The default value is None, then a new Variable will be created and returned.

    Returns:
        Variable: The sum of inputs. The shape and data type is the same with input. \
            If :code:`out` is not None, the returned value is :code:`out` .

    Examples:
        .. code-block:: python

            import paddle.fluid as fluid

            x0 = fluid.layers.fill_constant(shape=[16, 32], dtype='int64', value=1)
            x1 = fluid.layers.fill_constant(shape=[16, 32], dtype='int64', value=2)
            x2 = fluid.layers.fill_constant(shape=[16, 32], dtype='int64', value=3)
            x3 = fluid.layers.fill_constant(shape=[16, 32], dtype='int64', value=0)

            # Sum of multiple Tensors, the result is stored to a new Variable sum0 (sum0=x0+x1+x2, the value is [[6, ..., 6], ..., [6, ..., 6]])
            sum0 = fluid.layers.sums(input=[x0, x1, x2])

            # Sum of multiple Tensors, sum1 and x3 represents the same Variable (x3=x0+x1+x2, the value is [[6, ..., 6], ..., [6, ..., 6]])
            sum1 = fluid.layers.sums(input=[x0, x1, x2], out=x3)
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
    The OP copies the :attr:`input` to the :attr:`output`.

    Parameters:
        input (Variable|numpy.ndarray): A tensor or numpy ndarray, its data type supports
            float32, float64, int32 and int64.
        output (Variable, optional): A tensor. If :attr:`output` is None, a new tensor will
            be created as :attr:`output`. Default: None.

    Returns:
        Variable: A tensor with the same shape, data type and value as :attr:`input`.

    Examples:
        .. code-block:: python

          import paddle.fluid as fluid
          import numpy as np
          data = fluid.layers.fill_constant(shape=[3, 2], value=2.5, dtype='float64') # [[2.5, 2.5], [2.5, 2.5], [2.5, 2.5]]
          result1 = fluid.layers.create_tensor(dtype='float64')
          fluid.layers.assign(data, result1) # result1 = [[2.5, 2.5], [2.5, 2.5], [2.5, 2.5]]
          result2 = fluid.layers.assign(data)  # result2 = [[2.5, 2.5], [2.5, 2.5], [2.5, 2.5]]
          result3 = fluid.layers.assign(np.array([[2.5, 2.5], [2.5, 2.5], [2.5, 2.5]], dtype='float32')) # result3 = [[2.5, 2.5], [2.5, 2.5], [2.5, 2.5]]
    """
    helper = LayerHelper('assign', **locals())
    check_type(input, 'input', (Variable, numpy.ndarray), 'assign')
    if isinstance(input, Variable):
        check_dtype(input.dtype, 'input',
                    ['float32', 'float64', 'int32', 'int64', 'bool'], 'assign',
                    '(When the type of input in assign is Variable.)')
        if output is None:
            output = helper.create_variable_for_type_inference(
                dtype=input.dtype)
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
            raise TypeError(
                "When the type of 'input' in assign is numpy.ndarray, "
                "the data type of 'input' must be float32 or int32, but "
                "received %s." % convert_dtype(dtype))
        if input.size > 1024 * 1024:
            raise ValueError("The size of input is too big. Please consider "
                             "saving it to file and 'load_op' to load it")
        if output is None:
            output = helper.create_variable_for_type_inference(
                dtype=input.dtype)
        helper.append_op(
            type='assign_value',
            outputs={'Out': [output]},
            attrs={
                'dtype': dtype,
                'shape': list(input.shape),
                value_name: values
            })

    return output


def fill_constant(shape, dtype, value, force_cpu=False, out=None):
    """
    This OP creates a Tensor with specified `shape` and `dtype`, and
    initializes it with a constant specified by `value`.

    The attribute `stop_gradient` of the created Tensor is set to True.

    Args:
        shape(list|tuple|Variable): Shape of the Tensor to be created.
                The data type is ``int32`` or ``int64`` . If ``shape`` is a list or tuple,
                the elements of it should be integers or Tensors with shape [1].
                If ``shape`` is an Variable, it should be an 1-D Tensor .
        dtype(np.dtype|core.VarDesc.VarType|str): Data type of the output tensor which can
            be float16, float32, float64, int32, int64.
        value(float): The constant value used to initialize the Tensor to be created.
        force_cpu(True): data should be on CPU if it's true, default value is False.
        out(Variable, optional): Optional output which can be any created 
            Variable that meets the requirements to store the result of operation.
            if out is None, a new Varibale will be create to store the result.

    Returns:
        Variable: Tensor which is created according to shape and dtype.

    Raise:
        TypeError: The dtype must be one of bool, float16, float32, float64, int32 and int64
        and the data type of out Tensor must be the same as the dtype. 

    Examples:
        .. code-block:: python

          import paddle.fluid as fluid
          # attr shape is a list which doesn't contain Variable Tensor.
          data1 = fluid.layers.fill_constant(shape=[2,1], value=0, dtype='int64') # data1=[[0],[0]]
          data2 = fluid.layers.fill_constant(shape=[2,1], value=5, dtype='int64', out=data1)
          # data1=[[0], [0]] data2=[[5], [5]]

          # attr shape is a list which contains Variable Tensor.
          positive_2 = fluid.layers.fill_constant([1], "int32", 2)
          data3 = fluid.layers.fill_constant(shape=[1, positive_2], dtype='float32', value=1.5) # data3=[1.5, 1.5]

          # attr shape is an Variable Tensor.
          shape = fluid.layers.fill_constant([1,2], "int32", 2) # shape=[2,2]
          data4 = fluid.layers.fill_constant(shape=shape, dtype='bool', value=True) # data4=[[True,True],[True,True]]
    """
    attrs = {
        'value': float(value),
        'force_cpu': force_cpu or force_init_on_cpu()
    }

    if convert_dtype(dtype) in ['int64', 'int32']:
        attrs['str_value'] = str(int(value))
    else:
        attrs['str_value'] = str(float(value))

    if in_dygraph_mode():
        if isinstance(shape, (list, tuple)):
            if utils._contain_var(shape):
                raise TypeError(
                    "The type of 'shape' in fill_constant must be list[int] or tuple(int) in Dygraph mode, but "
                    "received %s, which contains Variable." % type(shape))
            attrs['shape'] = shape
        else:
            raise TypeError(
                "The type of 'shape' in fill_constant must be list[int] or tuple(int) in Dygraph mode, but "
                "received %s." % type(shape))
        if out is None:
            out = _varbase_creator(dtype=dtype)
        attrs['dtype'] = out.dtype
        outputs = {'Out': [out]}
        outs = core.ops.fill_constant({}, attrs, outputs)
        out.stop_gradient = True
        return out

    helper = LayerHelper("fill_constant", **locals())
    check_dtype(dtype, 'create data type',
                ['bool', 'float16', 'float32', 'float64', 'int32', 'int64'],
                'fill_constant')
    check_type(shape, 'shape', (Variable, list, tuple), 'fill_constant')
    inputs = {}
    attrs = {
        'value': float(value),
        'force_cpu': force_cpu or force_init_on_cpu()
    }

    if convert_dtype(dtype) in ['int64', 'int32']:
        attrs['str_value'] = str(int(value))
    else:
        attrs['str_value'] = str(float(value))

    def _get_attr_shape(list_shape):
        attr_shape = []
        for idx, dim in enumerate(list_shape):
            if isinstance(dim, Variable):
                attr_shape.append(-1)
            else:
                attr_shape.append(dim)
        return attr_shape

    def _get_shape_tensor(list_shape):
        new_shape_tensor = []
        for idx, dim in enumerate(list_shape):
            if isinstance(dim, Variable):
                dim.stop_gradient = True
                check_dtype(
                    dim.dtype, 'shape[' + str(idx) + ']', ['int32', 'int64'],
                    'fill_constant',
                    '(When type of shape in fill_constant is list or tuple.)')
                if convert_dtype(dim.dtype) == 'int64':
                    dim = cast(x=dim, dtype='int32')
                new_shape_tensor.append(dim)
            else:
                temp_out = helper.create_variable_for_type_inference('int32')
                fill_constant([1], 'int32', dim, force_cpu=True, out=temp_out)
                new_shape_tensor.append(temp_out)
        return new_shape_tensor

    if isinstance(shape, Variable):
        shape.stop_gradient = True
        check_dtype(shape.dtype, 'shape', ['int32', 'int64'], 'fill_constant',
                    '(When type of shape in fill_constant is Variable.)')
        if (convert_dtype(shape.dtype) == 'int64'):
            shape = cast(shape, 'int32')
        inputs["ShapeTensor"] = shape
    elif isinstance(shape, (list, tuple)):
        assert len(shape) > 0, (
            "The size of 'shape' in fill_constant can't be zero, "
            "but received %s." % len(shape))
        attrs["shape"] = _get_attr_shape(shape)
        if utils._contain_var(shape):
            inputs['ShapeTensorList'] = _get_shape_tensor(shape)

    if out is None:
        out = helper.create_variable_for_type_inference(dtype=dtype)
    else:
        check_dtype(
            dtype, 'create data type',
            convert_dtype(out.dtype), 'fill_constant',
            '(The create data type in fill_constant must be the same with out data type.)'
        )
    attrs['dtype'] = out.dtype
    helper.append_op(
        type='fill_constant',
        inputs=inputs,
        outputs={'Out': [out]},
        attrs=attrs,
        stop_gradient=True)
    out.stop_gradient = True
    return out


@templatedoc()
def fill_constant_batch_size_like(input,
                                  shape,
                                  dtype,
                                  value,
                                  input_dim_idx=0,
                                  output_dim_idx=0,
                                  force_cpu=False):
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
    helper = LayerHelper("fill_constant_batch_size_like", **locals())
    out = helper.create_variable_for_type_inference(dtype=dtype)
    attrs = {
        'shape': shape,
        'dtype': out.dtype,
        'value': float(value),
        'input_dim_idx': input_dim_idx,
        'output_dim_idx': output_dim_idx,
        'force_cpu': force_cpu or force_init_on_cpu()
    }
    if convert_dtype(dtype) in ['int64', 'int32']:
        attrs['str_value'] = str(int(value))
    else:
        attrs['str_value'] = str(float(value))
    helper.append_op(
        type='fill_constant_batch_size_like',
        inputs={'Input': input},
        outputs={'Out': [out]},
        attrs=attrs)
    out.stop_gradient = True
    return out


def argmin(x, axis=0):
    """
    **argmin**

    This OP computes the indices of the min elements of the input tensor's
    element along the provided axis.

    Args:
        x(Variable): An input N-D Tensor with type float32, float64, int16,
            int32, int64, uint8.
        axis(int, optional): Axis to compute indices along. The effective range
            is [-R, R), where R is Rank(x). when axis<0, it works the same way
            as axis+R. Default is 0.

    Returns:
        Variable: A Tensor with data type int64.

    Examples:
        .. code-block:: python

            import paddle.fluid as fluid
            import numpy as np

            in1 = np.array([[[5,8,9,5],
                            [0,0,1,7],
                            [6,9,2,4]],
                            [[5,2,4,2],
                            [4,7,7,9],
                            [1,7,0,6]]])
            with fluid.dygraph.guard():
                x = fluid.dygraph.to_variable(in1)
                out1 = fluid.layers.argmin(x=x, axis=-1)
                out2 = fluid.layers.argmin(x=x, axis=0)
                out3 = fluid.layers.argmin(x=x, axis=1)
                out4 = fluid.layers.argmin(x=x, axis=2)
                print(out1.numpy())
                # [[0 0 2]
                #  [1 0 2]]
                print(out2.numpy())
                # [[0 1 1 1]
                #  [0 0 0 0]
                #  [1 1 1 0]]
                print(out3.numpy())
                # [[1 1 1 2]
                #  [2 0 2 0]]
                print(out4.numpy())
                # [[0 0 2]
                #  [1 0 2]]
    """
    helper = LayerHelper("arg_min", **locals())
    out = helper.create_variable_for_type_inference(VarDesc.VarType.INT64)
    helper.append_op(
        type='arg_min',
        inputs={'X': x},
        outputs={'Out': [out]},
        attrs={'axis': axis})
    out.stop_gradient = True
    return out


def argmax(x, axis=0):
    """
    **argmax**

    This OP computes the indices of the max elements of the input tensor's
    element along the provided axis.

    Args:
        x(Variable): An input N-D Tensor with type float32, float64, int16,
            int32, int64, uint8.
        axis(int, optional): Axis to compute indices along. The effective range
            is [-R, R), where R is Rank(x). when axis<0, it works the same way
            as axis+R. Default is 0.

    Returns:
        Variable: A Tensor with data type int64.

    Examples:
        .. code-block:: python

            import paddle.fluid as fluid
            import numpy as np

            in1 = np.array([[[5,8,9,5],
                            [0,0,1,7],
                            [6,9,2,4]],
                            [[5,2,4,2],
                            [4,7,7,9],
                            [1,7,0,6]]])
            with fluid.dygraph.guard():
                x = fluid.dygraph.to_variable(in1)
                out1 = fluid.layers.argmax(x=x, axis=-1)
                out2 = fluid.layers.argmax(x=x, axis=0)
                out3 = fluid.layers.argmax(x=x, axis=1)
                out4 = fluid.layers.argmax(x=x, axis=2)
                print(out1.numpy())
                # [[2 3 1]
                #  [0 3 1]]
                print(out2.numpy())
                # [[0 0 0 0]
                #  [1 1 1 1]
                #  [0 0 0 1]]
                print(out3.numpy())
                # [[2 2 0 1]
                #  [0 1 1 1]]
                print(out4.numpy())
                # [[2 3 1]
                #  [0 3 1]]
    """
    helper = LayerHelper("arg_max", **locals())
    out = helper.create_variable_for_type_inference(VarDesc.VarType.INT64)
    helper.append_op(
        type='arg_max',
        inputs={'X': x},
        outputs={'Out': [out]},
        attrs={'axis': axis})
    out.stop_gradient = True
    return out


def argsort(input, axis=-1, descending=False, name=None):
    """
    This OP sorts the input along the given axis, and returns sorted output
    data Varibale and its corresponding index Variable with the same shape as
    :attr:`input`.

    Args:
        input(Variable): An input N-D Tensor with type float32, float64, int16,
            int32, int64, uint8.
        axis(int, optional): Axis to compute indices along. The effective range
            is [-R, R), where R is Rank(x). when axis<0, it works the same way
            as axis+R. Default is 0.
        descending(bool, optional) : Descending is a flag, if set to true,
            algorithm will sort by descending order, else sort by
            ascending order. Default is false.
        name(str, optional): The default value is None. Normally there is no
            need for user to set this property. For more information, please
            refer to :ref:`api_guide_Name`.

    Returns:
        tuple: A tuple of sorted data Variable(with the same shape and data
        type as input) and the sorted indices(with the same shape as input's
        and with data type int64).

    Examples:
        .. code-block:: python

            import paddle.fluid as fluid
            import numpy as np

            in1 = np.array([[[5,8,9,5],
                            [0,0,1,7],
                            [6,9,2,4]],
                            [[5,2,4,2],
                            [4,7,7,9],
                            [1,7,0,6]]]).astype(np.float32)
            with fluid.dygraph.guard():
                x = fluid.dygraph.to_variable(in1)
                out1 = fluid.layers.argsort(input=x, axis=-1)
                out2 = fluid.layers.argsort(input=x, axis=0)
                out3 = fluid.layers.argsort(input=x, axis=1)
                print(out1[0].numpy())
                # [[[5. 5. 8. 9.]
                #   [0. 0. 1. 7.]
                #   [2. 4. 6. 9.]]
                #  [[2. 2. 4. 5.]
                #   [4. 7. 7. 9.]
                #   [0. 1. 6. 7.]]]
                print(out1[1].numpy())
                # [[[0 3 1 2]
                #   [0 1 2 3]
                #   [2 3 0 1]]
                #  [[1 3 2 0]
                #   [0 1 2 3]
                #   [2 0 3 1]]]
                print(out2[0].numpy())
                # [[[5. 2. 4. 2.]
                #   [0. 0. 1. 7.]
                #   [1. 7. 0. 4.]]
                #  [[5. 8. 9. 5.]
                #   [4. 7. 7. 9.]
                #   [6. 9. 2. 6.]]]
                print(out3[0].numpy())
                # [[[0. 0. 1. 4.]
                #   [5. 8. 2. 5.]
                #   [6. 9. 9. 7.]]
                #  [[1. 2. 0. 2.]
                #   [4. 7. 4. 6.]
                #   [5. 7. 7. 9.]]]
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
        attrs={'axis': axis,
               'descending': descending})
    return out, ids


def ones(shape, dtype, force_cpu=False):
    """
    The OP creates a tensor of specified :attr:`shape` and :attr:`dtype`, and fills it with 1.
    Its :attr:`stop_gradient` will be set to True to stop gradient computation.

    Parameters:
        shape (tuple|list): Shape of output tensor.
        dtype (np.dtype|core.VarDesc.VarType|str): Data type of output tensor, it supports
            bool, float16, float32, float64, int32 and int64.
        force_cpu (bool, optional): Whether force to store the output tensor in CPU memory.
            If :attr:`force_cpu` is False, the output tensor will be stored in running device memory.
            Default: False.

    Returns:
        Variable: A tensor of data type :attr:`dtype` with shape :attr:`shape` and all elements set to 1.

    Examples:
        .. code-block:: python

          import paddle.fluid as fluid
          data = fluid.layers.ones(shape=[2, 4], dtype='float32') # [[1., 1., 1., 1.], [1., 1., 1., 1.]]
    """
    assert isinstance(shape, list) or isinstance(
        shape, tuple), "The shape's type should be list or tuple."
    assert reduce(lambda x, y: x * y,
                  shape) > 0, "The shape is invalid: %s." % (str(shape))
    return fill_constant(value=1.0, **locals())


def zeros(shape, dtype, force_cpu=False):
    """
    The OP creates a tensor of specified :attr:`shape` and :attr:`dtype`, and fills it with 0.
    Its :attr:`stop_gradient` will be set to True to stop gradient computation.

    Parameters:
        shape (tuple|list): Shape of output tensor.
        dtype (np.dtype|core.VarDesc.VarType|str): Data type of output tensor, it supports
            bool, float16, float32, float64, int32 and int64.
        force_cpu (bool, optional): Whether force to store the output tensor in CPU memory.
            If :attr:`force_cpu` is False, the output tensor will be stored in running device memory.
            Default: False.

    Returns:
        Variable: A tensor of data type :attr:`dtype` with shape :attr:`shape` and all elements set to 0.

    Examples:
        .. code-block:: python

          import paddle.fluid as fluid
          data = fluid.layers.zeros(shape=[3, 2], dtype='float32') # [[0., 0.], [0., 0.], [0., 0.]]
    """
    check_dtype(dtype, 'create data type',
                ['bool', 'float16', 'float32', 'float64', 'int32', 'int64'],
                'zeros')
    return fill_constant(value=0.0, **locals())


def reverse(x, axis):
    """
    The OP reverses the tensor :attr:`x` along the given :attr:`axis`.

    Parameters:
        x (Variable): A tensor to be reversed, its data type supports bool, float32, float64, int32, int64 and uint8.
        axis (int|tuple|list): A dimension or a set of dimensions of :attr:`x` to reverse. Must be
            in the range [-rank( :attr:`x` ), rank( :attr:`x` )). If it is a tuple or a list, reversing
            will be apply on each axis in the tuple or list.

    Returns:
        Variable: The reversed tensor with the same shape and data type as :attr:`x`.

    Examples:
        .. code-block:: python

          import paddle.fluid as fluid
          import numpy as np
          data = fluid.layers.assign(np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]], dtype='float32')) # [[0., 1., 2.], [3., 4., 5.], [6., 7., 8.]]
          result1 = fluid.layers.reverse(data, 0) # [[6., 7., 8.], [3., 4., 5.], [0., 1., 2.]]
          result2 = fluid.layers.reverse(data, [0, 1]) # [[8., 7., 6.], [5., 4., 3.], [2., 1., 0.]]
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
    Loads a list of variable from a single file.

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
       x (Variable): The Tensor/LoDTensor to be checked.

    Returns:
       Variable: The tensor variable storing the output, only a bool value, indicating that whether there is infinity number in x or not.
    
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
       x (Variable): The Tensor/LoDTensor to be checked.

    Returns:
       Variable: The tensor variable storing the output, only a bool value, indicating that whether there is NAN in x or not.
    
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
        dtype(str|core.VarDesc.VarType): the data type of the output tensor, can be float32, float64, int32, int64.

    Returns: a 1-D Tensor which is evenly spaced values within a given interval. Its data type is set by dtype.
    
    Return type: Variable

    examples:

        .. code-block:: python

             import paddle.fluid as fluid
             data = fluid.layers.range(0, 10, 2, 'int32')

    """
    helper = LayerHelper("range", **locals())

    check_dtype(dtype, 'create data type',
                ['float32', 'float64', 'int32', 'int64'], 'range')

    dtype = convert_dtype(dtype)
    if not isinstance(start, Variable):
        start = fill_constant([1], dtype, start)
    elif convert_dtype(start.dtype) != dtype:
        # make sure that start, end, step has the same dtype as
        # `dtype`
        start = cast(x=start, dtype=dtype)

    if not isinstance(end, Variable):
        end = fill_constant([1], dtype, end)
    elif convert_dtype(end.dtype) != dtype:
        end = cast(x=end, dtype=dtype)

    if not isinstance(step, Variable):
        step = fill_constant([1], dtype, step)
    elif convert_dtype(step.dtype) != dtype:
        step = cast(x=step, dtype=dtype)

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
            The default value is :attr:`None` .

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
