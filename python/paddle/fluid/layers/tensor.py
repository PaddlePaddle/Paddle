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

import numpy
import warnings

from ..layer_helper import LayerHelper
from ..framework import (
    _current_expected_place,
    convert_np_dtype_to_dtype_,
    _non_static_mode,
    _varbase_creator,
    _in_legacy_dygraph,
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
    'cast',
    'tensor_array_to_tensor',
    'concat',
    'sums',
    'assign',
    'fill_constant_batch_size_like',
    'fill_constant',
    'argmin',
    'argmax',
    'zeros',
]


def cast(x, dtype):
    """

    This OP takes in the Tensor :attr:`x` with :attr:`x.dtype` and casts it
    to the output with :attr:`dtype`. It's meaningless if the output dtype
    equals the input dtype, but it's fine if you do so.

    Args:
        x(Tensor): An input N-D Tensor with data type bool, float16,
            float32, float64, int32, int64, uint8.
        dtype(np.dtype|str): Data type of the output:
            bool, float16, float32, float64, int8, int32, int64, uint8.

    Returns:
        Tensor: A Tensor with the same shape as input's.

    Examples:
        .. code-block:: python

            import paddle

            x = paddle.to_tensor([2, 3, 4], 'float64')
            y = paddle.cast(x, 'uint8')
    """
    if in_dygraph_mode():
        if not isinstance(dtype, core.VarDesc.VarType):
            dtype = convert_np_dtype_to_dtype_(dtype)
        return _C_ops.cast(x, dtype)

    if _non_static_mode():
        if not isinstance(dtype, core.VarDesc.VarType):
            dtype = convert_np_dtype_to_dtype_(dtype)
        out = _legacy_C_ops.cast(x, 'in_dtype', x.dtype, 'out_dtype', dtype)
        return out

    check_variable_and_dtype(
        x,
        'x',
        [
            'bool',
            'float16',
            'float32',
            'float64',
            'int16',
            'int32',
            'int64',
            'uint8',
            'uint16',
        ],
        'cast',
    )
    check_dtype(
        dtype,
        'dtype',
        [
            'bool',
            'float16',
            'float32',
            'float64',
            'int8',
            'int16',
            'int32',
            'int64',
            'uint8',
            'uint16',
        ],
        'cast',
    )

    helper = LayerHelper('cast', **locals())
    out = helper.create_variable_for_type_inference(
        dtype=dtype, stop_gradient=x.stop_gradient
    )
    helper.append_op(
        type='cast',
        inputs={'X': [x]},
        outputs={'Out': [out]},
        attrs={'in_dtype': x.dtype, 'out_dtype': out.dtype},
    )
    return out


def concat(input, axis=0, name=None):
    """
    This OP concatenates the input along the axis.

    Args:
        input(list|tuple|Tensor): ``input`` can be Tensor, Tensor list or Tensor tuple which is with data type
            bool, float16, float32, float64, int32, int64. All the Tensors in ``input`` must have the same data type.
        axis(int|Tensor, optional): Specify the axis to operate on the input Tensors.
            It's a scalar with data type int or a Tensor with shape [1] and data type int32 or int64.
            The effective range is [-R, R), where R is Rank(x). When ``axis < 0``, it works the same way
            as ``axis+R``. Default is 0.
        name (str, optional): The default value is None. Normally there is no
            need for user to set this property. For more information, please
            refer to :ref:`api_guide_Name`.

    Returns:
        Tensor: A Tensor with the same data type as ``input``.

    Examples:
        .. code-block:: python

            import paddle.fluid as fluid
            import numpy as np

            in1 = np.array([[1, 2, 3],
                            [4, 5, 6]])
            in2 = np.array([[11, 12, 13],
                            [14, 15, 16]])
            in3 = np.array([[21, 22],
                            [23, 24]])
            with fluid.dygraph.guard():
                x1 = fluid.dygraph.to_variable(in1)
                x2 = fluid.dygraph.to_variable(in2)
                x3 = fluid.dygraph.to_variable(in3)
                # When the axis is negative, the real axis is (axis + Rank(x)).
                # As follows, axis is -1, Rank(x) is 2, the real axis is 1
                out1 = fluid.layers.concat(input=[x1, x2, x3], axis=-1)
                out2 = fluid.layers.concat(input=[x1, x2], axis=0)
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
        if isinstance(axis, Variable):
            axis = axis.numpy()
            axis = axis.item(0)
        if not isinstance(input, Variable):
            input = [t for t in input if t.shape.count(0) == 0]
        out = _C_ops.concat(input, axis)
        return out

    if _in_legacy_dygraph():
        if isinstance(axis, Variable):
            axis = axis.numpy()
            axis = axis.item(0)
        if not isinstance(input, Variable):
            input = [t for t in input if t.shape.count(0) == 0]
        out = _varbase_creator()
        _legacy_C_ops.concat(input, out, 'axis', axis)
        return out

    check_type(input, 'input', (list, tuple, Variable), 'concat')
    if not isinstance(input, Variable):
        for id, x in enumerate(input):
            check_variable_and_dtype(
                x,
                'input[' + str(id) + ']',
                ['bool', 'float16', 'float32', 'float64', 'int32', 'int64'],
                'concat',
            )
            if x.dtype != input[0].dtype:
                raise TypeError(
                    "All the Tensors in the input must have the same data type."
                )
    else:
        input = [input]
    check_type(axis, 'axis', (int, Variable), 'concat')

    if isinstance(axis, Variable):
        check_dtype(
            axis.dtype,
            'axis',
            ['int32', 'int64'],
            'concat',
            "The data type of axis must be int32 or int64 when axis is a Tensor",
        )

    helper = LayerHelper('concat', **locals())
    out = helper.create_variable_for_type_inference(dtype=helper.input_dtype())

    if input[0].desc.type() == core.VarDesc.VarType.LOD_TENSOR_ARRAY:
        # NOTE(liym27): Don't remove this if branch!
        # This feature is supported for Dynamic-to-Static, because after transformed, the type of inputs[0]
        # is LOD_TENSOR_ARRAY in some scenarios. And this feature can be used in static mode.

        assert len(input) == 1, (
            "If the elements of 'input' in concat are Variable(LoDTensorArray), "
            "number of the elements must be 1, but received %s." % len(input)
        )
        out_index = helper.create_variable_for_type_inference(dtype="int32")
        helper.append_op(
            type='tensor_array_to_tensor',
            inputs={'X': input[0]},
            outputs={'Out': [out], 'OutIndex': [out_index]},
            attrs={'axis': axis, 'use_stack': False},
        )
    else:
        inputs = {'X': input}
        attrs = {}
        if isinstance(axis, Variable):
            axis.stop_gradient = True
        attrs['axis'] = axis

        helper.append_op(
            type='concat', inputs=inputs, outputs={'Out': [out]}, attrs=attrs
        )
    return out


def tensor_array_to_tensor(input, axis=1, name=None, use_stack=False):
    r"""
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

            import paddle
            import paddle.fluid as fluid
            import numpy as np
            x0 = fluid.layers.assign(np.random.rand(2, 2).astype("float32"))
            x1 = fluid.layers.assign(np.random.rand(2, 2).astype("float32"))
            i = fluid.layers.fill_constant(shape=[1], dtype="int64", value=0)
            array = paddle.tensor.create_array(dtype='float32')
            paddle.tensor.array_write(x0, i, array)
            paddle.tensor.array_write(x1, i + 1, array)
            output, output_index = fluid.layers.tensor_array_to_tensor(input=array)
    """
    if _non_static_mode():
        assert isinstance(
            input, list
        ), "The 'input' in tensor_array_to_tensor must be list"
        from .nn import concat
        from ..dygraph import to_variable
        from paddle import stack

        op = stack if use_stack else concat
        res = op(input, axis=axis)
        sizes = to_variable(
            numpy.array(list(map(lambda x: int(x.shape[axis]), input)))
        )
        return res, sizes

    check_type(input, 'input', (list, Variable), 'tensor_array_to_tensor')
    if isinstance(input, list):
        for i, input_x in enumerate(input):
            check_type(
                input_x,
                'input[' + str(i) + ']',
                Variable,
                'tensor_array_to_tensor',
            )
    helper = LayerHelper('tensor_array_to_tensor', **locals())
    out = helper.create_variable_for_type_inference(dtype=helper.input_dtype())
    out_index = helper.create_variable_for_type_inference(dtype="int32")
    helper.append_op(
        type='tensor_array_to_tensor',
        inputs={'X': input},
        outputs={'Out': [out], 'OutIndex': [out_index]},
        attrs={'axis': axis, 'use_stack': use_stack},
    )
    return out, out_index


def sums(input, out=None):
    r"""
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
    check_type(input, 'input', (Variable, tuple, list), 'sums')
    if isinstance(input, list) or isinstance(input, tuple):
        for input_section in input:
            check_variable_and_dtype(
                input_section,
                "input",
                ['float16', 'float32', 'float64', 'int32', 'int64'],
                'sums',
            )
    else:
        check_variable_and_dtype(
            input,
            "input",
            ['float16', 'float32', 'float64', 'int32', 'int64'],
            'sums',
        )

    helper = LayerHelper('sum', **locals())
    if out is None:
        out = helper.create_variable_for_type_inference(
            dtype=helper.input_dtype()
        )
    else:
        check_variable_and_dtype(
            out, "out", ['float32', 'float64', 'int32', 'int64'], 'sums'
        )

    helper.append_op(
        type='sum',
        inputs={'X': input},
        outputs={'Out': out},
        attrs={'use_mkldnn': False},
    )
    return out


def assign(input, output=None):
    """

    The OP copies the :attr:`input` to the :attr:`output`.

    Parameters:
        input (Tensor|numpy.ndarray|list|tuple|scalar): A tensor, numpy ndarray, tuple/list of scalar,
            or scalar. Its data type supports float16, float32, float64, int32, int64, and bool.
            Note: the float64 data will be converted to float32 because of current platform protobuf
            data limitation.
        output (Tensor, optional): A tensor. If :attr:`output` is None, a new tensor will
            be created as :attr:`output`. Default: None.

    Returns:
        Tensor: A tensor with the same shape, data type and value as :attr:`input`.

    Examples:
        .. code-block:: python

          import paddle
          import numpy as np
          data = paddle.full(shape=[3, 2], fill_value=2.5, dtype='float64') # [[2.5, 2.5], [2.5, 2.5], [2.5, 2.5]]
          array = np.array([[1, 1],
                            [3, 4],
                            [1, 3]]).astype(np.int64)
          result1 = paddle.zeros(shape=[3, 3], dtype='float32')
          paddle.assign(array, result1) # result1 = [[1, 1], [3 4], [1, 3]]
          result2 = paddle.assign(data)  # result2 = [[2.5, 2.5], [2.5, 2.5], [2.5, 2.5]]
          result3 = paddle.assign(np.array([[2.5, 2.5], [2.5, 2.5], [2.5, 2.5]], dtype='float32')) # result3 = [[2.5, 2.5], [2.5, 2.5], [2.5, 2.5]]
    """
    helper = LayerHelper('assign', **locals())
    check_type(
        input,
        'input',
        (Variable, numpy.ndarray, list, tuple, float, int, bool),
        'assign',
    )
    is_inplace = True if output is not None else False

    if numpy.isscalar(input) and not isinstance(input, str):
        input = numpy.array([input])
    elif isinstance(input, (list, tuple)):
        input = numpy.array(input)
    # NOTE(Aurelius84): Why we judge core.VarBase?
    # In case of @to_static, a VarBase can be as input of `assign`,
    # but _non_static_mode()==False under @to_static, which means
    # isinstance(VarBase, Variable) == False. It will cause return None
    # after this api.
    if isinstance(input, (Variable, core.VarBase)):
        if _non_static_mode():
            if in_dygraph_mode() and output is None:
                output = _C_ops.assign(input)
            elif in_dygraph_mode() and output is not None:
                _C_ops.assign_out_(input, output)
            else:
                if output is None:
                    if _in_legacy_dygraph():
                        output = core.VarBase()
                    else:
                        output = core.eager.Tensor()
                _legacy_C_ops.assign(input, output)
        else:
            check_dtype(
                input.dtype,
                'input',
                [
                    'float16',
                    'uint16',
                    'float32',
                    'float64',
                    'int32',
                    'int64',
                    'uint8',
                    'bool',
                ],
                'assign',
                '(When the type of input in assign is Variable.)',
            )
            if output is None:
                output = helper.create_variable_for_type_inference(
                    dtype=input.dtype
                )
            helper.append_op(
                type='assign', inputs={'X': [input]}, outputs={'Out': [output]}
            )
    elif isinstance(input, numpy.ndarray):
        # Not support [var, var, ...] currently.
        if len(input.shape) > 0 and any(isinstance(x, Variable) for x in input):
            raise TypeError(
                "Required type(input) numpy.ndarray, but found `list(Variable)` in input."
            )
        dtype = convert_np_dtype_to_dtype_(input.dtype)
        if dtype == VarDesc.VarType.FP64:
            # Setting FP64 numpy data is not supported in Paddle, so we
            # use FP32 here
            warnings.warn(
                "paddle.assign doesn't support float64 input now due "
                "to current platform protobuf data limitation, we convert "
                "it to float32"
            )
            dtype = VarDesc.VarType.FP32
        if dtype == VarDesc.VarType.BOOL:
            value_name = "bool_values"
            values = [int(v) for v in input.flat]
        elif dtype == VarDesc.VarType.FP32:
            value_name = "fp32_values"
            values = [float(v) for v in input.flat]
        elif dtype == VarDesc.VarType.INT32:
            value_name = "int32_values"
            values = [int(v) for v in input.flat]
        elif dtype == VarDesc.VarType.INT64:
            value_name = "int64_values"
            values = [int(v) for v in input.flat]
        else:
            raise TypeError(
                "When the type of 'input' in assign is numpy.ndarray, "
                "the data type of 'input' must be bool, float32, int32 or int64, but "
                "received %s." % convert_dtype(dtype)
            )
        if input.size > 1024 * 1024:
            raise ValueError(
                "The size of input is too big. Please consider "
                "saving it to file and 'load_op' to load it"
            )
        if in_dygraph_mode():
            if output is None:
                output = zeros(list(input.shape), dtype)
            _C_ops.assign_value_(
                output,
                list(input.shape),
                dtype,
                values,
                _current_expected_place(),
            )
        elif _in_legacy_dygraph():
            if output is None:
                output = core.VarBase()
            _legacy_C_ops.assign_value(
                output,
                'shape',
                list(input.shape),
                'dtype',
                dtype,
                value_name,
                values,
            )
        else:
            if output is None:
                output = helper.create_variable_for_type_inference(
                    dtype=input.dtype
                )
            helper.append_op(
                type='assign_value',
                outputs={'Out': [output]},
                attrs={
                    'dtype': dtype,
                    'shape': list(input.shape),
                    value_name: values,
                },
            )

    if is_inplace and _non_static_mode():
        output._bump_inplace_version()

    return output


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

    attrs = {'force_cpu': force_cpu}
    dtype = convert_dtype(dtype)
    if not isinstance(value, Variable):
        if dtype in ['uint8', 'int16', 'int32', 'int64']:
            attrs['str_value'] = str(int(value))
            attrs['value'] = int(value)
        else:
            attrs['str_value'] = str(float(value))
            attrs['value'] = float(value)

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

    if _in_legacy_dygraph():
        shape = utils.convert_shape_to_list(shape)
        if out is None:
            out = _varbase_creator(dtype=dtype)

        if isinstance(value, Variable):
            if dtype in ['uint8', 'int16', 'int32', 'int64']:
                attrs['str_value'] = str(int(value.numpy().item(0)))
            else:
                attrs['str_value'] = str(float(value.numpy().item(0)))

        _legacy_C_ops.fill_constant(
            out,
            'value',
            float(value),
            'force_cpu',
            force_cpu,
            'dtype',
            out.dtype,
            'str_value',
            attrs['str_value'],
            'shape',
            shape,
        )
        out.stop_gradient = True
        return out

    helper = LayerHelper("fill_constant", **locals())
    inputs = {}
    if isinstance(value, Variable):
        if convert_dtype(value.dtype) != dtype:
            value = cast(value, dtype)
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


def argmin(x, axis=0):
    """
        :alias_main: paddle.argmin
        :alias: paddle.argmin,paddle.tensor.argmin,paddle.tensor.search.argmin
        :old_api: paddle.fluid.layers.argmin

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
    check_variable_and_dtype(
        x,
        'x',
        ['float32', 'float64', 'uint8', 'int16', 'int32', 'int64'],
        'argmin',
    )
    helper = LayerHelper("arg_min", **locals())
    out = helper.create_variable_for_type_inference(VarDesc.VarType.INT64)
    helper.append_op(
        type='arg_min',
        inputs={'X': x},
        outputs={'Out': [out]},
        attrs={'axis': axis},
    )
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
    check_variable_and_dtype(
        x,
        'x',
        ['float32', 'float64', 'uint8', 'int16', 'int32', 'int64'],
        'argmax',
    )
    helper = LayerHelper("arg_max", **locals())
    out = helper.create_variable_for_type_inference(VarDesc.VarType.INT64)
    helper.append_op(
        type='arg_max',
        inputs={'X': x},
        outputs={'Out': [out]},
        attrs={'axis': axis},
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
