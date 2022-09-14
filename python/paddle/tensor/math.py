# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
"""
math functions
"""
from __future__ import print_function
import numpy as np

from paddle.common_ops_import import VarDesc
from paddle.common_ops_import import dygraph_only
from paddle.common_ops_import import OpProtoHolder
from paddle.common_ops_import import templatedoc
from paddle.common_ops_import import dygraph_utils

from .manipulation import cast
from .creation import _complex_to_real_dtype
from .layer_function_generator import _generate_doc_string_, generate_activation_fn, generate_layer_fn

import paddle
from ..static import Variable
from ..framework import core, in_dygraph_mode, _non_static_mode, LayerHelper, _in_legacy_dygraph
from ..fluid.framework import _in_legacy_dygraph
from ..framework import _varbase_creator, convert_np_dtype_to_dtype_
from ..fluid.data_feeder import check_variable_and_dtype, check_type, check_dtype, convert_dtype
from ..fluid.dygraph.inplace_utils import inplace_apis_in_dygraph_only
from ..fluid.layers import utils

# TODO: define math functions
# yapf: disable
from .ops import abs    # noqa: F401
from .ops import acos    # noqa: F401
from .ops import asin    # noqa: F401
from .ops import ceil    # noqa: F401
from .ops import ceil_    # noqa: F401
from .ops import cos    # noqa: F401
from .ops import tan    # noqa: F401
from .ops import sinh    # noqa: F401
from .ops import cosh    # noqa: F401
from .ops import exp    # noqa: F401
from .ops import exp_    # noqa: F401
from .ops import expm1    # noqa: F401
from .ops import floor    # noqa: F401
from .ops import floor_    # noqa: F401
from .ops import reciprocal    # noqa: F401
from .ops import reciprocal_    # noqa: F401
from .ops import round    # noqa: F401
from .ops import round_    # noqa: F401
from .ops import rsqrt    # noqa: F401
from .ops import rsqrt_    # noqa: F401
from .ops import square    # noqa: F401
from .ops import atan    # noqa: F401
from .ops import erf    # noqa: F401
from .ops import sqrt    # noqa: F401
from .ops import sqrt_    # noqa: F401
from .ops import sin    # noqa: F401
from .ops import asinh    # noqa: F401
from .ops import acosh    # noqa: F401
from .ops import atanh    # noqa: F401


from ..fluid.layers import elementwise_sub
from paddle import _C_ops, _legacy_C_ops

__all__ = []

_supported_int_dtype_ = [
    VarDesc.VarType.UINT8,
    VarDesc.VarType.INT8,
    VarDesc.VarType.INT16,
    VarDesc.VarType.INT32,
    VarDesc.VarType.INT64,
]

_supported_float_dtype_ = [
    VarDesc.VarType.FP32,
    VarDesc.VarType.FP64,
]


def log(x, name=None):
    r"""
    Calculates the natural log of the given input Tensor, element-wise.

    .. math::

        Out = \ln(x)

    Args:
        x (Tensor): Input Tensor. Must be one of the following types: float32, float64.
        name (str|None): The default value is None. Normally there is no need for user to set this property. For more information, please refer to :ref:`api_guide_Name`


    Returns:
        Tensor: The natural log of the input Tensor computed element-wise.

    Examples:

        .. code-block:: python

            import paddle

            x = [[2,3,4], [7,8,9]]
            x = paddle.to_tensor(x, dtype='float32')
            res = paddle.log(x)
            # [[0.693147, 1.09861, 1.38629], [1.94591, 2.07944, 2.19722]]
    """
    if in_dygraph_mode():
        return _C_ops.log(x)
    if _in_legacy_dygraph():
        return _legacy_C_ops.log(x)

    check_variable_and_dtype(x, 'x', ['float32', 'float64'], "log")
    inputs = {'X': [x]}
    helper = LayerHelper('log', **locals())
    dtype = helper.input_dtype(input_param_name='x')
    out = helper.create_variable_for_type_inference(dtype)
    helper.append_op(type="log", inputs={"X": x}, outputs={"Out": out})
    return out


def scale(x, scale=1.0, bias=0.0, bias_after_scale=True, act=None, name=None):
    """
    Scale operator.

    Putting scale and bias to the input Tensor as following:

    ``bias_after_scale`` is True:

    .. math::
                            Out=scale*X+bias

    ``bias_after_scale`` is False:

    .. math::
                            Out=scale*(X+bias)

    Args:
        x (Tensor): Input N-D Tensor of scale operator. Data type can be float32, float64, int8, int16, int32, int64, uint8.
        scale (float|Tensor): The scale factor of the input, it should be a float number or a Tensor with shape [1] and data type as float32.
        bias (float): The bias to be put on the input.
        bias_after_scale (bool): Apply bias addition after or before scaling. It is useful for numeric stability in some circumstances.
        act (str, optional): Activation applied to the output such as tanh, softmax, sigmoid, relu.
        name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        Tensor: Output Tensor of scale operator, with shape and data type same as input.

    Examples:
        .. code-block:: python
            
            # scale as a float32 number
            import paddle

            data = paddle.randn(shape=[2,3], dtype='float32')
            res = paddle.scale(data, scale=2.0, bias=1.0)

        .. code-block:: python

            # scale with parameter scale as a Tensor
            import paddle

            data = paddle.randn(shape=[2, 3], dtype='float32')
            factor = paddle.to_tensor([2], dtype='float32')
            res = paddle.scale(data, scale=factor, bias=1.0)

    """

    if in_dygraph_mode():
        out = _C_ops.scale(x, scale, float(bias), bias_after_scale)
        return dygraph_utils._append_activation_in_dygraph(out, act)
    elif _in_legacy_dygraph():
        _scale = scale.numpy().item(0) if isinstance(scale, Variable) else scale
        out = _legacy_C_ops.scale(x, 'scale',
                           float(_scale), 'bias',
                           float(bias), 'bias_after_scale', bias_after_scale)
        return dygraph_utils._append_activation_in_dygraph(out, act)

    check_variable_and_dtype(x, "x", [
        'float16', 'uint16', 'float32', 'float64', 'int8', 'int16', 'int32',
        'int64', 'uint8'
    ], "scale")
    inputs = {'X': [x]}
    attrs = {
        'bias': float(bias),
        'bias_after_scale': bias_after_scale,
    }
    if isinstance(scale, Variable):
        inputs['ScaleTensor'] = [scale]
    else:
        attrs['scale'] = float(scale)
    helper = LayerHelper('scale', **locals())
    out = helper.create_variable_for_type_inference(dtype=x.dtype)

    helper.append_op(
        type='scale', inputs=inputs, outputs={'Out': out}, attrs=attrs)
    return helper.append_activation(out)


def stanh(x, scale_a=0.67, scale_b=1.7159, name=None):
    """
    stanh activation.

    .. math::

        out = b * \frac{e^{a * x} - e^{-a * x}}{e^{a * x} + e^{-a * x}}

    Parameters:
        x (Tensor): The input Tensor with data type float32, float64.
        scale_a (float, optional): The scale factor a of the input. Default is 0.67.
        scale_b (float, optional): The scale factor b of the output. Default is 1.7159.
        name (str, optional): Name for the operation (optional, default is None).
            For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        A Tensor with the same data type and shape as ``x`` .

    Examples:
        .. code-block:: python

            import paddle

            x = paddle.to_tensor([1.0, 2.0, 3.0, 4.0])
            out = paddle.stanh(x, scale_a=0.67, scale_b=1.72) # [1.00616539, 1.49927628, 1.65933108, 1.70390463]

    """

    if _non_static_mode():
        return _legacy_C_ops.stanh(x, 'scale_a', scale_a, 'scale_b', scale_b)

    check_variable_and_dtype(x, 'x', ['float16', 'float32', 'float64'], 'stanh')

    helper = LayerHelper('stanh', **locals())
    out = helper.create_variable_for_type_inference(dtype=x.dtype)
    helper.append_op(
        type='stanh',
        inputs={'X': x},
        outputs={'Out': out},
        attrs={'scale_a': scale_a,
               'scale_b': scale_b})
    return out

def multiplex(inputs, index, name=None):
    """

    Based on the given index parameter, the OP selects a specific row from each input Tensor to construct the output Tensor.

    If the input of this OP contains :math:`m` Tensors, where :math:`I_{i}` means the i-th input Tensor, :math:`i` between :math:`[0,m)` .

    And :math:`O` means the output, where :math:`O[i]` means the i-th row of the output, then the output satisfies that :math:`O[i] = I_{index[i]}[i]` .

    For Example:

            .. code-block:: text

                Given:

                inputs = [[[0,0,3,4], [0,1,3,4], [0,2,4,4], [0,3,3,4]],
                          [[1,0,3,4], [1,1,7,8], [1,2,4,2], [1,3,3,4]],
                          [[2,0,3,4], [2,1,7,8], [2,2,4,2], [2,3,3,4]],
                          [[3,0,3,4], [3,1,7,8], [3,2,4,2], [3,3,3,4]]]

                index = [[3],[0],[1],[2]]

                out = [[3,0,3,4],    # out[0] = inputs[index[0]][0] = inputs[3][0] = [3,0,3,4]
                       [0,1,3,4],    # out[1] = inputs[index[1]][1] = inputs[0][1] = [0,1,3,4]
                       [1,2,4,2],    # out[2] = inputs[index[2]][2] = inputs[1][2] = [1,2,4,2]
                       [2,3,3,4]]    # out[3] = inputs[index[3]][3] = inputs[2][3] = [2,3,3,4]


    Args:
        inputs (list): The input Tensor list. The list elements are N-D Tensors of data types float32, float64, int32, int64. All input Tensor shapes should be the same and rank must be at least 2.
        index (Tensor): Used to select some rows in the input Tensor to construct an index of the output Tensor. It is a 2-D Tensor with data type int32 or int64 and shape [M, 1], where M is the number of input Tensors.
        name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        Tensor: Output of multiplex OP, with data type being float32, float64, int32, int64.

    Examples:

        .. code-block:: python

            import paddle
            
            img1 = paddle.to_tensor([[1, 2], [3, 4]], dtype=paddle.float32)
            img2 = paddle.to_tensor([[5, 6], [7, 8]], dtype=paddle.float32)
            inputs = [img1, img2]
            index = paddle.to_tensor([[1], [0]], dtype=paddle.int32)
            res = paddle.multiplex(inputs, index)
            print(res) # Tensor([[5., 6.], [3., 4.]], dtype=float32)

    """
    if in_dygraph_mode():
        return _C_ops.multiplex(inputs, index)
    elif _in_legacy_dygraph():
        return _legacy_C_ops.multiplex(index, inputs)

    helper = LayerHelper('multiplex', **locals())

    check_type(inputs, 'inputs', (list), 'multiplex')
    if len(inputs) < 2:
        raise ValueError(
            "inputs should be a list object with at least 2 elements.")
    for id, x in enumerate(inputs):
        check_variable_and_dtype(x, 'input[' + str(id) + ']',
                                 ['float32', 'float64', 'int32', 'int64'],
                                 'multiplex')
    check_variable_and_dtype(index, "index", ['int32', 'int64'], 'multiplex')

    out = helper.create_variable_for_type_inference(inputs[0].dtype)
    helper.append_op(
        type='multiplex',
        inputs={'X': inputs,
                'Ids': index},
        outputs={'Out': [out]})
    return out

@inplace_apis_in_dygraph_only
def scale_(x, scale=1.0, bias=0.0, bias_after_scale=True, act=None, name=None):
    """
    Inplace version of ``scale`` API, the output Tensor will be inplaced with input ``x``.
    Please refer to :ref:`api_tensor_scale`.
    """
    if in_dygraph_mode():
        return _C_ops.scale_(x, scale, float(bias), bias_after_scale)
    if _in_legacy_dygraph():
        _scale = scale.numpy().item(0) if isinstance(scale, Variable) else scale
        return _legacy_C_ops.scale_(x, 'scale',
                                float(_scale), 'bias',
                                float(bias), 'bias_after_scale', bias_after_scale)


def pow(x, y, name=None):
    """
    Compute the power of Tensor elements. The equation is:

    .. math::
        out = x^{y} 

    Note:
        ``paddle.pow`` supports broadcasting. If you want know more about broadcasting, please refer to :ref:`user_guide_broadcasting` .


    Args:
        x (Tensor): An N-D Tensor, the data type is float16, float32, float64, int32 or int64.
        y (float|int|Tensor): If it is an N-D Tensor, its data type should be the same as `x`.
        name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.
    
    Returns:
        N-D Tensor. A location into which the result is stored. Its dimension and data type are the same as `x`.

    Examples:

        ..  code-block:: python

            import paddle

            x = paddle.to_tensor([1, 2, 3], dtype='float32')

            # example 1: y is a float or int
            res = paddle.pow(x, 2)
            print(res)
            # Tensor(shape=[3], dtype=float32, place=CUDAPlace(0), stop_gradient=True,
            #        [1., 4., 9.])
            res = paddle.pow(x, 2.5)
            print(res)
            # Tensor(shape=[3], dtype=float32, place=CUDAPlace(0), stop_gradient=True,
            #        [1.         , 5.65685415 , 15.58845711])

            # example 2: y is a Tensor
            y = paddle.to_tensor([2], dtype='float32')
            res = paddle.pow(x, y)
            print(res)
            # Tensor(shape=[3], dtype=float32, place=CUDAPlace(0), stop_gradient=True,
            #        [1., 4., 9.])

    """
    # in dynamic graph mode
    if in_dygraph_mode():
        if isinstance(y, (int, float)):
            return _C_ops.pow(x, y)
        elif isinstance(y, (paddle.Tensor, Variable)):
            return _C_ops.elementwise_pow(x, y)
        else:
            raise TypeError('y must be scalar or tensor type, but received: %s '% (y.dtype))
    if _in_legacy_dygraph():
        if isinstance(y, (int, float)):
            return _legacy_C_ops.pow(x, 'factor', y)
        elif isinstance(y, (paddle.Tensor, Variable)):
            return _elementwise_op_in_dygraph(
                x, y, axis=-1, act=None, op_name='elementwise_pow')
        else:
            raise TypeError('y must be scalar or tensor type, but received: %s '% (y.dtype))
    # in static graph mode
    if isinstance(y, (int, float)):
        helper = LayerHelper('pow', **locals())
        inputs = {'X': x}
        attrs = {'factor': y}
        out = helper.create_variable_for_type_inference(dtype=x.dtype)
        helper.append_op(
            type='pow', inputs=inputs, outputs={'Out': out}, attrs=attrs)
        return out
    elif isinstance(y, (paddle.Tensor, Variable)):
        # TODO A potential speed improvement is supporting different types in C++ and removing the cast ops here
        helper = LayerHelper('elementwise_pow', **locals())
        out = helper.create_variable_for_type_inference(dtype=x.dtype)
        return _elementwise_op(LayerHelper('elementwise_pow', **locals()))
    else:
        raise TypeError('y must be scalar or tensor type, but received: %s '% (type(y)))


OP_NAMEMAPPING = {
    'elementwise_max': 'maximum',
    'elementwise_min': 'minimum',
    'elementwise_pow': 'elementwise_pow',
    'elementwise_floordiv': 'floor_divide',
    'elementwise_add': 'add',
    'elementwise_sub': 'subtract',
    'elementwise_mul': 'multiply',
    'elementwise_div': 'divide',
    'elementwise_mod': 'remainder',
}

@dygraph_only
def _elementwise_op_in_dygraph(x,
                               y,
                               axis=-1,
                               act=None,
                               use_mkldnn=False,
                               op_name=None):
    def is_inplace(op_name):
        return  op_name[-1] == "_"

    if op_name not in OP_NAMEMAPPING.keys() or axis != -1:
        op = getattr(_legacy_C_ops, op_name)
        out = op(x, y, 'axis', axis, 'use_mkldnn', use_mkldnn)
    else:
        if in_dygraph_mode():
            op = getattr(_C_ops, OP_NAMEMAPPING[op_name] if not is_inplace(op_name) else op_name)
            out = op(x, y)

        if _in_legacy_dygraph():
            op = getattr(_legacy_C_ops, op_name)
            out = op(x, y, 'axis', axis, 'use_mkldnn', use_mkldnn)

    return dygraph_utils._append_activation_in_dygraph(
        out, act, use_mkldnn=use_mkldnn)

def _elementwise_op(helper):
    op_type = helper.layer_type
    original_op_type = helper.kwargs.get('original_op_type', op_type)
    x = helper.kwargs.get('x', None)
    y = helper.kwargs.get('y', None)

    out = helper.kwargs.get('out', None)

    assert x is not None, 'x cannot be None in {}'.format(original_op_type)
    assert y is not None, 'y cannot be None in {}'.format(original_op_type)
    check_variable_and_dtype(
        x, 'x', ['float16', 'float32', 'float64', 'int32', 'int64', 'bool'],
        original_op_type)
    check_variable_and_dtype(
        y, 'y', ['float16', 'float32', 'float64', 'int32', 'int64', 'bool'],
        original_op_type)

    axis = helper.kwargs.get('axis', -1)
    use_mkldnn = helper.kwargs.get('use_mkldnn', False)
    name = helper.kwargs.get('name', None)

    if out is None:
        if name is None:
            out = helper.create_variable_for_type_inference(dtype=x.dtype)
        else:
            out = helper.create_variable(name=name, dtype=x.dtype, persistable=False)

    helper.append_op(
        type=op_type,
        inputs={'X': x,
                'Y': y},
        outputs={'Out': out},
        attrs={'axis': axis,
               'use_mkldnn': use_mkldnn})
    return helper.append_activation(out)


def add(x, y, name=None):
    """
    Elementwise Add Operator.
    Add two tensors element-wise
    The equation is:

    ..  math::

        Out=X+Y

    $X$ the tensor of any dimension.
    $Y$ the tensor whose dimensions must be less than or equal to the dimensions of $X$.

    There are two cases for this operator:

    1. The shape of $Y$ is the same with $X$.
    2. The shape of $Y$ is a continuous subsequence of $X$.

    For case 2:

    1. Broadcast $Y$ to match the shape of $X$, where axis is the start dimension index for broadcasting $Y$ onto $X$.
    2. If $axis$ is -1 (default), $axis$=rank($X$)−rank($Y$).
    3. The trailing dimensions of size 1 for $Y$ will be ignored for the consideration of subsequence, such as shape($Y$) = (2, 1) => (2).

        For example:

        ..  code-block:: python

            shape(X) = (2, 3, 4, 5), shape(Y) = (,)
            shape(X) = (2, 3, 4, 5), shape(Y) = (5,)
            shape(X) = (2, 3, 4, 5), shape(Y) = (4, 5), with axis=-1(default) or axis=2
            shape(X) = (2, 3, 4, 5), shape(Y) = (3, 4), with axis=1
            shape(X) = (2, 3, 4, 5), shape(Y) = (2), with axis=0
            shape(X) = (2, 3, 4, 5), shape(Y) = (2, 1), with axis=0

    Args:
        x (Tensor): Tensor or LoDTensor of any dimensions. Its dtype should be int32, int64, float32, float64.
        y (Tensor): Tensor or LoDTensor of any dimensions. Its dtype should be int32, int64, float32, float64.
        name (string, optional): For details, please refer to :ref:`api_guide_Name`. Generally, no setting is required. Default: None.

    Returns:
        N-D Tensor. A location into which the result is stored. It’s dimension equals with x.

    Examples:

        ..  code-block:: python

            import paddle

            x = paddle.to_tensor([2, 3, 4], 'float64')
            y = paddle.to_tensor([1, 5, 2], 'float64')
            z = paddle.add(x, y)
            print(z)  # [3., 8., 6. ]
    """

    if in_dygraph_mode():
        return _C_ops.add( x, y)
    else:
        if _in_legacy_dygraph():
            return _legacy_C_ops.elementwise_add(x, y)
        else:
            return _elementwise_op(LayerHelper('elementwise_add', **locals()))


@inplace_apis_in_dygraph_only
def add_(x, y, name=None):
    """
    Inplace version of ``add`` API, the output Tensor will be inplaced with input ``x``.
    Please refer to :ref:`api_tensor_add`.
    """
    op_type = 'elementwise_add_'
    axis = -1

    out_shape = broadcast_shape(x.shape, y.shape)
    if out_shape != x.shape:
        raise ValueError("The shape of broadcast output {} is different from that of inplace tensor {} in the Inplace operation.".format(out_shape, x.shape))

    if in_dygraph_mode():
        return _C_ops.add_(x, y)
    else:
        out = _elementwise_op_in_dygraph(
            x, y, axis=axis, op_name=op_type)
        return out


def subtract(x, y, name=None):
    """
    Substract two tensors element-wise. The equation is:

    .. math::
        out = x - y

    Note:
        ``paddle.subtract`` supports broadcasting. If you want know more about broadcasting, please refer to :ref:`user_guide_broadcasting` .

    Args:
        x (Tensor): the input tensor, it's data type should be float32, float64, int32, int64.
        y (Tensor): the input tensor, it's data type should be float32, float64, int32, int64.
        name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        N-D Tensor. A location into which the result is stored. If x, y have different shapes and are "broadcastable", the resulting tensor shape is the shape of x and y after broadcasting. If x, y have the same shape,  its shape is the same as x and y.

    Examples:

        .. code-block:: python

            import paddle

            x = paddle.to_tensor([[1, 2], [7, 8]])
            y = paddle.to_tensor([[5, 6], [3, 4]])
            res = paddle.subtract(x, y)
            print(res)
            # Tensor(shape=[2, 2], dtype=int64, place=Place(cpu), stop_gradient=True,
            #        [[-4, -4],
            #         [ 4,  4]])

            x = paddle.to_tensor([[[1, 2, 3], [1, 2, 3]]])
            y = paddle.to_tensor([1, 0, 4])
            res = paddle.subtract(x, y)
            print(res)
            # Tensor(shape=[1, 2, 3], dtype=int64, place=Place(cpu), stop_gradient=True,
            #        [[[ 0,  2, -1],
            #          [ 0,  2, -1]]])

            x = paddle.to_tensor([2, float('nan'), 5], dtype='float32')
            y = paddle.to_tensor([1, 4, float('nan')], dtype='float32')
            res = paddle.subtract(x, y)
            print(res)
            # Tensor(shape=[3], dtype=float32, place=Place(cpu), stop_gradient=True,
            #        [1. , nan, nan])

            x = paddle.to_tensor([5, float('inf'), -float('inf')], dtype='float64')
            y = paddle.to_tensor([1, 4, 5], dtype='float64')
            res = paddle.subtract(x, y)
            print(res)
            # Tensor(shape=[3], dtype=float64, place=Place(cpu), stop_gradient=True,
            #        [ 4.  ,  inf., -inf.])
    """
    op_type = 'elementwise_sub'
    axis = -1
    act = None
    if in_dygraph_mode():
        return _C_ops.subtract(x, y)
    else:
        if _in_legacy_dygraph():
            return _elementwise_op_in_dygraph(
                x, y, axis=axis, act=act, op_name=op_type)
        else:
            return _elementwise_op(LayerHelper(op_type, **locals()))


@inplace_apis_in_dygraph_only
def subtract_(x, y, name=None):
    """
    Inplace version of ``subtract`` API, the output Tensor will be inplaced with input ``x``.
    Please refer to :ref:`api_tensor_subtract`.
    """
    axis = -1
    act = None

    out_shape = broadcast_shape(x.shape, y.shape)
    if out_shape != x.shape:
        raise ValueError("The shape of broadcast output {} is different from that of inplace tensor {} in the Inplace operation.".format(out_shape, x.shape))

    if in_dygraph_mode():
        return _C_ops.subtract_(x, y)
    else:
        out = _elementwise_op_in_dygraph(
            x, y, axis=axis, act=act, op_name='elementwise_sub_')
        return out


def divide(x, y, name=None):
    """
    Divide two tensors element-wise. The equation is:

    .. math::
        out = x / y

    Note:
        ``paddle.divide`` supports broadcasting. If you want know more about broadcasting, please refer to :ref:`user_guide_broadcasting` .

    Args:
        x (Tensor): the input tensor, it's data type should be float32, float64, int32, int64.
        y (Tensor): the input tensor, it's data type should be float32, float64, int32, int64.
        name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        N-D Tensor. A location into which the result is stored. If x, y have different shapes and are "broadcastable", the resulting tensor shape is the shape of x and y after broadcasting. If x, y have the same shape,  its shape is the same as x and y.

    Examples:

        ..  code-block:: python

            import paddle

            x = paddle.to_tensor([2, 3, 4], dtype='float64')
            y = paddle.to_tensor([1, 5, 2], dtype='float64')
            z = paddle.divide(x, y)
            print(z)  # [2., 0.6, 2.]

    """
    op_type = 'elementwise_div'
    axis = -1
    act = None
    if in_dygraph_mode():
        return _C_ops.divide( x, y)
    else:
        if _in_legacy_dygraph():
            return _elementwise_op_in_dygraph(
                x, y, axis=axis, act=act, op_name=op_type)
        else:
            return _elementwise_op(LayerHelper(op_type, **locals()))


def floor_divide(x, y, name=None):
    """
    Floor divide two tensors element-wise. The equation is:

    .. math::
        out = x // y

    Note:
        ``paddle.floor_divide`` supports broadcasting. If you want know more about broadcasting, please refer to :ref:`user_guide_broadcasting` .

    Args:
        x (Tensor): the input tensor, it's data type should be int32, int64.
        y (Tensor): the input tensor, it's data type should be int32, int64.
        name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        N-D Tensor. A location into which the result is stored. It's dimension equals with $x$.

    Examples:

        ..  code-block:: python

            import paddle

            x = paddle.to_tensor([2, 3, 8, 7])
            y = paddle.to_tensor([1, 5, 3, 3])
            z = paddle.floor_divide(x, y)
            print(z)  # [2, 0, 2, 2]

    """
    op_type = 'elementwise_floordiv'
    axis = -1
    if in_dygraph_mode():
        return _C_ops.floor_divide(x, y)
    elif _in_legacy_dygraph():
        return _elementwise_op_in_dygraph(
            x, y, axis=axis, op_name=op_type)

    return _elementwise_op(LayerHelper(op_type, **locals()))


def remainder(x, y, name=None):
    r"""
    Mod two tensors element-wise. The equation is:

    .. math::

        out = x \% y

    Note:
        ``paddle.remainder`` supports broadcasting. If you want know more about broadcasting, please refer to :ref:`user_guide_broadcasting` .

    Args:
        x (Tensor): the input tensor, it's data type should be float16, float32, float64, int32, int64.
        y (Tensor): the input tensor, it's data type should be float16, float32, float64, int32, int64.
        name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        N-D Tensor. A location into which the result is stored. If x, y have different shapes and are "broadcastable", the resulting tensor shape is the shape of x and y after broadcasting. If x, y have the same shape,  its shape is the same as x and y.

    Examples:

        ..  code-block:: python

            import paddle

            x = paddle.to_tensor([2, 3, 8, 7])
            y = paddle.to_tensor([1, 5, 3, 3])
            z = paddle.remainder(x, y)
            print(z)  # [0, 3, 2, 1]

    """
    op_type = 'elementwise_mod'
    axis = -1

    if in_dygraph_mode():
        return _C_ops.remainder(x, y)
    elif _in_legacy_dygraph():
        return _elementwise_op_in_dygraph(
            x, y, axis=axis, op_name=op_type)

    return _elementwise_op(LayerHelper(op_type, **locals()))


@inplace_apis_in_dygraph_only
def remainder_(x, y, name=None):
    r"""
    Inplace version of ``remainder`` API, the output Tensor will be inplaced with input ``x``.
    Please refer to :ref:`api_tensor_remainder`.
    """
    op_type = 'elementwise_mod_'
    axis = -1

    out_shape = broadcast_shape(x.shape, y.shape)
    if out_shape != x.shape:
        raise ValueError(
            "The shape of broadcast output {} is different from that of inplace tensor {} in the Inplace operation.".format(
                out_shape, x.shape))

    return _elementwise_op_in_dygraph(x, y, axis=axis, op_name=op_type)


mod = remainder  # noqa: F841
floor_mod = remainder  # noqa: F841


def multiply(x, y, name=None):
    """
    multiply two tensors element-wise. The equation is:

    .. math::
        out = x * y

    Note:
        ``paddle.multiply`` supports broadcasting. If you would like to know more about broadcasting, please refer to :ref:`user_guide_broadcasting`.

    Args:
        x (Tensor): the input tensor, its data type should be one of float32, float64, int32, int64, bool.
        y (Tensor): the input tensor, its data type should be one of float32, float64, int32, int64, bool.
        name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        N-D Tensor. A location into which the result is stored. If x, y have different shapes and are "broadcastable", the resulting tensor shape is the shape of x and y after broadcasting. If x, y have the same shape,  its shape is the same as x and y.

    Examples:

        ..  code-block:: python

            import paddle

            x = paddle.to_tensor([[1, 2], [3, 4]])
            y = paddle.to_tensor([[5, 6], [7, 8]])
            res = paddle.multiply(x, y)
            print(res) # [[5, 12], [21, 32]]

            x = paddle.to_tensor([[[1, 2, 3], [1, 2, 3]]])
            y = paddle.to_tensor([2])
            res = paddle.multiply(x, y)
            print(res) # [[[2, 4, 6], [2, 4, 6]]]

    """
    op_type = 'elementwise_mul'
    act = None
    axis = -1

    if in_dygraph_mode():
        return _C_ops.multiply(x, y)
    else:
        if _in_legacy_dygraph():
            return _elementwise_op_in_dygraph(
                x, y, axis=axis, act=act, op_name=op_type)
        else:
            if x.dtype != y.dtype:
                raise TypeError(
                    'Input tensors must be same type, but received type of x: %s, type of y: %s '
                    % (x.dtype, y.dtype))

            return _elementwise_op(LayerHelper(op_type, **locals()))

def maximum(x, y, name=None):
    """
    Compare two tensors and returns a new tensor containing the element-wise maxima. The equation is:

    .. math::
        out = max(x, y)

    Note:
        ``paddle.maximum`` supports broadcasting. If you want know more about broadcasting, please refer to :ref:`user_guide_broadcasting` .

    Args:
        x (Tensor): the input tensor, it's data type should be float32, float64, int32, int64.
        y (Tensor): the input tensor, it's data type should be float32, float64, int32, int64.
        name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        N-D Tensor. A location into which the result is stored. If x, y have different shapes and are "broadcastable", the resulting tensor shape is the shape of x and y after broadcasting. If x, y have the same shape,  its shape is the same as x and y.

    Examples:

        .. code-block:: python

            import numpy as np
            import paddle

            x = paddle.to_tensor([[1, 2], [7, 8]])
            y = paddle.to_tensor([[3, 4], [5, 6]])
            res = paddle.maximum(x, y)
            print(res)
            #    [[3, 4],
            #     [7, 8]]

            x = paddle.to_tensor([[1, 2, 3], [1, 2, 3]])
            y = paddle.to_tensor([3, 0, 4])
            res = paddle.maximum(x, y)
            print(res)
            #    [[3, 2, 4],
            #     [3, 2, 4]]

            x = paddle.to_tensor([2, 3, 5], dtype='float32')
            y = paddle.to_tensor([1, np.nan, np.nan], dtype='float32')
            res = paddle.maximum(x, y)
            print(res)
            #    [ 2., nan, nan]

            x = paddle.to_tensor([5, 3, np.inf], dtype='float32')
            y = paddle.to_tensor([1, -np.inf, 5], dtype='float32')
            res = paddle.maximum(x, y)
            print(res)
            #    [  5.,   3., inf.]
    """
    op_type = 'elementwise_max'
    axis = -1
    act = None
    if in_dygraph_mode():
        return _C_ops.maximum(x, y)
    elif _in_legacy_dygraph():
        return _elementwise_op_in_dygraph(
            x, y, axis=axis, act=act, op_name=op_type)
    return _elementwise_op(LayerHelper(op_type, **locals()))

def minimum(x, y, name=None):
    """
    Compare two tensors and return a new tensor containing the element-wise minima. The equation is:

    .. math::
        out = min(x, y)

    Note:
        ``paddle.minimum`` supports broadcasting. If you want know more about broadcasting, please refer to :ref:`user_guide_broadcasting` .

    Args:
        x (Tensor): the input tensor, it's data type should be float32, float64, int32, int64.
        y (Tensor): the input tensor, it's data type should be float32, float64, int32, int64.
        name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        Tensor. If x, y have different shapes and are "broadcastable", the resulting tensor shape is the shape of x and y after broadcasting. If x, y have the same shape,  its shape is the same as x and y.

    Examples:

        .. code-block:: python

            import numpy as np
            import paddle

            x = paddle.to_tensor([[1, 2], [7, 8]])
            y = paddle.to_tensor([[3, 4], [5, 6]])
            res = paddle.minimum(x, y)
            print(res)
            #       [[1, 2],
            #        [5, 6]]

            x = paddle.to_tensor([[[1, 2, 3], [1, 2, 3]]])
            y = paddle.to_tensor([3, 0, 4])
            res = paddle.minimum(x, y)
            print(res)
            #       [[[1, 0, 3],
            #         [1, 0, 3]]]

            x = paddle.to_tensor([2, 3, 5], dtype='float32')
            y = paddle.to_tensor([1, np.nan, np.nan], dtype='float32')
            res = paddle.minimum(x, y)
            print(res)
            #       [ 1., nan, nan]

            x = paddle.to_tensor([5, 3, np.inf], dtype='float64')
            y = paddle.to_tensor([1, -np.inf, 5], dtype='float64')
            res = paddle.minimum(x, y)
            print(res)
            #       [   1., -inf.,    5.]
    """
    op_type = 'elementwise_min'
    axis = -1
    act = None
    if in_dygraph_mode():
        return _C_ops.minimum(x, y)
    elif _in_legacy_dygraph():
        return _elementwise_op_in_dygraph(
            x, y, axis=axis, act=act, op_name=op_type)
    return _elementwise_op(LayerHelper(op_type, **locals()))

def fmax(x, y, name=None):
    """
    Compares the elements at the corresponding positions of the two tensors and returns a new tensor containing the maximum value of the element.
    If one of them is a nan value, the other value is directly returned, if both are nan values, then the first nan value is returned.
    The equation is:

    .. math::
        out = fmax(x, y)

    Note:
        ``paddle.fmax`` supports broadcasting. If you want know more about broadcasting, please refer to :ref:`user_guide_broadcasting` .

    Args:
        x (Tensor): the input tensor, it's data type should be float16, float32, float64, int32, int64.
        y (Tensor): the input tensor, it's data type should be float16, float32, float64, int32, int64.
        name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        N-D Tensor. A location into which the result is stored. If x, y have different shapes and are "broadcastable", the resulting tensor shape is the shape of x and y after broadcasting. If x, y have the same shape,  its shape is the same as x and y.

    Examples:

        .. code-block:: python

            import numpy as np
            import paddle

            x = paddle.to_tensor([[1, 2], [7, 8]])
            y = paddle.to_tensor([[3, 4], [5, 6]])
            res = paddle.fmax(x, y)
            print(res)
            #    [[3, 4],
            #     [7, 8]]

            x = paddle.to_tensor([[1, 2, 3], [1, 2, 3]])
            y = paddle.to_tensor([3, 0, 4])
            res = paddle.fmax(x, y)
            print(res)
            #    [[3, 2, 4],
            #     [3, 2, 4]]

            x = paddle.to_tensor([2, 3, 5], dtype='float32')
            y = paddle.to_tensor([1, np.nan, np.nan], dtype='float32')
            res = paddle.fmax(x, y)
            print(res)
            #    [ 2., 3., 5.]

            x = paddle.to_tensor([5, 3, np.inf], dtype='float32')
            y = paddle.to_tensor([1, -np.inf, 5], dtype='float32')
            res = paddle.fmax(x, y)
            print(res)
            #    [  5.,   3., inf.]
    """
    op_type = 'elementwise_fmax'
    axis = -1
    act = None
    if in_dygraph_mode():
        return _C_ops.fmax(x, y, axis)
    if _in_legacy_dygraph():
        return _elementwise_op_in_dygraph(
            x, y, axis=axis, act=act, op_name=op_type)
    return _elementwise_op(LayerHelper(op_type, **locals()))

def fmin(x, y, name=None):
    """
    Compares the elements at the corresponding positions of the two tensors and returns a new tensor containing the minimum value of the element.
    If one of them is a nan value, the other value is directly returned, if both are nan values, then the first nan value is returned.
    The equation is:

    .. math::
        out = fmin(x, y)

    Note:
        ``paddle.fmin`` supports broadcasting. If you want know more about broadcasting, please refer to :ref:`user_guide_broadcasting` .

    Args:
        x (Tensor): the input tensor, it's data type should be float16, float32, float64, int32, int64.
        y (Tensor): the input tensor, it's data type should be float16, float32, float64, int32, int64.
        name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        N-D Tensor. A location into which the result is stored. If x, y have different shapes and are "broadcastable", the resulting tensor shape is the shape of x and y after broadcasting. If x, y have the same shape,  its shape is the same as x and y.

    Examples:

        .. code-block:: python

            import numpy as np
            import paddle

            x = paddle.to_tensor([[1, 2], [7, 8]])
            y = paddle.to_tensor([[3, 4], [5, 6]])
            res = paddle.fmin(x, y)
            print(res)
            #       [[1, 2],
            #        [5, 6]]

            x = paddle.to_tensor([[[1, 2, 3], [1, 2, 3]]])
            y = paddle.to_tensor([3, 0, 4])
            res = paddle.fmin(x, y)
            print(res)
            #       [[[1, 0, 3],
            #         [1, 0, 3]]]

            x = paddle.to_tensor([2, 3, 5], dtype='float32')
            y = paddle.to_tensor([1, np.nan, np.nan], dtype='float32')
            res = paddle.fmin(x, y)
            print(res)
            #       [ 1., 3., 5.]

            x = paddle.to_tensor([5, 3, np.inf], dtype='float64')
            y = paddle.to_tensor([1, -np.inf, 5], dtype='float64')
            res = paddle.fmin(x, y)
            print(res)
            #       [   1., -inf.,    5.]
    """
    op_type = 'elementwise_fmin'
    axis = -1
    act = None
    if in_dygraph_mode():
        return _C_ops.fmin(x, y, axis)
    if _in_legacy_dygraph():
        return _elementwise_op_in_dygraph(
            x, y, axis=axis, act=act, op_name=op_type)
    return _elementwise_op(LayerHelper(op_type, **locals()))


def sum(x, axis=None, dtype=None, keepdim=False, name=None):
    """
    Computes the sum of tensor elements over the given dimension.

    Args:
        x (Tensor): An N-D Tensor, the data type is bool, float16, float32, float64, int32 or int64.
        axis (int|list|tuple, optional): The dimensions along which the sum is performed. If
            :attr:`None`, sum all elements of :attr:`x` and return a
            Tensor with a single element, otherwise must be in the
            range :math:`[-rank(x), rank(x))`. If :math:`axis[i] < 0`,
            the dimension to reduce is :math:`rank + axis[i]`.
        dtype (str, optional): The dtype of output Tensor. The default value is None, the dtype
            of output is the same as input Tensor `x`.
        keepdim (bool, optional): Whether to reserve the reduced dimension in the
            output Tensor. The result Tensor will have one fewer dimension
            than the :attr:`x` unless :attr:`keepdim` is true, default
            value is False.
        name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        Tensor: Results of summation operation on the specified axis of input Tensor `x`,
        if `x.dtype='bool'`, `x.dtype='int32'`, it's data type is `'int64'`, 
        otherwise it's data type is the same as `x`.

    Examples:
        .. code-block:: python

            import paddle

            # x is a Tensor with following elements:
            #    [[0.2, 0.3, 0.5, 0.9]
            #     [0.1, 0.2, 0.6, 0.7]]
            # Each example is followed by the corresponding output tensor.
            x = paddle.to_tensor([[0.2, 0.3, 0.5, 0.9],
                                  [0.1, 0.2, 0.6, 0.7]])
            out1 = paddle.sum(x)  # [3.5]
            out2 = paddle.sum(x, axis=0)  # [0.3, 0.5, 1.1, 1.6]
            out3 = paddle.sum(x, axis=-1)  # [1.9, 1.6]
            out4 = paddle.sum(x, axis=1, keepdim=True)  # [[1.9], [1.6]]

            # y is a Tensor with shape [2, 2, 2] and elements as below:
            #      [[[1, 2], [3, 4]],
            #      [[5, 6], [7, 8]]]
            # Each example is followed by the corresponding output tensor.
            y = paddle.to_tensor([[[1, 2], [3, 4]], 
                                  [[5, 6], [7, 8]]])
            out5 = paddle.sum(y, axis=[1, 2]) # [10, 26]
            out6 = paddle.sum(y, axis=[0, 1]) # [16, 20]
            
            # x is a Tensor with following elements:
            #    [[True, True, True, True]
            #     [False, False, False, False]]
            # Each example is followed by the corresponding output tensor.
            x = paddle.to_tensor([[True, True, True, True],
                                  [False, False, False, False]])
            out7 = paddle.sum(x)  # [4]
            out8 = paddle.sum(x, axis=0)  # [1, 1, 1, 1]
            out9 = paddle.sum(x, axis=1)  # [4, 0]
    """
    if isinstance(axis, Variable):
        reduce_all_flag = True if axis.shape[0] == len(x.shape) else False
    else:
        if axis is not None and not isinstance(axis, (list, tuple)):
            axis = [axis]

        if not axis:
            axis = []

        if len(axis) == 0:
            reduce_all_flag = True
        else:
            if len(axis) == len(x.shape):
                reduce_all_flag = True
            else:
                reduce_all_flag = False

    dtype_flag = False
    if dtype is not None:
        dtype_flag = True
        dtype = convert_np_dtype_to_dtype_(dtype)

    if in_dygraph_mode():
        return _C_ops.sum(x, axis, dtype, keepdim)

    if not isinstance(axis, Variable):
        axis = axis if axis != None and axis != [] and axis != () else [0]
        if utils._contain_var(axis):
            axis = utils._convert_to_tensor_list(axis)

    if _in_legacy_dygraph():
        if dtype_flag:
            return _legacy_C_ops.reduce_sum(x, 'dim', axis, 'keep_dim', keepdim,
                                       'reduce_all', reduce_all_flag, 'in_dtype',
                                       x.dtype, 'out_dtype', dtype)
        else:
            return _legacy_C_ops.reduce_sum(x, 'dim', axis, 'keep_dim', keepdim,
                                       'reduce_all', reduce_all_flag)

    attrs = {
        'dim': axis,
        'keep_dim': keepdim,
        'reduce_all': reduce_all_flag
    }

    if dtype_flag:
        attrs.update({
            'in_dtype': x.dtype,
            'out_dtype': dtype
        })

    check_variable_and_dtype(
        x, 'x', ['bool', 'float16', 'float32', 'float64',
                'int16', 'int32', 'int64', 'complex64', 'complex128',
                u'bool', u'float16', u'float32', u'float64',
                u'int32', u'int64', u'complex64', u'complex128'], 'sum')

    check_type(axis, 'axis', (int, list, tuple, type(None), Variable), 'sum')

    helper = LayerHelper('sum', **locals())
    if dtype_flag:
        out = helper.create_variable_for_type_inference(
            dtype=dtype)
    else:
        out = helper.create_variable_for_type_inference(dtype=x.dtype)
    helper.append_op(
        type='reduce_sum',
        inputs={'X': x},
        outputs={'Out': out},
        attrs=attrs)
    return out


def nansum(x, axis=None, dtype=None, keepdim=False, name=None):
    """
    Computes the sum of tensor elements over the given axis, treating Not a Numbers (NaNs) as zero.

    Args:
        x (Tensor): An N-D Tensor, the data type is float32, float64, int32 or int64.
        axis (int|list|tuple, optional): The dimensions along which the nansum is performed. If
            :attr:`None`, nansum all elements of :attr:`x` and return a
            Tensor with a single element, otherwise must be in the
            range :math:`[-rank(x), rank(x))`. If :math:`axis[i] < 0`,
            the dimension to reduce is :math:`rank + axis[i]`.
        dtype (str, optional): The dtype of output Tensor. The default value is None, the dtype
            of output is the same as input Tensor `x`.
        keepdim (bool, optional): Whether to reserve the reduced dimension in the
            output Tensor. The result Tensor will have one fewer dimension
            than the :attr:`x` unless :attr:`keepdim` is true, default
            value is False.
        name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        Tensor: Results of summation operation on the specified axis of input Tensor `x`,

    Examples:
        .. code-block:: python

            import paddle
            import numpy as np

            # x is a Tensor with following elements:
            #    [[nan, 0.3, 0.5, 0.9]
            #     [0.1, 0.2, -nan, 0.7]]
            # Each example is followed by the corresponding output tensor.
            x = np.array([[float('nan'), 0.3, 0.5, 0.9],
                            [0.1, 0.2, float('-nan'), 0.7]]).astype(np.float32)
            x = paddle.to_tensor(x)
            out1 = paddle.nansum(x)  # [2.7]
            out2 = paddle.nansum(x, axis=0)  # [0.1, 0.5, 0.5, 1.6]
            out3 = paddle.nansum(x, axis=-1)  # [1.7, 1.0]
            out4 = paddle.nansum(x, axis=1, keepdim=True)  # [[1.7], [1.0]]

            # y is a Tensor with shape [2, 2, 2] and elements as below:
            #      [[[1, nan], [3, 4]],
            #      [[5, 6], [-nan, 8]]]
            # Each example is followed by the corresponding output tensor.
            y = np.array([[[1, float('nan')], [3, 4]], 
                            [[5, 6], [float('-nan'), 8]]])
            y = paddle.to_tensor(y)
            out5 = paddle.nansum(y, axis=[1, 2]) # [8, 19]
            out6 = paddle.nansum(y, axis=[0, 1]) # [9, 18]
    """
    check_variable_and_dtype(
        x, 'x', ['float32', 'float64', 'int32', 'int64'], 'nansum')
    check_type(axis, 'axis', (int, list, tuple, type(None)), 'nansum')

    zero_tensor = paddle.zeros_like(x)
    tmp_tensor = paddle.where(isnan(x), zero_tensor, x)
    return sum(tmp_tensor, axis, dtype, keepdim, name)


def nanmean(x, axis=None, keepdim=False, name=None):
    r"""
    Compute the arithmetic mean along the specified axis, ignoring NaNs.

    Args:
        x (Tensor): The input Tensor with data type uint16, float16, float32, float64.
        axis (int|list|tuple, optional):The axis along which to perform nanmean
            calculations. ``axis`` should be int, list(int) or tuple(int). If
            ``axis`` is a list/tuple of dimension(s), nanmean is calculated along
            all element(s) of ``axis`` . ``axis`` or element(s) of ``axis``
            should be in range [-D, D), where D is the dimensions of ``x`` . If
            ``axis`` or element(s) of ``axis`` is less than 0, it works the
            same way as :math:`axis + D` . If ``axis`` is None, nanmean is
            calculated over all elements of ``x``. Default is None.
        keepdim (bool, optional): Whether to reserve the reduced dimension(s)
            in the output Tensor. If ``keepdim`` is True, the dimensions of
            the output Tensor is the same as ``x`` except in the reduced
            dimensions(it is of size 1 in this case). Otherwise, the shape of
            the output Tensor is squeezed in ``axis`` . Default is False.
        name (str, optional): Name for the operation (optional, default is None).
            For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        Tensor, results of arithmetic mean along ``axis`` of ``x``, with the same data
        type as ``x``.

    Examples:

        .. code-block:: python
            :name: code-example1

            import paddle
            # x is a 2-D Tensor:
            x = paddle.to_tensor([[float('nan'), 0.3, 0.5, 0.9],
                                  [0.1, 0.2, float('-nan'), 0.7]])
            out1 = paddle.nanmean(x)
            # [0.44999996]
            out2 = paddle.nanmean(x, axis=0)
            # [0.1, 0.25, 0.5, 0.79999995]
            out3 = paddle.nanmean(x, axis=0, keepdim=True)
            # [[0.1, 0.25, 0.5, 0.79999995]]
            out4 = paddle.nanmean(x, axis=1)
            # [0.56666666 0.33333334]
            out5 = paddle.nanmean(x, axis=1, keepdim=True)
            # [[0.56666666]
            #  [0.33333334]]

            # y is a 3-D Tensor:
            y = paddle.to_tensor([[[1, float('nan')], [3, 4]],
                                   [[5, 6], [float('-nan'), 8]]])
            out6 = paddle.nanmean(y, axis=[1, 2])
            # [2.66666675, 6.33333349]
            out7 = paddle.nanmean(y, axis=[0, 1])
            # [3., 6.]
    """
    if isinstance(axis, int):
        axis = [axis]
    check_variable_and_dtype(x, 'x/input',
                             ['uint16', 'float16', 'float32', 'float64'],
                             'nanmean' )
    if axis is not None:
        check_type(axis, 'axis/dim', (int, list, tuple), 'nanmean')

    cnt = paddle.sum(~paddle.isnan(x), axis = axis,keepdim=keepdim)
    return paddle.divide(paddle.nansum(x, axis=axis, keepdim=keepdim, name=name), cnt.astype(x.dtype))


def count_nonzero(x, axis=None, keepdim=False, name=None):
    r"""
    Counts the number of non-zero values in the tensor x along the specified axis.

    Args:
        x (Tensor): An N-D Tensor, the data type is bool, float16, float32, float64, int32 or int64.
        axis (int|list|tuple, optional): The dimensions along which the sum is performed. If
            :attr:`None`, sum all elements of :attr:`x` and return a
            Tensor with a single element, otherwise must be in the
            range :math:`[-rank(x), rank(x))`. If :math:`axis[i] < 0`,
            the dimension to reduce is :math:`rank + axis[i]`.
        keepdim (bool, optional): Whether to reserve the reduced dimension in the
            output Tensor. The result Tensor will have one fewer dimension
            than the :attr:`x` unless :attr:`keepdim` is true, default
            value is False.
        name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        Tensor: Results of count operation on the specified axis of input Tensor `x`, it's data type is `'int64'`.

    Examples:

        .. code-block:: python

            import paddle
            # x is a 2-D Tensor:
            x = paddle.to_tensor([[0., 1.1, 1.2], [0., 0., 1.3], [0., 0., 0.]])
            out1 = paddle.count_nonzero(x)
            # [3]
            out2 = paddle.count_nonzero(x, axis=0)
            # [0, 1, 2]
            out3 = paddle.count_nonzero(x, axis=0, keepdim=True)
            # [[0, 1, 2]]
            out4 = paddle.count_nonzero(x, axis=1)
            # [2, 1, 0]
            out5 = paddle.count_nonzero(x, axis=1, keepdim=True)
            #[[2],
            # [1],
            # [0]]

            # y is a 3-D Tensor:
            y = paddle.to_tensor([[[0., 1.1, 1.2], [0., 0., 1.3], [0., 0., 0.]],
                                  [[0., 2.5, 2.6], [0., 0., 2.4], [2.1, 2.2, 2.3]]])
            out6 = paddle.count_nonzero(y, axis=[1, 2])
            # [3, 6]
            out7 = paddle.count_nonzero(y, axis=[0, 1])
            # [1, 3, 5]
    """


    if axis is not None:
        if isinstance(axis, int):
            axis = [axis]
        dims = len(x.shape)
        for i in range(len(axis)):
            if not isinstance(axis[i], int) or not (axis[i] < dims and axis[i] >= -dims):
                raise ValueError(
                    "Axis should be None, int, or a list, element should in range [-rank(x), rank(x))."
                )

    bool_tensor = paddle.cast(x, 'bool')
    int_tensor = paddle.cast(bool_tensor, 'int64')
    return paddle.sum(int_tensor, axis=axis, keepdim=keepdim, name=name)


@templatedoc(op_type="sum")
def add_n(inputs, name=None):
    """
    Sum one or more Tensor of the input.
    
    For example:

    .. code-block:: text
    
        Case 1:

            Input:
                input.shape = [2, 3]
                input = [[1, 2, 3],
                         [4, 5, 6]]

            Output:
                output.shape = [2, 3]
                output = [[1, 2, 3],
                          [4, 5, 6]]

        Case 2:
       
            Input:
                First input:
                    input1.shape = [2, 3]
                    Input1 = [[1, 2, 3],
                              [4, 5, 6]]

                The second input:
                    input2.shape = [2, 3]
                    input2 = [[7, 8, 9],
                              [10, 11, 12]]

                Output:
                    output.shape = [2, 3]
                    output = [[8, 10, 12],
                              [14, 16, 18]]

    Args:
        inputs (Tensor|list[Tensor]|tuple[Tensor]):  A Tensor or a list/tuple of Tensors. The shape and data type of the list/tuple elements should be consistent.
            Input can be multi-dimensional Tensor, and data types can be: float32, float64, int32, int64.
        name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        Tensor, the sum of input :math:`inputs` , its shape and data types are consistent with :math:`inputs`.

    Examples:
        .. code-block:: python

            import paddle

            input0 = paddle.to_tensor([[1, 2, 3], [4, 5, 6]], dtype='float32')
            input1 = paddle.to_tensor([[7, 8, 9], [10, 11, 12]], dtype='float32')
            output = paddle.add_n([input0, input1])
            # [[8., 10., 12.], 
            #  [14., 16., 18.]]
    """
    if in_dygraph_mode():
        if isinstance(inputs, Variable):
            inputs = [inputs]
        for x in inputs:
            if not x.is_dense():
                return _legacy_C_ops.sum(inputs, 'use_mkldnn', False)
        return _C_ops.add_n(inputs)
    if _in_legacy_dygraph():
        if isinstance(inputs, Variable):
            inputs = [inputs]
        return _legacy_C_ops.sum(inputs, 'use_mkldnn', False)

    helper = LayerHelper('add_n', **locals())
    check_type(inputs, 'inputs', (Variable, tuple, list), 'add_n')
    if isinstance(inputs, list) or isinstance(inputs, tuple):
        if len(inputs) > 0:
            for input in inputs:
                check_variable_and_dtype(input, "inputs", \
                   ['float16', 'float32', 'float64', 'int32', 'int64'], 'add_n')
    else:
        check_variable_and_dtype(inputs, "inputs", \
                ['float16', 'float32', 'float64', 'int32', 'int64'], 'add_n')


    out = helper.create_variable_for_type_inference(
        dtype=helper.input_dtype('inputs'))
    helper.append_op(
        type='sum',
        inputs={'X': inputs},
        outputs={'Out': out},
        attrs={'use_mkldnn': False})

    return out


def trunc(input, name=None):
    '''
    This API is used to returns a new tensor with the truncated integer values of input.
    
    Args:
        input (Tensor): The input tensor, it's data type should be int32, int64, float32, float64.
        name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.
    
    Returns:
        Tensor: The output Tensor of trunc.
    
    Examples:
        .. code-block:: python

            import paddle

            input = paddle.rand([2,2],'float32')
            print(input)
            # Tensor(shape=[2, 2], dtype=float32, place=CUDAPlace(0), stop_gradient=True,
            #         [[0.02331470, 0.42374918],
            #         [0.79647720, 0.74970269]])

            output = paddle.trunc(input)
            print(output)
            # Tensor(shape=[2, 2], dtype=float32, place=CUDAPlace(0), stop_gradient=True,
            #         [[0., 0.],
            #         [0., 0.]]))
    '''
    if in_dygraph_mode():
        return  _C_ops.trunc(input)
    else:
        if _in_legacy_dygraph():
            return _legacy_C_ops.trunc(input)
        else:
            inputs = {"X": input}
            attrs = {}

            helper = LayerHelper("trunc", **locals())
            check_variable_and_dtype(input, 'X', ['int32', 'int64', 'float32', 'float64'], 'trunc')
            out = helper.create_variable_for_type_inference(dtype=input.dtype)

            helper.append_op(
                type="trunc", inputs=inputs, attrs=attrs, outputs={"Out": out})
            return out



def mm(input, mat2, name=None):
    """

    Applies matrix multiplication to two tensors.

    Currently, the input tensors' rank can be any, but when the rank of any
    inputs is bigger than 3, this two inputs' rank should be equal.


    Also note that if the raw tensor :math:`x` or :math:`mat2` is rank-1 and
    nontransposed, the prepended or appended dimension :math:`1` will be
    removed after matrix multiplication.

    Args:
        input (Tensor): The input tensor which is a Tensor.
        mat2 (Tensor): The input tensor which is a Tensor.
        name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        Tensor: The product Tensor.

    ::

        * example 1:

        input: [B, ..., M, K], mat2: [B, ..., K, N]
        out: [B, ..., M, N]

        * example 2:

        input: [B, M, K], mat2: [B, K, N]
        out: [B, M, N]

        * example 3:

        input: [B, M, K], mat2: [K, N]
        out: [B, M, N]

        * example 4:

        input: [M, K], mat2: [K, N]
        out: [M, N]

        * example 5:

        input: [B, M, K], mat2: [K]
        out: [B, M]

        * example 6:

        input: [K], mat2: [K]
        out: [1]

    Examples:
        .. code-block:: python

            import paddle
            input = paddle.arange(1, 7).reshape((3, 2)).astype('float32')
            mat2 = paddle.arange(1, 9).reshape((2, 4)).astype('float32')
            out = paddle.mm(input, mat2)
            print(out)
            #        [[11., 14., 17., 20.],
            #         [23., 30., 37., 44.],
            #         [35., 46., 57., 68.]])


    """
    if in_dygraph_mode():
        return _C_ops.matmul(input, mat2, False, False)
    elif paddle.in_dynamic_mode():
        return _legacy_C_ops.matmul_v2(input, mat2)

    def __check_input(x, y):
        var_names = {'x': x, 'y': y}
        for name, val in var_names.items():
            check_variable_and_dtype(val, name,
                                     ['float16', 'float32', 'float64'], 'mm')
        x_shape = list(x.shape)
        y_shape = list(y.shape)
        if len(x_shape) == 1:
            x_shape = [1] + x_shape
        if len(y_shape) == 1:
            y_shape = y_shape + [1]

        # check the inner 2 dimensions
        if x_shape[-1] != y_shape[-2]:
            if not ((x_shape[-1] == -1) or (y_shape[-2] == -1)):
                raise ValueError(
                    "After performing an optional transpose, Input X's width should be "
                    "equal to Y's width for multiplication "
                    "prerequisites. But received X's shape: %s, Y's shape: %s\n"
                    % (x_shape, y_shape))

        if len(y_shape) > 2 and len(x_shape) > 2:
            for i, dim_x in enumerate(x_shape[:-2]):
                # don't check neg shape
                if dim_x < 0 or y_shape[i] < 0:
                    continue
                if dim_x != y_shape[i]:
                    raise ValueError(
                        "When the matrix is larger than 2 dimensions, the higher "
                        "dimensional values of the two matrices need to be equal. "
                        "But received x_shape[%d] != y_shape[%d]. X's shape: %s, "
                        "Y's shape: %s.\n" % (i, i, x_shape, y_shape))

    __check_input(input, mat2)

    helper = LayerHelper('mm', **locals())
    out = helper.create_variable_for_type_inference(dtype=input.dtype)
    helper.append_op(
        type='matmul_v2', inputs={'X': input,
                               'Y': mat2}, outputs={'Out': out})
    return out


def addmm(input, x, y, beta=1.0, alpha=1.0, name=None):
    """
    **addmm**

    Perform matrix multiplication for input $x$ and $y$.
    $input$ is added to the final result.
    The equation is:

    ..  math::
        Out = alpha * x * y + beta * input

    $Input$, $x$ and $y$ can carry the LoD (Level of Details) information, or not. But the output only shares the LoD information with input $input$.

    Args:
        input (Tensor): The input Tensor to be added to the final result.
        x (Tensor): The first input Tensor for matrix multiplication.
        y (Tensor): The second input Tensor for matrix multiplication.
        beta (float, optional): Coefficient of $input$, default is 1.
        alpha (float, optional): Coefficient of $x*y$, default is 1.
        name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        Tensor: The output Tensor of addmm.

    Examples:
        ..  code-block:: python
            
            import paddle

            x = paddle.ones([2,2])
            y = paddle.ones([2,2])
            input = paddle.ones([2,2])

            out = paddle.addmm( input=input, x=x, y=y, beta=0.5, alpha=5.0 )

            print(out)
            # [[10.5 10.5]
            # [10.5 10.5]]
    """
    input_shape = input.shape
    x_shape = x.shape
    y_shape = y.shape
    if not len(x_shape) == len(y_shape) == 2:
        raise ValueError("The dimention of x, y should be 2 but receive x's shape: {}, y's shape: {}".format(x_shape, y_shape))
    if x_shape[1] != y_shape[0]:
        raise ValueError("The input Variable x's width must be equal with Variable y' height. But received x's shape = {}, y's shape = {}.".format(x_shape, y_shape))
    if len(input_shape) == 2:
        if input_shape[0] != x_shape[0]:
            if input_shape[0] != 1:
                raise ValueError( "When x's dimension[0] is not equal with input's dimension[0], input's dimension[0] must be 1 but got {}".format(input_shape[0]))
            if input_shape[1] != y_shape[1] and input_shape[1] != 1:
                raise ValueError( "When y's dimension[1] is not equal with input's dimension[1], input's dimension[1] must be 1 but got {}".format(input_shape[1]))
        if input_shape[1] != y_shape[1]:
            if input_shape[1] != 1:
                raise ValueError( "When y's dimension[1] is not equal with input's dimension[1], input's dimension[1] must be 1 but got {}".format(input_shape[1]))
    elif len(input_shape) == 1:
        if input_shape[0] not in (y_shape[1], 1):
            raise ValueError("The input's shape: {} is not broadcastable with [x.shape[0], y.shape[1]]: [{},{}]".format(input_shape, x_shape[0], y_shape[1]))
    else:
        raise ValueError("The dimention of input should be 2 or 1 but receive input's shape: {}".format(input_shape))



    if in_dygraph_mode():
        return _C_ops.addmm( input, x, y, alpha, beta)
    else:
        if _in_legacy_dygraph():
            out = _legacy_C_ops.addmm(input, x, y, "Alpha", alpha, "Beta", beta)
            return out
        else:
            inputs = {'Input': input, "X": x, "Y": y}
            attrs = {'Alpha': alpha, 'Beta': beta}

            helper = LayerHelper("addmm", **locals())
            check_variable_and_dtype(input, 'Input', ['float32', 'float64'], 'addmm')
            check_variable_and_dtype(x, 'X', ['float32', 'float64'], 'addmm')
            check_variable_and_dtype(y, 'Y', ['float32', 'float64'], 'addmm')
            out = helper.create_variable_for_type_inference(dtype=x.dtype)

            helper.append_op(
                type="addmm", inputs=inputs, attrs=attrs, outputs={"Out": out})
            return out

def renorm(x, p, axis, max_norm):
    """
    **renorm**

    This operator is used to calculate the p-norm along the axis,
    suppose the input-shape on axis dimension has the value of T, then
    the tensor is split into T parts, the p-norm should be calculated for each
    part, if the p-norm for part i is larger than max-norm, then each element 
    in part i should be re-normalized at the same scale so that part-i' p-norm equals
    max-norm exactly, otherwise part-i stays unchanged.

    Args:
        x (Tensor): The input Tensor
        p (float): The power of the norm operation.
        axis (int): the dimension to slice the tensor.
        max-norm (float): the maximal norm limit.

    Returns:
        Tensor: the renorm Tensor.

    Examples:
        ..  code-block:: python
            
            import paddle
            input = [[[2.0,2,-2],[3,0.3,3]],[[2,-8,2],[3.1,3.7,3]]]
            x = paddle.to_tensor(input,dtype='float32')
            y = paddle.renorm(x, 1.0, 2, 2.05)
            print(y)        
    #        [[[ 0.40594056,  0.29285714, -0.41000000],
    #          [ 0.60891086,  0.04392857,  0.61500001]],
    #         [[ 0.40594056, -1.17142856,  0.41000000],
    #          [ 0.62920785,  0.54178572,  0.61500001]]])
    
    """
    input_shape = x.shape
    check_variable_and_dtype(x, 'x', ['float32', 'float64'], 'renorm')
    if not axis < len(input_shape):
        raise ValueError("the axis:{} should be less then the shape's size {}:{}".format(axis,len(input_shape),input_shape))
    if not axis >=0:
        if not axis >= -1 * len(input_shape):
            raise ValueError("the axis:{} should not be less than -1 * length of input_shape:{}".format(axis,-1 * len(input_shape)))
        axis = axis + len(input_shape)
    if in_dygraph_mode():
        out = _C_ops.renorm(x, p, axis, max_norm)
        return out
    elif _in_legacy_dygraph():
        out = _legacy_C_ops.renorm(x, 'p',p, 'axis',axis, 'max_norm', max_norm)
        return out

    inputs = {'X': x}
    attrs = {'p': p, 'axis': axis, 'max_norm':max_norm}

    helper = LayerHelper("renorm", **locals())
    out = helper.create_variable_for_type_inference(dtype=x.dtype)

    helper.append_op(
        type="renorm", inputs=inputs, attrs=attrs, outputs={"Out": out})
    return out



def inner(x, y, name=None):
    """

    Inner product of two input Tensor.
    
    Ordinary inner product for 1-D Tensors, in higher dimensions a sum product over the last axes.

    Args:
        x (Tensor): An N-D Tensor or a Scalar Tensor. If its not a scalar Tensor, its last dimensions must match y's.
        y (Tensor): An N-D Tensor or a Scalar Tensor. If its not a scalar Tensor, its last dimensions must match x's.
        name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        Tensor: The inner-product Tensor, the output shape is x.shape[:-1] + y.shape[:-1].

    Examples:
        .. code-block:: python

            import paddle
            x = paddle.arange(1, 7).reshape((2, 3)).astype('float32')
            y = paddle.arange(1, 10).reshape((3, 3)).astype('float32')
            out = paddle.inner(x, y)
            print(out)
            #        ([[14, 32, 50],
            #         [32, 77, 122]])


    """
    if x.size == 1 or y.size == 1:
        return multiply(x, y)
    else:
        xshape = x.shape
        yshape = y.shape
        dstshape = list(xshape[:-1])+list(yshape[:-1])
        if len(dstshape)==0:
            dstshape = [1]
        nx = x.reshape((-1, xshape[-1]))
        ny = y.reshape((-1, yshape[-1]))

        if in_dygraph_mode():
            return _C_ops.matmul(nx, ny.T, False, False).reshape(dstshape)
        elif paddle.in_dynamic_mode():
            return _legacy_C_ops.matmul_v2(nx, ny.T).reshape(dstshape)

        def __check_input(x, y):
            var_names = {'x': x, 'y': y}
            for name, val in var_names.items():
                check_variable_and_dtype(val, name,
                                        ['float16', 'float32', 'float64'], 'inner')
            x_shape = list(xshape)
            y_shape = list(yshape)

            # check the inner 2 dimensions
            if x_shape[-1] != y_shape[-1]:
                if not ((x_shape[-1] == -1) or (y_shape[-1] == -1)):
                    raise ValueError(
                        "After performing an optional transpose, Input X's last dim should be "
                        "equal to Y's last dim for multiplication "
                        "prerequisites. But received X's shape: %s, Y's shape: %s\n"
                        % (x_shape, y_shape))

        __check_input(nx, ny)

        helper = LayerHelper('inner', **locals())
        out = helper.create_variable_for_type_inference(dtype=nx.dtype)
        helper.append_op(
            type='matmul_v2', inputs={'X': nx,
                                'Y': ny.T}, outputs={'Out': out})
        return out.reshape(dstshape)


def outer(x, y, name=None):
    """

    Outer product of two Tensors.

    Input is flattened if not already 1-dimensional.

    Args:
        x (Tensor): An N-D Tensor or a Scalar Tensor. 
        y (Tensor): An N-D Tensor or a Scalar Tensor. 
        name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        Tensor: The outer-product Tensor.

    Examples:
        .. code-block:: python

            import paddle
            x = paddle.arange(1, 4).astype('float32')
            y = paddle.arange(1, 6).astype('float32')
            out = paddle.outer(x, y)
            print(out)
            #        ([[1, 2, 3, 4, 5],
            #         [2, 4, 6, 8, 10],
            #         [3, 6, 9, 12, 15]])


    """
    nx = x.reshape((-1, 1))
    ny = y.reshape((1, -1))

    if in_dygraph_mode():
        return _C_ops.matmul(nx, ny, False, False)
    elif paddle.in_dynamic_mode():
        return _legacy_C_ops.matmul_v2(nx, ny)

    def __check_input(x, y):
        var_names = {'x': x, 'y': y}
        for name, val in var_names.items():
            check_variable_and_dtype(val, name,
                                     ['float16', 'float32', 'float64'], 'inner')

    __check_input(nx, ny)

    helper = LayerHelper('outer', **locals())
    out = helper.create_variable_for_type_inference(dtype=nx.dtype)
    helper.append_op(
        type='matmul_v2', inputs={'X': nx,
                               'Y': ny}, outputs={'Out': out})
    return out


def logsumexp(x, axis=None, keepdim=False, name=None):
    r"""
    Calculates the log of the sum of exponentials of ``x`` along ``axis`` .

    .. math::
       logsumexp(x) = \log\sum exp(x)

    Args:
        x (Tensor): The input Tensor with data type float32 or float64, which 
            have no more than 4 dimensions.
        axis (int|list|tuple, optional): The axis along which to perform
            logsumexp calculations. ``axis`` should be int, list(int) or
            tuple(int). If ``axis`` is a list/tuple of dimension(s), logsumexp
            is calculated along all element(s) of ``axis`` . ``axis`` or
            element(s) of ``axis`` should be in range [-D, D), where D is the
            dimensions of ``x`` . If ``axis`` or element(s) of ``axis`` is
            less than 0, it works the same way as :math:`axis + D` . If
            ``axis`` is None, logsumexp is calculated along all elements of
            ``x``. Default is None.
        keepdim (bool, optional): Whether to reserve the reduced dimension(s)
            in the output Tensor. If ``keep_dim`` is True, the dimensions of
            the output Tensor is the same as ``x`` except in the reduced
            dimensions(it is of size 1 in this case). Otherwise, the shape of
            the output Tensor is squeezed in ``axis`` . Default is False.
        name (str, optional): Name for the operation (optional, default is None).
            For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        Tensor, results of logsumexp along ``axis`` of ``x``, with the same data
        type as ``x``.

    Examples:

    .. code-block:: python

        import paddle

        x = paddle.to_tensor([[-1.5, 0., 2.], [3., 1.2, -2.4]])
        out1 = paddle.logsumexp(x) # [3.4691226]
        out2 = paddle.logsumexp(x, 1) # [2.15317821, 3.15684602]

    """
    if isinstance(axis, int):
        axis = [axis]
    reduce_all = True if axis is None \
        or len(axis)==0 \
        or len(axis) == len(x.shape) else False
    if axis is None or len(axis) == 0:
        axis = [0]

    if in_dygraph_mode():
        if reduce_all:
            axis = range(len(x.shape))
        return _C_ops.logsumexp(x, axis, keepdim, reduce_all)
    if _in_legacy_dygraph():
        return _legacy_C_ops.logsumexp(x, 'axis', axis, 'keepdim', keepdim, 'reduce_all', reduce_all)

    check_variable_and_dtype(x, 'x',
                             ['float32', 'float64'],
                             'logsumexp')

    helper = LayerHelper('logsumexp', **locals())
    attrs = {'axis': axis, 'keepdim': keepdim, 'reduce_all':reduce_all}
    out = helper.create_variable_for_type_inference(x.dtype)
    helper.append_op(
        type='logsumexp', inputs={'X': x}, outputs={'Out': out}, attrs=attrs)
    return out


def inverse(x, name=None):
    """
    Takes the inverse of the square matrix. A square matrix is a matrix with
    the same number of rows and columns. The input can be a square matrix
    (2-D Tensor) or batches of square matrices.

    Args:
        x (Tensor): The input tensor. The last two
            dimensions should be equal. When the number of dimensions is
            greater than 2, it is treated as batches of square matrix. The data
            type can be float32 and float64.
        name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        Tensor: A Tensor holds the inverse of x. The shape and data type
                        is the same as x.

    Examples:
        .. code-block:: python

            import paddle

            mat = paddle.to_tensor([[2, 0], [0, 2]], dtype='float32')
            inv = paddle.inverse(mat)
            print(inv) # [[0.5, 0], [0, 0.5]]

    """
    if in_dygraph_mode():
        return _C_ops.inverse(x)
    elif paddle.in_dynamic_mode():
        return _legacy_C_ops.inverse(x)

    def _check_input(x):
        check_variable_and_dtype(x, 'x',
                                 ['float32', 'float64'], 'inverse')
        if len(x.shape) < 2:
            raise ValueError(
                "The input of inverse is expected to be a Tensor whose number "
                "of dimensions is no less than 2. But reviced: %d, "
                "x's shape: %s." % (len(x.shape), x.shape))
    _check_input(x)
    helper = LayerHelper('inverse', **locals())
    out = helper.create_variable_for_type_inference(dtype=x.dtype)
    helper.append_op(
        type='inverse', inputs={'Input': [x] }, outputs={'Output': [out]})
    return out

def _get_reduce_axis(axis):
    """
    Internal function for max, min, amax and amin. 
    It computes the attribute reduce_all value based on axis.
    """
    if axis is not None and not isinstance(axis, list):
        if isinstance(axis, tuple):
            axis = list(axis)
        elif isinstance(axis, int):
            axis= [axis]
        else:
            raise TypeError(
                "The type of axis must be int, list or tuple, but received {}".format(type(axis)))
    reduce_all = True if axis == None or axis == [] else False
    if axis == None:
        axis = []
    return reduce_all, axis

def _get_reduce_axis_with_tensor(axis):
    if isinstance(axis, Variable):
        return False, axis
    return _get_reduce_axis(axis)

def _get_reduce_all_value(axis):
    """
    Internal function for max, min, amax and amin. 
    It computes the attribute reduce_all value based on axis.
    """
    if axis is not None and not isinstance(axis, list):
        if isinstance(axis, tuple):
            axis = list(axis)
        elif isinstance(axis, int):
            axis= [axis]
        else:
            raise TypeError(
                "The type of axis must be int, list or tuple, but received {}".format(type(axis)))

    reduce_all = True if axis == None or axis == [] else False
    axis = axis if axis != None and axis != [] else [0]
    return reduce_all, axis

def max(x, axis=None, keepdim=False, name=None):
    """

    Computes the maximum of tensor elements over the given axis.

    Note:
        The difference between max and amax is: If there are multiple maximum elements,
        amax evenly distributes gradient between these equal values, 
        while max propagates gradient to all of them.


    Args:
        x (Tensor): A tensor, the data type is float32, float64, int32, int64.
        axis (int|list|tuple, optional): The axis along which the maximum is computed.
            If :attr:`None`, compute the maximum over all elements of
            `x` and return a Tensor with a single element,
            otherwise must be in the range :math:`[-x.ndim(x), x.ndim(x))`.
            If :math:`axis[i] < 0`, the axis to reduce is :math:`x.ndim + axis[i]`.
        keepdim (bool, optional): Whether to reserve the reduced dimension in the
            output Tensor. The result tensor will have one fewer dimension
            than the `x` unless :attr:`keepdim` is true, default
            value is False.
        name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        Tensor, results of maximum on the specified axis of input tensor,
        it's data type is the same as `x`.

    Examples:
        .. code-block:: python

            import paddle

            # data_x is a Tensor with shape [2, 4]
            # the axis is a int element
            x = paddle.to_tensor([[0.2, 0.3, 0.5, 0.9],
                                  [0.1, 0.2, 0.6, 0.7]], 
                                 dtype='float64', stop_gradient=False)
            result1 = paddle.max(x)
            result1.backward()
            print(result1, x.grad) 
            #[0.9], [[0., 0., 0., 1.], [0., 0., 0., 0.]]

            x.clear_grad()
            result2 = paddle.max(x, axis=0)
            result2.backward()
            print(result2, x.grad) 
            #[0.2, 0.3, 0.6, 0.9], [[1., 1., 0., 1.], [0., 0., 1., 0.]]

            x.clear_grad()
            result3 = paddle.max(x, axis=-1)
            result3.backward()
            print(result3, x.grad) 
            #[0.9, 0.7], [[0., 0., 0., 1.], [0., 0., 0., 1.]]

            x.clear_grad()
            result4 = paddle.max(x, axis=1, keepdim=True)
            result4.backward()
            print(result4, x.grad) 
            #[[0.9], [0.7]], [[0., 0., 0., 1.], [0., 0., 0., 1.]]

            # data_y is a Tensor with shape [2, 2, 2]
            # the axis is list 
            y = paddle.to_tensor([[[1.0, 2.0], [3.0, 4.0]],
                                  [[5.0, 6.0], [7.0, 8.0]]],
                                 dtype='float64', stop_gradient=False)
            result5 = paddle.max(y, axis=[1, 2])
            result5.backward()
            print(result5, y.grad) 
            #[4., 8.], [[[0., 0.], [0., 1.]], [[0., 0.], [0., 1.]]]

            y.clear_grad()
            result6 = paddle.max(y, axis=[0, 1])
            result6.backward()
            print(result6, y.grad) 
            #[7., 8.], [[[0., 0.], [0., 0.]], [[0., 0.], [1., 1.]]]
    """

    reduce_all, axis = _get_reduce_axis_with_tensor(axis)
    if in_dygraph_mode():
        return _C_ops.max(x, axis, keepdim)
    if _in_legacy_dygraph():
        return _legacy_C_ops.reduce_max(x, 'dim', axis, 'keep_dim', keepdim,
                                   'reduce_all', reduce_all)

    helper = LayerHelper('max', **locals())
    check_variable_and_dtype(
        x, 'x', ['float32', 'float64', 'int32', 'int64'], 'max')
    if not isinstance(axis, Variable) and utils._contain_var(axis):
        axis = utils._convert_to_tensor_list(axis)

    out = helper.create_variable_for_type_inference(
            dtype=x.dtype)
    helper.append_op(
        type='reduce_max',
        inputs={'X': x},
        outputs={'Out': out},
        attrs={
            'dim': axis,
            'keep_dim': keepdim,
            'reduce_all': reduce_all
        })
    return out

def min(x, axis=None, keepdim=False, name=None):
    """

    Computes the minimum of tensor elements over the given axis

    Note:
        The difference between min and amin is: If there are multiple minimum elements,
        amin evenly distributes gradient between these equal values, 
        while min propagates gradient to all of them.

    Args:
        x (Tensor): A tensor, the data type is float32, float64, int32, int64.
        axis (int|list|tuple, optional): The axis along which the minimum is computed.
            If :attr:`None`, compute the minimum over all elements of
            `x` and return a Tensor with a single element,
            otherwise must be in the range :math:`[-x.ndim, x.ndim)`.
            If :math:`axis[i] < 0`, the axis to reduce is :math:`x.ndim + axis[i]`.
        keepdim (bool, optional): Whether to reserve the reduced dimension in the
            output Tensor. The result tensor will have one fewer dimension
            than the `x` unless :attr:`keepdim` is true, default
            value is False.
        name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        Tensor, results of minimum on the specified axis of input tensor,
        it's data type is the same as input's Tensor.

    Examples:
        .. code-block:: python

            import paddle

            # data_x is a Tensor with shape [2, 4]
            # the axis is a int element
            x = paddle.to_tensor([[0.2, 0.3, 0.5, 0.9],
                                  [0.1, 0.2, 0.6, 0.7]], 
                                 dtype='float64', stop_gradient=False)
            result1 = paddle.min(x)
            result1.backward()
            print(result1, x.grad) 
            #[0.1], [[0., 0., 0., 0.], [1., 0., 0., 0.]]

            x.clear_grad()
            result2 = paddle.min(x, axis=0)
            result2.backward()
            print(result2, x.grad) 
            #[0.1, 0.2, 0.5, 0.7], [[0., 0., 1., 0.], [1., 1., 0., 1.]]

            x.clear_grad()
            result3 = paddle.min(x, axis=-1)
            result3.backward()
            print(result3, x.grad) 
            #[0.2, 0.1], [[1., 0., 0., 0.], [1., 0., 0., 0.]]

            x.clear_grad()
            result4 = paddle.min(x, axis=1, keepdim=True)
            result4.backward()
            print(result4, x.grad) 
            #[[0.2], [0.1]], [[1., 0., 0., 0.], [1., 0., 0., 0.]]

            # data_y is a Tensor with shape [2, 2, 2]
            # the axis is list 
            y = paddle.to_tensor([[[1.0, 2.0], [3.0, 4.0]],
                                  [[5.0, 6.0], [7.0, 8.0]]],
                                 dtype='float64', stop_gradient=False)
            result5 = paddle.min(y, axis=[1, 2])
            result5.backward()
            print(result5, y.grad) 
            #[1., 5.], [[[1., 0.], [0., 0.]], [[1., 0.], [0., 0.]]]

            y.clear_grad()
            result6 = paddle.min(y, axis=[0, 1])
            result6.backward()
            print(result6, y.grad) 
            #[1., 2.], [[[1., 1.], [0., 0.]], [[0., 0.], [0., 0.]]]
    """

    reduce_all, axis = _get_reduce_axis_with_tensor(axis)
    if in_dygraph_mode():
        return _C_ops.min(x, axis, keepdim)

    if _in_legacy_dygraph():
        return _legacy_C_ops.reduce_min(x, 'dim', axis, 'keep_dim', keepdim,
                                   'reduce_all', reduce_all)

    helper = LayerHelper('min', **locals())
    check_variable_and_dtype(
        x, 'x', ['float32', 'float64', 'int32', 'int64'], 'min')
    if not isinstance(axis, Variable) and utils._contain_var(axis):
        axis = utils._convert_to_tensor_list(axis)

    out = helper.create_variable_for_type_inference(
            dtype=x.dtype)
    helper.append_op(
        type='reduce_min',
        inputs={'X': x},
        outputs={'Out': out},
        attrs={
            'dim': axis,
            'keep_dim': keepdim,
            'reduce_all': reduce_all
        })
    return out

def amax(x, axis=None, keepdim=False, name=None):
    """
    Computes the maximum of tensor elements over the given axis.

    Note:
        The difference between max and amax is: If there are multiple maximum elements,
        amax evenly distributes gradient between these equal values, 
        while max propagates gradient to all of them.

    Args:
        x (Tensor): A tensor, the data type is float32, float64, int32, int64,
            the dimension is no more than 4.
        axis (int|list|tuple, optional): The axis along which the maximum is computed.
            If :attr:`None`, compute the maximum over all elements of
            `x` and return a Tensor with a single element,
            otherwise must be in the range :math:`[-x.ndim(x), x.ndim(x))`.
            If :math:`axis[i] < 0`, the axis to reduce is :math:`x.ndim + axis[i]`.
        keepdim (bool, optional): Whether to reserve the reduced dimension in the
            output Tensor. The result tensor will have one fewer dimension
            than the `x` unless :attr:`keepdim` is true, default
            value is False.
        name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        Tensor, results of maximum on the specified axis of input tensor,
        it's data type is the same as `x`.

    Examples:
        .. code-block:: python

            import paddle
            # data_x is a Tensor with shape [2, 4] with multiple maximum elements
            # the axis is a int element

            x = paddle.to_tensor([[0.1, 0.9, 0.9, 0.9],
                                  [0.9, 0.9, 0.6, 0.7]], 
                                 dtype='float64', stop_gradient=False)
            # There are 5 maximum elements: 
            # 1) amax evenly distributes gradient between these equal values, 
            #    thus the corresponding gradients are 1/5=0.2;
            # 2) while max propagates gradient to all of them, 
            #    thus the corresponding gradient are 1.
            result1 = paddle.amax(x)
            result1.backward()
            print(result1, x.grad) 
            #[0.9], [[0., 0.2, 0.2, 0.2], [0.2, 0.2, 0., 0.]]

            x.clear_grad()
            result1_max = paddle.max(x)
            result1_max.backward()
            print(result1_max, x.grad) 
            #[0.9], [[0., 1.0, 1.0, 1.0], [1.0, 1.0, 0., 0.]]

            ###############################

            x.clear_grad()
            result2 = paddle.amax(x, axis=0)
            result2.backward()
            print(result2, x.grad) 
            #[0.9, 0.9, 0.9, 0.9], [[0., 0.5, 1., 1.], [1., 0.5, 0., 0.]]

            x.clear_grad()
            result3 = paddle.amax(x, axis=-1)
            result3.backward()
            print(result3, x.grad) 
            #[0.9, 0.9], [[0., 0.3333, 0.3333, 0.3333], [0.5, 0.5, 0., 0.]]

            x.clear_grad()
            result4 = paddle.amax(x, axis=1, keepdim=True)
            result4.backward()
            print(result4, x.grad) 
            #[[0.9], [0.9]], [[0., 0.3333, 0.3333, 0.3333.], [0.5, 0.5, 0., 0.]]

            # data_y is a Tensor with shape [2, 2, 2]
            # the axis is list 
            y = paddle.to_tensor([[[0.1, 0.9], [0.9, 0.9]],
                                  [[0.9, 0.9], [0.6, 0.7]]],
                                 dtype='float64', stop_gradient=False)
            result5 = paddle.amax(y, axis=[1, 2])
            result5.backward()
            print(result5, y.grad) 
            #[0.9., 0.9], [[[0., 0.3333], [0.3333, 0.3333]], [[0.5, 0.5], [0., 1.]]]

            y.clear_grad()
            result6 = paddle.amax(y, axis=[0, 1])
            result6.backward()
            print(result6, y.grad) 
            #[0.9., 0.9], [[[0., 0.3333], [0.5, 0.3333]], [[0.5, 0.3333], [1., 1.]]]
    """

    reduce_all, axis = _get_reduce_axis(axis)
    if in_dygraph_mode():
        return _C_ops.amax(x,  axis,  keepdim)
    if _in_legacy_dygraph():
        return _legacy_C_ops.reduce_amax(x, 'dim', axis, 'keep_dim', keepdim, 'reduce_all', reduce_all)

    helper = LayerHelper('amax', **locals())
    check_variable_and_dtype(
        x, 'x', ['float32', 'float64', 'int32', 'int64'], 'amax')

    out = helper.create_variable_for_type_inference(
            dtype=x.dtype)
    helper.append_op(
        type='reduce_amax',
        inputs={'X': x},
        outputs={'Out': out},
        attrs={
            'dim': axis,
            'keep_dim': keepdim,
            'reduce_all': reduce_all
        })
    return out

def amin(x, axis=None, keepdim=False, name=None):
    """

    Computes the minimum of tensor elements over the given axis

    Note:
        The difference between min and amin is: If there are multiple minimum elements,
        amin evenly distributes gradient between these equal values, 
        while min propagates gradient to all of them.

    Args:
        x (Tensor): A tensor, the data type is float32, float64, int32, int64, 
            the dimension is no more than 4.
        axis (int|list|tuple, optional): The axis along which the minimum is computed.
            If :attr:`None`, compute the minimum over all elements of
            `x` and return a Tensor with a single element,
            otherwise must be in the range :math:`[-x.ndim, x.ndim)`.
            If :math:`axis[i] < 0`, the axis to reduce is :math:`x.ndim + axis[i]`.
        keepdim (bool, optional): Whether to reserve the reduced dimension in the
            output Tensor. The result tensor will have one fewer dimension
            than the `x` unless :attr:`keepdim` is true, default
            value is False.
        name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        Tensor, results of minimum on the specified axis of input tensor,
        it's data type is the same as input's Tensor.

    Examples:
        .. code-block:: python

            import paddle
            # data_x is a Tensor with shape [2, 4] with multiple minimum elements
            # the axis is a int element

            x = paddle.to_tensor([[0.2, 0.1, 0.1, 0.1],
                                  [0.1, 0.1, 0.6, 0.7]], 
                                 dtype='float64', stop_gradient=False)
            # There are 5 minimum elements: 
            # 1) amin evenly distributes gradient between these equal values, 
            #    thus the corresponding gradients are 1/5=0.2;
            # 2) while min propagates gradient to all of them, 
            #    thus the corresponding gradient are 1.
            result1 = paddle.amin(x)
            result1.backward()
            print(result1, x.grad) 
            #[0.1], [[0., 0.2, 0.2, 0.2], [0.2, 0.2, 0., 0.]]

            x.clear_grad()
            result1_min = paddle.min(x)
            result1_min.backward()
            print(result1_min, x.grad) 
            #[0.1], [[0., 1.0, 1.0, 1.0], [1.0, 1.0, 0., 0.]]

            ###############################

            x.clear_grad()
            result2 = paddle.amin(x, axis=0)
            result2.backward()
            print(result2, x.grad) 
            #[0.1, 0.1, 0.1, 0.1], [[0., 0.5, 1., 1.], [1., 0.5, 0., 0.]]

            x.clear_grad()
            result3 = paddle.amin(x, axis=-1)
            result3.backward()
            print(result3, x.grad) 
            #[0.1, 0.1], [[0., 0.3333, 0.3333, 0.3333], [0.5, 0.5, 0., 0.]]

            x.clear_grad()
            result4 = paddle.amin(x, axis=1, keepdim=True)
            result4.backward()
            print(result4, x.grad) 
            #[[0.1], [0.1]], [[0., 0.3333, 0.3333, 0.3333.], [0.5, 0.5, 0., 0.]]

            # data_y is a Tensor with shape [2, 2, 2]
            # the axis is list 
            y = paddle.to_tensor([[[0.2, 0.1], [0.1, 0.1]],
                                  [[0.1, 0.1], [0.6, 0.7]]],
                                 dtype='float64', stop_gradient=False)
            result5 = paddle.amin(y, axis=[1, 2])
            result5.backward()
            print(result5, y.grad) 
            #[0.1., 0.1], [[[0., 0.3333], [0.3333, 0.3333]], [[0.5, 0.5], [0., 1.]]]

            y.clear_grad()
            result6 = paddle.amin(y, axis=[0, 1])
            result6.backward()
            print(result6, y.grad) 
            #[0.1., 0.1], [[[0., 0.3333], [0.5, 0.3333]], [[0.5, 0.3333], [1., 1.]]]
    """

    reduce_all, axis = _get_reduce_axis( axis )
    if in_dygraph_mode():
        return _C_ops.amin(x, axis, keepdim)
    elif _in_legacy_dygraph():
        return _legacy_C_ops.reduce_amin(x, 'dim', axis, 'keep_dim', keepdim, 'reduce_all', reduce_all)
    helper = LayerHelper('amin', **locals())
    check_variable_and_dtype(
        x, 'x', ['float32', 'float64', 'int32', 'int64'], 'amin')

    out = helper.create_variable_for_type_inference(
            dtype=x.dtype)
    helper.append_op(
        type='reduce_amin',
        inputs={'X': x},
        outputs={'Out': out},
        attrs={
            'dim': axis,
            'keep_dim': keepdim,
            'reduce_all': reduce_all
        })
    return out

def log1p(x, name=None):
    r"""
    Calculates the natural log of the given input tensor, element-wise.

    .. math::
        Out = \ln(x+1)

    Args:
        x (Tensor): Input Tensor. Must be one of the following types: float32, float64.
        name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.
        
    Returns:
        Tensor, the natural log of the input Tensor computed element-wise.

    Examples:
        .. code-block:: python

            import paddle

            data = paddle.to_tensor([[0], [1]], dtype='float32')
            res = paddle.log1p(data)
            # [[0.], [0.6931472]]
    """

    if in_dygraph_mode():
        return _C_ops.log1p(x)
    if _in_legacy_dygraph():
        return _legacy_C_ops.log1p(x)

    check_variable_and_dtype(x, 'x', ['float32', 'float64'], "log1p")
    inputs = {'X': [x]}
    helper = LayerHelper('log1p', **locals())
    dtype = helper.input_dtype(input_param_name='x')
    out = helper.create_variable_for_type_inference(dtype)
    helper.append_op(type="log1p", inputs={"X": x}, outputs={"Out": out})
    return out

def log2(x, name=None):
    r"""
    Calculates the log to the base 2 of the given input tensor, element-wise.

    .. math::

        Out = \log_2x

    Args:
        x (Tensor): Input tensor must be one of the following types: float32, float64.
        name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.


    Returns:
        Tensor: The log to the base 2 of the input Tensor computed element-wise.

    Examples:

        .. code-block:: python
        
            import paddle

            # example 1: x is a float
            x_i = paddle.to_tensor([[1.0], [2.0]])
            res = paddle.log2(x_i) # [[0.], [1.0]]

            # example 2: x is float32
            x_i = paddle.full(shape=[1], fill_value=2, dtype='float32')
            paddle.to_tensor(x_i)
            res = paddle.log2(x_i)
            print(res) # [1.0]

            # example 3: x is float64
            x_i = paddle.full(shape=[1], fill_value=2, dtype='float64')
            paddle.to_tensor(x_i)
            res = paddle.log2(x_i)
            print(res) # [1.0]
    """
    if in_dygraph_mode():
        return _C_ops.log2(x)
    if _in_legacy_dygraph():
        return _legacy_C_ops.log2(x)

    check_variable_and_dtype(x, 'x', ['float16', 'float32', 'float64'], "log2")
    inputs = {'X': [x]}
    helper = LayerHelper('log2', **locals())
    dtype = helper.input_dtype(input_param_name='x')
    out = helper.create_variable_for_type_inference(dtype)
    helper.append_op(type="log2", inputs={"X": x}, outputs={"Out": out})
    return out


def log10(x, name=None):
    r"""
    Calculates the log to the base 10 of the given input tensor, element-wise.

    .. math::

        Out = \log_10_x

    Args:
        x (Tensor): Input tensor must be one of the following types: float32, float64.
        name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.


    Returns:
        Tensor: The log to the base 10 of the input Tensor computed element-wise.

    Examples:

        .. code-block:: python
        
            import paddle

            # example 1: x is a float
            x_i = paddle.to_tensor([[1.0], [10.0]])
            res = paddle.log10(x_i) # [[0.], [1.0]]

            # example 2: x is float32
            x_i = paddle.full(shape=[1], fill_value=10, dtype='float32')
            paddle.to_tensor(x_i)
            res = paddle.log10(x_i)
            print(res) # [1.0]

            # example 3: x is float64
            x_i = paddle.full(shape=[1], fill_value=10, dtype='float64')
            paddle.to_tensor(x_i)
            res = paddle.log10(x_i)
            print(res) # [1.0]
    """
    if in_dygraph_mode():
        return _C_ops.log10(x)
    if _in_legacy_dygraph():
        return _legacy_C_ops.log10(x)

    check_variable_and_dtype(x, 'x', ['float16', 'float32', 'float64'], "log10")
    inputs = {'X': [x]}
    helper = LayerHelper('log10', **locals())
    dtype = helper.input_dtype(input_param_name='x')
    out = helper.create_variable_for_type_inference(dtype)
    helper.append_op(type="log10", inputs={"X": x}, outputs={"Out": out})
    return out


def clip(x, min=None, max=None, name=None):
    """
    This operator clip all elements in input into the range [ min, max ] and return
    a resulting tensor as the following equation:

    .. math::

        Out = MIN(MAX(x, min), max)

    Args:
        x (Tensor): An N-D Tensor with data type float32, float64, int32 or int64.
        min (float|int|Tensor, optional): The lower bound with type ``float`` , ``int`` or a ``Tensor``
            with shape [1] and type ``int32``, ``float32``, ``float64``.
        max (float|int|Tensor, optional): The upper bound with type ``float``, ``int`` or a ``Tensor``
            with shape [1] and type ``int32``, ``float32``, ``float64``.
        name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        Tensor: A Tensor with the same data type and data shape as input.

    Examples:
        .. code-block:: python

            import paddle

            x1 = paddle.to_tensor([[1.2, 3.5], [4.5, 6.4]], 'float32')
            out1 = paddle.clip(x1, min=3.5, max=5.0)
            out2 = paddle.clip(x1, min=2.5)
            print(out1)
            # [[3.5, 3.5]
            # [4.5, 5.0]]
            print(out2)
            # [[2.5, 3.5]
            # [[4.5, 6.4]
    """

    x_dtype = str(x.dtype)
    if x_dtype == 'paddle.int32':
        min_ = np.iinfo(np.int32).min
        max_ = np.iinfo(np.int32).max - 2**7
    elif x_dtype == 'paddle.int64':
        min_ = np.iinfo(np.int64).min
        max_ = np.iinfo(np.int64).max - 2**39
    else:
        min_ = float(np.finfo(np.float32).min)
        max_ = float(np.finfo(np.float32).max)

    if in_dygraph_mode():
        if isinstance(min, Variable):
            min = min.numpy().item(0)
        if isinstance(max, Variable):
            max = max.numpy().item(0)
        min = min_ if min is None else min
        max = max_ if max is None else max
        return _C_ops.clip(x, min, max)

    if _in_legacy_dygraph():
        if isinstance(min, Variable):
            min = min.numpy().item(0)
        if isinstance(max, Variable):
            max = max.numpy().item(0)
        min = min_ if min is None else min
        max = max_ if max is None else max
        return _legacy_C_ops.clip(x, "min", min, "max", max)

    if min is not None:
        check_type(min, 'min', (float, int, Variable), 'clip')
        if isinstance(min, Variable):
            check_dtype(min.dtype, 'min', ['float32', 'float64', 'int32'],
                        'clip', '(When the type of min in clip is Variable.)')
    if max is not None:
        check_type(max, 'max', (float, int, Variable), 'clip')
        if isinstance(max, Variable):
            check_dtype(max.dtype, 'max', ['float32', 'float64', 'int32'],
                        'clip', '(When the type of max in clip is Variable.)')

    check_variable_and_dtype(x, 'x', ['float32', 'float64', 'int32', 'int64'], 'clip')

    inputs = {'X': x}
    attrs = {'min': min_, 'max': max_}

    if isinstance(min, Variable):
        min.stop_gradient = True
        inputs['Min'] = min
    elif min is not None:
        attrs['min'] = min

    if isinstance(max, Variable):
        max.stop_gradient = True
        inputs['Max'] = max
    elif max is not None:
        attrs['max'] = max

    helper = LayerHelper('clip', **locals())
    output = helper.create_variable_for_type_inference(
        dtype=helper.input_dtype('x'))
    helper.append_op(
        type='clip', inputs=inputs, outputs={'Out': [output]}, attrs=attrs)

    return output


@inplace_apis_in_dygraph_only
def clip_(x, min=None, max=None, name=None):
    """
    Inplace version of ``clip`` API, the output Tensor will be inplaced with input ``x``.
    Please refer to :ref:`api_tensor_clip`.
    """
    fmin = float(np.finfo(np.float32).min)
    fmax = float(np.finfo(np.float32).max)
    if isinstance(min, Variable):
        min = min.numpy().item(0)
    if isinstance(max, Variable):
        max = max.numpy().item(0)
    min = fmin if min is None else min
    max = fmax if max is None else max

    if in_dygraph_mode():
        return _C_ops.clip_(x, min, max)

    if _in_legacy_dygraph():
        return _legacy_C_ops.clip_(x, "min", min, "max", max)



def trace(x, offset=0, axis1=0, axis2=1, name=None):
    """

    Computes the sum along diagonals of the input tensor x.

    If ``x`` is 2D, returns the sum of diagonal.

    If ``x`` has larger dimensions, then returns an tensor of diagonals sum, diagonals be taken from
    the 2D planes specified by axis1 and axis2. By default, the 2D planes formed by the first and second axes
    of the input tensor x.

    The argument ``offset`` determines where diagonals are taken from input tensor x:

    - If offset = 0, it is the main diagonal.
    - If offset > 0, it is above the main diagonal.
    - If offset < 0, it is below the main diagonal.
    - Note that if offset is out of input's shape indicated by axis1 and axis2, 0 will be returned.

    Args:
        x (Tensor): The input tensor x. Must be at least 2-dimensional. The input data type should be float32, float64, int32, int64.
        offset (int, optional): Which diagonals in input tensor x will be taken. Default: 0 (main diagonals).
        axis1 (int, optional): The first axis with respect to take diagonal. Default: 0.
        axis2 (int, optional): The second axis with respect to take diagonal. Default: 1.
        name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        Tensor: the output data type is the same as input data type.

    Examples:
        .. code-block:: python

            import paddle

            case1 = paddle.randn([2, 3])
            case2 = paddle.randn([3, 10, 10])
            case3 = paddle.randn([3, 10, 5, 10])
            data1 = paddle.trace(case1) # data1.shape = [1]
            data2 = paddle.trace(case2, offset=1, axis1=1, axis2=2) # data2.shape = [3]
            data3 = paddle.trace(case3, offset=-3, axis1=1, axis2=-1) # data2.shape = [3, 5]
    """
    def __check_input(x, offset, axis1, axis2):
        check_dtype(x.dtype, 'Input',
                    ['int32', 'int64', 'float16', 'float32', 'float64'],
                    'trace')

        input_shape = list(x.shape)
        assert len(input_shape) >= 2,                     \
                "The x must be at least 2-dimensional, "   \
                "But received Input x's dimensional: %s.\n" %  \
                len(input_shape)

        axis1_ = axis1 if axis1 >= 0 else len(input_shape) + axis1
        axis2_ = axis2 if axis2 >= 0 else len(input_shape) + axis2

        assert ((0 <= axis1_) and (axis1_ < len(input_shape))),     \
            "The argument axis1 is out of range (expected to be in range of [%d, %d], but got %d).\n"  \
            % (-(len(input_shape)), len(input_shape) - 1, axis1)

        assert ((0 <= axis2_) and (axis2_ < len(input_shape))),   \
            "The argument axis2 is out of range (expected to be in range of [%d, %d], but got %d).\n"   \
            % (-(len(input_shape)), len(input_shape) - 1, axis2)


        assert  axis1_ != axis2_,   \
               "axis1 and axis2 cannot be the same axis." \
                "But received axis1 = %d, axis2 = %d\n"%(axis1, axis2)

    if in_dygraph_mode():
        return _C_ops.trace( x, offset, axis1, axis2 )

    if _in_legacy_dygraph():
        return _legacy_C_ops.trace(x, 'offset', offset, 'axis1', axis1, 'axis2', axis2)

    __check_input(x, offset, axis1, axis2)

    helper = LayerHelper('trace', **locals())
    out = helper.create_variable_for_type_inference(dtype=x.dtype)

    helper.append_op(
        type='trace',
        inputs={'Input': [x]},
        attrs={'offset': offset,
               'axis1': axis1,
               'axis2': axis2},
        outputs={'Out': [out]})
    return out

def diagonal(x, offset=0, axis1=0, axis2=1, name=None):
    """
    This OP computes the diagonals of the input tensor x.

    If ``x`` is 2D, returns the diagonal.
    If ``x`` has larger dimensions, diagonals be taken from the 2D planes specified by axis1 and axis2. 
    By default, the 2D planes formed by the first and second axis of the input tensor x.

    The argument ``offset`` determines where diagonals are taken from input tensor x:

    - If offset = 0, it is the main diagonal.
    - If offset > 0, it is above the main diagonal.
    - If offset < 0, it is below the main diagonal.
    
    Args:
        x (Tensor): The input tensor x. Must be at least 2-dimensional. The input data type should be bool, int32, int64, float16, float32, float64.
        offset (int, optional): Which diagonals in input tensor x will be taken. Default: 0 (main diagonals).
        axis1 (int, optional): The first axis with respect to take diagonal. Default: 0.
        axis2 (int, optional): The second axis with respect to take diagonal. Default: 1.
        name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        Tensor: a partial view of input tensor in specify two dimensions, the output data type is the same as input data type.

    Examples:
        .. code-block:: python

            import paddle

            x = paddle.rand([2,2,3],'float32')
            print(x)
            # Tensor(shape=[2, 2, 3], dtype=float32, place=CUDAPlace(0), stop_gradient=True,
            #        [[[0.45661032, 0.03751532, 0.90191704],
            #          [0.43760979, 0.86177313, 0.65221709]],

            #         [[0.17020577, 0.00259554, 0.28954273],
            #          [0.51795638, 0.27325270, 0.18117726]]])

            out1 = paddle.diagonal(x)
            print(out1)
            #Tensor(shape=[3, 2], dtype=float32, place=CUDAPlace(0), stop_gradient=True,
            #       [[0.45661032, 0.51795638],
            #        [0.03751532, 0.27325270],
            #        [0.90191704, 0.18117726]])

            out2 = paddle.diagonal(x, offset=0, axis1=2, axis2=1)
            print(out2)
            #Tensor(shape=[2, 2], dtype=float32, place=CUDAPlace(0), stop_gradient=True,
            #       [[0.45661032, 0.86177313],
            #        [0.17020577, 0.27325270]])

            out3 = paddle.diagonal(x, offset=1, axis1=0, axis2=1)
            print(out3)
            #Tensor(shape=[3, 1], dtype=float32, place=CUDAPlace(0), stop_gradient=True,
            #       [[0.43760979],
            #        [0.86177313],
            #        [0.65221709]])

            out4 = paddle.diagonal(x, offset=0, axis1=1, axis2=2)
            print(out4)
            #Tensor(shape=[2, 2], dtype=float32, place=CUDAPlace(0), stop_gradient=True,
            #       [[0.45661032, 0.86177313],
            #        [0.17020577, 0.27325270]])
            
    """
    if in_dygraph_mode():
        return _C_ops.diagonal(x, offset, axis1, axis2)
    else:
        if _in_legacy_dygraph():
            return _legacy_C_ops.diagonal(x, 'offset', offset, 'axis1', axis1, 'axis2', axis2)

    def __check_input(x, offset, axis1, axis2):
        check_dtype(x.dtype, 'Input',
                    ['bool', 'int32', 'int64', 'float16', 'float32', 'float64'],
                    'diagonal')

        input_shape = list(x.shape)
        assert len(input_shape) >= 2,                     \
                "The x must be at least 2-dimensional, "   \
                "But received Input x's dimensional: %s.\n" %  \
                len(input_shape)

        axis1_ = axis1 if axis1 >= 0 else len(input_shape) + axis1
        axis2_ = axis2 if axis2 >= 0 else len(input_shape) + axis2

        assert axis1_ < len(input_shape),     \
            "The argument axis1 is out of range (expected to be in range of [%d, %d], but got %d).\n"  \
            % (-(len(input_shape)), len(input_shape) - 1, axis1)

        assert axis2_ < len(input_shape),   \
            "The argument axis2 is out of range (expected to be in range of [%d, %d], but got %d).\n"   \
            % (-(len(input_shape)), len(input_shape) - 1, axis2)

        assert  axis1_ != axis2_,   \
               "axis1 and axis2 cannot be the same axis." \
                "But received axis1 = %d, axis2 = %d\n"%(axis1, axis2)

    __check_input(x, offset, axis1, axis2)
    helper = LayerHelper('diagonal', **locals())
    out = helper.create_variable_for_type_inference(dtype=x.dtype)

    helper.append_op(
        type='diagonal',
        inputs={'Input': [x]},
        attrs={'offset': offset,
               'axis1': axis1,
               'axis2': axis2},
               outputs={'Out': [out]})
    return out


@templatedoc(op_type="kron")
def kron(x, y, name=None):
    """

    ${comment}

    Args:
        x (Tensor): the fist operand of kron op, data type: float16, float32, float64, int32 or int64.
        y (Tensor): the second operand of kron op, data type: float16, float32, float64, int32 or int64. Its data type should be the same with x.
        name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        Tensor: The output of kron, data type: float16, float32, float64, int32 or int64. Its data is the same with x.

    Examples:
        .. code-block:: python

            import paddle
            x = paddle.to_tensor([[1, 2], [3, 4]], dtype='int64')
            y = paddle.to_tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype='int64')
            out = paddle.kron(x, y)
            print(out)
            #        [[1, 2, 3, 2, 4, 6],
            #         [ 4,  5,  6,  8, 10, 12],
            #         [ 7,  8,  9, 14, 16, 18],
            #         [ 3,  6,  9,  4,  8, 12],
            #         [12, 15, 18, 16, 20, 24],
            #         [21, 24, 27, 28, 32, 36]])
    """
    if _in_legacy_dygraph():
        return _legacy_C_ops.kron(x, y)
    if in_dygraph_mode():
        return _C_ops.kron(x, y)
    helper = LayerHelper('kron', **locals())
    check_variable_and_dtype(x, 'x', ['float16', 'float32', 'float64', 'int32', 'int64'], 'kron')
    check_variable_and_dtype(y, 'y', ['float16', 'float32', 'float64', 'int32', 'int64'], 'kron')

    out = helper.create_variable_for_type_inference(dtype=x.dtype)
    helper.append_op(type="kron", inputs={"X": x, "Y": y}, outputs={"Out": out})
    return out


def cumsum(x, axis=None, dtype=None, name=None):
    """
    The cumulative sum of the elements along a given axis. 
    
    Note:
        The first element of the result is the same as the first element of the input. 

    Args:
        x (Tensor): The input tensor needed to be cumsumed.
        axis (int, optional): The dimension to accumulate along. -1 means the last dimension. The default (None) is to compute the cumsum over the flattened array.
        dtype (str, optional): The data type of the output tensor, can be float32, float64, int32, int64. If specified, the input tensor is casted to dtype before the operation is performed. This is useful for preventing data type overflows. The default value is None. 
        name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        Tensor, the result of cumsum operator. 

    Examples:
        .. code-block:: python
            
            import paddle
            
            data = paddle.arange(12)
            data = paddle.reshape(data, (3, 4))

            y = paddle.cumsum(data)
            # [ 0  1  3  6 10 15 21 28 36 45 55 66]

            y = paddle.cumsum(data, axis=0)
            # [[ 0  1  2  3]
            #  [ 4  6  8 10]
            #  [12 15 18 21]]
            
            y = paddle.cumsum(data, axis=-1)
            # [[ 0  1  3  6]
            #  [ 4  9 15 22]
            #  [ 8 17 27 38]]

            y = paddle.cumsum(data, dtype='float64')
            print(y.dtype)
            # paddle.float64
    """
    if axis is None:
        flatten = True
    else:
        flatten = False
    if dtype is not None and x.dtype != convert_np_dtype_to_dtype_(dtype):
        x = cast(x, dtype)

    if in_dygraph_mode():
        if axis is None: axis = -1
        return _C_ops.cumsum(x, axis, flatten, False, False)
    if _in_legacy_dygraph():
        if axis is None:
            return _legacy_C_ops.cumsum(x, 'flatten', flatten)
        else:
            return _legacy_C_ops.cumsum(x, 'axis', axis, 'flatten', flatten)

    check_type(x, 'x', (Variable), 'cumsum')
    locals_var = locals().copy()
    kwargs = dict()
    for name, val in locals_var.items():
        if val is not None:
            kwargs[name] = val
    _cum_sum_ = generate_layer_fn('cumsum')
    return _cum_sum_(**kwargs)


def logcumsumexp(x, axis=None, dtype=None, name=None):
    r"""
    The logarithm of the cumulative summation of the exponentiation of the elements along a given axis. 

    For summation index j given by `axis` and other indices i, the result is

    .. math::

        logcumsumexp(x)_{ij} = log \sum_{i=0}^{j}exp(x_{ij})
    
    Note:
        The first element of the result is the same as the first element of the input.

    Args:
        x (Tensor): The input tensor.
        axis (int, optional): The dimension to do the operation along. -1 means the last dimension. The default (None) is to compute the cumsum over the flattened array.
        dtype (str, optional): The data type of the output tensor, can be float32, float64. If specified, the input tensor is casted to dtype before the operation is performed. This is useful for preventing data type overflows. The default value is None. 
        name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        Tensor, the result of logcumsumexp operator. 

    Examples:
        .. code-block:: python
            
            import paddle
            
            data = paddle.arange(12, dtype='float64')
            data = paddle.reshape(data, (3, 4))

            y = paddle.logcumsumexp(data)
            # [ 0.         1.3132617  2.4076061  3.4401898  4.4519143  5.4561934
            #   6.4577627  7.4583397  8.458551   9.45863   10.458658  11.458669 ]

            y = paddle.logcumsumexp(data, axis=0)
            # [[ 0.        1.        2.        3.      ]
            #  [ 4.01815   5.01815   6.01815   7.01815 ]
            #  [ 8.018479  9.018479 10.018479 11.018479]]
            
            y = paddle.logcumsumexp(data, axis=-1)
            # [[ 0.         1.3132617  2.4076061  3.4401898]
            #  [ 4.         5.3132615  6.407606   7.44019  ]
            #  [ 8.         9.313262  10.407606  11.440189 ]]

            y = paddle.logcumsumexp(data, dtype='float64')
            print(y.dtype)
            # paddle.float64
    """
    if axis is None:
        flatten = True
    else:
        flatten = False
    if dtype is not None and x.dtype != convert_np_dtype_to_dtype_(dtype):
        x = cast(x, dtype)

    if in_dygraph_mode():
        if axis is None: axis = -1
        return _C_ops.logcumsumexp(x, axis, flatten, False, False)
    if _in_legacy_dygraph():
        if axis is None:
            return _legacy_C_ops.logcumsumexp(x, 'flatten', flatten)
        else:
            return _legacy_C_ops.logcumsumexp(x, 'axis', axis, 'flatten', flatten)

    check_variable_and_dtype(x, 'x', ['float32', 'float64'], "logcumsumexp")

    helper = LayerHelper('logcumsumexp', **locals())
    out = helper.create_variable_for_type_inference(x.dtype)
    helper.append_op(type='logcumsumexp', inputs={'X': x}, outputs={'Out': out}, attrs={'axis': axis, 'flatten': flatten})
    return out


def cumprod(x, dim=None, dtype=None, name=None):
    """
    Compute the cumulative product of the input tensor x along a given dimension dim.

    Note:
        The first element of the result is the same as the first element of the input.

    Args:
        x (Tensor): the input tensor need to be cumproded.
        dim (int): the dimension along which the input tensor will be accumulated. It need to be in the range of [-x.rank, x.rank), where x.rank means the dimensions of the input tensor x and -1 means the last dimension.
        dtype (str, optional): The data type of the output tensor, can be float32, float64, int32, int64, complex64, complex128. If specified, the input tensor is casted to dtype before the operation is performed. This is useful for preventing data type overflows. The default value is None.
        name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        Tensor, the result of cumprod operator.

    Examples:
        .. code-block:: python

            import paddle

            data = paddle.arange(12)
            data = paddle.reshape(data, (3, 4))
            # [[ 0  1  2  3 ]
            #  [ 4  5  6  7 ]
            #  [ 8  9  10 11]]

            y = paddle.cumprod(data, dim=0)
            # [[ 0  1   2   3]
            #  [ 0  5  12  21]
            #  [ 0 45 120 231]]

            y = paddle.cumprod(data, dim=-1)
            # [[ 0   0   0    0]
            #  [ 4  20 120  840]
            #  [ 8  72 720 7920]]

            y = paddle.cumprod(data, dim=1, dtype='float64')
            # [[ 0.   0.   0.    0.]
            #  [ 4.  20. 120.  840.]
            #  [ 8.  72. 720. 7920.]]

            print(y.dtype)
            # paddle.float64

    """

    if dtype is not None and x.dtype != convert_np_dtype_to_dtype_(dtype):
        x = cast(x, dtype)

    if in_dygraph_mode():
        return _C_ops.cumprod(x, dim)
    if _in_legacy_dygraph():
        return _legacy_C_ops.cumprod(x, 'dim', dim)

    check_variable_and_dtype(x, "x", ['complex64', 'complex128', 'float32', 'float64', 'int32', 'int64'], 'cumprod')
    check_type(dim, 'dim', int, 'cumprod')

    helper = LayerHelper('cumprod', **locals())
    out = helper.create_variable_for_type_inference(x.dtype)
    helper.append_op(type='cumprod', inputs={'X': x}, outputs={'Out': out}, attrs={'dim': dim})
    return out

def isfinite(x, name=None):
    """

    Return whether every element of input tensor is finite number or not.

    Args:
        x (Tensor): The input tensor, it's data type should be float16, float32, float64, int32, int64.
        name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        `Tensor`, the bool result which shows every element of `x` whether it is finite number or not.

    Examples:
        .. code-block:: python

            import paddle

            x = paddle.to_tensor([float('-inf'), -2, 3.6, float('inf'), 0, float('-nan'), float('nan')])
            out = paddle.isfinite(x)
            print(out)  # [False  True  True False  True False False]
    """
    if in_dygraph_mode():
        return _C_ops.isfinite( x )
    if _in_legacy_dygraph():
        return _legacy_C_ops.isfinite_v2(x)
    helper = LayerHelper("isfinite_v2", **locals())
    check_variable_and_dtype(x, 'x', ['float16', 'float32', 'float64', 'int32', 'int64'], 'isfinite')
    out = helper.create_variable_for_type_inference('bool')
    helper.append_op(type="isfinite_v2", inputs={"X": x}, outputs={"Out": out})
    return out

def isinf(x, name=None):
    """

    Return whether every element of input tensor is `+/-INF` or not.

    Args:
        x (Tensor): The input tensor, it's data type should be float16, float32, float64, int32, int64.
        name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        `Tensor`, the bool result which shows every element of `x` whether it is `+/-INF` or not.

    Examples:
        .. code-block:: python

            import paddle

            x = paddle.to_tensor([float('-inf'), -2, 3.6, float('inf'), 0, float('-nan'), float('nan')])
            out = paddle.isinf(x)
            print(out)  # [ True False False  True False False False]
    """
    if in_dygraph_mode():
        return _C_ops.isinf( x )
    if _in_legacy_dygraph():
        return _legacy_C_ops.isinf_v2(x)
    helper = LayerHelper("isinf_v2", **locals())
    check_variable_and_dtype(x, 'x', ['float16', 'float32', 'float64', 'int32', 'int64'], 'isinf')
    out = helper.create_variable_for_type_inference(dtype='bool')
    helper.append_op(type="isinf_v2", inputs={"X": x}, outputs={"Out": out})
    return out

def isnan(x, name=None):
    """

    Return whether every element of input tensor is `NaN` or not.

    Args:
        x (Tensor): The input tensor, it's data type should be float16, float32, float64, int32, int64.
        name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        `Tensor`, the bool result which shows every element of `x` whether it is `NaN` or not.

    Examples:
        .. code-block:: python

            import paddle
            
            x = paddle.to_tensor([float('-inf'), -2, 3.6, float('inf'), 0, float('-nan'), float('nan')])
            out = paddle.isnan(x)
            print(out)  # [False False False False False  True  True]
    """
    if in_dygraph_mode():
        return _C_ops.isnan( x )

    if _in_legacy_dygraph():
        return _legacy_C_ops.isnan_v2(x)
    helper = LayerHelper("isnan_v2", **locals())
    check_variable_and_dtype(x, 'x', ['float16', 'float32', 'float64', 'int32', 'int64'], 'isnan')
    out = helper.create_variable_for_type_inference(dtype='bool')
    helper.append_op(type="isnan_v2", inputs={"X": x}, outputs={"Out": out})
    return out


def prod(x, axis=None, keepdim=False, dtype=None, name=None):
    """
    Compute the product of tensor elements over the given axis.

    Args:
        x (Tensor): The input tensor, its data type should be float32, float64, int32, int64.
        axis (int|list|tuple, optional): The axis along which the product is computed. If :attr:`None`, 
            multiply all elements of `x` and return a Tensor with a single element, 
            otherwise must be in the range :math:`[-x.ndim, x.ndim)`. If :math:`axis[i]<0`, 
            the axis to reduce is :math:`x.ndim + axis[i]`. Default is None.
        keepdim (bool, optional): Whether to reserve the reduced dimension in the output Tensor. The result 
            tensor will have one fewer dimension than the input unless `keepdim` is true. Default is False.
        dtype (str|np.dtype, optional): The desired date type of returned tensor, can be float32, float64, 
            int32, int64. If specified, the input tensor is casted to dtype before operator performed. 
            This is very useful for avoiding data type overflows. The default value is None, the dtype 
            of output is the same as input Tensor `x`.
        name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        Tensor, result of product on the specified dim of input tensor.
    
    Examples:
        .. code-block:: python

            import paddle

            # the axis is a int element
            x = paddle.to_tensor([[0.2, 0.3, 0.5, 0.9],
                                  [0.1, 0.2, 0.6, 0.7]])
            out1 = paddle.prod(x)
            # [0.0002268]

            out2 = paddle.prod(x, -1)
            # [0.027  0.0084]

            out3 = paddle.prod(x, 0)
            # [0.02 0.06 0.3  0.63]

            out4 = paddle.prod(x, 0, keepdim=True)
            # [[0.02 0.06 0.3  0.63]]

            out5 = paddle.prod(x, 0, dtype='int64')
            # [0 0 0 0]

            # the axis is list
            y = paddle.to_tensor([[[1.0, 2.0], [3.0, 4.0]],
                                  [[5.0, 6.0], [7.0, 8.0]]])
            out6 = paddle.prod(y, [0, 1])
            # [105. 384.]

            out7 = paddle.prod(y, (1, 2))
            # [  24. 1680.]

    """
    if dtype is not None:
        check_dtype(dtype, 'dtype', ['float32', 'float64', 'int32', 'int64'], 'prod')
        if x.dtype != convert_np_dtype_to_dtype_(dtype):
            x = cast(x, dtype)

    dim = axis
    if isinstance(dim, Variable):
        reduce_all = True if axis.shape[0] == len(x.shape) else False
    else:
        if dim is not None and not isinstance(dim, list):
            if isinstance(dim, tuple):
                dim = list(dim)
            elif isinstance(dim, int):
                dim = [dim]
            else:
                raise TypeError(
                    "The type of axis must be int, list or tuple, but received {}".
                    format(type(dim)))

        reduce_all = True if dim is None or len(dim) == 0 or len(dim) == len(x.shape) else False
        if dim is None or len(dim) == 0:
            dim = [0]

    if in_dygraph_mode():
        return _C_ops.reduce_prod(x, dim, keepdim, reduce_all)
    if _in_legacy_dygraph():
        return _legacy_C_ops.reduce_prod(
            x, 'dim', dim, 'keep_dim', keepdim, 'reduce_all', reduce_all)

    helper = LayerHelper('reduce_prod', **locals())
    check_variable_and_dtype(
        x, 'x/input', ['float32', 'float64', 'int32', 'int64'], 'reduce_prod')
    out = helper.create_variable_for_type_inference(dtype=helper.input_dtype())
    if not isinstance(dim, Variable) and utils._contain_var(dim):
        dim = utils._convert_to_tensor_list(dim)
    helper.append_op(
        type='reduce_prod',
        inputs={'X': x},
        outputs={'Out': out},
        attrs={
            'dim': dim,
            'keep_dim': keepdim,
            'reduce_all': reduce_all
        })
    return out


def sign(x, name=None):
    """
    Returns sign of every element in `x`: 1 for positive, -1 for negative and 0 for zero.

    Args:
        x (Tensor): The input tensor. The data type can be float16, float32 or float64.
        name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        Tensor: The output sign tensor with identical shape and data type to the input :attr:`x`.

    Examples:
        .. code-block:: python

          import paddle

          x = paddle.to_tensor([3.0, 0.0, -2.0, 1.7], dtype='float32')
          out = paddle.sign(x=x)
          print(out)  # [1.0, 0.0, -1.0, 1.0]
    """
    if in_dygraph_mode():
        return _C_ops.sign(x)

    if _in_legacy_dygraph():
        return _legacy_C_ops.sign(x)

    check_variable_and_dtype(x, 'x', ['float16', 'float32', 'float64'], 'sign')
    helper = LayerHelper("sign", **locals())
    out = helper.create_variable_for_type_inference(dtype=x.dtype)

    helper.append_op(type='sign', inputs={'X': [x]}, outputs={'Out': [out]})

    return out


def tanh(x, name=None):
    r"""
    Tanh Activation Operator.

    .. math::
        out = \frac{e^{x} - e^{-x}}{e^{x} + e^{-x}}

    Args:
        x (Tensor): Input of Tanh operator, an N-D Tensor, with data type float32, float64 or float16.
        name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        Output of Tanh operator, a Tensor with same data type and shape as input.

    Examples:

        .. code-block:: python

            import paddle

            x = paddle.to_tensor([-0.4, -0.2, 0.1, 0.3])
            out = paddle.tanh(x)
            print(out)
            # [-0.37994896 -0.19737532  0.09966799  0.29131261]
    """
    if in_dygraph_mode():
        return _C_ops.tanh( x )

    if _in_legacy_dygraph():
        return _legacy_C_ops.tanh(x)

    check_variable_and_dtype(x, 'x', ['float16', 'float32', 'float64'], 'tanh')
    check_type(x, 'x', (Variable), 'tanh')
    helper = LayerHelper('tanh', **locals())
    out = helper.create_variable_for_type_inference(x.dtype)
    helper.append_op(type='tanh', inputs={'X': x}, outputs={'Out': out})
    return out

@inplace_apis_in_dygraph_only
def tanh_(x, name=None):
    r"""
    Inplace version of ``tanh`` API, the output Tensor will be inplaced with input ``x``.
    Please refer to :ref:`api_tensor_tanh`.
    """
    if in_dygraph_mode():
        return _C_ops.tanh_( x )
    return _legacy_C_ops.tanh_(x)


def increment(x, value=1.0, name=None):
    """
    The API is usually used for control flow to increment the data of :attr:`x` by an amount :attr:`value`.
    Notice that the number of elements in :attr:`x` must be equal to 1.

    Args:
        x (Tensor): A tensor that must always contain only one element, its data type supports float32, float64, int32 and int64.
        value (float, optional): The amount to increment the data of :attr:`x`. Default: 1.0.
        name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        Tensor, the elementwise-incremented tensor with the same shape and data type as :attr:`x`.

    Examples:
        .. code-block:: python

            import paddle

            data = paddle.zeros(shape=[1], dtype='float32')
            counter = paddle.increment(data)
            # [1.]

    """
    if in_dygraph_mode():
        return _C_ops.increment_(x, value)

    if _in_legacy_dygraph():
        return _legacy_C_ops.increment(x, 'step', value)

    check_variable_and_dtype(x, 'x', ['float32', 'float64', 'int32', 'int64'],
                             'increment')
    helper = LayerHelper("increment", **locals())
    helper.append_op(
        type='increment',
        inputs={'X': [x]},
        outputs={'Out': [x]},
        attrs={'step': float(value)})
    return x


def all(x, axis=None, keepdim=False, name=None):
    """
    Computes the ``logical and`` of tensor elements over the given dimension.

    Args:
        x (Tensor): An N-D Tensor, the input data type should be `bool`.
        axis (int|list|tuple, optional): The dimensions along which the ``logical and`` is compute. If
            :attr:`None`, and all elements of :attr:`x` and return a
            Tensor with a single element, otherwise must be in the
            range :math:`[-rank(x), rank(x))`. If :math:`axis[i] < 0`,
            the dimension to reduce is :math:`rank + axis[i]`.
        keepdim (bool, optional): Whether to reserve the reduced dimension in the
            output Tensor. The result Tensor will have one fewer dimension
            than the :attr:`x` unless :attr:`keepdim` is true, default
            value is False.
        name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        Tensor: Results the ``logical and`` on the specified axis of input Tensor `x`,  it's data type is bool.

    Examples:
        .. code-block:: python

            import paddle

            # x is a bool Tensor with following elements:
            #    [[True, False]
            #     [True, True]]
            x = paddle.to_tensor([[1, 0], [1, 1]], dtype='int32')
            print(x)
            x = paddle.cast(x, 'bool')

            # out1 should be [False]
            out1 = paddle.all(x)  # [False]
            print(out1)

            # out2 should be [True, False]
            out2 = paddle.all(x, axis=0)  # [True, False]
            print(out2)

            # keepdim=False, out3 should be [False, True], out.shape should be (2,)
            out3 = paddle.all(x, axis=-1)  # [False, True]
            print(out3)

            # keepdim=True, out4 should be [[False], [True]], out.shape should be (2,1)
            out4 = paddle.all(x, axis=1, keepdim=True) # [[False], [True]]
            print(out4)
            
    """
    if axis is not None and not isinstance(axis, (list, tuple)):
        axis = [axis]

    if not axis:
        reduce_all_flag = True
    else:
        if len(axis) == len(x.shape):
            reduce_all_flag = True
        else:
            reduce_all_flag = False

    if in_dygraph_mode():
        if reduce_all_flag:
            axis = range(len(x.shape))
        return _C_ops.all(x, axis, keepdim)

    if _in_legacy_dygraph():
        axis = axis if axis != None and axis != [] else [0]
        return _legacy_C_ops.reduce_all(x, 'dim', axis, 'keep_dim', keepdim,
                                       'reduce_all', reduce_all_flag)

    attrs = {
        'dim': axis if axis != None and axis != [] and axis != () else [0],
        'keep_dim': keepdim,
        'reduce_all': reduce_all_flag
    }
    check_variable_and_dtype(x, 'x', ['bool'], 'all')


    check_type(axis, 'axis', (int, list, tuple, type(None)), 'all')

    helper = LayerHelper('all', **locals())
    out = helper.create_variable_for_type_inference(dtype=x.dtype)
    helper.append_op(
        type='reduce_all',
        inputs={'X': x},
        outputs={'Out': out},
        attrs=attrs)
    return out


def any(x, axis=None, keepdim=False, name=None):
    """
    Computes the ``logical or`` of tensor elements over the given dimension, and return the result.

    Args:
        x (Tensor): An N-D Tensor, the input data type should be `bool`.
        axis (int|list|tuple, optional): The dimensions along which the ``logical or`` is compute. If
            :attr:`None`, and all elements of :attr:`x` and return a
            Tensor with a single element, otherwise must be in the
            range :math:`[-rank(x), rank(x))`. If :math:`axis[i] < 0`,
            the dimension to reduce is :math:`rank + axis[i]`.
        keepdim (bool, optional): Whether to reserve the reduced dimension in the
            output Tensor. The result Tensor will have one fewer dimension
            than the :attr:`x` unless :attr:`keepdim` is true, default
            value is False.
        name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        Tensor: Results the ``logical or`` on the specified axis of input Tensor `x`,  it's data type is bool.

    Examples:
        .. code-block:: python

            import paddle

            x = paddle.to_tensor([[1, 0], [1, 1]], dtype='int32')
            x = paddle.assign(x)
            print(x)
            x = paddle.cast(x, 'bool')
            # x is a bool Tensor with following elements:
            #    [[True, False]
            #     [True, True]]

            # out1 should be [True]
            out1 = paddle.any(x)  # [True]
            print(out1)

            # out2 should be [True, True]
            out2 = paddle.any(x, axis=0)  # [True, True]
            print(out2)

            # keepdim=False, out3 should be [True, True], out.shape should be (2,)
            out3 = paddle.any(x, axis=-1)  # [True, True]
            print(out3)

            # keepdim=True, result should be [[True], [True]], out.shape should be (2,1)
            out4 = paddle.any(x, axis=1, keepdim=True)  # [[True], [True]]
            print(out4) 
            
    """
    if axis is not None and not isinstance(axis, (list, tuple)):
        axis = [axis]

    if not axis:
        reduce_all_flag = True
    else:
        if len(axis) == len(x.shape):
            reduce_all_flag = True
        else:
            reduce_all_flag = False

    if in_dygraph_mode():
        if reduce_all_flag:
            axis = range(len(x.shape))
        return _C_ops.any(x, axis, keepdim)

    if _in_legacy_dygraph():
        axis = axis if axis != None and axis != [] else [0]
        return _legacy_C_ops.reduce_any(x, 'dim', axis, 'keep_dim', keepdim,
                                       'reduce_all', reduce_all_flag)

    attrs = {
        'dim': axis if axis != None and axis != [] and axis != () else [0],
        'keep_dim': keepdim,
        'reduce_all': reduce_all_flag
    }

    check_variable_and_dtype(x, 'x', ['bool'], 'any')


    check_type(axis, 'axis', (int, list, tuple, type(None)), 'any')

    helper = LayerHelper('any', **locals())
    out = helper.create_variable_for_type_inference(dtype=x.dtype)
    helper.append_op(
        type='reduce_any',
        inputs={'X': x},
        outputs={'Out': out},
        attrs=attrs)
    return out

def broadcast_shape(x_shape, y_shape):
    """
    The function returns the shape of doing operation with broadcasting on tensors of x_shape and y_shape, please refer to :ref:`user_guide_broadcasting` for more details.

    Args:
        x_shape (list[int]|tuple[int]): A shape of tensor.
        y_shape (list[int]|tuple[int]): A shape of tensor.
        

    Returns:
        list[int], the result shape.

    Examples:
        .. code-block:: python

            import paddle

            shape = paddle.broadcast_shape([2, 1, 3], [1, 3, 1])
            # [2, 3, 3]
            
            # shape = paddle.broadcast_shape([2, 1, 3], [3, 3, 1])
            # ValueError (terminated with error message).

    """

    return core.broadcast_shape(x_shape, y_shape)

def conj(x, name=None):
    r"""
    This function computes the conjugate of the Tensor elementwisely.

    Args:
        x (Tensor): The input Tensor which hold the complex numbers. 
            Optional data types are: complex64, complex128, float32, float64, int32 or int64.
        name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        out (Tensor): The conjugate of input. The shape and data type is the same with input. If the elements of tensor is real type such as float32, float64, int32 or int64, the out is the same with input.

    Examples:
        .. code-block:: python

          import paddle
          
          data=paddle.to_tensor([[1+1j, 2+2j, 3+3j], [4+4j, 5+5j, 6+6j]])
          #Tensor(shape=[2, 3], dtype=complex64, place=CUDAPlace(0), stop_gradient=True,
          #       [[(1+1j), (2+2j), (3+3j)],
          #        [(4+4j), (5+5j), (6+6j)]])

          conj_data=paddle.conj(data)
          #Tensor(shape=[2, 3], dtype=complex64, place=CUDAPlace(0), stop_gradient=True,
          #       [[(1-1j), (2-2j), (3-3j)],
          #        [(4-4j), (5-5j), (6-6j)]])

    """
    if in_dygraph_mode():
        return _C_ops.conj(x)

    if paddle.in_dynamic_mode():
        return _legacy_C_ops.conj(x)

    check_variable_and_dtype(x, "x", ['complex64', 'complex128', 'float32', 'float64', 'int32', 'int64'], 'conj')

    helper = LayerHelper('conj', **locals())
    out = helper.create_variable_for_type_inference(
            dtype=helper.input_dtype())

    helper.append_op(type='conj', inputs={'X': x}, outputs={'Out': [out]})
    return out

def digamma(x, name=None):
    r"""
    Calculates the digamma of the given input tensor, element-wise.

    .. math::
        Out = \Psi(x) = \frac{ \Gamma^{'}(x) }{ \Gamma(x) }

    Args:
        x (Tensor): Input Tensor. Must be one of the following types: float32, float64.
        name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.
    Returns:
        Tensor, the digamma of the input Tensor, the shape and data type is the same with input.

    Examples:
        .. code-block:: python

            import paddle

            data = paddle.to_tensor([[1, 1.5], [0, -2.2]], dtype='float32')
            res = paddle.digamma(data)
            print(res)
            # Tensor(shape=[2, 2], dtype=float32, place=CUDAPlace(0), stop_gradient=True,
            #       [[-0.57721591,  0.03648996],
            #        [ nan       ,  5.32286835]])
    """

    if in_dygraph_mode():
        return _C_ops.digamma(x)
    else:
        if _in_legacy_dygraph():
            return _legacy_C_ops.digamma(x)

    check_variable_and_dtype(x, 'x', ['float32', 'float64'], 'digamma')
    helper = LayerHelper('digamma', **locals())
    out = helper.create_variable_for_type_inference(x.dtype)
    helper.append_op(type='digamma', inputs={'X': x}, outputs={'Out': out})
    return out

def lgamma(x, name=None):
    r"""
    Calculates the lgamma of the given input tensor, element-wise.

    This operator performs elementwise lgamma for input $X$.
    :math:`out = log\Gamma(x)`


    Args:
        x (Tensor): Input Tensor. Must be one of the following types: float32, float64.
        name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        Tensor, the lgamma of the input Tensor, the shape and data type is the same with input.

    Examples:
        .. code-block:: python

            import paddle

            x = paddle.to_tensor([-0.4, -0.2, 0.1, 0.3])
            out = paddle.lgamma(x)
            print(out)
            # [1.31452441, 1.76149750, 2.25271273, 1.09579802]
    """
    if in_dygraph_mode():
        return _C_ops.lgamma(x)
    elif _in_legacy_dygraph():
        return _legacy_C_ops.lgamma(x)

    check_variable_and_dtype(x, 'x', ['float32', 'float64'], 'lgamma')
    helper = LayerHelper('lgamma', **locals())
    out = helper.create_variable_for_type_inference(x.dtype)
    helper.append_op(type='lgamma', inputs={'X': x}, outputs={'Out': out})
    return out


def neg(x, name=None):
    """
    This function computes the negative of the Tensor elementwisely.

    Args:
        x (Tensor): Input of neg operator, an N-D Tensor, with data type float32, float64, int8, int16, int32, or int64.
        name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        out (Tensor): The negative of input Tensor. The shape and data type are the same with input Tensor.

    Examples:
        .. code-block:: python

            import paddle

            x = paddle.to_tensor([-0.4, -0.2, 0.1, 0.3])
            out = paddle.neg(x)
            print(out)
            # [0.4 0.2 -0.1 -0.3]
    """

    return scale(x, scale=-1.0, bias=0.0, bias_after_scale=True, act=None, name=name)

def atan2(x, y, name=None):
    r"""
    Element-wise arctangent of x/y with consideration of the quadrant.

    Equation:
        .. math::

            atan2(x,y)=\left\{\begin{matrix}
            & tan^{-1}(\frac{x}{y}) & y > 0 \\
            & tan^{-1}(\frac{x}{y}) + \pi & x>=0, y < 0 \\
            & tan^{-1}(\frac{x}{y}) - \pi & x<0, y < 0 \\
            & +\frac{\pi}{2} & x>0, y = 0 \\
            & -\frac{\pi}{2} & x<0, y = 0 \\
            &\text{undefined} & x=0, y = 0
            \end{matrix}\right.

    Args:
        x (Tensor): An N-D Tensor, the data type is int32, int64, float16, float32, float64.
        y (Tensor): An N-D Tensor, must have the same type as `x`.
        name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        out (Tensor): An N-D Tensor, the shape and data type is the same with input (The output data type is float64 when the input data type is int).

    Examples:
        .. code-block:: python

            import paddle

            x = paddle.to_tensor([-1, +1, +1, -1]).astype('float32')
            #Tensor(shape=[4], dtype=float32, place=CUDAPlace(0), stop_gradient=True,
            #       [-1,  1,  1, -1])

            y = paddle.to_tensor([-1, -1, +1, +1]).astype('float32')
            #Tensor(shape=[4], dtype=float32, place=CUDAPlace(0), stop_gradient=True,
            #       [-1,  -1,  1, 1])

            out = paddle.atan2(x, y)
            #Tensor(shape=[4], dtype=float32, place=CUDAPlace(0), stop_gradient=True,
            #       [-2.35619450,  2.35619450,  0.78539819, -0.78539819])

    """

    if in_dygraph_mode():
        return _C_ops.atan2( x, y)
    else:
        if _in_legacy_dygraph():
            return _legacy_C_ops.atan2(x, y)
        else:
            check_variable_and_dtype(x, 'x', ['int32', 'int64', 'float16', 'float32', 'float64'], 'atan2')
            check_variable_and_dtype(y, 'y', ['int32', 'int64', 'float16', 'float32', 'float64'], 'atan2')

            helper = LayerHelper('atan2', **locals())
            inputs = {'X1' : x, 'X2' : y}
            out = helper.create_variable_for_type_inference(dtype=x.dtype)
            helper.append_op(
                    type='atan2', inputs=inputs, outputs={'Out': out})
            return out

def logit(x, eps=None, name=None):
    r"""
    This function generates a new tensor with the logit of the elements of input x. x is clamped to [eps, 1-eps] when eps is not zero. When eps is zero and x < 0 or x > 1, the function will yields NaN.

    .. math::
 
        logit(x) = ln(\frac{x}{1 - x})

    where

    .. math::

        x_i=
            \left\{\begin{array}{rcl}
                x_i & &\text{if } eps == Default \\
                eps & &\text{if } x_i < eps \\
                x_i & &\text{if } eps <= x_i <= 1-eps \\
                1-eps & &\text{if } x_i > 1-eps
            \end{array}\right.

    Args:
        x (Tensor): The input Tensor with data type float32, float64.
        eps (float, optional):  the epsilon for input clamp bound. Default is None.
        name (str, optional): Name for the operation (optional, default is None).
            For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        out(Tensor): A Tensor with the same data type and shape as ``x`` .

    Examples:
        .. code-block:: python

            import paddle

            x = paddle.to_tensor([0.2635, 0.0106, 0.2780, 0.2097, 0.8095])
            out1 = paddle.logit(x)
            print(out1)
            # [-1.0277, -4.5365, -0.9544, -1.3269,  1.4468]  

    """

    if eps == None:
        eps = 0.0
    if _in_legacy_dygraph():
        return _legacy_C_ops.logit(x, 'eps', eps)
    if in_dygraph_mode():
        return _C_ops.logit(x, eps)
    check_variable_and_dtype(x, 'x', ['float16', 'float32', 'float64'], 'logit')
    helper = LayerHelper("logit", **locals())
    out = helper.create_variable_for_type_inference(x.dtype)
    helper.append_op(
        type='logit',
        inputs={'X': x},
        outputs={'Out': out},
        attrs={'eps': eps})
    return out

def lerp(x, y, weight, name=None):
    r"""
    Does a linear interpolation between x and y based on weight.

    Equation:
        .. math::

            lerp(x, y, weight) = x + weight * (y - x).

    Args:
        x (Tensor): An N-D Tensor with starting points, the data type is float32, float64.
        y (Tensor): An N-D Tensor with ending points, the data type is float32, float64.
        weight (float|Tensor): The weight for the interpolation formula. When weight is Tensor, the data type is float32, float64.
        name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        out (Tensor): An N-D Tensor, the shape and data type is the same with input.

    Example:
        .. code-block:: python

            import paddle
            
            x = paddle.arange(1., 5., dtype='float32')
            y = paddle.empty([4], dtype='float32')
            y.fill_(10.)
            out = paddle.lerp(x, y, 0.5)
            # out: [5.5, 6., 6.5, 7.]

    """
    if in_dygraph_mode():
        check_type(weight, 'weight', (float, paddle.Tensor, Variable), 'lerp')
        if isinstance(weight, float):
            weight = paddle.to_tensor(weight, dtype=x.dtype)

        return _C_ops.lerp( x, y, weight)
    if _in_legacy_dygraph():
        if isinstance(weight, float):
            weight = paddle.to_tensor(weight, dtype=x.dtype)
        return _legacy_C_ops.lerp(x, y, weight)

    if isinstance(weight, float):
        weight = paddle.full(shape=[1], fill_value=weight, dtype=x.dtype)

    check_variable_and_dtype(x, 'x', ['float32', 'float64'], 'lerp')
    check_variable_and_dtype(y, 'y', ['float32', 'float64'], 'lerp')
    check_variable_and_dtype(weight, 'weight', ['float32', 'float64'], 'lerp')

    helper = LayerHelper('lerp', **locals())
    inputs = {'X': x, 'Y': y, 'Weight': weight}
    out = helper.create_variable_for_type_inference(dtype=x.dtype)
    helper.append_op(type='lerp', inputs=inputs, outputs={'Out': out})
    return out

@inplace_apis_in_dygraph_only
def lerp_(x, y, weight, name=None):
    r"""
    Inplace version of ``lerp`` API, the output Tensor will be inplaced with input ``x``.
    Please refer to :ref:`api_tensor_lerp`.
    """
    out_shape = broadcast_shape(x.shape, y.shape)
    check_type(weight, 'weight', (float, paddle.Tensor, Variable), 'lerp')
    if isinstance(weight, float):
        weight = paddle.to_tensor([weight], dtype=x.dtype)
    elif isinstance(weight, (paddle.Tensor, Variable)):
        out_shape = broadcast_shape(out_shape, weight.shape)
    if out_shape != x.shape:
        raise ValueError("The shape of broadcast output {} is different from that of inplace tensor {} in the Inplace operation.".format(out_shape, x.shape))
    if in_dygraph_mode():
        return _C_ops.lerp_( x, y, weight)
    return _legacy_C_ops.lerp_(x, y, weight)

def erfinv(x, name=None):
    r"""
    The inverse error function of x.

    Equation:
        .. math::

            erfinv(erf(x)) = x.

    Args:
        x (Tensor): An N-D Tensor, the data type is float32, float64.
        name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        out (Tensor): An N-D Tensor, the shape and data type is the same with input.

    Example:
        .. code-block:: python

            import paddle
            
            x = paddle.to_tensor([0, 0.5, -1.], dtype="float32")
            out = paddle.erfinv(x)
            # out: [0, 0.4769, -inf]

    """
    if in_dygraph_mode():
        return _C_ops.erfinv( x )

    check_variable_and_dtype(x, 'x', ['float32', 'float64'], 'erfinv')

    if paddle.in_dynamic_mode():
        return _legacy_C_ops.erfinv(x)

    helper = LayerHelper('erfinv', **locals())
    out = helper.create_variable_for_type_inference(dtype=x.dtype)
    helper.append_op(type='erfinv', inputs={'X': x}, outputs={'Out': out})
    return out

@inplace_apis_in_dygraph_only
def erfinv_(x, name=None):
    r"""
    Inplace version of ``erfinv`` API, the output Tensor will be inplaced with input ``x``.
    Please refer to :ref:`api_tensor_erfinv`.
    """
    check_type(x, 'x', (paddle.Tensor, Variable), 'erfinv')
    if in_dygraph_mode():
        return _C_ops.erfinv_( x )
    return _legacy_C_ops.erfinv_(x)

def rad2deg(x, name=None):
    r"""
    Convert each of the elements of input x from angles in radians to degrees.
    
    Equation:
        .. math::

            rad2deg(x)=180/ \pi * x

    Args:
        x (Tensor): An N-D Tensor, the data type is float32, float64, int32, int64.
        name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        out (Tensor): An N-D Tensor, the shape and data type is the same with input (The output data type is float32 when the input data type is int).

    Examples:
        .. code-block:: python

            import paddle
            import numpy as np
            
            x1 = paddle.to_tensor([3.142, -3.142, 6.283, -6.283, 1.570, -1.570])
            result1 = paddle.rad2deg(x1)
            print(result1)
            # Tensor(shape=[6], dtype=float32, place=CUDAPlace(0), stop_gradient=True,
            #         [180.02334595, -180.02334595,  359.98937988, -359.98937988,
            #           9.95437622 , -89.95437622])

            x2 = paddle.to_tensor(np.pi/2)
            result2 = paddle.rad2deg(x2)
            print(result2)
            # Tensor(shape=[1], dtype=float32, place=CUDAPlace(0), stop_gradient=True,
            #         [90.])
                     
            x3 = paddle.to_tensor(1)
            result3 = paddle.rad2deg(x3)
            print(result3)
            # Tensor(shape=[1], dtype=float32, place=CUDAPlace(0), stop_gradient=True,
            #         [57.29578018])
    """
    rad2deg_scale = 180 / np.pi
    if in_dygraph_mode():
        if convert_dtype(x.dtype) in ['int32', 'int64']:
            x = cast(x, dtype="float32")
        return _C_ops.scale(x, rad2deg_scale, 0.0, True)
    elif paddle.in_dynamic_mode():
        if convert_dtype(x.dtype) in ['int32', 'int64']:
            x = cast(x, dtype="float32")
        return _legacy_C_ops.scale(x, 'scale', rad2deg_scale)
    else:
        check_variable_and_dtype(x, 'x', ['int32', 'int64', 'float32', 'float64'], 'rad2deg')
        helper = LayerHelper('rad2deg', **locals())
        out_cast = x
        if convert_dtype(x.dtype) in ['int32', 'int64']:
            out_cast = helper.create_variable_for_type_inference(dtype=paddle.float32)
            helper.append_op(
                    type='cast', inputs={'X':x}, outputs={'Out': out_cast}, attrs={'in_dtype': x.dtype,'out_dtype': paddle.float32})
        out = helper.create_variable_for_type_inference(dtype=out_cast.dtype)
        helper.append_op(
            type='scale', inputs={'X':out_cast}, outputs={'Out': out}, attrs={'scale': rad2deg_scale})
        return out

def deg2rad(x, name=None):
    r"""
    Convert each of the elements of input x from degrees to angles in radians.
    
    Equation:
        .. math::

            deg2rad(x)=\pi * x / 180

    Args:
        x (Tensor): An N-D Tensor, the data type is float32, float64, int32, int64.
        name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        out (Tensor): An N-D Tensor, the shape and data type is the same with input (The output data type is float32 when the input data type is int).

    Examples:
        .. code-block:: python

            import paddle
            import numpy as np
            
            x1 = paddle.to_tensor([180.0, -180.0, 360.0, -360.0, 90.0, -90.0])
            result1 = paddle.deg2rad(x1)
            print(result1)
            # Tensor(shape=[6], dtype=float32, place=CUDAPlace(0), stop_gradient=True,
            #         [3.14159274, -3.14159274,  6.28318548, -6.28318548,  1.57079637,
            #           -1.57079637])

            x2 = paddle.to_tensor(180)
            result2 = paddle.deg2rad(x2)
            print(result2)
            # Tensor(shape=[1], dtype=float32, place=CUDAPlace(0), stop_gradient=True,
            #         [3.14159274])
    """
    deg2rad_scale = np.pi / 180.0
    if in_dygraph_mode():
        if convert_dtype(x.dtype) in ['int32', 'int64']:
            x = cast(x, dtype="float32")
        return _C_ops.scale(x, deg2rad_scale, 0.0, True)
    elif paddle.in_dynamic_mode():
        if convert_dtype(x.dtype) in ['int32', 'int64']:
            x = cast(x, dtype="float32")
        return _legacy_C_ops.scale(x, 'scale', deg2rad_scale)
    else:
        check_variable_and_dtype(x, 'x', ['int32', 'int64', 'float32', 'float64'], 'deg2rad')
        helper = LayerHelper('deg2rad', **locals())
        out_cast = x
        if convert_dtype(x.dtype) in ['int32', 'int64']:
            out_cast = helper.create_variable_for_type_inference(dtype=paddle.float32)
            helper.append_op(
                    type='cast', inputs={'X':x}, outputs={'Out': out_cast}, attrs={'in_dtype': x.dtype,'out_dtype': paddle.float32})
        out = helper.create_variable_for_type_inference(dtype=out_cast.dtype)
        helper.append_op(
            type='scale', inputs={'X':out_cast}, outputs={'Out': out}, attrs={'scale': deg2rad_scale})
        return out

def gcd(x, y, name=None):
    """
    Computes the element-wise greatest common divisor (GCD) of input |x| and |y|.
    Both x and y must have integer types.
    
    Note:
        gcd(0,0)=0, gcd(0, y)=|y|

        If x.shape != y.shape, they must be broadcastable to a common shape (which becomes the shape of the output).

    Args:
        x (Tensor): An N-D Tensor, the data type is int32，int64. 
        y (Tensor): An N-D Tensor, the data type is int32，int64. 
        name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        out (Tensor): An N-D Tensor, the data type is the same with input.

    Examples:
        .. code-block:: python

            import paddle
            
            x1 = paddle.to_tensor(12)
            x2 = paddle.to_tensor(20)
            paddle.gcd(x1, x2)
            # Tensor(shape=[1], dtype=int64, place=CUDAPlace(0), stop_gradient=True,
            #        [4])

            x3 = paddle.arange(6)
            paddle.gcd(x3, x2)
            # Tensor(shape=[6], dtype=int64, place=CUDAPlace(0), stop_gradient=True,
            #        [20, 1 , 2 , 1 , 4 , 5])

            x4 = paddle.to_tensor(0)
            paddle.gcd(x4, x2)
            # Tensor(shape=[1], dtype=int64, place=CUDAPlace(0), stop_gradient=True,
            #        [20])

            paddle.gcd(x4, x4)
            # Tensor(shape=[1], dtype=int64, place=CUDAPlace(0), stop_gradient=True,
            #        [0])
            
            x5 = paddle.to_tensor(-20)
            paddle.gcd(x1, x5)
            # Tensor(shape=[1], dtype=int64, place=CUDAPlace(0), stop_gradient=True,
            #        [4])
    """
    shape = paddle.broadcast_shape(x.shape, y.shape)
    x = paddle.broadcast_to(x, shape)
    y = paddle.broadcast_to(y, shape)
    x = paddle.abs(x)
    y = paddle.abs(y)

    def _gcd_cond_fn(x, y):
        return paddle.any(y != 0)

    def _gcd_body_fn(x, y):
        # paddle.mod will raise an error when any element of y is 0. To avoid
        # that, we change those zeros to ones. Their values don't matter because
        # they won't be used.
        y_not_equal_0 = (y != 0)
        y_safe = paddle.where(y_not_equal_0, y, paddle.ones(y.shape, y.dtype))
        x, y = (paddle.where(y_not_equal_0, y, x),
                  paddle.where(y_not_equal_0, paddle.mod(x, y_safe),paddle.zeros(y.shape, y.dtype)))
        return (paddle.where(x < y, y, x), paddle.where(x < y, x, y))

    if paddle.in_dynamic_mode():
        while _gcd_cond_fn(x, y):
            x, y = _gcd_body_fn(x, y)

        return x
    else:
        check_variable_and_dtype(x, 'x', ['int32', 'int64'], 'gcd')
        check_variable_and_dtype(y, 'y', ['int32', 'int64'], 'gcd')
        out, _ = paddle.static.nn.while_loop(_gcd_cond_fn, _gcd_body_fn, [x, y])
        return out

def lcm(x, y, name=None):
    """
    Computes the element-wise least common multiple (LCM) of input |x| and |y|.
    Both x and y must have integer types.
    
    Note:
        lcm(0,0)=0, lcm(0, y)=0

        If x.shape != y.shape, they must be broadcastable to a common shape (which becomes the shape of the output).

    Args:
        x (Tensor): An N-D Tensor, the data type is int32，int64. 
        y (Tensor): An N-D Tensor, the data type is int32，int64. 
        name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        out (Tensor): An N-D Tensor, the data type is the same with input.

    Examples:
        .. code-block:: python

            import paddle
            
            x1 = paddle.to_tensor(12)
            x2 = paddle.to_tensor(20)
            paddle.lcm(x1, x2)
            # Tensor(shape=[1], dtype=int64, place=CUDAPlace(0), stop_gradient=True,
            #        [60])

            x3 = paddle.arange(6)
            paddle.lcm(x3, x2)
            # Tensor(shape=[6], dtype=int64, place=CUDAPlace(0), stop_gradient=True,
            #        [0, 20, 20, 60, 20, 20])

            x4 = paddle.to_tensor(0)
            paddle.lcm(x4, x2)
            # Tensor(shape=[1], dtype=int64, place=CUDAPlace(0), stop_gradient=True,
            #        [0])

            paddle.lcm(x4, x4)
            # Tensor(shape=[1], dtype=int64, place=CUDAPlace(0), stop_gradient=True,
            #        [0])
            
            x5 = paddle.to_tensor(-20)
            paddle.lcm(x1, x5)
            # Tensor(shape=[1], dtype=int64, place=CUDAPlace(0), stop_gradient=True,
            #        [60])
    """
    d = paddle.gcd(x, y)
    # paddle.mod will raise an error when any element of y is 0. To avoid
    # that, we change those zeros to ones. Their values don't matter because
    # they won't be used.
    d_equal_0 = paddle.equal(d, 0)
    d_safe = paddle.where(d_equal_0, paddle.ones(d.shape, d.dtype), d)
    out = paddle.where(d_equal_0, paddle.zeros(d.shape, d.dtype), paddle.abs(x * y) // d_safe)
    return out

def diff(x, n=1, axis=-1, prepend=None, append=None, name=None):
    r"""
    Computes the n-th forward difference along the given axis.
    The first-order differences is computed by using the following formula: 

    .. math::

        out[i] = x[i+1] - x[i]
    
    Higher-order differences are computed by using paddle.diff() recursively. 
    Only n=1 is currently supported.

    Args:
        x (Tensor): The input tensor to compute the forward difference on
        n (int, optional): The number of times to recursively compute the difference. 
                          Only support n=1. Default:1
        axis (int, optional): The axis to compute the difference along. Default:-1
        prepend (Tensor, optional): The tensor to prepend to input along axis before computing the difference.
                                   It's dimensions must be equivalent to that of x, 
                                   and its shapes must match x's shape except on axis.
        append (Tensor, optional): The tensor to append to input along axis before computing the difference, 
                                   It's dimensions must be equivalent to that of x, 
                                   and its shapes must match x's shape except on axis.
        name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.
    
    Returns:
        Tensor: The output tensor with same dtype with x.

    Examples:
        .. code-block:: python

            import paddle

            x = paddle.to_tensor([1, 4, 5, 2])
            out = paddle.diff(x)
            print(out)
            # out:
            # [3, 1, -3]

            y = paddle.to_tensor([7, 9])
            out = paddle.diff(x, append=y)
            print(out)
            # out: 
            # [3, 1, -3, 5, 2]

            z = paddle.to_tensor([[1, 2, 3], [4, 5, 6]])
            out = paddle.diff(z, axis=0)
            print(out)
            # out:
            # [[3, 3, 3]]
            out = paddle.diff(z, axis=1)
            print(out)
            # out:
            # [[1, 1], [1, 1]]
    """

    if axis < 0:
        axis = axis + len(x.shape)
    if axis > len(x.shape):
        axis = len(x.shape)
    if axis < 0:
        axis = 0
    dtype = x.dtype
    axes = [axis]
    infer_flags = list(1 for i in range(len(axes)))
    if in_dygraph_mode():
        has_pend = False
        input_list = []
        if prepend is not None and append is not None:
            input_list = [prepend, x, append]
            has_pend = True
        elif prepend is not None:
            input_list = [prepend, x]
            has_pend = True
        elif append is not None:
            input_list = [x, append]
            has_pend = True
        if has_pend:
            new_input = _C_ops.concat(input_list, axis)
        else:
            new_input = x

        attrs_1 = ()
        attrs_2 = ()

        dim_len = new_input.shape[axis]

        starts_1 = [0]
        attrs_1 += ('starts', starts_1)
        ends_1 = [dim_len - 1]
        attrs_1 += ('ends', ends_1)
        input_front = _C_ops.slice(new_input, axes, starts_1, ends_1, infer_flags,
                                            [])
        starts_2 = [1]
        attrs_2 += ('starts', starts_2)
        ends_2 = [dim_len]
        attrs_2 += ('ends', ends_2)
        input_back = _C_ops.slice(new_input, axes, starts_2, ends_2, infer_flags,
                                            [])

        if x.dtype == paddle.bool:
            return _C_ops.logical_xor(input_back, input_front)
        else:
            return _C_ops.subtract(input_back, input_front)
    elif _in_legacy_dygraph():
        has_pend = False
        input_list = []
        if prepend is not None and append is not None:
            input_list = [prepend, x, append]
            has_pend = True
        elif prepend is not None:
            input_list = [prepend, x]
            has_pend = True
        elif append is not None:
            input_list = [x, append]
            has_pend = True
        if has_pend:
            new_input = _varbase_creator()
            _legacy_C_ops.concat(input_list, new_input, 'axis', axis)
        else:
            new_input = x

        attrs_1 = ()
        attrs_2 = ()

        dim_len = new_input.shape[axis]

        starts_1 = [0]
        attrs_1 += ('starts', starts_1)
        ends_1 = [dim_len - 1]
        attrs_1 += ('ends', ends_1)
        input_front = _legacy_C_ops.slice(new_input, None, None, None, None, 'axes', axes, \
                'infer_flags', infer_flags, *attrs_1)
        starts_2 = [1]
        attrs_2 += ('starts', starts_2)
        ends_2 = [dim_len]
        attrs_2 += ('ends', ends_2)
        input_back = _legacy_C_ops.slice(new_input, None, None, None, None, 'axes', axes, \
                'infer_flags', infer_flags, *attrs_2)

        if x.dtype == paddle.bool:
            return _legacy_C_ops.logical_xor(input_back, input_front)
        else:
            return elementwise_sub(input_back, input_front, axis=axis)
    else:
        check_variable_and_dtype(x, 'x', ['float32', 'float64', 'bool', 'int32', 'int64'], 'diff')
        check_type(axis, 'axis', (int), 'diff')
        helper = LayerHelper('diff', **locals())
        has_pend = False
        input_list = []
        if prepend is not None and append is not None:
            input_list = [prepend, x, append]
            has_pend = True
        elif prepend is not None:
            input_list = [prepend, x]
            has_pend = True
        elif append is not None:
            input_list = [x, append]
            has_pend = True

        if has_pend:
            new_input = helper.create_variable_for_type_inference(dtype)
            helper.append_op(
                type='concat', inputs={'X': input_list}, outputs={'Out': [new_input]}, attrs={'axis': axis}
            )
        else:
            new_input = x

        dim_len = new_input.shape[axis]
        attrs_1 = {'axes': axes}
        starts_1 = [0]
        ends_1 = [dim_len - 1]
        attrs_1['starts'] = starts_1
        attrs_1['ends'] = ends_1
        input_front = helper.create_variable_for_type_inference(dtype)
        helper.append_op(
            type='slice', inputs={'Input': new_input}, attrs=attrs_1, outputs={'Out': input_front}
        )
        attrs_2 = {'axes': axes}
        starts_2 = [1]
        ends_2 = [dim_len]
        attrs_2['starts'] = starts_2
        attrs_2['ends'] = ends_2
        input_back = helper.create_variable_for_type_inference(dtype)
        helper.append_op(
            type='slice', inputs={'Input': new_input}, attrs=attrs_2, outputs={'Out': input_back}
        )

        if dtype == paddle.bool:
            out = helper.create_variable_for_type_inference(dtype)
            helper.append_op(
                type='logical_xor', inputs={"X": input_back, "Y": input_front}, outputs={"Out": out}
            )
        else:
            out = elementwise_sub(input_back, input_front, axis=axis)

        return out

def angle(x, name=None):
    r"""
    Element-wise angle of complex numbers. For non-negative real numbers, the angle is 0 while 
    for negative real numbers, the angle is :math:`\pi`.

    Equation:
        .. math::

            angle(x)=arctan2(x.imag, x.real)

    Args:
        x (Tensor): An N-D Tensor, the data type is complex64, complex128, or float32, float64 .
        name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        Tensor: An N-D Tensor of real data type with the same precision as that of x's data type.

    Examples:
        .. code-block:: python

            import paddle

            x = paddle.to_tensor([-2, -1, 0, 1]).unsqueeze(-1).astype('float32')
            y = paddle.to_tensor([-2, -1, 0, 1]).astype('float32')
            z = x + 1j * y
            print(z.numpy())
            # [[-2.-2.j -2.-1.j -2.+0.j -2.+1.j]
            #  [-1.-2.j -1.-1.j -1.+0.j -1.+1.j]
            #  [ 0.-2.j  0.-1.j  0.+0.j  0.+1.j]
            #  [ 1.-2.j  1.-1.j  1.+0.j  1.+1.j]]

            theta = paddle.angle(z)
            print(theta.numpy())
            # [[-2.3561945 -2.6779451  3.1415927  2.6779451]
            #  [-2.0344439 -2.3561945  3.1415927  2.3561945]
            #  [-1.5707964 -1.5707964  0.         1.5707964]
            #  [-1.1071488 -0.7853982  0.         0.7853982]]
    """

    if in_dygraph_mode():
        return _C_ops.angle(x)
    elif paddle.in_dynamic_mode():
        return _legacy_C_ops.angle(x)

    check_variable_and_dtype(x, 'x',
        ['float32', 'float64', 'complex64', 'complex128'], 'angle')
    op_type = "angle"
    helper = LayerHelper(op_type, **locals())
    inputs = {"X": x}
    out = helper.create_variable_for_type_inference(
        dtype=_complex_to_real_dtype(x.dtype))
    outputs = {"Out": out}
    helper.append_op(type=op_type, inputs=inputs, outputs=outputs)
    return out

def heaviside(x, y, name=None):
    """
    Computes the Heaviside step function determined by corresponding element in y for each element in x. The equation is

    .. math::
        heaviside(x, y)=
            \left\{
                \\begin{array}{lcl}
                0,& &\\text{if} \ x < 0, \\\\
                y,& &\\text{if} \ x = 0, \\\\
                1,& &\\text{if} \ x > 0.
                \end{array}
            \\right.

    Note:
        ``paddle.heaviside`` supports broadcasting. If you want know more about broadcasting, please refer to :ref:`user_guide_broadcasting`.

    Args:
        x (Tensor): The input tensor of Heaviside step function, it's data type should be float16, float32, float64, int32 or int64.
        y (Tensor): The tensor that determines a Heaviside step function, it's data type should be float16, float32, float64, int32 or int64.
        name (str, optional): Name for the operation (optional, default is None). Normally there is no need for user to set this property. For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        N-D Tensor. A location into which the result is stored. If x and y have different shapes and are broadcastable, the resulting tensor shape is the shape of x and y after broadcasting. If x, y have the same shape, its shape is the same as x and y.

    Examples:
        .. code-block:: python

            import paddle
            x = paddle.to_tensor([-0.5, 0, 0.5])
            y = paddle.to_tensor([0.1])
            paddle.heaviside(x, y)
            #    [0.        , 0.10000000, 1.        ]
            x = paddle.to_tensor([[-0.5, 0, 0.5], [-0.5, 0.5, 0]])
            y = paddle.to_tensor([0.1, 0.2, 0.3])
            paddle.heaviside(x, y)
            #    [[0.        , 0.20000000, 1.        ],
            #     [0.        , 1.        , 0.30000001]]
     """
    op_type = 'elementwise_heaviside'
    axis = -1
    act = None
    if _non_static_mode():
        return _elementwise_op_in_dygraph(
            x, y, axis=axis, act=act, op_name=op_type)
    return _elementwise_op(LayerHelper(op_type, **locals()))

def frac(x, name=None):
    """
    This API is used to return the fractional portion of each element in input.

    Args:
        x (Tensor): The input tensor, which data type should be int32, int64, float32, float64.
        name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        Tensor: The output Tensor of frac.

    Examples:
        .. code-block:: python

            import paddle
            import numpy as np

            input = paddle.rand([3, 3], 'float32')
            print(input.numpy())
            # [[ 1.2203873  -1.0035421  -0.35193074]
            #  [-0.00928353  0.58917075 -0.8407828 ]
            #  [-1.5131804   0.5850153  -0.17597814]]

            output = paddle.frac(input)
            print(output.numpy())
            # [[ 0.22038734 -0.00354207 -0.35193074]
            #  [-0.00928353  0.58917075 -0.8407828 ]
            #  [-0.5131804   0.5850153  -0.17597814]]
    """
    op_type = 'elementwise_sub'
    axis = -1
    act = None
    if x.dtype not in [paddle.int32, paddle.int64, paddle.float32, paddle.float64]:
        raise TypeError(
            "The data type of input must be one of ['int32', 'int64', 'float32', 'float64'], but got {}".format(x.dtype))
    if in_dygraph_mode():
        y = _C_ops.trunc(x)
        return _C_ops.subtract(x, y)
    else:
        if _in_legacy_dygraph():
            y = _legacy_C_ops.trunc(x)
            return _elementwise_op_in_dygraph(
                x, y, axis=axis, act=act, op_name=op_type)
        else:
            inputs = {"X": x}
            attrs = {}

            helper = LayerHelper("trunc", **locals())
            check_variable_and_dtype(x, "X", ['int32', 'int64', 'float32', 'float64'], 'trunc')
            y = helper.create_variable_for_type_inference(dtype=x.dtype)
            helper.append_op(
                type="trunc", inputs=inputs, attrs=attrs, outputs={"Out": y})
            return _elementwise_op(LayerHelper(op_type, **locals()))

def sgn(x, name=None):
    """
    For complex tensor, this API returns a new tensor whose elements have the same angles as the corresponding
    elements of input and absolute values of one.
    For other float dtype tensor,
    this API returns sign of every element in `x`: 1 for positive, -1 for negative and 0 for zero, same as paddle.sign.

    Args:
        x (Tensor): The input tensor, which data type should be float16, float32, float64, complex64, complex128.
        name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        Tensor: A sign Tensor for real input, or normalized Tensor for complex input, shape and data type are same as input.

    Examples:
        .. code-block:: Python

            import paddle

            x = paddle.to_tensor([[3 + 4j, 7 - 24j, 0, 1 + 2j], [6 + 8j, 3, 0, -2]])
            print(paddle.sgn(x))
            #[[0.6+0.8j       0.28-0.96j      0.+0.j      0.4472136+0.8944272j]
            # [0.6+0.8j       1.+0.j          0.+0.j      -1.+0.j]]

    """
    if x.dtype not in [paddle.float16, paddle.float32, paddle.float64, paddle.complex64, paddle.complex128]:
        raise TypeError(
            "The data type of input must be one of ['float16', 'float32', 'float64', 'complex64', 'complex128'], but got {}"
                .format(x.dtype))
    if paddle.is_complex(x):
        expand_x = paddle.as_real(x)
        x_abs = paddle.abs(x)
        x_abs = paddle.unsqueeze(x_abs, axis=-1)
        output = expand_x / x_abs
        zeros = paddle.zeros_like(output)
        output = paddle.where(paddle.isnan(output), zeros, output)

        return paddle.as_complex(output)
    else:
        return paddle.sign(x)

def take(x, index, mode='raise', name=None):
    """
    Returns a new tensor with the elements of input tensor x at the given index.
    The input tensor is treated as if it were viewed as a 1-D tensor.
    The result takes the same shape as the index.

    Args:
        x (Tensor): An N-D Tensor, its data type should be int32, int64, float32, float64.
        index (Tensor): An N-D Tensor, its data type should be int32, int64.
        mode (str, optional): Specifies how out-of-bounds index will behave. the candicates are ``'raise'``, ``'wrap'`` and ``'clip'``.

            - ``'raise'``: raise an error (default);
            - ``'wrap'``: wrap around;
            - ``'clip'``: clip to the range. ``'clip'`` mode means that all indices that are too large are replaced by the index that addresses the last element. Note that this disables indexing with negative numbers.

        name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        Tensor, Tensor with the same shape as index, the data type is the same with input.

    Examples:
        .. code-block:: python

            import paddle

            x_int = paddle.arange(0, 12).reshape([3, 4])
            x_float = x_int.astype(paddle.float64)

            idx_pos = paddle.arange(4, 10).reshape([2, 3])  # positive index
            idx_neg = paddle.arange(-2, 4).reshape([2, 3])  # negative index
            idx_err = paddle.arange(-2, 13).reshape([3, 5])  # index out of range

            paddle.take(x_int, idx_pos)
            # Tensor(shape=[2, 3], dtype=int64, place=Place(cpu), stop_gradient=True,
            #        [[4, 5, 6],
            #         [7, 8, 9]])

            paddle.take(x_int, idx_neg)
            # Tensor(shape=[2, 3], dtype=int64, place=Place(cpu), stop_gradient=True,
            #        [[10, 11, 0 ],
            #         [1 , 2 , 3 ]])

            paddle.take(x_float, idx_pos)
            # Tensor(shape=[2, 3], dtype=float64, place=Place(cpu), stop_gradient=True,
            #        [[4., 5., 6.],
            #         [7., 8., 9.]])

            x_int.take(idx_pos)
            # Tensor(shape=[2, 3], dtype=int64, place=Place(cpu), stop_gradient=True,
            #        [[4, 5, 6],
            #         [7, 8, 9]])

            paddle.take(x_int, idx_err, mode='wrap')
            # Tensor(shape=[3, 5], dtype=int32, place=Place(cpu), stop_gradient=True,
            #        [[10, 11, 0 , 1 , 2 ],
            #         [3 , 4 , 5 , 6 , 7 ],
            #         [8 , 9 , 10, 11, 0 ]])

            paddle.take(x_int, idx_err, mode='clip')
            # Tensor(shape=[3, 5], dtype=int32, place=Place(cpu), stop_gradient=True,
            #        [[0 , 0 , 0 , 1 , 2 ],
            #         [3 , 4 , 5 , 6 , 7 ],
            #         [8 , 9 , 10, 11, 11]])

    """
    if mode not in ['raise', 'wrap', 'clip']:
        raise ValueError(
            "'mode' in 'take' should be 'raise', 'wrap', 'clip', but received {}.".format(mode))

    if paddle.in_dynamic_mode():
        if not isinstance(index, (paddle.Tensor, Variable)):
            raise TypeError(
                "The type of 'index' must be Tensor, but got {}".format(type(index)))
        if index.dtype not in [paddle.int32, paddle.int64]:
            raise TypeError(
                "The data type of 'index' must be one of ['int32', 'int64'], but got {}".format(
                    index.dtype))

    else:
        check_variable_and_dtype(index, 'index', ['int32', 'int64'], 'take')

    input_1d = x.flatten()
    index_1d = index.flatten()
    max_index = input_1d.shape[-1]

    if mode == 'raise':
        # This processing enables 'take' to handle negative indexes within the correct range.
        index_1d = paddle.where(index_1d < 0, index_1d + max_index, index_1d)
    elif mode == 'wrap':
        # The out of range indices are constrained by taking the remainder.
        index_1d = paddle.where(index_1d < 0,
                                index_1d % max_index, index_1d)
        index_1d = paddle.where(index_1d >= max_index,
                                index_1d % max_index, index_1d)
    elif mode == 'clip':
        # 'clip' mode disables indexing with negative numbers.
        index_1d = clip(index_1d, 0, max_index - 1)

    out = input_1d.index_select(index_1d).reshape(index.shape)

    return out
