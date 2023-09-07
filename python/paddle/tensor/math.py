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

import numpy as np

import paddle
from paddle import _C_ops, _legacy_C_ops
from paddle.common_ops_import import VarDesc, dygraph_only, dygraph_utils
from paddle.utils.inplace_utils import inplace_apis_in_dygraph_only

from ..common_ops_import import Variable
from ..fluid.data_feeder import (
    check_dtype,
    check_type,
    check_variable_and_dtype,
    convert_dtype,
)
from ..framework import (
    LayerHelper,
    convert_np_dtype_to_dtype_,
    core,
    in_dynamic_mode,
)
from .creation import _complex_to_real_dtype
from .layer_function_generator import generate_layer_fn, templatedoc
from .manipulation import cast, cast_
from .ops import abs  # noqa: F401
from .ops import abs_  # noqa: F401
from .ops import acos  # noqa: F401
from .ops import acos_  # noqa: F401
from .ops import acosh  # noqa: F401
from .ops import acosh_  # noqa: F401
from .ops import asin  # noqa: F401
from .ops import asin_  # noqa: F401
from .ops import asinh  # noqa: F401
from .ops import asinh_  # noqa: F401
from .ops import atan  # noqa: F401
from .ops import atan_  # noqa: F401
from .ops import atanh  # noqa: F401
from .ops import atanh_  # noqa: F401
from .ops import ceil  # noqa: F401
from .ops import ceil_  # noqa: F401
from .ops import cos  # noqa: F401
from .ops import cos_  # noqa: F401
from .ops import cosh  # noqa: F401
from .ops import cosh_  # noqa: F401
from .ops import erf  # noqa: F401
from .ops import erf_  # noqa: F401
from .ops import exp  # noqa: F401
from .ops import exp_  # noqa: F401
from .ops import expm1  # noqa: F401
from .ops import expm1_  # noqa: F401
from .ops import floor  # noqa: F401
from .ops import floor_  # noqa: F401
from .ops import reciprocal  # noqa: F401
from .ops import reciprocal_  # noqa: F401
from .ops import round  # noqa: F401
from .ops import round_  # noqa: F401
from .ops import rsqrt  # noqa: F401
from .ops import rsqrt_  # noqa: F401
from .ops import sigmoid  # noqa: F401
from .ops import sigmoid_  # noqa: F401
from .ops import sin  # noqa: F401
from .ops import sin_  # noqa: F401
from .ops import sinh  # noqa: F401
from .ops import sinh_  # noqa: F401
from .ops import sqrt  # noqa: F401
from .ops import sqrt_  # noqa: F401
from .ops import square  # noqa: F401
from .ops import square_  # noqa: F401
from .ops import tan  # noqa: F401
from .ops import tan_  # noqa: F401

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


def _get_reduce_axis(axis, x):
    """
    Internal function for max, min, amax and amin.
    It computes the attribute reduce_all value based on axis.
    """
    if axis is not None and not isinstance(axis, list):
        if isinstance(axis, (tuple, range)):
            axis = list(axis)
        elif isinstance(axis, int):
            axis = [axis]
        else:
            raise TypeError(
                "The type of axis must be int, list or tuple, but received {}".format(
                    type(axis)
                )
            )
    if axis is None:
        axis = []
    if axis == [] or len(axis) == len(x.shape):
        reduce_all = True
    else:
        reduce_all = False
    return reduce_all, axis


def _get_reduce_axis_with_tensor(axis, x):
    if isinstance(axis, Variable):
        if axis.shape[0] == len(x.shape):
            reduce_all = True
        else:
            reduce_all = False
    else:
        reduce_all, axis = _get_reduce_axis(axis, x)
        if paddle.utils._contain_var(axis):
            axis = paddle.utils._convert_to_tensor_list(axis)
    return reduce_all, axis


def log(x, name=None):
    r"""
    Calculates the natural log of the given input Tensor, element-wise.

    .. math::

        Out = \ln(x)

    Args:
        x (Tensor): Input Tensor. Must be one of the following types: int32, int64, float16, bfloat16, float32, float64.
        name (str|None): The default value is None. Normally there is no need for user to set this property. For more information, please refer to :ref:`api_guide_Name`


    Returns:
        Tensor: The natural log of the input Tensor computed element-wise.

    Examples:

        .. code-block:: python

            >>> import paddle

            >>> x = [[2, 3, 4], [7, 8, 9]]
            >>> x = paddle.to_tensor(x, dtype='float32')
            >>> print(paddle.log(x))
            Tensor(shape=[2, 3], dtype=float32, place=Place(cpu), stop_gradient=True,
            [[0.69314718, 1.09861231, 1.38629436],
             [1.94591010, 2.07944155, 2.19722462]])
    """
    if in_dynamic_mode():
        return _C_ops.log(x)
    else:
        check_variable_and_dtype(
            x,
            'x',
            ['int32', 'int64', 'uint16', 'float16', 'float32', 'float64'],
            "log",
        )
        inputs = {'X': [x]}
        helper = LayerHelper('log', **locals())
        dtype = helper.input_dtype(input_param_name='x')
        out = helper.create_variable_for_type_inference(dtype)
        helper.append_op(type="log", inputs={"X": x}, outputs={"Out": out})
        return out


@inplace_apis_in_dygraph_only
def log_(x, name=None):
    r"""
    Inplace version of ``log`` API, the output Tensor will be inplaced with input ``x``.
    Please refer to :ref:`api_paddle_log`.
    """

    if in_dynamic_mode():
        return _C_ops.log_(x)


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
        scale (float|Tensor): The scale factor of the input, it should be a float number or a 0-D Tensor with shape [] and data type as float32.
        bias (float): The bias to be put on the input.
        bias_after_scale (bool): Apply bias addition after or before scaling. It is useful for numeric stability in some circumstances.
        act (str, optional): Activation applied to the output such as tanh, softmax, sigmoid, relu.
        name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        Tensor: Output Tensor of scale operator, with shape and data type same as input.

    Examples:
        .. code-block:: python

            >>> # scale as a float32 number
            >>> import paddle

            >>> data = paddle.arange(6).astype("float32").reshape([2, 3])
            >>> print(data)
            Tensor(shape=[2, 3], dtype=float32, place=Place(cpu), stop_gradient=True,
            [[0., 1., 2.],
             [3., 4., 5.]])
            >>> res = paddle.scale(data, scale=2.0, bias=1.0)
            >>> print(res)
            Tensor(shape=[2, 3], dtype=float32, place=Place(cpu), stop_gradient=True,
            [[1. , 3. , 5. ],
             [7. , 9. , 11.]])

        .. code-block:: python

            >>> # scale with parameter scale as a Tensor
            >>> import paddle

            >>> data = paddle.arange(6).astype("float32").reshape([2, 3])
            >>> print(data)
            Tensor(shape=[2, 3], dtype=float32, place=Place(cpu), stop_gradient=True,
            [[0., 1., 2.],
             [3., 4., 5.]])
            >>> factor = paddle.to_tensor([2], dtype='float32')
            >>> res = paddle.scale(data, scale=factor, bias=1.0)
            >>> print(res)
            Tensor(shape=[2, 3], dtype=float32, place=Place(cpu), stop_gradient=True,
            [[1. , 3. , 5. ],
             [7. , 9. , 11.]])

    """

    if in_dynamic_mode():
        if act is None:
            return _C_ops.scale(x, scale, float(bias), bias_after_scale)
        out = _C_ops.scale(x, scale, float(bias), bias_after_scale)
        return dygraph_utils._append_activation_in_dygraph(out, act)
    else:
        check_variable_and_dtype(
            x,
            "x",
            [
                'float16',
                'bfloat16',
                'uint16',
                'float32',
                'float64',
                'int8',
                'int16',
                'int32',
                'int64',
                'uint8',
                'complex64',
                'complex128',
            ],
            "scale",
        )
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
            type='scale', inputs=inputs, outputs={'Out': out}, attrs=attrs
        )
        return helper.append_activation(out)


def stanh(x, scale_a=0.67, scale_b=1.7159, name=None):
    r"""

    stanh activation.

    .. math::

        out = b * \frac{e^{a * x} - e^{-a * x}}{e^{a * x} + e^{-a * x}}

    Parameters:
        x (Tensor): The input Tensor with data type float32, float64.
        scale_a (float, optional): The scale factor a of the input. Default is 0.67.
        scale_b (float, optional): The scale factor b of the output. Default is 1.7159.
        name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        A Tensor with the same data type and shape as ``x`` .

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> x = paddle.to_tensor([1.0, 2.0, 3.0, 4.0])
            >>> out = paddle.stanh(x, scale_a=0.67, scale_b=1.72)
            >>> print(out)
            Tensor(shape=[4], dtype=float32, place=Place(cpu), stop_gradient=True,
            [1.00616539, 1.49927628, 1.65933096, 1.70390463])

    """

    if in_dynamic_mode():
        return _C_ops.stanh(x, scale_a, scale_b)
    else:
        check_variable_and_dtype(
            x, 'x', ['float16', 'uint16', 'float32', 'float64'], 'stanh'
        )

        helper = LayerHelper('stanh', **locals())
        out = helper.create_variable_for_type_inference(dtype=x.dtype)
        helper.append_op(
            type='stanh',
            inputs={'X': x},
            outputs={'Out': out},
            attrs={'scale_a': scale_a, 'scale_b': scale_b},
        )
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

            >>> import paddle

            >>> img1 = paddle.to_tensor([[1, 2], [3, 4]], dtype=paddle.float32)
            >>> img2 = paddle.to_tensor([[5, 6], [7, 8]], dtype=paddle.float32)
            >>> inputs = [img1, img2]
            >>> index = paddle.to_tensor([[1], [0]], dtype=paddle.int32)
            >>> res = paddle.multiplex(inputs, index)
            >>> print(res)
            Tensor(shape=[2, 2], dtype=float32, place=Place(cpu), stop_gradient=True,
            [[5., 6.],
             [3., 4.]])

    """
    if in_dynamic_mode():
        return _C_ops.multiplex(inputs, index)
    else:
        helper = LayerHelper('multiplex', **locals())

        check_type(inputs, 'inputs', (list), 'multiplex')
        if len(inputs) < 2:
            raise ValueError(
                "inputs should be a list object with at least 2 elements."
            )
        for id, x in enumerate(inputs):
            check_variable_and_dtype(
                x,
                'input[' + str(id) + ']',
                ['float32', 'float64', 'int32', 'int64'],
                'multiplex',
            )
        check_variable_and_dtype(
            index, "index", ['int32', 'int64'], 'multiplex'
        )

        out = helper.create_variable_for_type_inference(inputs[0].dtype)
        helper.append_op(
            type='multiplex',
            inputs={'X': inputs, 'Ids': index},
            outputs={'Out': [out]},
        )
        return out


@inplace_apis_in_dygraph_only
def scale_(x, scale=1.0, bias=0.0, bias_after_scale=True, act=None, name=None):
    """
    Inplace version of ``scale`` API, the output Tensor will be inplaced with input ``x``.
    Please refer to :ref:`api_paddle_scale`.
    """
    if in_dynamic_mode():
        return _C_ops.scale_(x, scale, float(bias), bias_after_scale)


def pow(x, y, name=None):
    """
    Compute the power of Tensor elements. The equation is:

    .. math::
        out = x^{y}

    Note:
        ``paddle.pow`` supports broadcasting. If you want know more about broadcasting, please refer to `Introduction to Tensor`_ .

        .. _Introduction to Tensor: ../../guides/beginner/tensor_en.html#chapter5-broadcasting-of-tensors


    Args:
        x (Tensor): An N-D Tensor, the data type is float16, float32, float64, int32 or int64.
        y (float|int|Tensor): If it is an N-D Tensor, its data type should be the same as `x`.
        name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        N-D Tensor. A location into which the result is stored. Its dimension and data type are the same as `x`.

    Examples:

        .. code-block:: python

            >>> import paddle

            >>> x = paddle.to_tensor([1, 2, 3], dtype='float32')

            >>> # example 1: y is a float or int
            >>> res = paddle.pow(x, 2)
            >>> print(res)
            Tensor(shape=[3], dtype=float32, place=Place(cpu), stop_gradient=True,
            [1., 4., 9.])
            >>> res = paddle.pow(x, 2.5)
            >>> print(res)
            Tensor(shape=[3], dtype=float32, place=Place(cpu), stop_gradient=True,
            [1.         , 5.65685415 , 15.58845711])

            >>> # example 2: y is a Tensor
            >>> y = paddle.to_tensor([2], dtype='float32')
            >>> res = paddle.pow(x, y)
            >>> print(res)
            Tensor(shape=[3], dtype=float32, place=Place(cpu), stop_gradient=True,
            [1., 4., 9.])

    """
    # in dynamic graph mode
    if in_dynamic_mode():
        if isinstance(y, (int, float)):
            return _C_ops.pow(x, y)
        elif isinstance(y, (paddle.Tensor, Variable)):
            return _C_ops.elementwise_pow(x, y)
        else:
            raise TypeError(
                'y must be scalar or tensor type, but received: %s ' % (y.dtype)
            )
    else:
        # in static graph mode
        if isinstance(y, (int, float)):
            helper = LayerHelper('pow', **locals())
            inputs = {'X': x}
            attrs = {'factor': y}
            out = helper.create_variable_for_type_inference(dtype=x.dtype)
            helper.append_op(
                type='pow', inputs=inputs, outputs={'Out': out}, attrs=attrs
            )
            return out
        elif isinstance(y, (paddle.Tensor, Variable)):
            # TODO A potential speed improvement is supporting different types in C++ and removing the cast ops here
            helper = LayerHelper('elementwise_pow', **locals())
            out = helper.create_variable_for_type_inference(dtype=x.dtype)
            return _elementwise_op(LayerHelper('elementwise_pow', **locals()))
        else:
            raise TypeError(
                'y must be scalar or tensor type, but received: %s ' % (type(y))
            )


@inplace_apis_in_dygraph_only
def pow_(x, y, name=None):
    """
    Inplace version of ``pow`` API, the output Tensor will be inplaced with input ``x``.
    Please refer to :ref:`api_paddle_pow`.
    """
    if isinstance(y, (int, float)):
        return _C_ops.pow_(x, y)
    else:
        raise TypeError('y must be scalar type, but received: %s ' % (type(y)))


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


def _elementwise_op(helper):
    op_type = helper.layer_type
    original_op_type = helper.kwargs.get('original_op_type', op_type)
    x = helper.kwargs.get('x', None)
    y = helper.kwargs.get('y', None)

    out = helper.kwargs.get('out', None)

    assert x is not None, f'x cannot be None in {original_op_type}'
    assert y is not None, f'y cannot be None in {original_op_type}'
    bf16_and_complex_supported_ops = [
        "elementwise_add",
        "elementwise_sub",
        "elementwise_mul",
        "elementwise_div",
        "elementwise_max",
    ]
    if original_op_type in bf16_and_complex_supported_ops:
        data_type = [
            'uint16',
            'float16',
            'float32',
            'float64',
            'int32',
            'int64',
            'bool',
            'complex64',
            'complex128',
        ]
    else:
        data_type = [
            'float16',
            'uint16',
            'float32',
            'float64',
            'int32',
            'int64',
            'bool',
        ]
    check_variable_and_dtype(
        x,
        'x',
        data_type,
        original_op_type,
    )
    check_variable_and_dtype(
        y,
        'y',
        data_type,
        original_op_type,
    )

    axis = helper.kwargs.get('axis', -1)
    use_mkldnn = helper.kwargs.get('use_mkldnn', False)
    name = helper.kwargs.get('name', None)

    if out is None:
        if name is None:
            out = helper.create_variable_for_type_inference(dtype=x.dtype)
        else:
            out = helper.create_variable(
                name=name, dtype=x.dtype, persistable=False
            )

    helper.append_op(
        type=op_type,
        inputs={'X': x, 'Y': y},
        outputs={'Out': out},
        attrs={'axis': axis, 'use_mkldnn': use_mkldnn},
    )
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
    2. If $axis$ is -1 (default), $axis$=rank($X$)-rank($Y$).
    3. The trailing dimensions of size 1 for $Y$ will be ignored for the consideration of subsequence, such as shape($Y$) = (2, 1) => (2).

        For example:

        .. code-block:: text

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
        N-D Tensor. A location into which the result is stored. It's dimension equals with x.

    Examples:

        .. code-block:: python

            >>> import paddle

            >>> x = paddle.to_tensor([2, 3, 4], 'float64')
            >>> y = paddle.to_tensor([1, 5, 2], 'float64')
            >>> z = paddle.add(x, y)
            >>> print(z)
            Tensor(shape=[3], dtype=float64, place=Place(cpu), stop_gradient=True,
            [3., 8., 6.])
    """

    if in_dynamic_mode():
        return _C_ops.add(x, y)
    else:
        return _elementwise_op(LayerHelper('elementwise_add', **locals()))


@inplace_apis_in_dygraph_only
def add_(x, y, name=None):
    """
    Inplace version of ``add`` API, the output Tensor will be inplaced with input ``x``.
    Please refer to :ref:`api_paddle_add`.
    """

    out_shape = broadcast_shape(x.shape, y.shape)
    if out_shape != x.shape:
        raise ValueError(
            "The shape of broadcast output {} is different from that of inplace tensor {} in the Inplace operation.".format(
                out_shape, x.shape
            )
        )

    return _C_ops.add_(x, y)


def logaddexp(x, y, name=None):
    """
    Elementwise LogAddExp Operator.
    Add of exponentiations of the inputs
    The equation is:

    ..  math::

        Out=log(X.exp()+Y.exp())

    $X$ the tensor of any dimension.
    $Y$ the tensor whose dimensions must be less than or equal to the dimensions of $X$.

    There are two cases for this operator:

    1. The shape of $Y$ is the same with $X$.
    2. The shape of $Y$ is a continuous subsequence of $X$.

    For case 2:

    1. Broadcast $Y$ to match the shape of $X$, where axis is the start dimension index for broadcasting $Y$ onto $X$.
    2. If $axis$ is -1 (default), $axis$=rank($X$)-rank($Y$).
    3. The trailing dimensions of size 1 for $Y$ will be ignored for the consideration of subsequence, such as shape($Y$) = (2, 1) => (2).

        For example:

        .. code-block:: text

            shape(X) = (2, 3, 4, 5), shape(Y) = (,)
            shape(X) = (2, 3, 4, 5), shape(Y) = (5,)
            shape(X) = (2, 3, 4, 5), shape(Y) = (4, 5), with axis=-1(default) or axis=2
            shape(X) = (2, 3, 4, 5), shape(Y) = (3, 4), with axis=1
            shape(X) = (2, 3, 4, 5), shape(Y) = (2), with axis=0
            shape(X) = (2, 3, 4, 5), shape(Y) = (2, 1), with axis=0

    Args:
        x (Tensor): Tensor or LoDTensor of any dimensions. Its dtype should be int32, int64, float32, float64, float16.
        y (Tensor): Tensor or LoDTensor of any dimensions. Its dtype should be int32, int64, float32, float64, float16.
        name (string, optional): For details, please refer to :ref:`api_guide_Name`. Generally, no setting is required. Default: None.

    Returns:
        N-D Tensor. A location into which the result is stored. It's dimension equals with x.

    Examples:

        .. code-block:: python

            >>> import paddle

            >>> x = paddle.to_tensor([-1, -2, -3], 'float64')
            >>> y = paddle.to_tensor([-1], 'float64')
            >>> z = paddle.logaddexp(x, y)
            >>> print(z)
            Tensor(shape=[3], dtype=float64, place=Place(cpu), stop_gradient=True,
            [-0.30685282, -0.68673831, -0.87307199])
    """

    return paddle.log1p(paddle.exp(-paddle.abs(x - y))) + paddle.maximum(x, y)


def subtract(x, y, name=None):
    """
    Substract two tensors element-wise. The equation is:

    .. math::
        out = x - y

    Note:
        ``paddle.subtract`` supports broadcasting. If you want know more about broadcasting, please refer to `Introduction to Tensor`_ .

        .. _Introduction to Tensor: ../../guides/beginner/tensor_en.html#chapter5-broadcasting-of-tensor

    Args:
        x (Tensor): the input tensor, it's data type should be float32, float64, int32, int64.
        y (Tensor): the input tensor, it's data type should be float32, float64, int32, int64.
        name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        N-D Tensor. A location into which the result is stored. If x, y have different shapes and are "broadcastable", the resulting tensor shape is the shape of x and y after broadcasting. If x, y have the same shape,  its shape is the same as x and y.

    Examples:

        .. code-block:: python

            >>> import paddle

            >>> x = paddle.to_tensor([[1, 2], [7, 8]])
            >>> y = paddle.to_tensor([[5, 6], [3, 4]])
            >>> res = paddle.subtract(x, y)
            >>> print(res)
            Tensor(shape=[2, 2], dtype=int64, place=Place(cpu), stop_gradient=True,
            [[-4, -4],
             [ 4,  4]])

            >>> x = paddle.to_tensor([[[1, 2, 3], [1, 2, 3]]])
            >>> y = paddle.to_tensor([1, 0, 4])
            >>> res = paddle.subtract(x, y)
            >>> print(res)
            Tensor(shape=[1, 2, 3], dtype=int64, place=Place(cpu), stop_gradient=True,
            [[[ 0,  2, -1],
              [ 0,  2, -1]]])

            >>> x = paddle.to_tensor([2, float('nan'), 5], dtype='float32')
            >>> y = paddle.to_tensor([1, 4, float('nan')], dtype='float32')
            >>> res = paddle.subtract(x, y)
            >>> print(res)
            Tensor(shape=[3], dtype=float32, place=Place(cpu), stop_gradient=True,
            [1. , nan, nan])

            >>> x = paddle.to_tensor([5, float('inf'), -float('inf')], dtype='float64')
            >>> y = paddle.to_tensor([1, 4, 5], dtype='float64')
            >>> res = paddle.subtract(x, y)
            >>> print(res)
            Tensor(shape=[3], dtype=float64, place=Place(cpu), stop_gradient=True,
            [ 4.  ,  inf., -inf.])
    """
    if in_dynamic_mode():
        return _C_ops.subtract(x, y)
    else:
        return _elementwise_op(LayerHelper('elementwise_sub', **locals()))


@inplace_apis_in_dygraph_only
def subtract_(x, y, name=None):
    """
    Inplace version of ``subtract`` API, the output Tensor will be inplaced with input ``x``.
    Please refer to :ref:`api_paddle_subtract`.
    """

    out_shape = broadcast_shape(x.shape, y.shape)
    if out_shape != x.shape:
        raise ValueError(
            "The shape of broadcast output {} is different from that of inplace tensor {} in the Inplace operation.".format(
                out_shape, x.shape
            )
        )

    return _C_ops.subtract_(x, y)


def divide(x, y, name=None):
    """
    Divide two tensors element-wise. The equation is:

    .. math::
        out = x / y

    Note:
        ``paddle.divide`` supports broadcasting. If you want know more about broadcasting, please refer to `Introduction to Tensor`_ .

        .. _Introduction to Tensor: ../../guides/beginner/tensor_en.html#chapter5-broadcasting-of-tensor

    Args:
        x (Tensor): the input tensor, it's data type should be float32, float64, int32, int64.
        y (Tensor): the input tensor, it's data type should be float32, float64, int32, int64.
        name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        N-D Tensor. A location into which the result is stored. If x, y have different shapes and are "broadcastable", the resulting tensor shape is the shape of x and y after broadcasting. If x, y have the same shape,  its shape is the same as x and y.

    Examples:

        .. code-block:: python

            >>> import paddle

            >>> x = paddle.to_tensor([2, 3, 4], dtype='float64')
            >>> y = paddle.to_tensor([1, 5, 2], dtype='float64')
            >>> z = paddle.divide(x, y)
            >>> print(z)
            Tensor(shape=[3], dtype=float64, place=Place(cpu), stop_gradient=True,
            [2.        , 0.60000000, 2.        ])

    """
    if in_dynamic_mode():
        return _C_ops.divide(x, y)
    else:
        if paddle.ir.core._use_new_ir_api():
            return paddle._ir_ops.divide(x, y)
        return _elementwise_op(LayerHelper('elementwise_div', **locals()))


@inplace_apis_in_dygraph_only
def divide_(x, y, name=None):
    r"""
    Inplace version of ``divide`` API, the output Tensor will be inplaced with input ``x``.
    Please refer to :ref:`api_paddle_divide`.
    """
    out_shape = broadcast_shape(x.shape, y.shape)
    if out_shape != x.shape:
        raise ValueError(
            "The shape of broadcast output {} is different from that of inplace tensor {} in the Inplace operation.".format(
                out_shape, x.shape
            )
        )
    return _C_ops.divide_(x, y)


def floor_divide(x, y, name=None):
    """
    Floor divide two tensors element-wise and rounds the quotinents to the nearest integer toward zero. The equation is:

    .. math::
        out = trunc(x / y)

    - :math:`x`: Multidimensional Tensor.
    - :math:`y`: Multidimensional Tensor.

    Note:
        ``paddle.floor_divide`` supports broadcasting. If you want know more about broadcasting, please refer to `Introduction to Tensor`_ .

        .. _Introduction to Tensor: ../../guides/beginner/tensor_en.html#chapter5-broadcasting-of-tensor

        Also note that the name ``floor_divide`` can be misleading, as the quotinents are actually rounded toward zero, not toward negative infinite.

    Args:
        x (Tensor): the input tensor, it's data type should be uint8, int8, int32, int64, float32, float64, float16, bfloat16.
        y (Tensor): the input tensor, it's data type should be uint8, int8, int32, int64, float32, float64, float16, bfloat16.
        name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        N-D Tensor. A location into which the result is stored. It's dimension equals with $x$.

    Examples:

        .. code-block:: python

            >>> import paddle

            >>> x = paddle.to_tensor([2, 3, 8, 7])
            >>> y = paddle.to_tensor([1, 5, 3, 3])
            >>> z = paddle.floor_divide(x, y)
            >>> print(z)
            Tensor(shape=[4], dtype=int64, place=Place(cpu), stop_gradient=True,
            [2, 0, 2, 2])

    """
    if in_dynamic_mode():
        return _C_ops.floor_divide(x, y)
    else:
        return _elementwise_op(LayerHelper('elementwise_floordiv', **locals()))


@inplace_apis_in_dygraph_only
def floor_divide_(x, y, name=None):
    r"""
    Inplace version of ``floor_divide`` API, the output Tensor will be inplaced with input ``x``.
    Please refer to :ref:`api_paddle_floor_divide`.
    """
    out_shape = broadcast_shape(x.shape, y.shape)
    if out_shape != x.shape:
        raise ValueError(
            "The shape of broadcast output {} is different from that of inplace tensor {} in the Inplace operation.".format(
                out_shape, x.shape
            )
        )
    return _C_ops.floor_divide_(x, y)


def remainder(x, y, name=None):
    r"""
    Mod two tensors element-wise. The equation is:

    .. math::

        out = x \% y

    Note:
        ``paddle.remainder`` supports broadcasting. If you want know more about broadcasting, please refer to `Introduction to Tensor`_ .

        .. _Introduction to Tensor: ../../guides/beginner/tensor_en.html#chapter5-broadcasting-of-tensor

    Args:
        x (Tensor): the input tensor, it's data type should be float16, float32, float64, int32, int64.
        y (Tensor): the input tensor, it's data type should be float16, float32, float64, int32, int64.
        name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        N-D Tensor. A location into which the result is stored. If x, y have different shapes and are "broadcastable", the resulting tensor shape is the shape of x and y after broadcasting. If x, y have the same shape,  its shape is the same as x and y.

    Examples:

        .. code-block:: python

            >>> import paddle

            >>> x = paddle.to_tensor([2, 3, 8, 7])
            >>> y = paddle.to_tensor([1, 5, 3, 3])
            >>> z = paddle.remainder(x, y)
            >>> print(z)
            Tensor(shape=[4], dtype=int64, place=Place(cpu), stop_gradient=True,
            [0, 3, 2, 1])

    """
    if in_dynamic_mode():
        return _C_ops.remainder(x, y)
    else:
        return _elementwise_op(LayerHelper('elementwise_mod', **locals()))


@inplace_apis_in_dygraph_only
def remainder_(x, y, name=None):
    r"""
    Inplace version of ``remainder`` API, the output Tensor will be inplaced with input ``x``.
    Please refer to :ref:`api_paddle_remainder`.
    """
    out_shape = broadcast_shape(x.shape, y.shape)
    if out_shape != x.shape:
        raise ValueError(
            "The shape of broadcast output {} is different from that of inplace tensor {} in the Inplace operation.".format(
                out_shape, x.shape
            )
        )
    return _C_ops.remainder_(x, y)


mod = remainder  # noqa: F841
floor_mod = remainder  # noqa: F841
mod_ = remainder_  # noqa: F841
mod_.__doc__ = r"""
    Inplace version of ``mod`` API, the output Tensor will be inplaced with input ``x``.
    Please refer to :ref:`api_paddle_mod`.
    """
floor_mod_ = remainder_  # noqa: F841
floor_mod_.__doc__ = r"""
    Inplace version of ``floor_mod_`` API, the output Tensor will be inplaced with input ``x``.
    Please refer to :ref:`api_paddle_floor_mod_`.
    """


def multiply(x, y, name=None):
    """
    multiply two tensors element-wise. The equation is:

    .. math::
        out = x * y

    Note:
        ``paddle.multiply`` supports broadcasting. If you would like to know more about broadcasting, please refer to `Introduction to Tensor`_ .

        .. _Introduction to Tensor: ../../guides/beginner/tensor_en.html#chapter5-broadcasting-of-tensor

    Args:
        x (Tensor): the input tensor, its data type should be one of float32, float64, int32, int64, bool.
        y (Tensor): the input tensor, its data type should be one of float32, float64, int32, int64, bool.
        name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        N-D Tensor. A location into which the result is stored. If x, y have different shapes and are "broadcastable", the resulting tensor shape is the shape of x and y after broadcasting. If x, y have the same shape,  its shape is the same as x and y.

    Examples:

        .. code-block:: python

            >>> import paddle

            >>> x = paddle.to_tensor([[1, 2], [3, 4]])
            >>> y = paddle.to_tensor([[5, 6], [7, 8]])
            >>> res = paddle.multiply(x, y)
            >>> print(res)
            Tensor(shape=[2, 2], dtype=int64, place=Place(cpu), stop_gradient=True,
            [[5 , 12],
             [21, 32]])
            >>> x = paddle.to_tensor([[[1, 2, 3], [1, 2, 3]]])
            >>> y = paddle.to_tensor([2])
            >>> res = paddle.multiply(x, y)
            >>> print(res)
            Tensor(shape=[1, 2, 3], dtype=int64, place=Place(cpu), stop_gradient=True,
            [[[2, 4, 6],
              [2, 4, 6]]])

    """
    if in_dynamic_mode():
        return _C_ops.multiply(x, y)
    else:
        if x.dtype != y.dtype:
            raise TypeError(
                f'Input tensors must be same type, but received type of x: {x.dtype}, type of y: {y.dtype} '
            )

        return _elementwise_op(LayerHelper('elementwise_mul', **locals()))


@inplace_apis_in_dygraph_only
def multiply_(x, y, name=None):
    """
    Inplace version of ``multiply`` API, the output Tensor will be inplaced with input ``x``.
    Please refer to :ref:`api_paddle_multiply`.
    """

    out_shape = broadcast_shape(x.shape, y.shape)
    if out_shape != x.shape:
        raise ValueError(
            "The shape of broadcast output {} is different from that of inplace tensor {} in the Inplace operation.".format(
                out_shape, x.shape
            )
        )

    return _C_ops.multiply_(x, y)


@dygraph_only
def _elementwise_op_with_axis_in_dygraph(
    x, y, axis=-1, name=None, op_type="Undifined"
):
    assert (
        in_dynamic_mode()
    ), "You can only call `_elementwise_op_with_axis_in_dygraph` function within in_dynamic_mode"
    assert op_type in ["add", "subtract", "multiply", "divide"], (
        "op_name input error! _elementwise_op_with_axis is an inner function to replace elementwise_add/sub/mul/div. Input op_name=%s, Expect op_name=[add|subtract|multiply|divide]\n"
        % op_type
    )
    op = getattr(_C_ops, op_type)
    x_shape = list(x.shape)
    y_shape = list(y.shape)
    if axis == -1 or len(x_shape) == len(y_shape):
        return op(x, y)
    if len(x_shape) > len(y_shape):
        padding = len(x_shape) - len(y_shape) - axis
        y = paddle.reshape(y, [1] * axis + y_shape + [1] * padding)
    else:
        padding = len(y_shape) - len(x_shape) - axis
        x = paddle.reshape(x, [1] * axis + y_shape + [1] * padding)
    return op(x, y)


def _add_with_axis(x, y, axis=-1, name=None):
    # opt performance, only dynamic mode needs reshape
    if in_dynamic_mode():
        return _elementwise_op_with_axis_in_dygraph(x, y, axis, name, "add")
    else:
        op_type = 'elementwise_add'
        return _elementwise_op(LayerHelper(op_type, **locals()))


def _subtract_with_axis(x, y, axis=-1, name=None):
    # opt performance, only dynamic mode needs reshape
    if in_dynamic_mode():
        return _elementwise_op_with_axis_in_dygraph(
            x, y, axis, name, "subtract"
        )
    else:
        op_type = 'elementwise_sub'
        return _elementwise_op(LayerHelper(op_type, **locals()))


def _multiply_with_axis(x, y, axis=-1, name=None):
    # opt performance, only dynamic mode needs reshape
    if in_dynamic_mode():
        return _elementwise_op_with_axis_in_dygraph(
            x, y, axis, name, "multiply"
        )
    else:
        op_type = 'elementwise_mul'
        return _elementwise_op(LayerHelper(op_type, **locals()))


def _divide_with_axis(x, y, axis=-1, name=None):
    # opt performance, only dynamic mode needs reshape
    if in_dynamic_mode():
        return _elementwise_op_with_axis_in_dygraph(x, y, axis, name, "divide")
    else:
        op_type = 'elementwise_div'
        return _elementwise_op(LayerHelper(op_type, **locals()))


def maximum(x, y, name=None):
    """
    Compare two tensors and returns a new tensor containing the element-wise maxima. The equation is:

    .. math::
        out = max(x, y)

    Note:
        ``paddle.maximum`` supports broadcasting. If you want know more about broadcasting, please refer to  `Introduction to Tensor`_ .

        .. _Introduction to Tensor: ../../guides/beginner/tensor_en.html#chapter5-broadcasting-of-tensor

    Args:
        x (Tensor): the input tensor, it's data type should be float32, float64, int32, int64.
        y (Tensor): the input tensor, it's data type should be float32, float64, int32, int64.
        name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        N-D Tensor. A location into which the result is stored. If x, y have different shapes and are "broadcastable", the resulting tensor shape is the shape of x and y after broadcasting. If x, y have the same shape,  its shape is the same as x and y.

    Examples:

        .. code-block:: python

            >>> import paddle

            >>> x = paddle.to_tensor([[1, 2], [7, 8]])
            >>> y = paddle.to_tensor([[3, 4], [5, 6]])
            >>> res = paddle.maximum(x, y)
            >>> print(res)
            Tensor(shape=[2, 2], dtype=int64, place=Place(cpu), stop_gradient=True,
            [[3, 4],
             [7, 8]])

            >>> x = paddle.to_tensor([[1, 2, 3], [1, 2, 3]])
            >>> y = paddle.to_tensor([3, 0, 4])
            >>> res = paddle.maximum(x, y)
            >>> print(res)
            Tensor(shape=[2, 3], dtype=int64, place=Place(cpu), stop_gradient=True,
            [[3, 2, 4],
             [3, 2, 4]])

            >>> x = paddle.to_tensor([2, 3, 5], dtype='float32')
            >>> y = paddle.to_tensor([1, float("nan"), float("nan")], dtype='float32')
            >>> res = paddle.maximum(x, y)
            >>> print(res)
            Tensor(shape=[3], dtype=float32, place=Place(cpu), stop_gradient=True,
            [2. , nan, nan])

            >>> x = paddle.to_tensor([5, 3, float("inf")], dtype='float32')
            >>> y = paddle.to_tensor([1, -float("inf"), 5], dtype='float32')
            >>> res = paddle.maximum(x, y)
            >>> print(res)
            Tensor(shape=[3], dtype=float32, place=Place(cpu), stop_gradient=True,
            [5.  , 3.  , inf.])
    """
    if in_dynamic_mode():
        return _C_ops.maximum(x, y)
    else:
        return _elementwise_op(LayerHelper('elementwise_max', **locals()))


def minimum(x, y, name=None):
    """
    Compare two tensors and return a new tensor containing the element-wise minima. The equation is:

    .. math::
        out = min(x, y)

    Note:
        ``paddle.minimum`` supports broadcasting. If you want know more about broadcasting, please refer to `Introduction to Tensor`_ .

        .. _Introduction to Tensor: ../../guides/beginner/tensor_en.html#chapter5-broadcasting-of-tensor

    Args:
        x (Tensor): the input tensor, it's data type should be float32, float64, int32, int64.
        y (Tensor): the input tensor, it's data type should be float32, float64, int32, int64.
        name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        Tensor. If x, y have different shapes and are "broadcastable", the resulting tensor shape is the shape of x and y after broadcasting. If x, y have the same shape,  its shape is the same as x and y.

    Examples:

        .. code-block:: python

            >>> import paddle

            >>> x = paddle.to_tensor([[1, 2], [7, 8]])
            >>> y = paddle.to_tensor([[3, 4], [5, 6]])
            >>> res = paddle.minimum(x, y)
            >>> print(res)
            Tensor(shape=[2, 2], dtype=int64, place=Place(cpu), stop_gradient=True,
            [[1, 2],
             [5, 6]])

            >>> x = paddle.to_tensor([[[1, 2, 3], [1, 2, 3]]])
            >>> y = paddle.to_tensor([3, 0, 4])
            >>> res = paddle.minimum(x, y)
            >>> print(res)
            Tensor(shape=[1, 2, 3], dtype=int64, place=Place(cpu), stop_gradient=True,
            [[[1, 0, 3],
              [1, 0, 3]]])

            >>> x = paddle.to_tensor([2, 3, 5], dtype='float32')
            >>> y = paddle.to_tensor([1, float("nan"), float("nan")], dtype='float32')
            >>> res = paddle.minimum(x, y)
            >>> print(res)
            Tensor(shape=[3], dtype=float32, place=Place(cpu), stop_gradient=True,
            [1. , nan, nan])

            >>> x = paddle.to_tensor([5, 3, float("inf")], dtype='float64')
            >>> y = paddle.to_tensor([1, -float("inf"), 5], dtype='float64')
            >>> res = paddle.minimum(x, y)
            >>> print(res)
            Tensor(shape=[3], dtype=float64, place=Place(cpu), stop_gradient=True,
            [ 1.  , -inf.,  5.  ])
    """
    if in_dynamic_mode():
        return _C_ops.minimum(x, y)
    else:
        return _elementwise_op(LayerHelper('elementwise_min', **locals()))


def fmax(x, y, name=None):
    """
    Compares the elements at the corresponding positions of the two tensors and returns a new tensor containing the maximum value of the element.
    If one of them is a nan value, the other value is directly returned, if both are nan values, then the first nan value is returned.
    The equation is:

    .. math::
        out = fmax(x, y)

    Note:
        ``paddle.fmax`` supports broadcasting. If you want know more about broadcasting, please refer to `Introduction to Tensor`_ .

        .. _Introduction to Tensor: ../../guides/beginner/tensor_en.html#chapter5-broadcasting-of-tensor

    Args:
        x (Tensor): the input tensor, it's data type should be float16, float32, float64, int32, int64.
        y (Tensor): the input tensor, it's data type should be float16, float32, float64, int32, int64.
        name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        N-D Tensor. A location into which the result is stored. If x, y have different shapes and are "broadcastable", the resulting tensor shape is the shape of x and y after broadcasting. If x, y have the same shape,  its shape is the same as x and y.

    Examples:

        .. code-block:: python

            >>> import paddle

            >>> x = paddle.to_tensor([[1, 2], [7, 8]])
            >>> y = paddle.to_tensor([[3, 4], [5, 6]])
            >>> res = paddle.fmax(x, y)
            >>> print(res)
            Tensor(shape=[2, 2], dtype=int64, place=Place(cpu), stop_gradient=True,
            [[3, 4],
             [7, 8]])

            >>> x = paddle.to_tensor([[1, 2, 3], [1, 2, 3]])
            >>> y = paddle.to_tensor([3, 0, 4])
            >>> res = paddle.fmax(x, y)
            >>> print(res)
            Tensor(shape=[2, 3], dtype=int64, place=Place(cpu), stop_gradient=True,
            [[3, 2, 4],
             [3, 2, 4]])

            >>> x = paddle.to_tensor([2, 3, 5], dtype='float32')
            >>> y = paddle.to_tensor([1, float("nan"), float("nan")], dtype='float32')
            >>> res = paddle.fmax(x, y)
            >>> print(res)
            Tensor(shape=[3], dtype=float32, place=Place(cpu), stop_gradient=True,
            [2., 3., 5.])

            >>> x = paddle.to_tensor([5, 3, float("inf")], dtype='float32')
            >>> y = paddle.to_tensor([1, -float("inf"), 5], dtype='float32')
            >>> res = paddle.fmax(x, y)
            >>> print(res)
            Tensor(shape=[3], dtype=float32, place=Place(cpu), stop_gradient=True,
            [5.  , 3.  , inf.])
    """
    if in_dynamic_mode():
        return _C_ops.fmax(x, y)
    else:
        return _elementwise_op(LayerHelper('elementwise_fmax', **locals()))


def fmin(x, y, name=None):
    """
    Compares the elements at the corresponding positions of the two tensors and returns a new tensor containing the minimum value of the element.
    If one of them is a nan value, the other value is directly returned, if both are nan values, then the first nan value is returned.
    The equation is:

    .. math::
        out = fmin(x, y)

    Note:
        ``paddle.fmin`` supports broadcasting. If you want know more about broadcasting, please refer to `Introduction to Tensor`_ .

        .. _Introduction to Tensor: ../../guides/beginner/tensor_en.html#chapter5-broadcasting-of-tensor

    Args:
        x (Tensor): the input tensor, it's data type should be float16, float32, float64, int32, int64.
        y (Tensor): the input tensor, it's data type should be float16, float32, float64, int32, int64.
        name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        N-D Tensor. A location into which the result is stored. If x, y have different shapes and are "broadcastable", the resulting tensor shape is the shape of x and y after broadcasting. If x, y have the same shape,  its shape is the same as x and y.

    Examples:

        .. code-block:: python

            >>> import paddle

            >>> x = paddle.to_tensor([[1, 2], [7, 8]])
            >>> y = paddle.to_tensor([[3, 4], [5, 6]])
            >>> res = paddle.fmin(x, y)
            >>> print(res)
            Tensor(shape=[2, 2], dtype=int64, place=Place(cpu), stop_gradient=True,
            [[1, 2],
             [5, 6]])

            >>> x = paddle.to_tensor([[[1, 2, 3], [1, 2, 3]]])
            >>> y = paddle.to_tensor([3, 0, 4])
            >>> res = paddle.fmin(x, y)
            >>> print(res)
            Tensor(shape=[1, 2, 3], dtype=int64, place=Place(cpu), stop_gradient=True,
            [[[1, 0, 3],
              [1, 0, 3]]])

            >>> x = paddle.to_tensor([2, 3, 5], dtype='float32')
            >>> y = paddle.to_tensor([1, float("nan"), float("nan")], dtype='float32')
            >>> res = paddle.fmin(x, y)
            >>> print(res)
            Tensor(shape=[3], dtype=float32, place=Place(cpu), stop_gradient=True,
            [1., 3., 5.])

            >>> x = paddle.to_tensor([5, 3, float("inf")], dtype='float64')
            >>> y = paddle.to_tensor([1, -float("inf"), 5], dtype='float64')
            >>> res = paddle.fmin(x, y)
            >>> print(res)
            Tensor(shape=[3], dtype=float64, place=Place(cpu), stop_gradient=True,
            [ 1.  , -inf.,  5.  ])
    """
    if in_dynamic_mode():
        return _C_ops.fmin(x, y)
    else:
        return _elementwise_op(LayerHelper('elementwise_fmin', **locals()))


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

            >>> import paddle

            >>> # x is a Tensor with following elements:
            >>> #    [[0.2, 0.3, 0.5, 0.9]
            >>> #     [0.1, 0.2, 0.6, 0.7]]
            >>> # Each example is followed by the corresponding output tensor.
            >>> x = paddle.to_tensor([[0.2, 0.3, 0.5, 0.9],
            ...                       [0.1, 0.2, 0.6, 0.7]])
            >>> out1 = paddle.sum(x)
            >>> out1
            Tensor(shape=[], dtype=float32, place=Place(cpu), stop_gradient=True,
            3.50000000)
            >>> out2 = paddle.sum(x, axis=0)
            >>> out2
            Tensor(shape=[4], dtype=float32, place=Place(cpu), stop_gradient=True,
            [0.30000001, 0.50000000, 1.10000002, 1.59999990])
            >>> out3 = paddle.sum(x, axis=-1)
            >>> out3
            Tensor(shape=[2], dtype=float32, place=Place(cpu), stop_gradient=True,
            [1.89999998, 1.60000002])
            >>> out4 = paddle.sum(x, axis=1, keepdim=True)
            >>> out4
            Tensor(shape=[2, 1], dtype=float32, place=Place(cpu), stop_gradient=True,
            [[1.89999998],
             [1.60000002]])

            >>> # y is a Tensor with shape [2, 2, 2] and elements as below:
            >>> #      [[[1, 2], [3, 4]],
            >>> #      [[5, 6], [7, 8]]]
            >>> # Each example is followed by the corresponding output tensor.
            >>> y = paddle.to_tensor([[[1, 2], [3, 4]],
            ...                       [[5, 6], [7, 8]]])
            >>> out5 = paddle.sum(y, axis=[1, 2])
            >>> out5
            Tensor(shape=[2], dtype=int64, place=Place(cpu), stop_gradient=True,
            [10, 26])
            >>> out6 = paddle.sum(y, axis=[0, 1])
            >>> out6
            Tensor(shape=[2], dtype=int64, place=Place(cpu), stop_gradient=True,
            [16, 20])

            >>> # x is a Tensor with following elements:
            >>> #    [[True, True, True, True]
            >>> #     [False, False, False, False]]
            >>> # Each example is followed by the corresponding output tensor.
            >>> x = paddle.to_tensor([[True, True, True, True],
            ...                       [False, False, False, False]])
            >>> out7 = paddle.sum(x)
            >>> out7
            Tensor(shape=[], dtype=int64, place=Place(cpu), stop_gradient=True,
            4)
            >>> out8 = paddle.sum(x, axis=0)
            >>> out8
            Tensor(shape=[4], dtype=int64, place=Place(cpu), stop_gradient=True,
            [1, 1, 1, 1])
            >>> out9 = paddle.sum(x, axis=1)
            >>> out9
            Tensor(shape=[2], dtype=int64, place=Place(cpu), stop_gradient=True,
            [4, 0])
    """

    dtype_flag = False
    if dtype is not None:
        if paddle.ir.core._use_new_ir_api():
            dtype = paddle.ir.core.convert_np_dtype_to_dtype_(dtype)
        else:
            dtype_flag = True
            dtype = convert_np_dtype_to_dtype_(dtype)

    if in_dynamic_mode():
        return _C_ops.sum(x, axis, dtype, keepdim)
    else:
        if paddle.ir.core._use_new_ir_api():
            return paddle._ir_ops.sum(x, axis, dtype, keepdim)
        reduce_all, axis = _get_reduce_axis_with_tensor(axis, x)
        attrs = {'dim': axis, 'keep_dim': keepdim, 'reduce_all': reduce_all}

        if dtype_flag:
            attrs.update({'in_dtype': x.dtype, 'out_dtype': dtype})

        check_variable_and_dtype(
            x,
            'x',
            [
                'bool',
                'uint16',
                'float16',
                'float32',
                'float64',
                'int16',
                'int32',
                'int64',
                'complex64',
                'complex128',
            ],
            'sum',
        )

        check_type(
            axis, 'axis', (int, list, tuple, type(None), Variable), 'sum'
        )

        helper = LayerHelper('sum', **locals())
        if dtype_flag:
            out = helper.create_variable_for_type_inference(dtype=dtype)
        else:
            out = helper.create_variable_for_type_inference(dtype=x.dtype)
        helper.append_op(
            type='reduce_sum',
            inputs={'X': x},
            outputs={'Out': out},
            attrs=attrs,
        )
        return out


def nan_to_num(x, nan=0.0, posinf=None, neginf=None, name=None):
    """
    Replaces NaN, positive infinity, and negative infinity values in input tensor.

    Args:
        x (Tensor): An N-D Tensor, the data type is float32, float64.
        nan (float, optional): the value to replace NaNs with. Default is 0.
        posinf (float, optional): if a Number, the value to replace positive infinity values with. If None, positive infinity values are replaced with the greatest finite value representable by input’s dtype. Default is None.
        neginf (float, optional): if a Number, the value to replace negative infinity values with. If None, negative infinity values are replaced with the lowest finite value representable by input’s dtype. Default is None.
        name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        Tensor: Results of nan_to_num operation input Tensor ``x``.

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> x = paddle.to_tensor([float('nan'), 0.3, float('+inf'), float('-inf')], dtype='float32')
            >>> out1 = paddle.nan_to_num(x)
            >>> out1
            Tensor(shape=[4], dtype=float32, place=Place(cpu), stop_gradient=True,
            [ 0.                                      ,
              0.30000001                              ,
              340282346638528859811704183484516925440.,
             -340282346638528859811704183484516925440.])
            >>> out2 = paddle.nan_to_num(x, nan=1)
            >>> out2
            Tensor(shape=[4], dtype=float32, place=Place(cpu), stop_gradient=True,
            [ 1.                                      ,
              0.30000001                              ,
              340282346638528859811704183484516925440.,
             -340282346638528859811704183484516925440.])
            >>> out3 = paddle.nan_to_num(x, posinf=5)
            >>> out3
            Tensor(shape=[4], dtype=float32, place=Place(cpu), stop_gradient=True,
            [ 0.                                      ,
              0.30000001                              ,
              5.                                      ,
             -340282346638528859811704183484516925440.])
            >>> out4 = paddle.nan_to_num(x, nan=10, neginf=-99)
            >>> out4
            Tensor(shape=[4], dtype=float32, place=Place(cpu), stop_gradient=True,
            [ 10.                                    ,
              0.30000001                             ,
             340282346638528859811704183484516925440.,
             -99.                                    ])
    """
    # NOTE(tiancaishaonvjituizi): it seems that paddle handles the dtype of python float number
    # incorrectly, so we have to explicitly contruct tensors here
    posinf_value = paddle.full_like(x, float("+inf"))
    neginf_value = paddle.full_like(x, float("-inf"))
    nan = paddle.full_like(x, nan)
    assert x.dtype in [paddle.float32, paddle.float64]
    is_float32 = x.dtype == paddle.float32
    if posinf is None:
        posinf = (
            np.finfo(np.float32).max if is_float32 else np.finfo(np.float64).max
        )
    posinf = paddle.full_like(x, posinf)
    if neginf is None:
        neginf = (
            np.finfo(np.float32).min if is_float32 else np.finfo(np.float64).min
        )
    neginf = paddle.full_like(x, neginf)
    x = paddle.where(paddle.isnan(x), nan, x)
    x = paddle.where(x == posinf_value, posinf, x)
    x = paddle.where(x == neginf_value, neginf, x)
    return x


@inplace_apis_in_dygraph_only
def nan_to_num_(x, nan=0.0, posinf=None, neginf=None, name=None):
    r"""
    Inplace version of ``nan_to_num`` API, the output Tensor will be inplaced with input ``x``.
    Please refer to :ref:`api_paddle_nan_to_num`.
    """
    # NOTE(tiancaishaonvjituizi): it seems that paddle handles the dtype of python float number
    # incorrectly, so we have to explicitly contruct tensors here
    posinf_value = paddle.full_like(x, float("+inf"))
    neginf_value = paddle.full_like(x, float("-inf"))
    nan = paddle.full_like(x, nan)
    assert x.dtype in [paddle.float32, paddle.float64]
    is_float32 = x.dtype == paddle.float32
    if posinf is None:
        posinf = (
            np.finfo(np.float32).max if is_float32 else np.finfo(np.float64).max
        )
    posinf = paddle.full_like(x, posinf)
    if neginf is None:
        neginf = (
            np.finfo(np.float32).min if is_float32 else np.finfo(np.float64).min
        )
    neginf = paddle.full_like(x, neginf)
    x_not_nan = paddle.logical_not(paddle.isnan(x))
    x = paddle.where_(x_not_nan, x, nan)
    x = paddle.where_(x != posinf_value, x, posinf)
    x = paddle.where_(x != neginf_value, x, neginf)
    return x


def nansum(x, axis=None, dtype=None, keepdim=False, name=None):
    """
    Computes the sum of tensor elements over the given axis, treating Not a Numbers (NaNs) as zero.

    Args:
        x (Tensor): An N-D Tensor, the data type is float16, float32, float64, int32 or int64.
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

            >>> import paddle

            >>> # x is a Tensor with following elements:
            >>> #    [[nan, 0.3, 0.5, 0.9]
            >>> #     [0.1, 0.2, -nan, 0.7]]
            >>> # Each example is followed by the corresponding output tensor.
            >>> x = paddle.to_tensor([[float('nan'), 0.3, 0.5, 0.9],
            ...                       [0.1, 0.2, float('-nan'), 0.7]],dtype="float32")
            >>> out1 = paddle.nansum(x)
            >>> out1
            Tensor(shape=[], dtype=float32, place=Place(cpu), stop_gradient=True,
            2.69999981)
            >>> out2 = paddle.nansum(x, axis=0)
            >>> out2
            Tensor(shape=[4], dtype=float32, place=Place(cpu), stop_gradient=True,
            [0.10000000, 0.50000000, 0.50000000, 1.59999990])
            >>> out3 = paddle.nansum(x, axis=-1)
            >>> out3
            Tensor(shape=[2], dtype=float32, place=Place(cpu), stop_gradient=True,
            [1.70000005, 1.        ])
            >>> out4 = paddle.nansum(x, axis=1, keepdim=True)
            >>> out4
            Tensor(shape=[2, 1], dtype=float32, place=Place(cpu), stop_gradient=True,
            [[1.70000005],
             [1.        ]])

            >>> # y is a Tensor with shape [2, 2, 2] and elements as below:
            >>> #      [[[1, nan], [3, 4]],
            >>> #       [[5, 6], [-nan, 8]]]
            >>> # Each example is followed by the corresponding output tensor.
            >>> y = paddle.to_tensor([[[1, float('nan')], [3, 4]],
            ...                       [[5, 6], [float('-nan'), 8]]])
            >>> out5 = paddle.nansum(y, axis=[1, 2])
            >>> out5
            Tensor(shape=[2], dtype=float32, place=Place(cpu), stop_gradient=True,
            [8. , 19.])
            >>> out6 = paddle.nansum(y, axis=[0, 1])
            >>> out6
            Tensor(shape=[2], dtype=float32, place=Place(cpu), stop_gradient=True,
            [9. , 18.])
    """
    check_variable_and_dtype(
        x, 'x', ['float16', 'float32', 'float64', 'int32', 'int64'], 'nansum'
    )
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

            >>> import paddle
            >>> # x is a 2-D Tensor:
            >>> x = paddle.to_tensor([[float('nan'), 0.3, 0.5, 0.9],
            ...                       [0.1, 0.2, float('-nan'), 0.7]])
            >>> out1 = paddle.nanmean(x)
            >>> out1
            Tensor(shape=[], dtype=float32, place=Place(cpu), stop_gradient=True,
            0.44999996)
            >>> out2 = paddle.nanmean(x, axis=0)
            >>> out2
            Tensor(shape=[4], dtype=float32, place=Place(cpu), stop_gradient=True,
            [0.10000000, 0.25000000, 0.50000000, 0.79999995])
            >>> out3 = paddle.nanmean(x, axis=0, keepdim=True)
            >>> out3
            Tensor(shape=[1, 4], dtype=float32, place=Place(cpu), stop_gradient=True,
            [[0.10000000, 0.25000000, 0.50000000, 0.79999995]])
            >>> out4 = paddle.nanmean(x, axis=1)
            >>> out4
            Tensor(shape=[2], dtype=float32, place=Place(cpu), stop_gradient=True,
            [0.56666666, 0.33333334])
            >>> out5 = paddle.nanmean(x, axis=1, keepdim=True)
            >>> out5
            Tensor(shape=[2, 1], dtype=float32, place=Place(cpu), stop_gradient=True,
            [[0.56666666],
             [0.33333334]])

            >>> # y is a 3-D Tensor:
            >>> y = paddle.to_tensor([[[1, float('nan')], [3, 4]],
            ...                       [[5, 6], [float('-nan'), 8]]])
            >>> out6 = paddle.nanmean(y, axis=[1, 2])
            >>> out6
            Tensor(shape=[2], dtype=float32, place=Place(cpu), stop_gradient=True,
            [2.66666675, 6.33333349])
            >>> out7 = paddle.nanmean(y, axis=[0, 1])
            >>> out7
            Tensor(shape=[2], dtype=float32, place=Place(cpu), stop_gradient=True,
            [3., 6.])
    """
    if isinstance(axis, int):
        axis = [axis]
    check_variable_and_dtype(
        x, 'x/input', ['uint16', 'float16', 'float32', 'float64'], 'nanmean'
    )
    if axis is not None:
        check_type(axis, 'axis/dim', (int, list, tuple), 'nanmean')

    cnt = paddle.sum(~paddle.isnan(x), axis=axis, keepdim=keepdim)
    return paddle.divide(
        paddle.nansum(x, axis=axis, keepdim=keepdim, name=name),
        cnt.astype(x.dtype),
    )


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

            >>> import paddle
            >>> # x is a 2-D Tensor:
            >>> x = paddle.to_tensor([[0., 1.1, 1.2], [0., 0., 1.3], [0., 0., 0.]])
            >>> out1 = paddle.count_nonzero(x)
            >>> out1
            Tensor(shape=[], dtype=int64, place=Place(cpu), stop_gradient=True,
            3)
            >>> out2 = paddle.count_nonzero(x, axis=0)
            >>> out2
            Tensor(shape=[3], dtype=int64, place=Place(cpu), stop_gradient=True,
            [0, 1, 2])
            >>> out3 = paddle.count_nonzero(x, axis=0, keepdim=True)
            >>> out3
            Tensor(shape=[1, 3], dtype=int64, place=Place(cpu), stop_gradient=True,
            [[0, 1, 2]])
            >>> out4 = paddle.count_nonzero(x, axis=1)
            >>> out4
            Tensor(shape=[3], dtype=int64, place=Place(cpu), stop_gradient=True,
            [2, 1, 0])
            >>> out5 = paddle.count_nonzero(x, axis=1, keepdim=True)
            >>> out5
            Tensor(shape=[3, 1], dtype=int64, place=Place(cpu), stop_gradient=True,
            [[2],
             [1],
             [0]])

            >>> # y is a 3-D Tensor:
            >>> y = paddle.to_tensor([[[0., 1.1, 1.2], [0., 0., 1.3], [0., 0., 0.]],
            ...                         [[0., 2.5, 2.6], [0., 0., 2.4], [2.1, 2.2, 2.3]]])
            >>> out6 = paddle.count_nonzero(y, axis=[1, 2])
            >>> out6
            Tensor(shape=[2], dtype=int64, place=Place(cpu), stop_gradient=True,
            [3, 6])
            >>> out7 = paddle.count_nonzero(y, axis=[0, 1])
            >>> out7
            Tensor(shape=[3], dtype=int64, place=Place(cpu), stop_gradient=True,
            [1, 3, 5])
    """

    if isinstance(axis, int):
        axis = [axis]

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

            >>> import paddle

            >>> input0 = paddle.to_tensor([[1, 2, 3], [4, 5, 6]], dtype='float32')
            >>> input1 = paddle.to_tensor([[7, 8, 9], [10, 11, 12]], dtype='float32')
            >>> output = paddle.add_n([input0, input1])
            >>> output
            Tensor(shape=[2, 3], dtype=float32, place=Place(cpu), stop_gradient=True,
            [[8. , 10., 12.],
             [14., 16., 18.]])
    """
    if in_dynamic_mode():
        if isinstance(inputs, Variable):
            inputs = [inputs]
        return _C_ops.add_n(inputs)
    else:
        if paddle.ir.core._use_new_ir_api():
            return paddle._ir_ops.add_n(inputs)

        helper = LayerHelper('add_n', **locals())
        check_type(inputs, 'inputs', (Variable, tuple, list), 'add_n')
        if isinstance(inputs, (list, tuple)):
            if len(inputs) > 0:
                for input in inputs:
                    check_variable_and_dtype(
                        input,
                        "inputs",
                        [
                            'float16',
                            'float32',
                            'float64',
                            'int32',
                            'int64',
                            'uint16',
                        ],
                        'add_n',
                    )
        else:
            check_variable_and_dtype(
                inputs,
                "inputs",
                ['float16', 'float32', 'float64', 'int32', 'int64', 'uint16'],
                'add_n',
            )

        out = helper.create_variable_for_type_inference(
            dtype=helper.input_dtype('inputs')
        )
        helper.append_op(
            type='sum',
            inputs={'X': inputs},
            outputs={'Out': out},
            attrs={'use_mkldnn': False},
        )

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

            >>> import paddle

            >>> input = paddle.to_tensor([[0.1, 1.5], [-0.2, -2.4]], 'float32')
            >>> output = paddle.trunc(input)
            >>> output
            Tensor(shape=[2, 2], dtype=float32, place=Place(cpu), stop_gradient=True,
            [[ 0.,  1.],
             [-0., -2.]])
    '''
    if in_dynamic_mode():
        return _C_ops.trunc(input)
    else:
        inputs = {"X": input}
        attrs = {}

        helper = LayerHelper("trunc", **locals())
        check_variable_and_dtype(
            input, 'X', ['int32', 'int64', 'float32', 'float64'], 'trunc'
        )
        out = helper.create_variable_for_type_inference(dtype=input.dtype)

        helper.append_op(
            type="trunc", inputs=inputs, attrs=attrs, outputs={"Out": out}
        )
        return out


@inplace_apis_in_dygraph_only
def trunc_(input, name=None):
    r"""
    Inplace version of ``trunc`` API, the output Tensor will be inplaced with input ``x``.
    Please refer to :ref:`api_paddle_trunc`.
    """
    if in_dynamic_mode():
        return _C_ops.trunc_(input)


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

            >>> import paddle
            >>> input = paddle.arange(1, 7).reshape((3, 2)).astype('float32')
            >>> mat2 = paddle.arange(1, 9).reshape((2, 4)).astype('float32')
            >>> out = paddle.mm(input, mat2)
            >>> out
            Tensor(shape=[3, 4], dtype=float32, place=Place(cpu), stop_gradient=True,
            [[11., 14., 17., 20.],
             [23., 30., 37., 44.],
             [35., 46., 57., 68.]])


    """
    if in_dynamic_mode():
        return _C_ops.matmul(input, mat2, False, False)
    else:

        def __check_input(x, y):
            var_names = {'x': x, 'y': y}
            for name, val in var_names.items():
                check_variable_and_dtype(
                    val, name, ['float16', 'float32', 'float64'], 'mm'
                )
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
                        "prerequisites. But received X's shape: {}, Y's shape: {}\n".format(
                            x_shape, y_shape
                        )
                    )

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
                            "Y's shape: %s.\n" % (i, i, x_shape, y_shape)
                        )

        __check_input(input, mat2)

        helper = LayerHelper('mm', **locals())
        out = helper.create_variable_for_type_inference(dtype=input.dtype)
        helper.append_op(
            type='matmul_v2',
            inputs={'X': input, 'Y': mat2},
            outputs={'Out': out},
        )
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
        .. code-block:: python

            >>> import paddle

            >>> x = paddle.ones([2, 2])
            >>> y = paddle.ones([2, 2])
            >>> input = paddle.ones([2, 2])

            >>> out = paddle.addmm(input=input, x=x, y=y, beta=0.5, alpha=5.0)

            >>> print(out)
            Tensor(shape=[2, 2], dtype=float32, place=Place(cpu), stop_gradient=True,
            [[10.50000000, 10.50000000],
             [10.50000000, 10.50000000]])
    """
    input_shape = input.shape
    x_shape = x.shape
    y_shape = y.shape
    if not len(x_shape) == len(y_shape) == 2:
        raise ValueError(
            "The dimention of x, y should be 2 but receive x's shape: {}, y's shape: {}".format(
                x_shape, y_shape
            )
        )
    if x_shape[1] != y_shape[0]:
        raise ValueError(
            "The input Variable x's width must be equal with Variable y' height. But received x's shape = {}, y's shape = {}.".format(
                x_shape, y_shape
            )
        )
    if len(input_shape) == 2:
        if input_shape[0] != x_shape[0]:
            if input_shape[0] != 1:
                raise ValueError(
                    "When x's dimension[0] is not equal with input's dimension[0], input's dimension[0] must be 1 but got {}".format(
                        input_shape[0]
                    )
                )
            if input_shape[1] != y_shape[1] and input_shape[1] != 1:
                raise ValueError(
                    "When y's dimension[1] is not equal with input's dimension[1], input's dimension[1] must be 1 but got {}".format(
                        input_shape[1]
                    )
                )
        if input_shape[1] != y_shape[1]:
            if input_shape[1] != 1:
                raise ValueError(
                    "When y's dimension[1] is not equal with input's dimension[1], input's dimension[1] must be 1 but got {}".format(
                        input_shape[1]
                    )
                )
    elif len(input_shape) == 1:
        if input_shape[0] not in (y_shape[1], 1):
            raise ValueError(
                "The input's shape: {} is not broadcastable with [x.shape[0], y.shape[1]]: [{},{}]".format(
                    input_shape, x_shape[0], y_shape[1]
                )
            )
    else:
        raise ValueError(
            "The dimention of input should be 2 or 1 but receive input's shape: {}".format(
                input_shape
            )
        )

    if in_dynamic_mode():
        return _C_ops.addmm(input, x, y, beta, alpha)
    else:
        inputs = {'Input': input, "X": x, "Y": y}
        attrs = {'Alpha': alpha, 'Beta': beta}

        helper = LayerHelper("addmm", **locals())
        check_variable_and_dtype(
            input, 'Input', ['float16', 'float32', 'float64', 'uint16'], 'addmm'
        )
        check_variable_and_dtype(
            x, 'X', ['float16', 'float32', 'float64', 'uint16'], 'addmm'
        )
        check_variable_and_dtype(
            y, 'Y', ['float16', 'float32', 'float64', 'uint16'], 'addmm'
        )
        out = helper.create_variable_for_type_inference(dtype=x.dtype)

        helper.append_op(
            type="addmm", inputs=inputs, attrs=attrs, outputs={"Out": out}
        )
        return out


@inplace_apis_in_dygraph_only
def addmm_(input, x, y, beta=1.0, alpha=1.0, name=None):
    """
    Inplace version of ``addmm`` API, the output Tensor will be inplaced with input ``x``.
    Please refer to :ref:`api_paddle_addmm`.
    """
    input_shape = input.shape
    x_shape = x.shape
    y_shape = y.shape
    if not len(x_shape) == len(y_shape) == 2:
        raise ValueError(
            "The dimention of x, y should be 2 but receive x's shape: {}, y's shape: {}".format(
                x_shape, y_shape
            )
        )
    if x_shape[1] != y_shape[0]:
        raise ValueError(
            "The input Variable x's width must be equal with Variable y' height. But received x's shape = {}, y's shape = {}.".format(
                x_shape, y_shape
            )
        )
    if len(input_shape) == 2:
        if input_shape[0] != x_shape[0]:
            if input_shape[0] != 1:
                raise ValueError(
                    "When x's dimension[0] is not equal with input's dimension[0], input's dimension[0] must be 1 but got {}".format(
                        input_shape[0]
                    )
                )
            if input_shape[1] != y_shape[1] and input_shape[1] != 1:
                raise ValueError(
                    "When y's dimension[1] is not equal with input's dimension[1], input's dimension[1] must be 1 but got {}".format(
                        input_shape[1]
                    )
                )
        if input_shape[1] != y_shape[1]:
            if input_shape[1] != 1:
                raise ValueError(
                    "When y's dimension[1] is not equal with input's dimension[1], input's dimension[1] must be 1 but got {}".format(
                        input_shape[1]
                    )
                )
    elif len(input_shape) == 1:
        if input_shape[0] not in (y_shape[1], 1):
            raise ValueError(
                "The input's shape: {} is not broadcastable with [x.shape[0], y.shape[1]]: [{},{}]".format(
                    input_shape, x_shape[0], y_shape[1]
                )
            )
    else:
        raise ValueError(
            "The dimention of input should be 2 or 1 but receive input's shape: {}".format(
                input_shape
            )
        )

    if in_dynamic_mode():
        return _C_ops.addmm_(input, x, y, beta, alpha)


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
        .. code-block:: python

            >>> import paddle
            >>> input = [[[2.0, 2, -2], [3, 0.3, 3]],
            ...          [[2, -8, 2],   [3.1, 3.7, 3]]]
            >>> x = paddle.to_tensor(input,dtype='float32')
            >>> y = paddle.renorm(x, 1.0, 2, 2.05)
            >>> print(y)
            Tensor(shape=[2, 2, 3], dtype=float32, place=Place(cpu), stop_gradient=True,
            [[[ 0.40594056,  0.29285714, -0.41000000],
              [ 0.60891086,  0.04392857,  0.61500001]],
             [[ 0.40594056, -1.17142856,  0.41000000],
              [ 0.62920785,  0.54178572,  0.61500001]]])

    """
    input_shape = x.shape
    if not axis < len(input_shape):
        raise ValueError(
            "the axis:{} should be less then the shape's size {}:{}".format(
                axis, len(input_shape), input_shape
            )
        )
    if not axis >= 0:
        if not axis >= -1 * len(input_shape):
            raise ValueError(
                "the axis:{} should not be less than -1 * length of input_shape:{}".format(
                    axis, -1 * len(input_shape)
                )
            )
        axis = axis + len(input_shape)
    if in_dynamic_mode():
        out = _C_ops.renorm(x, p, axis, max_norm)
        return out
    else:
        check_variable_and_dtype(x, 'x', ['float32', 'float64'], 'renorm')
        inputs = {'X': x}
        attrs = {'p': p, 'axis': axis, 'max_norm': max_norm}

        helper = LayerHelper("renorm", **locals())
        out = helper.create_variable_for_type_inference(dtype=x.dtype)

        helper.append_op(
            type="renorm", inputs=inputs, attrs=attrs, outputs={"Out": out}
        )
        return out


@inplace_apis_in_dygraph_only
def renorm_(x, p, axis, max_norm):
    """
    Inplace version of ``renorm`` API, the output Tensor will be inplaced with input ``x``.
    Please refer to :ref:`api_paddle_renorm`.
    """
    input_shape = x.shape
    if not axis < len(input_shape):
        raise ValueError(
            "the axis:{} should be less then the shape's size {}:{}".format(
                axis, len(input_shape), input_shape
            )
        )
    if not axis >= 0:
        if not axis >= -1 * len(input_shape):
            raise ValueError(
                "the axis:{} should not be less than -1 * length of input_shape:{}".format(
                    axis, -1 * len(input_shape)
                )
            )
        axis = axis + len(input_shape)
    if in_dynamic_mode():
        out = _C_ops.renorm_(x, p, axis, max_norm)
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

            >>> import paddle
            >>> x = paddle.arange(1, 7).reshape((2, 3)).astype('float32')
            >>> y = paddle.arange(1, 10).reshape((3, 3)).astype('float32')
            >>> out = paddle.inner(x, y)
            >>> print(out)
            Tensor(shape=[2, 3], dtype=float32, place=Place(cpu), stop_gradient=True,
            [[14. , 32. , 50. ],
             [32. , 77. , 122.]])


    """
    if x.size == 1 or y.size == 1:
        return multiply(x, y)
    else:
        xshape = x.shape
        yshape = y.shape
        dstshape = list(xshape[:-1]) + list(yshape[:-1])

        nx = x.reshape((-1, xshape[-1]))
        ny = y.reshape((-1, yshape[-1]))

        if in_dynamic_mode():
            return _C_ops.matmul(nx, ny.T, False, False).reshape(dstshape)
        else:

            def __check_input(x, y):
                var_names = {'x': x, 'y': y}
                for name, val in var_names.items():
                    check_variable_and_dtype(
                        val, name, ['float16', 'float32', 'float64'], 'inner'
                    )
                x_shape = list(xshape)
                y_shape = list(yshape)

                # check the inner 2 dimensions
                if x_shape[-1] != y_shape[-1]:
                    if not ((x_shape[-1] == -1) or (y_shape[-1] == -1)):
                        raise ValueError(
                            "After performing an optional transpose, Input X's last dim should be "
                            "equal to Y's last dim for multiplication "
                            "prerequisites. But received X's shape: {}, Y's shape: {}\n".format(
                                x_shape, y_shape
                            )
                        )

            __check_input(nx, ny)

            helper = LayerHelper('inner', **locals())
            out = helper.create_variable_for_type_inference(dtype=nx.dtype)
            helper.append_op(
                type='matmul_v2',
                inputs={'X': nx, 'Y': ny.T},
                outputs={'Out': out},
            )
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

            >>> import paddle
            >>> x = paddle.arange(1, 4).astype('float32')
            >>> y = paddle.arange(1, 6).astype('float32')
            >>> out = paddle.outer(x, y)
            >>> print(out)
            Tensor(shape=[3, 5], dtype=float32, place=Place(cpu), stop_gradient=True,
            [[1. , 2. , 3. , 4. , 5. ],
             [2. , 4. , 6. , 8. , 10.],
             [3. , 6. , 9. , 12., 15.]])


    """
    nx = x.reshape((-1, 1))
    ny = y.reshape((1, -1))

    if in_dynamic_mode():
        return _C_ops.matmul(nx, ny, False, False)
    else:

        def __check_input(x, y):
            var_names = {'x': x, 'y': y}
            for name, val in var_names.items():
                check_variable_and_dtype(
                    val,
                    name,
                    ['float16', 'float32', 'float64', 'int32', 'int64'],
                    'outer',
                )

        __check_input(nx, ny)

        helper = LayerHelper('outer', **locals())
        out = helper.create_variable_for_type_inference(dtype=nx.dtype)
        helper.append_op(
            type='matmul_v2', inputs={'X': nx, 'Y': ny}, outputs={'Out': out}
        )
        return out


def logsumexp(x, axis=None, keepdim=False, name=None):
    r"""
    Calculates the log of the sum of exponentials of ``x`` along ``axis`` .

    .. math::
       logsumexp(x) = \log\sum exp(x)

    Args:
        x (Tensor): The input Tensor with data type float16, float32 or float64, which
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

        >>> import paddle

        >>> x = paddle.to_tensor([[-1.5, 0., 2.], [3., 1.2, -2.4]])
        >>> out1 = paddle.logsumexp(x)
        >>> out1
        Tensor(shape=[], dtype=float32, place=Place(cpu), stop_gradient=True,
        3.46912265)
        >>> out2 = paddle.logsumexp(x, 1)
        >>> out2
        Tensor(shape=[2], dtype=float32, place=Place(cpu), stop_gradient=True,
        [2.15317822, 3.15684605])

    """
    reduce_all, axis = _get_reduce_axis(axis, x)

    if in_dynamic_mode():
        return _C_ops.logsumexp(x, axis, keepdim, reduce_all)
    else:
        check_variable_and_dtype(
            x, 'x', ['float16', 'float32', 'float64', 'uint16'], 'logsumexp'
        )

        helper = LayerHelper('logsumexp', **locals())
        attrs = {'axis': axis, 'keepdim': keepdim, 'reduce_all': reduce_all}
        out = helper.create_variable_for_type_inference(x.dtype)
        helper.append_op(
            type='logsumexp', inputs={'X': x}, outputs={'Out': out}, attrs=attrs
        )
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

            >>> import paddle

            >>> mat = paddle.to_tensor([[2, 0], [0, 2]], dtype='float32')
            >>> inv = paddle.inverse(mat)
            >>> print(inv)
            Tensor(shape=[2, 2], dtype=float32, place=Place(cpu), stop_gradient=True,
            [[0.50000000, 0.        ],
             [0.        , 0.50000000]])

    """
    if in_dynamic_mode():
        return _C_ops.inverse(x)
    else:

        def _check_input(x):
            check_variable_and_dtype(x, 'x', ['float32', 'float64'], 'inverse')
            if len(x.shape) < 2:
                raise ValueError(
                    "The input of inverse is expected to be a Tensor whose number "
                    "of dimensions is no less than 2. But reviced: %d, "
                    "x's shape: %s." % (len(x.shape), x.shape)
                )

        _check_input(x)
        helper = LayerHelper('inverse', **locals())
        out = helper.create_variable_for_type_inference(dtype=x.dtype)
        helper.append_op(
            type='inverse', inputs={'Input': [x]}, outputs={'Output': [out]}
        )
        return out


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

            >>> import paddle

            >>> # data_x is a Tensor with shape [2, 4]
            >>> # the axis is a int element
            >>> x = paddle.to_tensor([[0.2, 0.3, 0.5, 0.9],
            ...                       [0.1, 0.2, 0.6, 0.7]],
            ...                       dtype='float64', stop_gradient=False)
            >>> result1 = paddle.max(x)
            >>> result1.backward()
            >>> result1
            Tensor(shape=[], dtype=float64, place=Place(cpu), stop_gradient=False,
            0.90000000)
            >>> x.grad
            Tensor(shape=[2, 4], dtype=float64, place=Place(cpu), stop_gradient=False,
            [[0., 0., 0., 1.],
             [0., 0., 0., 0.]])

            >>> x.clear_grad()
            >>> result2 = paddle.max(x, axis=0)
            >>> result2.backward()
            >>> result2
            Tensor(shape=[4], dtype=float64, place=Place(cpu), stop_gradient=False,
            [0.20000000, 0.30000000, 0.60000000, 0.90000000])
            >>> x.grad
            Tensor(shape=[2, 4], dtype=float64, place=Place(cpu), stop_gradient=False,
            [[1., 1., 0., 1.],
             [0., 0., 1., 0.]])

            >>> x.clear_grad()
            >>> result3 = paddle.max(x, axis=-1)
            >>> result3.backward()
            >>> result3
            Tensor(shape=[2], dtype=float64, place=Place(cpu), stop_gradient=False,
            [0.90000000, 0.70000000])
            >>> x.grad
            Tensor(shape=[2, 4], dtype=float64, place=Place(cpu), stop_gradient=False,
            [[0., 0., 0., 1.],
             [0., 0., 0., 1.]])

            >>> x.clear_grad()
            >>> result4 = paddle.max(x, axis=1, keepdim=True)
            >>> result4.backward()
            >>> result4
            Tensor(shape=[2, 1], dtype=float64, place=Place(cpu), stop_gradient=False,
            [[0.90000000],
             [0.70000000]])
            >>> x.grad
            Tensor(shape=[2, 4], dtype=float64, place=Place(cpu), stop_gradient=False,
            [[0., 0., 0., 1.],
             [0., 0., 0., 1.]])

            >>> # data_y is a Tensor with shape [2, 2, 2]
            >>> # the axis is list
            >>> y = paddle.to_tensor([[[1.0, 2.0], [3.0, 4.0]],
            ...                         [[5.0, 6.0], [7.0, 8.0]]],
            ...                         dtype='float64', stop_gradient=False)
            >>> result5 = paddle.max(y, axis=[1, 2])
            >>> result5.backward()
            >>> result5
            Tensor(shape=[2], dtype=float64, place=Place(cpu), stop_gradient=False,
            [4., 8.])
            >>> y.grad
            Tensor(shape=[2, 2, 2], dtype=float64, place=Place(cpu), stop_gradient=False,
            [[[0., 0.],
              [0., 1.]],
             [[0., 0.],
              [0., 1.]]])

            >>> y.clear_grad()
            >>> result6 = paddle.max(y, axis=[0, 1])
            >>> result6.backward()
            >>> result6
            Tensor(shape=[2], dtype=float64, place=Place(cpu), stop_gradient=False,
            [7., 8.])
            >>> y.grad
            Tensor(shape=[2, 2, 2], dtype=float64, place=Place(cpu), stop_gradient=False,
            [[[0., 0.],
              [0., 0.]],
             [[0., 0.],
              [1., 1.]]])
    """

    if in_dynamic_mode():
        return _C_ops.max(x, axis, keepdim)
    else:
        reduce_all, axis = _get_reduce_axis_with_tensor(axis, x)
        helper = LayerHelper('max', **locals())
        check_variable_and_dtype(
            x,
            'x',
            ['float16', 'uint16', 'float32', 'float64', 'int32', 'int64'],
            'max',
        )
        if not isinstance(axis, Variable) and paddle.utils._contain_var(axis):
            axis = paddle.utils._convert_to_tensor_list(axis)

        out = helper.create_variable_for_type_inference(dtype=x.dtype)
        helper.append_op(
            type='reduce_max',
            inputs={'X': x},
            outputs={'Out': out},
            attrs={'dim': axis, 'keep_dim': keepdim, 'reduce_all': reduce_all},
        )
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

            >>> import paddle

            >>> # data_x is a Tensor with shape [2, 4]
            >>> # the axis is a int element
            >>> x = paddle.to_tensor([[0.2, 0.3, 0.5, 0.9],
            ...                       [0.1, 0.2, 0.6, 0.7]],
            ...                       dtype='float64', stop_gradient=False)
            >>> result1 = paddle.min(x)
            >>> result1.backward()
            >>> result1
            Tensor(shape=[], dtype=float64, place=Place(cpu), stop_gradient=False,
            0.10000000)
            >>> x.grad
            Tensor(shape=[2, 4], dtype=float64, place=Place(cpu), stop_gradient=False,
            [[0., 0., 0., 0.],
             [1., 0., 0., 0.]])

            >>> x.clear_grad()
            >>> result2 = paddle.min(x, axis=0)
            >>> result2.backward()
            >>> result2
            Tensor(shape=[4], dtype=float64, place=Place(cpu), stop_gradient=False,
            [0.10000000, 0.20000000, 0.50000000, 0.70000000])
            >>> x.grad
            Tensor(shape=[2, 4], dtype=float64, place=Place(cpu), stop_gradient=False,
            [[0., 0., 1., 0.],
             [1., 1., 0., 1.]])

            >>> x.clear_grad()
            >>> result3 = paddle.min(x, axis=-1)
            >>> result3.backward()
            >>> result3
            Tensor(shape=[2], dtype=float64, place=Place(cpu), stop_gradient=False,
            [0.20000000, 0.10000000])
            >>> x.grad
            Tensor(shape=[2, 4], dtype=float64, place=Place(cpu), stop_gradient=False,
            [[1., 0., 0., 0.],
             [1., 0., 0., 0.]])

            >>> x.clear_grad()
            >>> result4 = paddle.min(x, axis=1, keepdim=True)
            >>> result4.backward()
            >>> result4
            Tensor(shape=[2, 1], dtype=float64, place=Place(cpu), stop_gradient=False,
            [[0.20000000],
             [0.10000000]])
            >>> x.grad
            Tensor(shape=[2, 4], dtype=float64, place=Place(cpu), stop_gradient=False,
            [[1., 0., 0., 0.],
             [1., 0., 0., 0.]])

            >>> # data_y is a Tensor with shape [2, 2, 2]
            >>> # the axis is list
            >>> y = paddle.to_tensor([[[1.0, 2.0], [3.0, 4.0]],
            ...                       [[5.0, 6.0], [7.0, 8.0]]],
            ...                       dtype='float64', stop_gradient=False)
            >>> result5 = paddle.min(y, axis=[1, 2])
            >>> result5.backward()
            >>> result5
            Tensor(shape=[2], dtype=float64, place=Place(cpu), stop_gradient=False,
            [1., 5.])
            >>> y.grad
            Tensor(shape=[2, 2, 2], dtype=float64, place=Place(cpu), stop_gradient=False,
            [[[1., 0.],
              [0., 0.]],
             [[1., 0.],
              [0., 0.]]])

            >>> y.clear_grad()
            >>> result6 = paddle.min(y, axis=[0, 1])
            >>> result6.backward()
            >>> result6
            Tensor(shape=[2], dtype=float64, place=Place(cpu), stop_gradient=False,
            [1., 2.])
            >>> y.grad
            Tensor(shape=[2, 2, 2], dtype=float64, place=Place(cpu), stop_gradient=False,
            [[[1., 1.],
              [0., 0.]],
             [[0., 0.],
              [0., 0.]]])
    """

    if in_dynamic_mode():
        return _C_ops.min(x, axis, keepdim)
    else:
        reduce_all, axis = _get_reduce_axis_with_tensor(axis, x)
        helper = LayerHelper('min', **locals())
        check_variable_and_dtype(
            x,
            'x',
            ['float16', 'uint16', 'float32', 'float64', 'int32', 'int64'],
            'min',
        )

        out = helper.create_variable_for_type_inference(dtype=x.dtype)
        helper.append_op(
            type='reduce_min',
            inputs={'X': x},
            outputs={'Out': out},
            attrs={'dim': axis, 'keep_dim': keepdim, 'reduce_all': reduce_all},
        )
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

            >>> import paddle
            >>> # data_x is a Tensor with shape [2, 4] with multiple maximum elements
            >>> # the axis is a int element

            >>> x = paddle.to_tensor([[0.1, 0.9, 0.9, 0.9],
            ...                         [0.9, 0.9, 0.6, 0.7]],
            ...                         dtype='float64', stop_gradient=False)
            >>> # There are 5 maximum elements:
            >>> # 1) amax evenly distributes gradient between these equal values,
            >>> #    thus the corresponding gradients are 1/5=0.2;
            >>> # 2) while max propagates gradient to all of them,
            >>> #    thus the corresponding gradient are 1.
            >>> result1 = paddle.amax(x)
            >>> result1.backward()
            >>> result1
            Tensor(shape=[], dtype=float64, place=Place(cpu), stop_gradient=False,
            0.90000000)
            >>> x.grad
            Tensor(shape=[2, 4], dtype=float64, place=Place(cpu), stop_gradient=False,
            [[0.        , 0.20000000, 0.20000000, 0.20000000],
             [0.20000000, 0.20000000, 0.        , 0.        ]])

            >>> x.clear_grad()
            >>> result1_max = paddle.max(x)
            >>> result1_max.backward()
            >>> result1_max
            Tensor(shape=[], dtype=float64, place=Place(cpu), stop_gradient=False,
            0.90000000)
            >>> x.grad
            Tensor(shape=[2, 4], dtype=float64, place=Place(cpu), stop_gradient=False,
            [[0., 1., 1., 1.],
             [1., 1., 0., 0.]])

            >>> x.clear_grad()
            >>> result2 = paddle.amax(x, axis=0)
            >>> result2.backward()
            >>> result2
            Tensor(shape=[4], dtype=float64, place=Place(cpu), stop_gradient=False,
            [0.90000000, 0.90000000, 0.90000000, 0.90000000])
            >>> x.grad
            Tensor(shape=[2, 4], dtype=float64, place=Place(cpu), stop_gradient=False,
            [[0.        , 0.50000000, 1.        , 1.        ],
             [1.        , 0.50000000, 0.        , 0.        ]])

            >>> x.clear_grad()
            >>> result3 = paddle.amax(x, axis=-1)
            >>> result3.backward()
            >>> result3
            Tensor(shape=[2], dtype=float64, place=Place(cpu), stop_gradient=False,
            [0.90000000, 0.90000000])
            >>> x.grad
            Tensor(shape=[2, 4], dtype=float64, place=Place(cpu), stop_gradient=False,
            [[0.        , 0.33333333, 0.33333333, 0.33333333],
             [0.50000000, 0.50000000, 0.        , 0.        ]])

            >>> x.clear_grad()
            >>> result4 = paddle.amax(x, axis=1, keepdim=True)
            >>> result4.backward()
            >>> result4
            Tensor(shape=[2, 1], dtype=float64, place=Place(cpu), stop_gradient=False,
            [[0.90000000],
             [0.90000000]])
            >>> x.grad
            Tensor(shape=[2, 4], dtype=float64, place=Place(cpu), stop_gradient=False,
            [[0.        , 0.33333333, 0.33333333, 0.33333333],
             [0.50000000, 0.50000000, 0.        , 0.        ]])

            >>> # data_y is a Tensor with shape [2, 2, 2]
            >>> # the axis is list
            >>> y = paddle.to_tensor([[[0.1, 0.9], [0.9, 0.9]],
            ...                         [[0.9, 0.9], [0.6, 0.7]]],
            ...                         dtype='float64', stop_gradient=False)
            >>> result5 = paddle.amax(y, axis=[1, 2])
            >>> result5.backward()
            >>> result5
            Tensor(shape=[2], dtype=float64, place=Place(cpu), stop_gradient=False,
            [0.90000000, 0.90000000])
            >>> y.grad
            Tensor(shape=[2, 2, 2], dtype=float64, place=Place(cpu), stop_gradient=False,
            [[[0.        , 0.33333333],
              [0.33333333, 0.33333333]],
             [[0.50000000, 0.50000000],
              [0.        , 0.        ]]])

            >>> y.clear_grad()
            >>> result6 = paddle.amax(y, axis=[0, 1])
            >>> result6.backward()
            >>> result6
            Tensor(shape=[2], dtype=float64, place=Place(cpu), stop_gradient=False,
            [0.90000000, 0.90000000])
            >>> y.grad
            Tensor(shape=[2, 2, 2], dtype=float64, place=Place(cpu), stop_gradient=False,
            [[[0.        , 0.33333333],
              [0.50000000, 0.33333333]],
             [[0.50000000, 0.33333333],
              [0.        , 0.        ]]])
    """
    if in_dynamic_mode():
        return _C_ops.amax(x, axis, keepdim)

    else:
        reduce_all, axis = _get_reduce_axis(axis, x)
        helper = LayerHelper('amax', **locals())
        check_variable_and_dtype(
            x, 'x', ['float32', 'float64', 'int32', 'int64'], 'amax'
        )

        out = helper.create_variable_for_type_inference(dtype=x.dtype)
        helper.append_op(
            type='reduce_amax',
            inputs={'X': x},
            outputs={'Out': out},
            attrs={'dim': axis, 'keep_dim': keepdim, 'reduce_all': reduce_all},
        )
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

            >>> import paddle
            >>> # data_x is a Tensor with shape [2, 4] with multiple minimum elements
            >>> # the axis is a int element

            >>> x = paddle.to_tensor([[0.2, 0.1, 0.1, 0.1],
            ...                         [0.1, 0.1, 0.6, 0.7]],
            ...                         dtype='float64', stop_gradient=False)
            >>> # There are 5 minimum elements:
            >>> # 1) amin evenly distributes gradient between these equal values,
            >>> #    thus the corresponding gradients are 1/5=0.2;
            >>> # 2) while min propagates gradient to all of them,
            >>> #    thus the corresponding gradient are 1.
            >>> result1 = paddle.amin(x)
            >>> result1.backward()
            >>> result1
            Tensor(shape=[], dtype=float64, place=Place(cpu), stop_gradient=False,
            0.10000000)
            >>> x.grad
            Tensor(shape=[2, 4], dtype=float64, place=Place(cpu), stop_gradient=False,
            [[0.        , 0.20000000, 0.20000000, 0.20000000],
             [0.20000000, 0.20000000, 0.        , 0.        ]])

            >>> x.clear_grad()
            >>> result1_min = paddle.min(x)
            >>> result1_min.backward()
            >>> result1_min
            Tensor(shape=[], dtype=float64, place=Place(cpu), stop_gradient=False,
            0.10000000)
            >>> x.grad
            Tensor(shape=[2, 4], dtype=float64, place=Place(cpu), stop_gradient=False,
            [[0., 1., 1., 1.],
             [1., 1., 0., 0.]])

            >>> x.clear_grad()
            >>> result2 = paddle.amin(x, axis=0)
            >>> result2.backward()
            >>> result2
            Tensor(shape=[4], dtype=float64, place=Place(cpu), stop_gradient=False,
            [0.10000000, 0.10000000, 0.10000000, 0.10000000])
            >>> x.grad
            Tensor(shape=[2, 4], dtype=float64, place=Place(cpu), stop_gradient=False,
            [[0.        , 0.50000000, 1.        , 1.        ],
             [1.        , 0.50000000, 0.        , 0.        ]])

            >>> x.clear_grad()
            >>> result3 = paddle.amin(x, axis=-1)
            >>> result3.backward()
            >>> result3
            Tensor(shape=[2], dtype=float64, place=Place(cpu), stop_gradient=False,
            [0.10000000, 0.10000000])
            >>> x.grad
            Tensor(shape=[2, 4], dtype=float64, place=Place(cpu), stop_gradient=False,
            [[0.        , 0.33333333, 0.33333333, 0.33333333],
             [0.50000000, 0.50000000, 0.        , 0.        ]])

            >>> x.clear_grad()
            >>> result4 = paddle.amin(x, axis=1, keepdim=True)
            >>> result4.backward()
            >>> result4
            Tensor(shape=[2, 1], dtype=float64, place=Place(cpu), stop_gradient=False,
            [[0.10000000],
             [0.10000000]])
            >>> x.grad
            Tensor(shape=[2, 4], dtype=float64, place=Place(cpu), stop_gradient=False,
            [[0.        , 0.33333333, 0.33333333, 0.33333333],
             [0.50000000, 0.50000000, 0.        , 0.        ]])

            >>> # data_y is a Tensor with shape [2, 2, 2]
            >>> # the axis is list
            >>> y = paddle.to_tensor([[[0.2, 0.1], [0.1, 0.1]],
            ...                       [[0.1, 0.1], [0.6, 0.7]]],
            ...                       dtype='float64', stop_gradient=False)
            >>> result5 = paddle.amin(y, axis=[1, 2])
            >>> result5.backward()
            >>> result5
            Tensor(shape=[2], dtype=float64, place=Place(cpu), stop_gradient=False,
            [0.10000000, 0.10000000])
            >>> y.grad
            Tensor(shape=[2, 2, 2], dtype=float64, place=Place(cpu), stop_gradient=False,
            [[[0.        , 0.33333333],
              [0.33333333, 0.33333333]],
             [[0.50000000, 0.50000000],
              [0.        , 0.        ]]])

            >>> y.clear_grad()
            >>> result6 = paddle.amin(y, axis=[0, 1])
            >>> result6.backward()
            >>> result6
            Tensor(shape=[2], dtype=float64, place=Place(cpu), stop_gradient=False,
            [0.10000000, 0.10000000])
            >>> y.grad
            Tensor(shape=[2, 2, 2], dtype=float64, place=Place(cpu), stop_gradient=False,
            [[[0.        , 0.33333333],
              [0.50000000, 0.33333333]],
             [[0.50000000, 0.33333333],
              [0.        , 0.        ]]])
    """
    if in_dynamic_mode():
        return _C_ops.amin(x, axis, keepdim)

    else:
        reduce_all, axis = _get_reduce_axis(axis, x)
        helper = LayerHelper('amin', **locals())
        check_variable_and_dtype(
            x, 'x', ['float32', 'float64', 'int32', 'int64'], 'amin'
        )

        out = helper.create_variable_for_type_inference(dtype=x.dtype)
        helper.append_op(
            type='reduce_amin',
            inputs={'X': x},
            outputs={'Out': out},
            attrs={'dim': axis, 'keep_dim': keepdim, 'reduce_all': reduce_all},
        )
        return out


def log1p(x, name=None):
    r"""
    Calculates the natural log of the given input tensor, element-wise.

    .. math::
        Out = \ln(x+1)

    Args:
        x (Tensor): Input Tensor. Must be one of the following types: int32, int64, float16, bfloat16, float32, float64.
        name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        Tensor, the natural log of the input Tensor computed element-wise.

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> data = paddle.to_tensor([[0], [1]], dtype='float32')
            >>> res = paddle.log1p(data)
            >>> res
            Tensor(shape=[2, 1], dtype=float32, place=Place(cpu), stop_gradient=True,
            [[0.        ],
             [0.69314718]])
    """

    if in_dynamic_mode():
        return _C_ops.log1p(x)
    else:
        check_variable_and_dtype(
            x,
            'x',
            ['int32', 'int64', 'float16', 'uint16', 'float32', 'float64'],
            "log1p",
        )
        inputs = {'X': [x]}
        helper = LayerHelper('log1p', **locals())
        dtype = helper.input_dtype(input_param_name='x')
        out = helper.create_variable_for_type_inference(dtype)
        helper.append_op(type="log1p", inputs={"X": x}, outputs={"Out": out})
        return out


@inplace_apis_in_dygraph_only
def log1p_(x, name=None):
    r"""
    Inplace version of ``log1p`` API, the output Tensor will be inplaced with input ``x``.
    Please refer to :ref:`api_paddle_log1p`.
    """

    if in_dynamic_mode():
        return _C_ops.log1p_(x)


def log2(x, name=None):
    r"""
    Calculates the log to the base 2 of the given input tensor, element-wise.

    .. math::

        Out = \log_2x

    Args:
        x (Tensor): Input tensor must be one of the following types: int32, int64, float16, bfloat16, float32, float64.
        name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.


    Returns:
        Tensor: The log to the base 2 of the input Tensor computed element-wise.

    Examples:

        .. code-block:: python

            >>> import paddle

            >>> # example 1: x is a float
            >>> x_i = paddle.to_tensor([[1.0], [2.0]])
            >>> res = paddle.log2(x_i)
            >>> res
            Tensor(shape=[2, 1], dtype=float32, place=Place(cpu), stop_gradient=True,
            [[0.],
             [1.]])

            >>> # example 2: x is float32
            >>> x_i = paddle.full(shape=[1], fill_value=2, dtype='float32')
            >>> paddle.to_tensor(x_i)
            >>> res = paddle.log2(x_i)
            >>> res
            Tensor(shape=[1], dtype=float32, place=Place(cpu), stop_gradient=True,
            [1.])

            >>> # example 3: x is float64
            >>> x_i = paddle.full(shape=[1], fill_value=2, dtype='float64')
            >>> paddle.to_tensor(x_i)
            >>> res = paddle.log2(x_i)
            >>> res
            Tensor(shape=[1], dtype=float64, place=Place(cpu), stop_gradient=True,
            [1.])
    """
    if in_dynamic_mode():
        return _C_ops.log2(x)
    else:
        check_variable_and_dtype(
            x,
            'x',
            ['int32', 'int64', 'float16', 'uint16', 'float32', 'float64'],
            "log2",
        )
        inputs = {'X': [x]}
        helper = LayerHelper('log2', **locals())
        dtype = helper.input_dtype(input_param_name='x')
        out = helper.create_variable_for_type_inference(dtype)
        helper.append_op(type="log2", inputs={"X": x}, outputs={"Out": out})
        return out


@inplace_apis_in_dygraph_only
def log2_(x, name=None):
    r"""
    Inplace version of ``log2`` API, the output Tensor will be inplaced with input ``x``.
    Please refer to :ref:`api_paddle_log2`.
    """

    if in_dynamic_mode():
        return _C_ops.log2_(x)


def log10(x, name=None):
    r"""
    Calculates the log to the base 10 of the given input tensor, element-wise.

    .. math::

        Out = \log_10_x

    Args:
        x (Tensor): Input tensor must be one of the following types: int32, int64, float16, bfloat16, float32, float64.
        name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.


    Returns:
        Tensor: The log to the base 10 of the input Tensor computed element-wise.

    Examples:

        .. code-block:: python

            >>> import paddle

            >>> # example 1: x is a float
            >>> x_i = paddle.to_tensor([[1.0], [10.0]])
            >>> res = paddle.log10(x_i)
            >>> res
            Tensor(shape=[2, 1], dtype=float32, place=Place(cpu), stop_gradient=True,
            [[0.],
             [1.]])

            >>> # example 2: x is float32
            >>> x_i = paddle.full(shape=[1], fill_value=10, dtype='float32')
            >>> paddle.to_tensor(x_i)
            >>> res = paddle.log10(x_i)
            >>> res
            Tensor(shape=[1], dtype=float32, place=Place(cpu), stop_gradient=True,
            [1.])

            >>> # example 3: x is float64
            >>> x_i = paddle.full(shape=[1], fill_value=10, dtype='float64')
            >>> paddle.to_tensor(x_i)
            >>> res = paddle.log10(x_i)
            >>> res
            Tensor(shape=[1], dtype=float64, place=Place(cpu), stop_gradient=True,
            [1.])
    """
    if in_dynamic_mode():
        return _C_ops.log10(x)
    else:
        check_variable_and_dtype(
            x,
            'x',
            ['int32', 'int64', 'float16', 'uint16', 'float32', 'float64'],
            "log10",
        )
        inputs = {'X': [x]}
        helper = LayerHelper('log10', **locals())
        dtype = helper.input_dtype(input_param_name='x')
        out = helper.create_variable_for_type_inference(dtype)
        helper.append_op(type="log10", inputs={"X": x}, outputs={"Out": out})
        return out


@inplace_apis_in_dygraph_only
def log10_(x, name=None):
    r"""
    Inplace version of ``log10`` API, the output Tensor will be inplaced with input ``x``.
    Please refer to :ref:`api_paddle_log10`.
    """

    if in_dynamic_mode():
        return _C_ops.log10_(x)


def clip(x, min=None, max=None, name=None):
    """
    This operator clip all elements in input into the range [ min, max ] and return
    a resulting tensor as the following equation:

    .. math::

        Out = MIN(MAX(x, min), max)

    Args:
        x (Tensor): An N-D Tensor with data type float16, float32, float64, int32 or int64.
        min (float|int|Tensor, optional): The lower bound with type ``float`` , ``int`` or a ``0-D Tensor``
            with shape [] and type ``int32``, ``float16``, ``float32``, ``float64``.
        max (float|int|Tensor, optional): The upper bound with type ``float``, ``int`` or a ``0-D Tensor``
            with shape [] and type ``int32``, ``float16``, ``float32``, ``float64``.
        name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        Tensor: A Tensor with the same data type and data shape as input.

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> x1 = paddle.to_tensor([[1.2, 3.5], [4.5, 6.4]], 'float32')
            >>> out1 = paddle.clip(x1, min=3.5, max=5.0)
            >>> out1
            Tensor(shape=[2, 2], dtype=float32, place=Place(cpu), stop_gradient=True,
            [[3.50000000, 3.50000000],
             [4.50000000, 5.        ]])
            >>> out2 = paddle.clip(x1, min=2.5)
            >>> out2
            Tensor(shape=[2, 2], dtype=float32, place=Place(cpu), stop_gradient=True,
            [[2.50000000, 3.50000000],
             [4.50000000, 6.40000010]])
    """

    x_dtype = str(x.dtype)
    if x_dtype == 'paddle.int32':
        min_ = np.iinfo(np.int32).min
        max_ = np.iinfo(np.int32).max - 2**7
    elif x_dtype == 'paddle.int64':
        min_ = np.iinfo(np.int64).min
        max_ = np.iinfo(np.int64).max - 2**39
    elif x_dtype == 'paddle.float16':
        min_ = float(np.finfo(np.float16).min)
        max_ = float(np.finfo(np.float16).max)
    else:
        min_ = float(np.finfo(np.float32).min)
        max_ = float(np.finfo(np.float32).max)

    if in_dynamic_mode():
        if isinstance(min, Variable):
            min = min.item(0)
        if isinstance(max, Variable):
            max = max.item(0)
        min = min_ if min is None else min
        max = max_ if max is None else max
        return _C_ops.clip(x, min, max)
    else:
        if min is not None:
            check_type(min, 'min', (float, int, Variable), 'clip')
            if isinstance(min, Variable):
                check_dtype(
                    min.dtype,
                    'min',
                    ['float16', 'float32', 'float64', 'int32', 'uint16'],
                    'clip',
                    '(When the type of min in clip is Variable.)',
                )
        if max is not None:
            check_type(max, 'max', (float, int, Variable), 'clip')
            if isinstance(max, Variable):
                check_dtype(
                    max.dtype,
                    'max',
                    ['float16', 'float32', 'float64', 'int32', 'uint16'],
                    'clip',
                    '(When the type of max in clip is Variable.)',
                )

        check_variable_and_dtype(
            x,
            'x',
            ['float16', 'float32', 'float64', 'int32', 'int64', 'uint16'],
            'clip',
        )

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
            dtype=helper.input_dtype('x')
        )
        helper.append_op(
            type='clip', inputs=inputs, outputs={'Out': [output]}, attrs=attrs
        )

        return output


@inplace_apis_in_dygraph_only
def clip_(x, min=None, max=None, name=None):
    """
    Inplace version of ``clip`` API, the output Tensor will be inplaced with input ``x``.
    Please refer to :ref:`api_paddle_clip`.
    """
    fmin = float(np.finfo(np.float32).min)
    fmax = float(np.finfo(np.float32).max)
    if isinstance(min, Variable):
        min = min.item(0)
    if isinstance(max, Variable):
        max = max.item(0)
    min = fmin if min is None else min
    max = fmax if max is None else max

    if in_dynamic_mode():
        return _C_ops.clip_(x, min, max)


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

            >>> import paddle

            >>> case1 = paddle.randn([2, 3])
            >>> case2 = paddle.randn([3, 10, 10])
            >>> case3 = paddle.randn([3, 10, 5, 10])
            >>> data1 = paddle.trace(case1)
            >>> data1.shape
            []
            >>> data2 = paddle.trace(case2, offset=1, axis1=1, axis2=2)
            >>> data2.shape
            [3]
            >>> data3 = paddle.trace(case3, offset=-3, axis1=1, axis2=-1)
            >>> data3.shape
            [3, 5]
    """

    def __check_input(x, offset, axis1, axis2):
        check_dtype(
            x.dtype,
            'Input',
            ['int32', 'int64', 'float16', 'float32', 'float64'],
            'trace',
        )

        input_shape = list(x.shape)
        assert len(input_shape) >= 2, (
            "The x must be at least 2-dimensional, "
            "But received Input x's dimensional: %s.\n" % len(input_shape)
        )

        axis1_ = axis1 if axis1 >= 0 else len(input_shape) + axis1
        axis2_ = axis2 if axis2 >= 0 else len(input_shape) + axis2

        assert (0 <= axis1_) and (axis1_ < len(input_shape)), (
            "The argument axis1 is out of range (expected to be in range of [%d, %d], but got %d).\n"
            % (-(len(input_shape)), len(input_shape) - 1, axis1)
        )

        assert (0 <= axis2_) and (axis2_ < len(input_shape)), (
            "The argument axis2 is out of range (expected to be in range of [%d, %d], but got %d).\n"
            % (-(len(input_shape)), len(input_shape) - 1, axis2)
        )

        assert axis1_ != axis2_, (
            "axis1 and axis2 cannot be the same axis."
            "But received axis1 = %d, axis2 = %d\n" % (axis1, axis2)
        )

    if in_dynamic_mode():
        return _C_ops.trace(x, offset, axis1, axis2)
    else:
        __check_input(x, offset, axis1, axis2)

        helper = LayerHelper('trace', **locals())
        out = helper.create_variable_for_type_inference(dtype=x.dtype)

        helper.append_op(
            type='trace',
            inputs={'Input': [x]},
            attrs={'offset': offset, 'axis1': axis1, 'axis2': axis2},
            outputs={'Out': [out]},
        )
        return out


def diagonal(x, offset=0, axis1=0, axis2=1, name=None):
    """
    Computes the diagonals of the input tensor x.

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

            >>> import paddle

            >>> paddle.seed(2023)
            >>> x = paddle.rand([2, 2, 3],'float32')
            >>> print(x)
            Tensor(shape=[2, 2, 3], dtype=float32, place=Place(cpu), stop_gradient=True,
            [[[0.86583614, 0.52014720, 0.25960937],
              [0.90525323, 0.42400089, 0.40641287]],
             [[0.97020894, 0.74437362, 0.51785129],
              [0.73292869, 0.97786582, 0.04315904]]])

            >>> out1 = paddle.diagonal(x)
            >>> print(out1)
            Tensor(shape=[3, 2], dtype=float32, place=Place(cpu), stop_gradient=True,
            [[0.86583614, 0.73292869],
             [0.52014720, 0.97786582],
             [0.25960937, 0.04315904]])

            >>> out2 = paddle.diagonal(x, offset=0, axis1=2, axis2=1)
            >>> print(out2)
            Tensor(shape=[2, 2], dtype=float32, place=Place(cpu), stop_gradient=True,
            [[0.86583614, 0.42400089],
             [0.97020894, 0.97786582]])

            >>> out3 = paddle.diagonal(x, offset=1, axis1=0, axis2=1)
            >>> print(out3)
            Tensor(shape=[3, 1], dtype=float32, place=Place(cpu), stop_gradient=True,
            [[0.90525323],
             [0.42400089],
             [0.40641287]])

            >>> out4 = paddle.diagonal(x, offset=0, axis1=1, axis2=2)
            >>> print(out4)
            Tensor(shape=[2, 2], dtype=float32, place=Place(cpu), stop_gradient=True,
            [[0.86583614, 0.42400089],
             [0.97020894, 0.97786582]])

    """
    if in_dynamic_mode():
        return _C_ops.diagonal(x, offset, axis1, axis2)
    else:

        def __check_input(x, offset, axis1, axis2):
            check_dtype(
                x.dtype,
                'Input',
                [
                    'bool',
                    'int32',
                    'int64',
                    'float16',
                    'uint16',
                    'float32',
                    'float64',
                ],
                'diagonal',
            )

            input_shape = list(x.shape)
            assert len(input_shape) >= 2, (
                "The x must be at least 2-dimensional, "
                "But received Input x's dimensional: %s.\n" % len(input_shape)
            )

            axis1_ = axis1 if axis1 >= 0 else len(input_shape) + axis1
            axis2_ = axis2 if axis2 >= 0 else len(input_shape) + axis2

            assert axis1_ < len(input_shape), (
                "The argument axis1 is out of range (expected to be in range of [%d, %d], but got %d).\n"
                % (-(len(input_shape)), len(input_shape) - 1, axis1)
            )

            assert axis2_ < len(input_shape), (
                "The argument axis2 is out of range (expected to be in range of [%d, %d], but got %d).\n"
                % (-(len(input_shape)), len(input_shape) - 1, axis2)
            )

            assert axis1_ != axis2_, (
                "axis1 and axis2 cannot be the same axis."
                "But received axis1 = %d, axis2 = %d\n" % (axis1, axis2)
            )

        __check_input(x, offset, axis1, axis2)
        helper = LayerHelper('diagonal', **locals())
        out = helper.create_variable_for_type_inference(dtype=x.dtype)

        helper.append_op(
            type='diagonal',
            inputs={'Input': [x]},
            attrs={'offset': offset, 'axis1': axis1, 'axis2': axis2},
            outputs={'Out': [out]},
        )
        return out


def kron(x, y, name=None):
    r"""
    Compute the Kronecker product of two tensors, a
    composite tensor made of blocks of the second tensor scaled by the
    first.
    Assume that the rank of the two tensors, $X$ and $Y$
    are the same, if necessary prepending the smallest with ones. If the
    shape of $X$ is [$r_0$, $r_1$, ..., $r_N$] and the shape of $Y$ is
    [$s_0$, $s_1$, ..., $s_N$], then the shape of the output tensor is
    [$r_{0}s_{0}$, $r_{1}s_{1}$, ..., $r_{N}s_{N}$]. The elements are
    products of elements from $X$ and $Y$.
    The equation is:
    $$
    output[k_{0}, k_{1}, ..., k_{N}] = X[i_{0}, i_{1}, ..., i_{N}] *
    Y[j_{0}, j_{1}, ..., j_{N}]
    $$
    where
    $$
    k_{t} = i_{t} * s_{t} + j_{t}, t = 0, 1, ..., N
    $$

    Args:
        x (Tensor): the fist operand of kron op, data type: float16, float32, float64, int32 or int64.
        y (Tensor): the second operand of kron op, data type: float16, float32, float64, int32 or int64. Its data type should be the same with x.
        name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        Tensor: The output of kron, data type: float16, float32, float64, int32 or int64. Its data is the same with x.

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> x = paddle.to_tensor([[1, 2], [3, 4]], dtype='int64')
            >>> y = paddle.to_tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype='int64')
            >>> out = paddle.kron(x, y)
            >>> out
            Tensor(shape=[6, 6], dtype=int64, place=Place(cpu), stop_gradient=True,
            [[1 , 2 , 3 , 2 , 4 , 6 ],
             [4 , 5 , 6 , 8 , 10, 12],
             [7 , 8 , 9 , 14, 16, 18],
             [3 , 6 , 9 , 4 , 8 , 12],
             [12, 15, 18, 16, 20, 24],
             [21, 24, 27, 28, 32, 36]])
    """
    if in_dynamic_mode():
        return _legacy_C_ops.kron(x, y)
    else:
        helper = LayerHelper('kron', **locals())
        check_variable_and_dtype(
            x, 'x', ['float16', 'float32', 'float64', 'int32', 'int64'], 'kron'
        )
        check_variable_and_dtype(
            y, 'y', ['float16', 'float32', 'float64', 'int32', 'int64'], 'kron'
        )

        out = helper.create_variable_for_type_inference(dtype=x.dtype)
        helper.append_op(
            type="kron", inputs={"X": x, "Y": y}, outputs={"Out": out}
        )
        return out


def cumsum(x, axis=None, dtype=None, name=None):
    """
    The cumulative sum of the elements along a given axis.

    Note:
        The first element of the result is the same as the first element of the input.

    Args:
        x (Tensor): The input tensor needed to be cumsumed.
        axis (int, optional): The dimension to accumulate along. -1 means the last dimension. The default (None) is to compute the cumsum over the flattened array.
        dtype (str, optional): The data type of the output tensor, can be float16, float32, float64, int32, int64. If specified, the input tensor is casted to dtype before the operation is performed. This is useful for preventing data type overflows. The default value is None.
        name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        Tensor, the result of cumsum operator.

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> data = paddle.arange(12)
            >>> data = paddle.reshape(data, (3, 4))

            >>> y = paddle.cumsum(data)
            >>> y
            Tensor(shape=[12], dtype=int64, place=Place(cpu), stop_gradient=True,
            [0 , 1 , 3 , 6 , 10, 15, 21, 28, 36, 45, 55, 66])

            >>> y = paddle.cumsum(data, axis=0)
            >>> y
            Tensor(shape=[3, 4], dtype=int64, place=Place(cpu), stop_gradient=True,
            [[0 , 1 , 2 , 3 ],
             [4 , 6 , 8 , 10],
             [12, 15, 18, 21]])

            >>> y = paddle.cumsum(data, axis=-1)
            >>> y
            Tensor(shape=[3, 4], dtype=int64, place=Place(cpu), stop_gradient=True,
            [[0 , 1 , 3 , 6 ],
             [4 , 9 , 15, 22],
             [8 , 17, 27, 38]])

            >>> y = paddle.cumsum(data, dtype='float64')
            >>> assert y.dtype == paddle.float64
    """
    if axis is None:
        flatten = True
    else:
        flatten = False
    if dtype is not None and x.dtype != convert_np_dtype_to_dtype_(dtype):
        x = cast(x, dtype)

    if in_dynamic_mode():
        if axis is None:
            axis = -1
        return _C_ops.cumsum(x, axis, flatten, False, False)
    else:
        check_variable_and_dtype(
            x,
            'x',
            ['float16', 'uint16', 'float32', 'float64', 'int32', 'int64'],
            'cumsum',
        )
        check_type(x, 'x', (Variable), 'cumsum')
        locals_var = locals().copy()
        kwargs = {}
        for name, val in locals_var.items():
            if val is not None:
                kwargs[name] = val
        _cum_sum_ = generate_layer_fn('cumsum')
        return _cum_sum_(**kwargs)


@inplace_apis_in_dygraph_only
def cumsum_(x, axis=None, dtype=None, name=None):
    r"""
    Inplace version of ``cumprod`` API, the output Tensor will be inplaced with input ``x``.
    Please refer to :ref:`api_paddle_cumprod`.
    """
    if axis is None:
        flatten = True
    else:
        flatten = False
    if dtype is not None and x.dtype != convert_np_dtype_to_dtype_(dtype):
        x = cast_(x, dtype)

    if in_dynamic_mode():
        if axis is None:
            axis = -1
        return _C_ops.cumsum_(x, axis, flatten, False, False)


def cummax(x, axis=None, dtype='int64', name=None):
    """
    The cumulative max of the elements along a given axis.

    Note:
        The first element of the result is the same as the first element of the input.

    Args:
        x (Tensor): The input tensor needed to be cummaxed.
        axis (int, optional): The dimension to accumulate along. -1 means the last dimension. The default (None) is to compute the cummax over the flattened array.
        dtype (str, optional): The data type of the indices tensor, can be int32, int64. The default value is int64.
        name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        out (Tensor), The result of cummax operation. The dtype of cummax result is same with input x.

        indices (Tensor), The corresponding index results of cummax operation.

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> data = paddle.to_tensor([-1, 5, 0, -2, -3, 2])
            >>> data = paddle.reshape(data, (2, 3))

            >>> value, indices = paddle.cummax(data)
            >>> value
            Tensor(shape=[6], dtype=int64, place=Place(cpu), stop_gradient=True,
            [-1,  5,  5,  5,  5,  5])
            >>> indices
            Tensor(shape=[6], dtype=int64, place=Place(cpu), stop_gradient=True,
            [0, 1, 1, 1, 1, 1])

            >>> value, indices = paddle.cummax(data, axis=0)
            >>> value
            Tensor(shape=[2, 3], dtype=int64, place=Place(cpu), stop_gradient=True,
            [[-1,  5,  0],
             [-1,  5,  2]])
            >>> indices
            Tensor(shape=[2, 3], dtype=int64, place=Place(cpu), stop_gradient=True,
            [[0, 0, 0],
             [0, 0, 1]])

            >>> value, indices = paddle.cummax(data, axis=-1)
            >>> value
            Tensor(shape=[2, 3], dtype=int64, place=Place(cpu), stop_gradient=True,
            [[-1,  5,  5],
             [-2, -2,  2]])
            >>> indices
            Tensor(shape=[2, 3], dtype=int64, place=Place(cpu), stop_gradient=True,
            [[0, 1, 1],
             [0, 0, 2]])

            >>> value, indices = paddle.cummax(data, dtype='int64')
            >>> assert indices.dtype == paddle.int64
    """
    if axis is None:
        axis = -1
        x = x.flatten(0, len(x.shape) - 1)

    check_dtype(dtype, 'dtype', ['int32', 'int64'], 'cummax')
    dtype = convert_np_dtype_to_dtype_(dtype)

    if in_dynamic_mode():
        return _C_ops.cummax(x, axis, dtype)
    else:
        check_variable_and_dtype(
            x,
            'x',
            ['float32', 'float64', 'int32', 'int64'],
            'cummax',
        )
        check_type(x, 'x', (Variable), 'cummax')
        helper = LayerHelper('cummax', **locals())
        out = helper.create_variable_for_type_inference(x.dtype)
        indices = helper.create_variable_for_type_inference(dtype='int64')
        helper.append_op(
            type='cummax',
            inputs={'x': x},
            outputs={'out': out, 'indices': indices},
            attrs={'axis': axis, 'dtype': dtype},
        )
        return out, indices


def cummin(x, axis=None, dtype='int64', name=None):
    """
    The cumulative min of the elements along a given axis.

    Note:
        The first element of the result is the same as the first element of the input.

    Args:
        x (Tensor): The input tensor needed to be cummined.
        axis (int, optional): The dimension to accumulate along. -1 means the last dimension. The default (None) is to compute the cummin over the flattened array.
        dtype (str, optional): The data type of the indices tensor, can be int32, int64. The default value is int64.
        name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        out (Tensor), The result of cummin operation. The dtype of cummin result is same with input x.

        indices (Tensor), The corresponding index results of cummin operation.

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> data = paddle.to_tensor([-1, 5, 0, -2, -3, 2])
            >>> data = paddle.reshape(data, (2, 3))

            >>> value, indices = paddle.cummin(data)
            >>> value
            Tensor(shape=[6], dtype=int64, place=Place(cpu), stop_gradient=True,
            [-1, -1, -1, -2, -3, -3])
            >>> indices
            Tensor(shape=[6], dtype=int64, place=Place(cpu), stop_gradient=True,
            [0, 0, 0, 3, 4, 4])

            >>> value, indices = paddle.cummin(data, axis=0)
            >>> value
            Tensor(shape=[2, 3], dtype=int64, place=Place(cpu), stop_gradient=True,
            [[-1,  5,  0],
             [-2, -3,  0]])
            >>> indices
            Tensor(shape=[2, 3], dtype=int64, place=Place(cpu), stop_gradient=True,
            [[0, 0, 0],
             [1, 1, 0]])

            >>> value, indices = paddle.cummin(data, axis=-1)
            >>> value
            Tensor(shape=[2, 3], dtype=int64, place=Place(cpu), stop_gradient=True,
            [[-1, -1, -1],
             [-2, -3, -3]])
            >>> indices
            Tensor(shape=[2, 3], dtype=int64, place=Place(cpu), stop_gradient=True,
            [[0, 0, 0],
             [0, 1, 1]])

            >>> value, indices = paddle.cummin(data, dtype='int64')
            >>> assert indices.dtype == paddle.int64
    """
    if axis is None:
        axis = -1
        x = x.flatten(0, len(x.shape) - 1)

    check_dtype(dtype, 'dtype', ['int32', 'int64'], 'cummin')
    dtype = convert_np_dtype_to_dtype_(dtype)

    if in_dynamic_mode():
        return _C_ops.cummin(x, axis, dtype)
    else:
        check_variable_and_dtype(
            x,
            'x',
            ['float32', 'float64', 'int32', 'int64'],
            'cummin',
        )
        check_type(x, 'x', (Variable), 'cummin')
        helper = LayerHelper('cummin', **locals())
        out = helper.create_variable_for_type_inference(x.dtype)
        indices = helper.create_variable_for_type_inference(dtype='int64')
        helper.append_op(
            type='cummin',
            inputs={'x': x},
            outputs={'out': out, 'indices': indices},
            attrs={'axis': axis, 'dtype': dtype},
        )
        return out, indices


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
        dtype (str, optional): The data type of the output tensor, can be float16, float32, float64. If specified, the input tensor is casted to dtype before the operation is performed. This is useful for preventing data type overflows. The default value is None.
        name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        Tensor, the result of logcumsumexp operator.

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> data = paddle.arange(12, dtype='float64')
            >>> data = paddle.reshape(data, (3, 4))

            >>> y = paddle.logcumsumexp(data)
            >>> y
            Tensor(shape=[12], dtype=float64, place=Place(cpu), stop_gradient=True,
            [0.         , 1.31326169 , 2.40760596 , 3.44018970 , 4.45191440 ,
             5.45619332 , 6.45776285 , 7.45833963 , 8.45855173 , 9.45862974 ,
             10.45865844, 11.45866900])

            >>> y = paddle.logcumsumexp(data, axis=0)
            >>> y
            Tensor(shape=[3, 4], dtype=float64, place=Place(cpu), stop_gradient=True,
            [[0.         , 1.         , 2.         , 3.         ],
             [4.01814993 , 5.01814993 , 6.01814993 , 7.01814993 ],
             [8.01847930 , 9.01847930 , 10.01847930, 11.01847930]])

            >>> y = paddle.logcumsumexp(data, axis=-1)
            >>> y
            Tensor(shape=[3, 4], dtype=float64, place=Place(cpu), stop_gradient=True,
            [[0.         , 1.31326169 , 2.40760596 , 3.44018970 ],
             [4.         , 5.31326169 , 6.40760596 , 7.44018970 ],
             [8.         , 9.31326169 , 10.40760596, 11.44018970]])

            >>> y = paddle.logcumsumexp(data, dtype='float64')
            >>> assert y.dtype == paddle.float64
    """
    if axis is None:
        flatten = True
    else:
        flatten = False
    if dtype is not None and x.dtype != convert_np_dtype_to_dtype_(dtype):
        x = cast(x, dtype)

    if in_dynamic_mode():
        if axis is None:
            axis = -1
        return _C_ops.logcumsumexp(x, axis, flatten, False, False)
    else:
        check_variable_and_dtype(
            x, 'x', ['float16', 'float32', 'float64', 'uint16'], "logcumsumexp"
        )

        helper = LayerHelper('logcumsumexp', **locals())
        out = helper.create_variable_for_type_inference(x.dtype)
        helper.append_op(
            type='logcumsumexp',
            inputs={'X': x},
            outputs={'Out': out},
            attrs={'axis': axis, 'flatten': flatten},
        )
        return out


def cumprod(x, dim=None, dtype=None, name=None):
    """
    Compute the cumulative product of the input tensor x along a given dimension dim.

    Note:
        The first element of the result is the same as the first element of the input.

    Args:
        x (Tensor): the input tensor need to be cumproded.
        dim (int, optional): the dimension along which the input tensor will be accumulated. It need to be in the range of [-x.rank, x.rank),
                    where x.rank means the dimensions of the input tensor x and -1 means the last dimension.
        dtype (str, optional): The data type of the output tensor, can be float32, float64, int32, int64, complex64,
                    complex128. If specified, the input tensor is casted to dtype before the operation is performed.
                    This is useful for preventing data type overflows. The default value is None.
        name (str, optional): Name for the operation (optional, default is None). For more information,
                    please refer to :ref:`api_guide_Name`.

    Returns:
        Tensor, the result of cumprod operator.

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> data = paddle.arange(12)
            >>> data = paddle.reshape(data, (3, 4))
            >>> data
            Tensor(shape=[3, 4], dtype=int64, place=Place(cpu), stop_gradient=True,
            [[0 , 1 , 2 , 3 ],
             [4 , 5 , 6 , 7 ],
             [8 , 9 , 10, 11]])

            >>> y = paddle.cumprod(data, dim=0)
            >>> y
            Tensor(shape=[3, 4], dtype=int64, place=Place(cpu), stop_gradient=True,
            [[0  , 1  , 2  , 3  ],
             [0  , 5  , 12 , 21 ],
             [0  , 45 , 120, 231]])

            >>> y = paddle.cumprod(data, dim=-1)
            >>> y
            Tensor(shape=[3, 4], dtype=int64, place=Place(cpu), stop_gradient=True,
            [[0   , 0   , 0   , 0   ],
             [4   , 20  , 120 , 840 ],
             [8   , 72  , 720 , 7920]])

            >>> y = paddle.cumprod(data, dim=1, dtype='float64')
            >>> y
            Tensor(shape=[3, 4], dtype=float64, place=Place(cpu), stop_gradient=True,
            [[0.   , 0.   , 0.   , 0.   ],
             [4.   , 20.  , 120. , 840. ],
             [8.   , 72.  , 720. , 7920.]])

            >>> assert y.dtype == paddle.float64

    """

    if dtype is not None and x.dtype != convert_np_dtype_to_dtype_(dtype):
        x = cast(x, dtype)

    if in_dynamic_mode():
        return _C_ops.cumprod(x, dim)
    else:
        check_variable_and_dtype(
            x,
            "x",
            [
                'complex64',
                'complex128',
                'float16',
                'uint16',
                'float32',
                'float64',
                'int32',
                'int64',
            ],
            'cumprod',
        )
        check_type(dim, 'dim', int, 'cumprod')

        helper = LayerHelper('cumprod', **locals())
        out = helper.create_variable_for_type_inference(x.dtype)
        helper.append_op(
            type='cumprod',
            inputs={'X': x},
            outputs={'Out': out},
            attrs={'dim': dim},
        )
        return out


@inplace_apis_in_dygraph_only
def cumprod_(x, dim=None, dtype=None, name=None):
    r"""
    Inplace version of ``cumprod`` API, the output Tensor will be inplaced with input ``x``.
    Please refer to :ref:`api_paddle_cumprod`.
    """
    if dtype is not None and x.dtype != convert_np_dtype_to_dtype_(dtype):
        x = cast_(x, dtype)

    if in_dynamic_mode():
        return _C_ops.cumprod_(x, dim)


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

            >>> import paddle

            >>> x = paddle.to_tensor([float('-inf'), -2, 3.6, float('inf'), 0, float('-nan'), float('nan')])
            >>> out = paddle.isfinite(x)
            >>> out
            Tensor(shape=[7], dtype=bool, place=Place(cpu), stop_gradient=True,
            [False, True , True , False, True , False, False])
    """
    if in_dynamic_mode():
        return _C_ops.isfinite(x)
    else:
        helper = LayerHelper("isfinite_v2", **locals())
        check_variable_and_dtype(
            x,
            'x',
            [
                'float16',
                'float32',
                'float64',
                'int32',
                'int64',
                'uint16',
            ],
            'isfinite',
        )
        out = helper.create_variable_for_type_inference('bool')
        helper.append_op(
            type="isfinite_v2", inputs={"X": x}, outputs={"Out": out}
        )
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

            >>> import paddle

            >>> x = paddle.to_tensor([float('-inf'), -2, 3.6, float('inf'), 0, float('-nan'), float('nan')])
            >>> out = paddle.isinf(x)
            >>> out
            Tensor(shape=[7], dtype=bool, place=Place(cpu), stop_gradient=True,
            [True , False, False, True , False, False, False])
    """
    if in_dynamic_mode():
        return _C_ops.isinf(x)
    else:
        helper = LayerHelper("isinf_v2", **locals())
        check_variable_and_dtype(
            x,
            'x',
            [
                'float16',
                'float32',
                'float64',
                'int32',
                'int64',
                'uint16',
            ],
            'isinf',
        )
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

            >>> import paddle

            >>> x = paddle.to_tensor([float('-inf'), -2, 3.6, float('inf'), 0, float('-nan'), float('nan')])
            >>> out = paddle.isnan(x)
            >>> out
            Tensor(shape=[7], dtype=bool, place=Place(cpu), stop_gradient=True,
            [False, False, False, False, False, True , True ])
    """
    if in_dynamic_mode():
        return _C_ops.isnan(x)
    else:
        helper = LayerHelper("isnan_v2", **locals())
        check_variable_and_dtype(
            x,
            'x',
            [
                'float16',
                'float32',
                'float64',
                'int32',
                'int64',
                'uint16',
            ],
            'isnan',
        )
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

            >>> import paddle

            >>> # the axis is a int element
            >>> x = paddle.to_tensor([[0.2, 0.3, 0.5, 0.9],
            ...                       [0.1, 0.2, 0.6, 0.7]])
            >>> out1 = paddle.prod(x)
            >>> out1
            Tensor(shape=[], dtype=float32, place=Place(cpu), stop_gradient=True,
            0.00022680)

            >>> out2 = paddle.prod(x, -1)
            >>> out2
            Tensor(shape=[2], dtype=float32, place=Place(cpu), stop_gradient=True,
            [0.02700000, 0.00840000])

            >>> out3 = paddle.prod(x, 0)
            >>> out3
            Tensor(shape=[4], dtype=float32, place=Place(cpu), stop_gradient=True,
            [0.02000000, 0.06000000, 0.30000001, 0.63000000])

            >>> out4 = paddle.prod(x, 0, keepdim=True)
            >>> out4
            Tensor(shape=[1, 4], dtype=float32, place=Place(cpu), stop_gradient=True,
            [[0.02000000, 0.06000000, 0.30000001, 0.63000000]])

            >>> out5 = paddle.prod(x, 0, dtype='int64')
            >>> out5
            Tensor(shape=[4], dtype=int64, place=Place(cpu), stop_gradient=True,
            [0, 0, 0, 0])

            >>> # the axis is list
            >>> y = paddle.to_tensor([[[1.0, 2.0], [3.0, 4.0]],
            ...                         [[5.0, 6.0], [7.0, 8.0]]])
            >>> out6 = paddle.prod(y, [0, 1])
            >>> out6
            Tensor(shape=[2], dtype=float32, place=Place(cpu), stop_gradient=True,
            [105., 384.])

            >>> out7 = paddle.prod(y, (1, 2))
            >>> out7
            Tensor(shape=[2], dtype=float32, place=Place(cpu), stop_gradient=True,
            [24.  , 1680.])

    """
    if dtype is not None:
        check_dtype(
            dtype,
            'dtype',
            ['float32', 'float64', 'int32', 'int64', "float16", "uint16"],
            'prod',
        )
        if x.dtype != convert_np_dtype_to_dtype_(dtype):
            x = cast(x, dtype)

    reduce_all, axis = _get_reduce_axis_with_tensor(axis, x)
    if in_dynamic_mode():
        return _C_ops.prod(x, axis, keepdim, reduce_all)
    else:
        helper = LayerHelper('reduce_prod', **locals())
        check_variable_and_dtype(
            x,
            'x/input',
            ['float32', 'float64', 'int32', 'int64', "float16", "uint16"],
            'reduce_prod',
        )
        out = helper.create_variable_for_type_inference(
            dtype=helper.input_dtype()
        )
        helper.append_op(
            type='reduce_prod',
            inputs={'X': x},
            outputs={'Out': out},
            attrs={'dim': axis, 'keep_dim': keepdim, 'reduce_all': reduce_all},
        )
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

            >>> import paddle

            >>> x = paddle.to_tensor([3.0, 0.0, -2.0, 1.7], dtype='float32')
            >>> out = paddle.sign(x=x)
            >>> out
            Tensor(shape=[4], dtype=float32, place=Place(cpu), stop_gradient=True,
            [ 1.,  0., -1.,  1.])
    """
    if in_dynamic_mode():
        return _C_ops.sign(x)
    else:
        check_variable_and_dtype(
            x, 'x', ['float16', 'float32', 'float64', 'uint16'], 'sign'
        )
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
        x (Tensor): Input of Tanh operator, an N-D Tensor, with data type bfloat16, float32, float64 or float16.
        name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        Output of Tanh operator, a Tensor with same data type and shape as input.

    Examples:

        .. code-block:: python

            >>> import paddle

            >>> x = paddle.to_tensor([-0.4, -0.2, 0.1, 0.3])
            >>> out = paddle.tanh(x)
            >>> out
            Tensor(shape=[4], dtype=float32, place=Place(cpu), stop_gradient=True,
            [-0.37994900, -0.19737528,  0.09966799,  0.29131261])
    """
    if in_dynamic_mode():
        return _C_ops.tanh(x)
    else:
        check_variable_and_dtype(
            x, 'x', ['uint16', 'float16', 'float32', 'float64'], 'tanh'
        )
        check_type(x, 'x', (Variable), 'tanh')
        helper = LayerHelper('tanh', **locals())
        out = helper.create_variable_for_type_inference(x.dtype)
        helper.append_op(type='tanh', inputs={'X': x}, outputs={'Out': out})
        return out


@inplace_apis_in_dygraph_only
def tanh_(x, name=None):
    r"""
    Inplace version of ``tanh`` API, the output Tensor will be inplaced with input ``x``.
    Please refer to :ref:`api_paddle_tanh`.
    """
    return _C_ops.tanh_(x)


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

            >>> import paddle

            >>> data = paddle.zeros(shape=[1], dtype='float32')
            >>> counter = paddle.increment(data)
            >>> counter
            Tensor(shape=[1], dtype=float32, place=Place(cpu), stop_gradient=True,
            [1.])

    """
    if in_dynamic_mode():
        return _C_ops.increment_(x, value)
    else:
        check_variable_and_dtype(
            x, 'x', ['float32', 'float64', 'int32', 'int64'], 'increment'
        )
        helper = LayerHelper("increment", **locals())
        helper.append_op(
            type='increment',
            inputs={'X': [x]},
            outputs={'Out': [x]},
            attrs={'step': float(value)},
        )
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

            >>> import paddle

            >>> # x is a bool Tensor with following elements:
            >>> #    [[True, False]
            >>> #     [True, True]]
            >>> x = paddle.to_tensor([[1, 0], [1, 1]], dtype='int32')
            >>> x
            Tensor(shape=[2, 2], dtype=int32, place=Place(cpu), stop_gradient=True,
            [[1, 0],
             [1, 1]])
            >>> x = paddle.cast(x, 'bool')

            >>> # out1 should be False
            >>> out1 = paddle.all(x)
            >>> out1
            Tensor(shape=[], dtype=bool, place=Place(cpu), stop_gradient=True,
            False)

            >>> # out2 should be [True, False]
            >>> out2 = paddle.all(x, axis=0)
            >>> out2
            Tensor(shape=[2], dtype=bool, place=Place(cpu), stop_gradient=True,
            [True , False])

            >>> # keepdim=False, out3 should be [False, True], out.shape should be (2,)
            >>> out3 = paddle.all(x, axis=-1)
            >>> out3
            Tensor(shape=[2], dtype=bool, place=Place(cpu), stop_gradient=True,
            [False, True ])

            >>> # keepdim=True, out4 should be [[False], [True]], out.shape should be (2, 1)
            >>> out4 = paddle.all(x, axis=1, keepdim=True)
            >>> out4
            Tensor(shape=[2, 1], dtype=bool, place=Place(cpu), stop_gradient=True,
            [[False],
             [True ]])

    """
    if in_dynamic_mode():
        return _C_ops.all(x, axis, keepdim)
    else:
        reduce_all, axis = _get_reduce_axis(axis, x)
        attrs = {
            'dim': axis,
            'keep_dim': keepdim,
            'reduce_all': reduce_all,
        }
        check_variable_and_dtype(
            x, 'x', ['bool', 'float32', 'float64', 'int32', 'int64'], 'all'
        )
        check_type(axis, 'axis', (int, list, tuple, type(None)), 'all')

        helper = LayerHelper('all', **locals())
        out = helper.create_variable_for_type_inference(dtype=paddle.bool)
        helper.append_op(
            type='reduce_all',
            inputs={'X': x},
            outputs={'Out': out},
            attrs=attrs,
        )
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

            >>> import paddle

            >>> x = paddle.to_tensor([[1, 0], [1, 1]], dtype='int32')
            >>> x = paddle.assign(x)
            >>> x
            Tensor(shape=[2, 2], dtype=int32, place=Place(cpu), stop_gradient=True,
            [[1, 0],
             [1, 1]])
            >>> x = paddle.cast(x, 'bool')
            >>> # x is a bool Tensor with following elements:
            >>> #    [[True, False]
            >>> #     [True, True]]

            >>> # out1 should be True
            >>> out1 = paddle.any(x)
            >>> out1
            Tensor(shape=[], dtype=bool, place=Place(cpu), stop_gradient=True,
            True)

            >>> # out2 should be [True, True]
            >>> out2 = paddle.any(x, axis=0)
            >>> out2
            Tensor(shape=[2], dtype=bool, place=Place(cpu), stop_gradient=True,
            [True, True])

            >>> # keepdim=False, out3 should be [True, True], out.shape should be (2,)
            >>> out3 = paddle.any(x, axis=-1)
            >>> out3
            Tensor(shape=[2], dtype=bool, place=Place(cpu), stop_gradient=True,
            [True, True])

            >>> # keepdim=True, result should be [[True], [True]], out.shape should be (2,1)
            >>> out4 = paddle.any(x, axis=1, keepdim=True)
            >>> out4
            Tensor(shape=[2, 1], dtype=bool, place=Place(cpu), stop_gradient=True,
            [[True],
             [True]])

    """
    if in_dynamic_mode():
        return _C_ops.any(x, axis, keepdim)
    else:
        reduce_all, axis = _get_reduce_axis(axis, x)
        attrs = {
            'dim': axis,
            'keep_dim': keepdim,
            'reduce_all': reduce_all,
        }
        check_variable_and_dtype(
            x, 'x', ['bool', 'float32', 'float64', 'int32', 'int64'], 'any'
        )
        check_type(axis, 'axis', (int, list, tuple, type(None)), 'any')

        helper = LayerHelper('any', **locals())
        out = helper.create_variable_for_type_inference(dtype=paddle.bool)
        helper.append_op(
            type='reduce_any',
            inputs={'X': x},
            outputs={'Out': out},
            attrs=attrs,
        )
        return out


def broadcast_shape(x_shape, y_shape):
    """
    The function returns the shape of doing operation with broadcasting on tensors of x_shape and y_shape.

    Note:
        If you want know more about broadcasting, please refer to `Introduction to Tensor`_ .

        .. _Introduction to Tensor: ../../guides/beginner/tensor_en.html#chapter5-broadcasting-of-tensor

    Args:
        x_shape (list[int]|tuple[int]): A shape of tensor.
        y_shape (list[int]|tuple[int]): A shape of tensor.


    Returns:
        list[int], the result shape.

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> shape = paddle.broadcast_shape([2, 1, 3], [1, 3, 1])
            >>> shape
            [2, 3, 3]

            >>> # shape = paddle.broadcast_shape([2, 1, 3], [3, 3, 1])
            >>> # ValueError (terminated with error message).

    """

    return core.broadcast_shape(x_shape, y_shape)


def conj(x, name=None):
    r"""
    This function computes the conjugate of the Tensor elementwisely.

    Args:
        x (Tensor): The input Tensor which hold the complex numbers.
            Optional data types are:float16, complex64, complex128, float32, float64, int32 or int64.
        name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        out (Tensor): The conjugate of input. The shape and data type is the same with input. If the elements of tensor is real type such as float32, float64, int32 or int64, the out is the same with input.

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> data = paddle.to_tensor([[1+1j, 2+2j, 3+3j], [4+4j, 5+5j, 6+6j]])
            >>> data
            Tensor(shape=[2, 3], dtype=complex64, place=Place(cpu), stop_gradient=True,
            [[(1+1j), (2+2j), (3+3j)],
             [(4+4j), (5+5j), (6+6j)]])

            >>> conj_data = paddle.conj(data)
            >>> conj_data
            Tensor(shape=[2, 3], dtype=complex64, place=Place(cpu), stop_gradient=True,
            [[(1-1j), (2-2j), (3-3j)],
             [(4-4j), (5-5j), (6-6j)]])

    """
    if in_dynamic_mode():
        return _C_ops.conj(x)
    else:
        check_variable_and_dtype(
            x,
            "x",
            [
                'complex64',
                'complex128',
                'float16',
                'uint16',
                'float32',
                'float64',
                'int32',
                'int64',
            ],
            'conj',
        )

        helper = LayerHelper('conj', **locals())
        out = helper.create_variable_for_type_inference(
            dtype=helper.input_dtype()
        )

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

            >>> import paddle

            >>> data = paddle.to_tensor([[1, 1.5], [0, -2.2]], dtype='float32')
            >>> res = paddle.digamma(data)
            >>> res
            Tensor(shape=[2, 2], dtype=float32, place=Place(cpu), stop_gradient=True,
            [[-0.57721591,  0.03648996],
             [ nan       ,  5.32286835]])
    """

    if in_dynamic_mode():
        return _C_ops.digamma(x)
    else:
        check_variable_and_dtype(
            x, 'x', ['float16', 'float32', 'float64', 'uint16'], 'digamma'
        )
        helper = LayerHelper('digamma', **locals())
        out = helper.create_variable_for_type_inference(x.dtype)
        helper.append_op(type='digamma', inputs={'X': x}, outputs={'Out': out})
        return out


@inplace_apis_in_dygraph_only
def digamma_(x, name=None):
    r"""
    Inplace version of ``digamma`` API, the output Tensor will be inplaced with input ``x``.
    Please refer to :ref:`api_paddle_digamma`.
    """
    if in_dynamic_mode():
        return _C_ops.digamma_(x)


def lgamma(x, name=None):
    r"""
    Calculates the lgamma of the given input tensor, element-wise.

    This operator performs elementwise lgamma for input $X$.
    :math:`out = log\Gamma(x)`


    Args:
        x (Tensor): Input Tensor. Must be one of the following types: float16, float32, float64, uint16.
        name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        Tensor, the lgamma of the input Tensor, the shape and data type is the same with input.

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> x = paddle.to_tensor([-0.4, -0.2, 0.1, 0.3])
            >>> out = paddle.lgamma(x)
            >>> out
            Tensor(shape=[4], dtype=float32, place=Place(cpu), stop_gradient=True,
            [1.31452453, 1.76149762, 2.25271273, 1.09579790])
    """
    if in_dynamic_mode():
        return _C_ops.lgamma(x)
    else:
        check_variable_and_dtype(
            x, 'x', ['float16', 'float32', 'float64', 'uint16'], 'lgamma'
        )
        helper = LayerHelper('lgamma', **locals())
        out = helper.create_variable_for_type_inference(x.dtype)
        helper.append_op(type='lgamma', inputs={'X': x}, outputs={'Out': out})
        return out


@inplace_apis_in_dygraph_only
def lgamma_(x, name=None):
    r"""
    Inplace version of ``lgamma`` API, the output Tensor will be inplaced with input ``x``.
    Please refer to :ref:`api_paddle_lgamma`.
    """
    if in_dynamic_mode():
        return _C_ops.lgamma_(x)


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

            >>> import paddle

            >>> x = paddle.to_tensor([-0.4, -0.2, 0.1, 0.3])
            >>> out = paddle.neg(x)
            >>> out
            Tensor(shape=[4], dtype=float32, place=Place(cpu), stop_gradient=True,
            [ 0.40000001,  0.20000000, -0.10000000, -0.30000001])
    """

    return scale(
        x, scale=-1.0, bias=0.0, bias_after_scale=True, act=None, name=name
    )


@inplace_apis_in_dygraph_only
def neg_(x, name=None):
    r"""
    Inplace version of ``neg`` API, the output Tensor will be inplaced with input ``x``.
    Please refer to :ref:`api_paddle_neg`.
    """
    return x.scale_(
        scale=-1.0, bias=0.0, bias_after_scale=True, act=None, name=name
    )


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

            >>> import paddle

            >>> x = paddle.to_tensor([-1, +1, +1, -1]).astype('float32')
            >>> x
            Tensor(shape=[4], dtype=float32, place=Place(cpu), stop_gradient=True,
            [-1,  1,  1, -1])

            >>> y = paddle.to_tensor([-1, -1, +1, +1]).astype('float32')
            >>> y
            Tensor(shape=[4], dtype=float32, place=Place(cpu), stop_gradient=True,
            [-1,  -1,  1, 1])

            >>> out = paddle.atan2(x, y)
            >>> out
            Tensor(shape=[4], dtype=float32, place=Place(cpu), stop_gradient=True,
            [-2.35619450,  2.35619450,  0.78539819, -0.78539819])

    """

    if in_dynamic_mode():
        return _C_ops.atan2(x, y)
    else:
        check_variable_and_dtype(
            x,
            'x',
            ['int32', 'int64', 'float16', 'float32', 'float64'],
            'atan2',
        )
        check_variable_and_dtype(
            y,
            'y',
            ['int32', 'int64', 'float16', 'float32', 'float64'],
            'atan2',
        )

        helper = LayerHelper('atan2', **locals())
        inputs = {'X1': x, 'X2': y}
        out = helper.create_variable_for_type_inference(dtype=x.dtype)
        helper.append_op(type='atan2', inputs=inputs, outputs={'Out': out})
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
        x (Tensor): The input Tensor with data type bfloat16, float16, float32, float64.
        eps (float, optional):  the epsilon for input clamp bound. Default is None.
        name (str, optional): Name for the operation (optional, default is None).
            For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        out(Tensor): A Tensor with the same data type and shape as ``x`` .

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> x = paddle.to_tensor([0.2635, 0.0106, 0.2780, 0.2097, 0.8095])
            >>> out1 = paddle.logit(x)
            >>> out1
            Tensor(shape=[5], dtype=float32, place=Place(cpu), stop_gradient=True,
            [-1.02785587, -4.53624487, -0.95440406, -1.32673466,  1.44676447])

    """
    if eps is None:
        eps = 0.0
    if in_dynamic_mode():
        return _C_ops.logit(x, eps)
    else:
        check_variable_and_dtype(
            x, 'x', ['float16', 'uint16', 'float32', 'float64'], 'logit'
        )
        helper = LayerHelper("logit", **locals())
        out = helper.create_variable_for_type_inference(x.dtype)
        helper.append_op(
            type='logit',
            inputs={'X': x},
            outputs={'Out': out},
            attrs={'eps': eps},
        )
        return out


@inplace_apis_in_dygraph_only
def logit_(x, eps=None, name=None):
    r"""
    Inplace version of ``logit`` API, the output Tensor will be inplaced with input ``x``.
    Please refer to :ref:`api_paddle_logit`.
    """
    if eps is None:
        eps = 0.0
    if in_dynamic_mode():
        return _C_ops.logit_(x, eps)


def lerp(x, y, weight, name=None):
    r"""
    Does a linear interpolation between x and y based on weight.

    Equation:
        .. math::

            lerp(x, y, weight) = x + weight * (y - x).

    Args:
        x (Tensor): An N-D Tensor with starting points, the data type is bfloat16, float16, float32, float64.
        y (Tensor): An N-D Tensor with ending points, the data type is bfloat16, float16, float32, float64.
        weight (float|Tensor): The weight for the interpolation formula. When weight is Tensor, the data type is bfloat16, float16, float32, float64.
        name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        out (Tensor): An N-D Tensor, the shape and data type is the same with input.

    Example:
        .. code-block:: python

            >>> import paddle

            >>> x = paddle.arange(1., 5., dtype='float32')
            >>> y = paddle.empty([4], dtype='float32')
            >>> y.fill_(10.)
            >>> out = paddle.lerp(x, y, 0.5)
            >>> out
            Tensor(shape=[4], dtype=float32, place=Place(cpu), stop_gradient=True,
            [5.50000000, 6.        , 6.50000000, 7.        ])

    """
    if isinstance(weight, float):
        weight = paddle.full(shape=[], fill_value=weight, dtype=x.dtype)

    if in_dynamic_mode():
        return _C_ops.lerp(x, y, weight)
    else:
        check_variable_and_dtype(
            x, 'x', ['uint16', 'float16', 'float32', 'float64'], 'lerp'
        )
        check_variable_and_dtype(
            y, 'y', ['uint16', 'float16', 'float32', 'float64'], 'lerp'
        )
        check_variable_and_dtype(
            weight,
            'weight',
            ['uint16', 'float16', 'float32', 'float64'],
            'lerp',
        )

        helper = LayerHelper('lerp', **locals())
        inputs = {'X': x, 'Y': y, 'Weight': weight}
        out = helper.create_variable_for_type_inference(dtype=x.dtype)
        helper.append_op(type='lerp', inputs=inputs, outputs={'Out': out})
        return out


@inplace_apis_in_dygraph_only
def lerp_(x, y, weight, name=None):
    r"""
    Inplace version of ``lerp`` API, the output Tensor will be inplaced with input ``x``.
    Please refer to :ref:`api_paddle_lerp`.
    """
    out_shape = broadcast_shape(x.shape, y.shape)
    check_type(weight, 'weight', (float, paddle.Tensor, Variable), 'lerp')
    if isinstance(weight, float):
        weight = paddle.to_tensor([weight], dtype=x.dtype)
    elif isinstance(weight, (paddle.Tensor, Variable)):
        out_shape = broadcast_shape(out_shape, weight.shape)
    if out_shape != x.shape:
        raise ValueError(
            "The shape of broadcast output {} is different from that of inplace tensor {} in the Inplace operation.".format(
                out_shape, x.shape
            )
        )
    return _C_ops.lerp_(x, y, weight)


def erfinv(x, name=None):
    r"""
    The inverse error function of x. Please refer to :ref:`api_paddle_erf`

        .. math::

            erfinv(erf(x)) = x.

    Args:
        x (Tensor): An N-D Tensor, the data type is float16, bfloat16, float32, float64.
        name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        out (Tensor), an N-D Tensor, the shape and data type is the same with input.

    Example:
        .. code-block:: python

            >>> import paddle

            >>> x = paddle.to_tensor([0, 0.5, -1.], dtype="float32")
            >>> out = paddle.erfinv(x)
            >>> out
            Tensor(shape=[3], dtype=float32, place=Place(cpu), stop_gradient=True,
            [ 0.       , 0.47693631, -inf.     ])

    """
    if in_dynamic_mode():
        return _C_ops.erfinv(x)
    else:
        check_variable_and_dtype(
            x, 'x', ['float32', 'float64', 'float16', 'uint16'], 'erfinv'
        )
        helper = LayerHelper('erfinv', **locals())
        out = helper.create_variable_for_type_inference(dtype=x.dtype)
        helper.append_op(type='erfinv', inputs={'X': x}, outputs={'Out': out})
        return out


@inplace_apis_in_dygraph_only
def erfinv_(x, name=None):
    r"""
    Inplace version of ``erfinv`` API, the output Tensor will be inplaced with input ``x``.
    Please refer to :ref:`api_paddle_erfinv`.
    """
    check_type(x, 'x', (paddle.Tensor, Variable), 'erfinv')
    return _C_ops.erfinv_(x)


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

            >>> import paddle
            >>> import math

            >>> x1 = paddle.to_tensor([3.142, -3.142, 6.283, -6.283, 1.570, -1.570])
            >>> result1 = paddle.rad2deg(x1)
            >>> result1
            Tensor(shape=[6], dtype=float32, place=Place(cpu), stop_gradient=True,
            [ 180.02334595, -180.02334595,  359.98937988, -359.98937988,
              89.95437622 , -89.95437622 ])

            >>> x2 = paddle.to_tensor(math.pi/2)
            >>> result2 = paddle.rad2deg(x2)
            >>> result2
            Tensor(shape=[], dtype=float32, place=Place(cpu), stop_gradient=True,
            90.)

            >>> x3 = paddle.to_tensor(1)
            >>> result3 = paddle.rad2deg(x3)
            >>> result3
            Tensor(shape=[], dtype=float32, place=Place(cpu), stop_gradient=True,
            57.29578018)
    """
    rad2deg_scale = 180 / np.pi
    if in_dynamic_mode():
        if convert_dtype(x.dtype) in ['int32', 'int64']:
            x = cast(x, dtype="float32")
        return _C_ops.scale(x, rad2deg_scale, 0.0, True)
    else:
        check_variable_and_dtype(
            x, 'x', ['int32', 'int64', 'float32', 'float64'], 'rad2deg'
        )
        helper = LayerHelper('rad2deg', **locals())
        out_cast = x
        if convert_dtype(x.dtype) in ['int32', 'int64']:
            out_cast = helper.create_variable_for_type_inference(
                dtype=paddle.float32
            )
            helper.append_op(
                type='cast',
                inputs={'X': x},
                outputs={'Out': out_cast},
                attrs={'in_dtype': x.dtype, 'out_dtype': paddle.float32},
            )
        out = helper.create_variable_for_type_inference(dtype=out_cast.dtype)
        helper.append_op(
            type='scale',
            inputs={'X': out_cast},
            outputs={'Out': out},
            attrs={'scale': rad2deg_scale},
        )
        return out


def deg2rad(x, name=None):
    r"""
    Convert each of the elements of input x from degrees to angles in radians.

        .. math::

            deg2rad(x)=\pi * x / 180

    Args:
        x (Tensor): An N-D Tensor, the data type is float32, float64, int32, int64.
        name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        out (Tensor): An N-D Tensor, the shape and data type is the same with input (The output data type is float32 when the input data type is int).

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> x1 = paddle.to_tensor([180.0, -180.0, 360.0, -360.0, 90.0, -90.0])
            >>> result1 = paddle.deg2rad(x1)
            >>> result1
            Tensor(shape=[6], dtype=float32, place=Place(cpu), stop_gradient=True,
            [3.14159274, -3.14159274,  6.28318548, -6.28318548,  1.57079637,
            -1.57079637])

            >>> x2 = paddle.to_tensor(180)
            >>> result2 = paddle.deg2rad(x2)
            >>> result2
            Tensor(shape=[], dtype=float32, place=Place(cpu), stop_gradient=True,
            3.14159274)
    """
    deg2rad_scale = np.pi / 180.0
    if in_dynamic_mode():
        if convert_dtype(x.dtype) in ['int32', 'int64']:
            x = cast(x, dtype="float32")
        return _C_ops.scale(x, deg2rad_scale, 0.0, True)
    else:
        check_variable_and_dtype(
            x, 'x', ['int32', 'int64', 'float32', 'float64'], 'deg2rad'
        )
        helper = LayerHelper('deg2rad', **locals())
        out_cast = x
        if convert_dtype(x.dtype) in ['int32', 'int64']:
            out_cast = helper.create_variable_for_type_inference(
                dtype=paddle.float32
            )
            helper.append_op(
                type='cast',
                inputs={'X': x},
                outputs={'Out': out_cast},
                attrs={'in_dtype': x.dtype, 'out_dtype': paddle.float32},
            )
        out = helper.create_variable_for_type_inference(dtype=out_cast.dtype)
        helper.append_op(
            type='scale',
            inputs={'X': out_cast},
            outputs={'Out': out},
            attrs={'scale': deg2rad_scale},
        )
        return out


def gcd(x, y, name=None):
    """
    Computes the element-wise greatest common divisor (GCD) of input |x| and |y|.
    Both x and y must have integer types.

    Note:
        gcd(0,0)=0, gcd(0, y)=|y|

        If x.shape != y.shape, they must be broadcastable to a common shape (which becomes the shape of the output).

    Args:
        x (Tensor): An N-D Tensor, the data type is int32, int64.
        y (Tensor): An N-D Tensor, the data type is int32, int64.
        name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        out (Tensor): An N-D Tensor, the data type is the same with input.

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> x1 = paddle.to_tensor(12)
            >>> x2 = paddle.to_tensor(20)
            >>> paddle.gcd(x1, x2)
            Tensor(shape=[], dtype=int64, place=Place(cpu), stop_gradient=True,
            4)

            >>> x3 = paddle.arange(6)
            >>> paddle.gcd(x3, x2)
            Tensor(shape=[6], dtype=int64, place=Place(cpu), stop_gradient=True,
            [20, 1 , 2 , 1 , 4 , 5])

            >>> x4 = paddle.to_tensor(0)
            >>> paddle.gcd(x4, x2)
            Tensor(shape=[], dtype=int64, place=Place(cpu), stop_gradient=True,
            20)

            >>> paddle.gcd(x4, x4)
            Tensor(shape=[], dtype=int64, place=Place(cpu), stop_gradient=True,
            0)

            >>> x5 = paddle.to_tensor(-20)
            >>> paddle.gcd(x1, x5)
            Tensor(shape=[], dtype=int64, place=Place(cpu), stop_gradient=True,
            4)
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
        y_not_equal_0 = y != 0
        y_safe = paddle.where(y_not_equal_0, y, paddle.ones(y.shape, y.dtype))
        x, y = (
            paddle.where(y_not_equal_0, y, x),
            paddle.where(
                y_not_equal_0,
                paddle.mod(x, y_safe),
                paddle.zeros(y.shape, y.dtype),
            ),
        )
        return (paddle.where(x < y, y, x), paddle.where(x < y, x, y))

    if in_dynamic_mode():
        while _gcd_cond_fn(x, y):
            x, y = _gcd_body_fn(x, y)

        return x
    else:
        check_variable_and_dtype(x, 'x', ['int32', 'int64'], 'gcd')
        check_variable_and_dtype(y, 'y', ['int32', 'int64'], 'gcd')
        out, _ = paddle.static.nn.while_loop(_gcd_cond_fn, _gcd_body_fn, [x, y])
        return out


def gcd_(x, y, name=None):
    r"""
    Inplace version of ``gcd`` API, the output Tensor will be inplaced with input ``x``.
    Please refer to :ref:`api_paddle_gcd`.
    """
    shape = paddle.broadcast_shape(x.shape, y.shape)
    if shape != x.shape:
        raise ValueError(
            "The shape of broadcast output {} is different from that of inplace tensor {} in the Inplace operation.".format(
                shape, x.shape
            )
        )
    y = paddle.broadcast_to(y, shape)
    x = paddle.abs_(x)
    y = paddle.abs(y)

    def _gcd_cond_fn(x, y):
        return paddle.any(y != 0)

    def _gcd_body_fn(x, y):
        # paddle.mod will raise an error when any element of y is 0. To avoid
        # that, we change those zeros to ones. Their values don't matter because
        # they won't be used.
        y_equal_0 = y == 0
        y_safe = paddle.where(y_equal_0, paddle.ones(y.shape, y.dtype), y)
        y, x = (
            paddle.where(
                y_equal_0,
                paddle.zeros(y.shape, y.dtype),
                paddle.mod(x, y_safe),
            ),
            paddle.where_(y_equal_0, x, y),
        )
        return (
            paddle.where(x < y, x, y),
            paddle.where_(x >= y, x, y),
        )

    if in_dynamic_mode():
        while _gcd_cond_fn(x, y):
            y, x = _gcd_body_fn(x, y)

        return x


def lcm(x, y, name=None):
    """
    Computes the element-wise least common multiple (LCM) of input |x| and |y|.
    Both x and y must have integer types.

    Note:
        lcm(0,0)=0, lcm(0, y)=0

        If x.shape != y.shape, they must be broadcastable to a common shape (which becomes the shape of the output).

    Args:
        x (Tensor): An N-D Tensor, the data type is int32, int64.
        y (Tensor): An N-D Tensor, the data type is int32, int64.
        name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        out (Tensor): An N-D Tensor, the data type is the same with input.

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> x1 = paddle.to_tensor(12)
            >>> x2 = paddle.to_tensor(20)
            >>> paddle.lcm(x1, x2)
            Tensor(shape=[], dtype=int64, place=Place(cpu), stop_gradient=True,
            60)

            >>> x3 = paddle.arange(6)
            >>> paddle.lcm(x3, x2)
            Tensor(shape=[6], dtype=int64, place=Place(cpu), stop_gradient=True,
            [0, 20, 20, 60, 20, 20])

            >>> x4 = paddle.to_tensor(0)
            >>> paddle.lcm(x4, x2)
            Tensor(shape=[], dtype=int64, place=Place(cpu), stop_gradient=True,
            0)

            >>> paddle.lcm(x4, x4)
            Tensor(shape=[], dtype=int64, place=Place(cpu), stop_gradient=True,
            0)

            >>> x5 = paddle.to_tensor(-20)
            >>> paddle.lcm(x1, x5)
            Tensor(shape=[], dtype=int64, place=Place(cpu), stop_gradient=True,
            60)
    """
    d = paddle.gcd(x, y)
    # paddle.mod will raise an error when any element of y is 0. To avoid
    # that, we change those zeros to ones. Their values don't matter because
    # they won't be used.
    d_equal_0 = paddle.equal(d, 0)
    d_safe = paddle.where(d_equal_0, paddle.ones(d.shape, d.dtype), d)
    out = paddle.where(
        d_equal_0, paddle.zeros(d.shape, d.dtype), paddle.abs(x * y) // d_safe
    )
    return out


def lcm_(x, y, name=None):
    r"""
    Inplace version of ``lcm`` API, the output Tensor will be inplaced with input ``x``.
    Please refer to :ref:`api_paddle_lcm`.
    """
    d = paddle.gcd(x, y)
    # paddle.mod will raise an error when any element of y is 0. To avoid
    # that, we change those zeros to ones. Their values don't matter because
    # they won't be used.
    d_not_equal_0 = d != 0
    d_safe = paddle.where(d_not_equal_0, d, paddle.ones(d.shape, d.dtype))
    out = paddle.where_(
        d_not_equal_0,
        paddle.abs_(x.multiply_(y)).floor_divide_(d_safe),
        paddle.zeros(d.shape, d.dtype),
    )
    return out


def diff(x, n=1, axis=-1, prepend=None, append=None, name=None):
    r"""
    Computes the n-th forward difference along the given axis.
    The first-order differences is computed by using the following formula:

    .. math::

        out[i] = x[i+1] - x[i]

    Higher-order differences are computed by using paddle.diff() recursively.
    The number of n supports any positive integer value.

    Args:
        x (Tensor): The input tensor to compute the forward difference on, the data type is float16, float32, float64, bool, int32, int64.
        n (int, optional): The number of times to recursively compute the difference.
                            Supports any positive integer value. Default:1
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

            >>> import paddle

            >>> x = paddle.to_tensor([1, 4, 5, 2])
            >>> out = paddle.diff(x)
            >>> out
            Tensor(shape=[3], dtype=int64, place=Place(cpu), stop_gradient=True,
            [ 3,  1, -3])

            >>> x_2 = paddle.to_tensor([1, 4, 5, 2])
            >>> out = paddle.diff(x_2, n=2)
            >>> out
            Tensor(shape=[2], dtype=int64, place=Place(cpu), stop_gradient=True,
            [ -2,  -4])

            >>> y = paddle.to_tensor([7, 9])
            >>> out = paddle.diff(x, append=y)
            >>> out
            Tensor(shape=[5], dtype=int64, place=Place(cpu), stop_gradient=True,
            [ 3,  1, -3,  5,  2])

            >>> z = paddle.to_tensor([[1, 2, 3], [4, 5, 6]])
            >>> out = paddle.diff(z, axis=0)
            >>> out
            Tensor(shape=[1, 3], dtype=int64, place=Place(cpu), stop_gradient=True,
            [[3, 3, 3]])
            >>> out = paddle.diff(z, axis=1)
            >>> out
            Tensor(shape=[2, 2], dtype=int64, place=Place(cpu), stop_gradient=True,
            [[1, 1],
             [1, 1]])
    """
    if n < 1:
        raise ValueError(
            "Diff expects input to be at least one-dimensional but got {}".format(
                n
            )
        )

    def _diff_handler(x, n=1, axis=-1, prepend=None, append=None, name=None):
        if axis < 0:
            axis = axis + len(x.shape)
        if axis > len(x.shape):
            axis = len(x.shape)
        if axis < 0:
            axis = 0
        dtype = x.dtype
        axes = [axis]
        infer_flags = [1 for i in range(len(axes))]
        if in_dynamic_mode():
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
            input_front = _C_ops.slice(
                new_input, axes, starts_1, ends_1, infer_flags, []
            )
            starts_2 = [1]
            attrs_2 += ('starts', starts_2)
            ends_2 = [dim_len]
            attrs_2 += ('ends', ends_2)
            input_back = _C_ops.slice(
                new_input, axes, starts_2, ends_2, infer_flags, []
            )

            if x.dtype == paddle.bool:
                return _C_ops.logical_xor(input_back, input_front)
            else:
                return _C_ops.subtract(input_back, input_front)
        else:
            check_variable_and_dtype(
                x,
                'x',
                ['float16', 'float32', 'float64', 'bool', 'int32', 'int64'],
                'diff',
            )
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
                    type='concat',
                    inputs={'X': input_list},
                    outputs={'Out': [new_input]},
                    attrs={'axis': axis},
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
                type='slice',
                inputs={'Input': new_input},
                attrs=attrs_1,
                outputs={'Out': input_front},
            )
            attrs_2 = {'axes': axes}
            starts_2 = [1]
            ends_2 = [dim_len]
            attrs_2['starts'] = starts_2
            attrs_2['ends'] = ends_2
            input_back = helper.create_variable_for_type_inference(dtype)
            helper.append_op(
                type='slice',
                inputs={'Input': new_input},
                attrs=attrs_2,
                outputs={'Out': input_back},
            )

            if dtype == paddle.bool:
                out = helper.create_variable_for_type_inference(dtype)
                helper.append_op(
                    type='logical_xor',
                    inputs={"X": input_back, "Y": input_front},
                    outputs={"Out": out},
                )
            else:
                out = paddle.tensor.math.subtract(input_back, input_front)
            return out

    out = _diff_handler(
        x, n=1, axis=axis, prepend=prepend, append=append, name=name
    )
    if n > 1:
        for _ in range(n - 1):
            out = _diff_handler(
                out, n=1, axis=axis, prepend=prepend, append=append, name=name
            )
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

            >>> import paddle

            >>> x = paddle.to_tensor([-2, -1, 0, 1]).unsqueeze(-1).astype('float32')
            >>> y = paddle.to_tensor([-2, -1, 0, 1]).astype('float32')
            >>> z = x + 1j * y
            >>> z
            Tensor(shape=[4, 4], dtype=complex64, place=Place(cpu), stop_gradient=True,
            [[(-2-2j), (-2-1j), (-2+0j), (-2+1j)],
             [(-1-2j), (-1-1j), (-1+0j), (-1+1j)],
             [-2j    , -1j    ,  0j    ,  1j    ],
             [ (1-2j),  (1-1j),  (1+0j),  (1+1j)]])

            >>> theta = paddle.angle(z)
            >>> theta
            Tensor(shape=[4, 4], dtype=float32, place=Place(cpu), stop_gradient=True,
            [[-2.35619450, -2.67794514,  3.14159274,  2.67794514],
             [-2.03444386, -2.35619450,  3.14159274,  2.35619450],
             [-1.57079637, -1.57079637,  0.        ,  1.57079637],
             [-1.10714877, -0.78539819,  0.        ,  0.78539819]])
    """

    if in_dynamic_mode():
        return _C_ops.angle(x)
    else:
        check_variable_and_dtype(
            x,
            'x',
            [
                'float16',
                'float32',
                'float64',
                'complex64',
                'complex128',
                'uint16',
            ],
            'angle',
        )
        op_type = "angle"
        helper = LayerHelper(op_type, **locals())
        inputs = {"X": x}
        out = helper.create_variable_for_type_inference(
            dtype=_complex_to_real_dtype(x.dtype)
        )
        outputs = {"Out": out}
        helper.append_op(type=op_type, inputs=inputs, outputs=outputs)
        return out


def heaviside(x, y, name=None):
    r"""
    Computes the Heaviside step function determined by corresponding element in y for each element in x. The equation is

    .. math::
        heaviside(x, y)=
            \left\{
                \begin{array}{lcl}
                0,& &\text{if} \ x < 0, \\
                y,& &\text{if} \ x = 0, \\
                1,& &\text{if} \ x > 0.
                \end{array}
            \right.

    Note:
        ``paddle.heaviside`` supports broadcasting. If you want know more about broadcasting, please refer to `Introduction to Tensor`_ .

        .. _Introduction to Tensor: ../../guides/beginner/tensor_en.html#chapter5-broadcasting-of-tensor

    Args:
        x (Tensor): The input tensor of Heaviside step function, it's data type should be float16, float32, float64, int32 or int64.
        y (Tensor): The tensor that determines a Heaviside step function, it's data type should be float16, float32, float64, int32 or int64.
        name (str, optional): Name for the operation (optional, default is None). Normally there is no need for user to set this property. For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        N-D Tensor. A location into which the result is stored. If x and y have different shapes and are broadcastable, the resulting tensor shape is the shape of x and y after broadcasting. If x, y have the same shape, its shape is the same as x and y.

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> x = paddle.to_tensor([-0.5, 0, 0.5])
            >>> y = paddle.to_tensor([0.1])
            >>> paddle.heaviside(x, y)
            Tensor(shape=[3], dtype=float32, place=Place(cpu), stop_gradient=True,
            [0.        , 0.10000000, 1.        ])
            >>> x = paddle.to_tensor([[-0.5, 0, 0.5], [-0.5, 0.5, 0]])
            >>> y = paddle.to_tensor([0.1, 0.2, 0.3])
            >>> paddle.heaviside(x, y)
            Tensor(shape=[2, 3], dtype=float32, place=Place(cpu), stop_gradient=True,
            [[0.        , 0.20000000, 1.        ],
             [0.        , 1.        , 0.30000001]])
    """
    if in_dynamic_mode():
        return _C_ops.heaviside(x, y)
    else:
        op_type = 'elementwise_heaviside'
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

            >>> import paddle

            >>> input = paddle.to_tensor([[12.22000003, -1.02999997],
            ...                           [-0.54999995, 0.66000003]])
            >>> output = paddle.frac(input)
            >>> output
            Tensor(shape=[2, 2], dtype=float32, place=Place(cpu), stop_gradient=True,
            [[ 0.22000003, -0.02999997],
             [-0.54999995,  0.66000003]])
    """
    if x.dtype not in [
        paddle.int32,
        paddle.int64,
        paddle.float32,
        paddle.float64,
    ]:
        raise TypeError(
            "The data type of input must be one of ['int32', 'int64', 'float32', 'float64'], but got {}".format(
                x.dtype
            )
        )
    if in_dynamic_mode():
        y = _C_ops.trunc(x)
        return _C_ops.subtract(x, y)
    else:
        inputs = {"X": x}
        attrs = {}

        helper = LayerHelper("trunc", **locals())
        check_variable_and_dtype(
            x, "X", ['int32', 'int64', 'float32', 'float64'], 'trunc'
        )
        y = helper.create_variable_for_type_inference(dtype=x.dtype)
        helper.append_op(
            type="trunc", inputs=inputs, attrs=attrs, outputs={"Out": y}
        )
        return _elementwise_op(LayerHelper('elementwise_sub', **locals()))


@inplace_apis_in_dygraph_only
def frac_(x, name=None):
    r"""
    Inplace version of ``frac`` API, the output Tensor will be inplaced with input ``x``.
    Please refer to :ref:`api_paddle_frac`.
    """

    if x.dtype not in [
        paddle.int32,
        paddle.int64,
        paddle.float32,
        paddle.float64,
    ]:
        raise TypeError(
            "The data type of input must be one of ['int32', 'int64', 'float32', 'float64'], but got {}".format(
                x.dtype
            )
        )
    if in_dynamic_mode():
        y = _C_ops.trunc(x)
        return _C_ops.subtract_(x, y)


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
        .. code-block:: python

            >>> import paddle

            >>> x = paddle.to_tensor([[3 + 4j, 7 - 24j, 0, 1 + 2j], [6 + 8j, 3, 0, -2]])
            >>> paddle.sgn(x)
            Tensor(shape=[2, 4], dtype=complex64, place=Place(cpu), stop_gradient=True,
            [[ (0.6000000238418579+0.800000011920929j),
              (0.2800000011920929-0.9599999785423279j),
               0j                                     ,
              (0.4472135901451111+0.8944271802902222j)],
             [ (0.6000000238418579+0.800000011920929j),
               (1+0j)                                 ,
               0j                                     ,
              (-1+0j)                                 ]])

    """
    if x.dtype not in [
        paddle.float16,
        paddle.float32,
        paddle.float64,
        paddle.complex64,
        paddle.complex128,
    ]:
        raise TypeError(
            "The data type of input must be one of ['float16', 'float32', 'float64', 'complex64', 'complex128'], but got {}".format(
                x.dtype
            )
        )
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

            >>> import paddle

            >>> x_int = paddle.arange(0, 12).reshape([3, 4])
            >>> x_float = x_int.astype(paddle.float64)

            >>> idx_pos = paddle.arange(4, 10).reshape([2, 3])  # positive index
            >>> idx_neg = paddle.arange(-2, 4).reshape([2, 3])  # negative index
            >>> idx_err = paddle.arange(-2, 13).reshape([3, 5])  # index out of range

            >>> paddle.take(x_int, idx_pos)
            Tensor(shape=[2, 3], dtype=int64, place=Place(cpu), stop_gradient=True,
            [[4, 5, 6],
             [7, 8, 9]])

            >>> paddle.take(x_int, idx_neg)
            Tensor(shape=[2, 3], dtype=int64, place=Place(cpu), stop_gradient=True,
            [[10, 11, 0 ],
             [1 , 2 , 3 ]])

            >>> paddle.take(x_float, idx_pos)
            Tensor(shape=[2, 3], dtype=float64, place=Place(cpu), stop_gradient=True,
            [[4., 5., 6.],
             [7., 8., 9.]])

            >>> x_int.take(idx_pos)
            Tensor(shape=[2, 3], dtype=int64, place=Place(cpu), stop_gradient=True,
            [[4, 5, 6],
             [7, 8, 9]])

            >>> paddle.take(x_int, idx_err, mode='wrap')
            Tensor(shape=[3, 5], dtype=int64, place=Place(cpu), stop_gradient=True,
            [[10, 11, 0 , 1 , 2 ],
             [3 , 4 , 5 , 6 , 7 ],
             [8 , 9 , 10, 11, 0 ]])

            >>> paddle.take(x_int, idx_err, mode='clip')
            Tensor(shape=[3, 5], dtype=int64, place=Place(cpu), stop_gradient=True,
            [[0 , 0 , 0 , 1 , 2 ],
             [3 , 4 , 5 , 6 , 7 ],
             [8 , 9 , 10, 11, 11]])

    """
    if mode not in ['raise', 'wrap', 'clip']:
        raise ValueError(
            "'mode' in 'take' should be 'raise', 'wrap', 'clip', but received {}.".format(
                mode
            )
        )

    if in_dynamic_mode():
        if not isinstance(index, (paddle.Tensor, Variable)):
            raise TypeError(
                "The type of 'index' must be Tensor, but got {}".format(
                    type(index)
                )
            )
        if index.dtype not in [paddle.int32, paddle.int64]:
            raise TypeError(
                "The data type of 'index' must be one of ['int32', 'int64'], but got {}".format(
                    index.dtype
                )
            )

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
        index_1d = paddle.where(index_1d < 0, index_1d % max_index, index_1d)
        index_1d = paddle.where(
            index_1d >= max_index, index_1d % max_index, index_1d
        )
    elif mode == 'clip':
        # 'clip' mode disables indexing with negative numbers.
        index_1d = clip(index_1d, 0, max_index - 1)

    out = input_1d.index_select(index_1d).reshape(index.shape)

    return out


def frexp(x, name=None):
    """
    The function used to decompose a floating point number into mantissa and exponent.

    Args:
        x (Tensor): The input tensor, it's data type should be float32, float64.
        name (str, optional): For details, please refer to :ref:`api_guide_Name`. Generally, no setting is required. Default: None.

    Returns:

        - mantissa (Tensor), A mantissa Tensor. The shape and data type of mantissa tensor and exponential tensor are
            the same as those of input.

        - exponent (Tensor), A exponent Tensor. The shape and data type of mantissa tensor and exponential tensor are
            the same as those of input.

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> x = paddle.to_tensor([[1, 2, 3, 4]], dtype="float32")
            >>> mantissa, exponent = paddle.tensor.math.frexp(x)
            >>> mantissa
            Tensor(shape=[1, 4], dtype=float32, place=Place(cpu), stop_gradient=True,
            [[0.50000000, 0.50000000, 0.75000000, 0.50000000]])
            >>> exponent
            Tensor(shape=[1, 4], dtype=float32, place=Place(cpu), stop_gradient=True,
            [[1., 2., 2., 3.]])
    """
    if x.dtype not in [paddle.float32, paddle.float64]:
        raise TypeError(
            "The data type of input must be one of ['float32', 'float64'], but got {}".format(
                x.dtype
            )
        )
    input_x = paddle.abs(x)
    exponent = paddle.floor(paddle.log2(input_x))
    exponent = paddle.where(
        paddle.isinf(exponent), paddle.full_like(exponent, 0), exponent
    )

    # 0填充
    mantissa = paddle.divide(input_x, 2**exponent)
    # 计算exponent
    exponent = paddle.where(
        (mantissa >= 1),
        paddle.add(exponent, paddle.ones_like(exponent)),
        exponent,
    )
    mantissa = paddle.where(
        (mantissa >= 1),
        paddle.divide(mantissa, 2 ** paddle.ones_like(exponent)),
        mantissa,
    )

    mantissa = paddle.where((x < 0), mantissa * -1, mantissa)
    return mantissa, exponent


def _trapezoid(y, x=None, dx=None, axis=-1, mode='sum'):
    """
    Integrate along the given axis using the composite trapezoidal rule.

    Args:
        y (Tensor): Input tensor to integrate. It's data type should be float16, float32, float64.
        x (Tensor, optional): The sample points corresponding to the :attr:`y` values, the same type as :attr:`y`.
            It is known that the size of :attr:`y` is `[d_1, d_2, ... , d_n]` and :math:`axis=k`, then the size of :attr:`x` can only be `[d_k]` or `[d_1, d_2, ... , d_n ]`.
            If :attr:`x` is None, the sample points are assumed to be evenly spaced :attr:`dx` apart. The default is None.
        dx (float, optional): The spacing between sample points when :attr:`x` is None. If neither :attr:`x` nor :attr:`dx` is provided then the default is :math:`dx = 1`.
        axis (int, optional): The axis along which to integrate. The default is -1.
        name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.
        sum_mode (str): use a different summation. The default is `sum`.

    Returns:
        Tensor, Definite integral of :attr:`y` is N-D tensor as approximated along a single axis by the trapezoidal rule.
    """
    if mode == 'sum':
        sum_mode = paddle.sum
    elif mode == 'cumsum':
        sum_mode = paddle.cumsum

    if not (x is None or dx is None):
        raise ValueError("Not permitted to specify both x and dx input args.")
    if y.dtype not in [paddle.float16, paddle.float32, paddle.float64]:
        raise TypeError(
            "The data type of input must be Tensor, and dtype should be one of ['paddle.float16', 'paddle.float32', 'paddle.float64'], but got {}".format(
                y.dtype
            )
        )

    y_shape = y.shape
    length = y_shape[axis]
    if axis < 0:
        axis += y.dim()
    if x is None:
        if dx is None:
            dx = 1.0
        dx = paddle.to_tensor(dx)
        if dx.dim() > 1:
            raise ValueError(f'Expected dx to be a scalar, got dx={dx}')
    else:
        if x.dtype not in [paddle.float16, paddle.float32, paddle.float64]:
            raise TypeError(
                "The data type of input must be Tensor, and dtype should be one of ['paddle.float16', 'paddle.float32', 'paddle.float64'], but got {}".format(
                    x.dtype
                )
            )
        # Reshape to correct shape
        if x.dim() == 1:
            dx = paddle.diff(x)
            shape = [1] * y.dim()
            shape[axis] = dx.shape[0]
            dx = dx.reshape(shape)
        else:
            dx = paddle.diff(x, axis=axis)
    return 0.5 * sum_mode(
        (
            paddle.gather(y, paddle.arange(1, length), axis=axis)
            + paddle.gather(y, paddle.arange(0, length - 1), axis=axis)
        )
        * dx,
        axis=axis,
    )


def trapezoid(y, x=None, dx=None, axis=-1, name=None):
    """
    Integrate along the given axis using the composite trapezoidal rule. Use the sum method.

    Args:
        y (Tensor): Input tensor to integrate. It's data type should be float16, float32, float64.
        x (Tensor, optional): The sample points corresponding to the :attr:`y` values, the same type as :attr:`y`.
            It is known that the size of :attr:`y` is `[d_1, d_2, ... , d_n]` and :math:`axis=k`, then the size of :attr:`x` can only be `[d_k]` or `[d_1, d_2, ... , d_n ]`.
            If :attr:`x` is None, the sample points are assumed to be evenly spaced :attr:`dx` apart. The default is None.
        dx (float, optional): The spacing between sample points when :attr:`x` is None. If neither :attr:`x` nor :attr:`dx` is provided then the default is :math:`dx = 1`.
        axis (int, optional): The axis along which to integrate. The default is -1.
        name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        Tensor, Definite integral of :attr:`y` is N-D tensor as approximated along a single axis by the trapezoidal rule.
        If :attr:`y` is a 1D tensor, then the result is a float. If N is greater than 1, then the result is an (N-1)-D tensor.

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> y = paddle.to_tensor([4, 5, 6], dtype='float32')

            >>> paddle.trapezoid(y)
            Tensor(shape=[], dtype=float32, place=Place(cpu), stop_gradient=True,
            10.)

            >>> paddle.trapezoid(y, dx=2.)
            Tensor(shape=[], dtype=float32, place=Place(cpu), stop_gradient=True,
            20.)

            >>> y = paddle.to_tensor([4, 5, 6], dtype='float32')
            >>> x = paddle.to_tensor([1, 2, 3], dtype='float32')

            >>> paddle.trapezoid(y, x)
            Tensor(shape=[], dtype=float32, place=Place(cpu), stop_gradient=True,
            10.)

            >>> y = paddle.to_tensor([1, 2, 3], dtype='float64')
            >>> x = paddle.to_tensor([8, 6, 4], dtype='float64')

            >>> paddle.trapezoid(y, x)
            Tensor(shape=[], dtype=float64, place=Place(cpu), stop_gradient=True,
            -8.)
            >>> y = paddle.arange(6).reshape((2, 3)).astype('float32')

            >>> paddle.trapezoid(y, axis=0)
            Tensor(shape=[3], dtype=float32, place=Place(cpu), stop_gradient=True,
            [1.50000000, 2.50000000, 3.50000000])
            >>> paddle.trapezoid(y, axis=1)
            Tensor(shape=[2], dtype=float32, place=Place(cpu), stop_gradient=True,
            [2., 8.])
    """
    return _trapezoid(y, x, dx, axis, mode='sum')


def cumulative_trapezoid(y, x=None, dx=None, axis=-1, name=None):
    """
    Integrate along the given axis using the composite trapezoidal rule. Use the cumsum method

    Args:
        y (Tensor): Input tensor to integrate. It's data type should be float16, float32, float64.
        x (Tensor, optional): The sample points corresponding to the :attr:`y` values, the same type as :attr:`y`.
            It is known that the size of :attr:`y` is `[d_1, d_2, ... , d_n]` and :math:`axis=k`, then the size of :attr:`x` can only be `[d_k]` or `[d_1, d_2, ... , d_n ]`.
            If :attr:`x` is None, the sample points are assumed to be evenly spaced :attr:`dx` apart. The default is None.
        dx (float, optional): The spacing between sample points when :attr:`x` is None. If neither :attr:`x` nor :attr:`dx` is provided then the default is :math:`dx = 1`.
        axis (int, optional): The axis along which to integrate. The default is -1.
        name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        Tensor, Definite integral of :attr:`y` is N-D tensor as approximated along a single axis by the trapezoidal rule.
        The result is an N-D tensor.

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> y = paddle.to_tensor([4, 5, 6], dtype='float32')

            >>> paddle.cumulative_trapezoid(y)
            Tensor(shape=[2], dtype=float32, place=Place(cpu), stop_gradient=True,
            [4.50000000, 10.       ])

            >>> paddle.cumulative_trapezoid(y, dx=2.)
            >>> # Tensor(shape=[2], dtype=float32, place=Place(cpu), stop_gradient=True,
            >>> #        [9. , 20.])

            >>> y = paddle.to_tensor([4, 5, 6], dtype='float32')
            >>> x = paddle.to_tensor([1, 2, 3], dtype='float32')

            >>> paddle.cumulative_trapezoid(y, x)
            Tensor(shape=[2], dtype=float32, place=Place(cpu), stop_gradient=True,
            [4.50000000, 10.       ])

            >>> y = paddle.to_tensor([1, 2, 3], dtype='float64')
            >>> x = paddle.to_tensor([8, 6, 4], dtype='float64')

            >>> paddle.cumulative_trapezoid(y, x)
            Tensor(shape=[2], dtype=float64, place=Place(cpu), stop_gradient=True,
            [-3., -8.])

            >>> y = paddle.arange(6).reshape((2, 3)).astype('float32')

            >>> paddle.cumulative_trapezoid(y, axis=0)
            Tensor(shape=[1, 3], dtype=float32, place=Place(cpu), stop_gradient=True,
            [[1.50000000, 2.50000000, 3.50000000]])
            >>> paddle.cumulative_trapezoid(y, axis=1)
            Tensor(shape=[2, 2], dtype=float32, place=Place(cpu), stop_gradient=True,
            [[0.50000000, 2.        ],
             [3.50000000, 8.        ]])
    """
    return _trapezoid(y, x, dx, axis, mode='cumsum')


def vander(x, n=None, increasing=False, name=None):
    """
    Generate a Vandermonde matrix.

    The columns of the output matrix are powers of the input vector. Order of the powers is
    determined by the increasing Boolean parameter. Specifically, when the increment is
    "false", the ith output column is a step-up in the order of the elements of the input
    vector to the N - i - 1 power. Such a matrix with a geometric progression in each row
    is named after Alexandre-Theophile Vandermonde.

    Args:
        x (Tensor): The input tensor, it must be 1-D Tensor, and it's data type should be ['complex64', 'complex128', 'float32', 'float64', 'int32', 'int64'].
        n (int): Number of columns in the output. If n is not specified, a square array is returned (n = len(x)).
        increasing(bool): Order of the powers of the columns. If True, the powers increase from left to right, if False (the default) they are reversed.
        name (str, optional): For details, please refer to :ref:`api_guide_Name`. Generally, no setting is required. Default: None.
    Returns:
        Tensor, A vandermonde matrix with shape (len(x), N). If increasing is False, the first column is :math:`x^{(N-1)}`, the second :math:`x^{(N-2)}` and so forth.
        If increasing is True, the columns are :math:`x^0`, :math:`x^1`, ..., :math:`x^{(N-1)}`.

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> x = paddle.to_tensor([1., 2., 3.], dtype="float32")
            >>> out = paddle.vander(x)
            >>> out
            Tensor(shape=[3, 3], dtype=float32, place=Place(cpu), stop_gradient=True,
            [[1., 1., 1.],
             [4., 2., 1.],
             [9., 3., 1.]])
            >>> out1 = paddle.vander(x,2)
            >>> out1
            Tensor(shape=[3, 2], dtype=float32, place=Place(cpu), stop_gradient=True,
            [[1., 1.],
             [2., 1.],
             [3., 1.]])
            >>> out2 = paddle.vander(x, increasing = True)
            >>> out2
            Tensor(shape=[3, 3], dtype=float32, place=Place(cpu), stop_gradient=True,
            [[1., 1., 1.],
             [1., 2., 4.],
             [1., 3., 9.]])
            >>> real = paddle.to_tensor([2., 4.])
            >>> imag = paddle.to_tensor([1., 3.])
            >>> complex = paddle.complex(real, imag)
            >>> out3 = paddle.vander(complex)
            >>> out3
            Tensor(shape=[2, 2], dtype=complex64, place=Place(cpu), stop_gradient=True,
            [[(2+1j), (1+0j)],
             [(4+3j), (1+0j)]])
    """
    check_variable_and_dtype(
        x,
        'x',
        ['complex64', 'complex128', 'float32', 'float64', 'int32', 'int64'],
        'vander',
    )
    if x.dim() != 1:
        raise ValueError(
            "The input of x is expected to be a 1-D Tensor."
            "But now the dims of Input(X) is %d." % x.dim()
        )

    if n is None:
        n = x.shape[0]

    if n < 0:
        raise ValueError("N must be non-negative.")

    res = paddle.empty([x.shape[0], n], dtype=x.dtype)

    if paddle.in_dynamic_mode():
        if n > 0:
            res[:, 0] = paddle.to_tensor([1], dtype=x.dtype)
        if n > 1:
            res[:, 1:] = x[:, None]
            res[:, 1:] = paddle.cumprod(res[:, 1:], dim=-1)
    else:
        if n > 0:
            res = paddle.static.setitem(
                res, (slice(None), 0), paddle.to_tensor([1], dtype=x.dtype)
            )
        if n > 1:
            res = paddle.static.setitem(
                res, (slice(None), slice(1, None)), x[:, None]
            )
            res = paddle.static.setitem(
                res,
                (slice(None), slice(1, None)),
                paddle.cumprod(res[:, 1:], dim=-1),
            )
    res = res[:, ::-1] if not increasing else res
    return res


def nextafter(x, y, name=None):
    r"""
    Return the next floating-point value after input towards other, elementwise.
    The shapes of input and other must be broadcastable.

    Args:
        x (Tensor): An N-D Tensor, the data type is float32, float64.
        y (Tensor): An N-D Tensor, the data type is float32, float64.
        name(str, optional):Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        out (Tensor): An N-D Tensor, the shape and data type is the same with input.

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> out = paddle.nextafter(paddle.to_tensor([1.0,2.0]),paddle.to_tensor([2.0,1.0]))
            >>> out
            Tensor(shape=[2], dtype=float32, place=Place(cpu), stop_gradient=True,
            [1.00000012, 1.99999988])
    """
    if in_dynamic_mode():
        return _C_ops.nextafter(x, y)
    else:
        check_variable_and_dtype(x, 'x', ['float32', 'float64'], 'nextafter')
        check_variable_and_dtype(y, 'y', ['float32', 'float64'], 'nextafter')
        op_type = "nextafter"
        helper = LayerHelper(op_type, **locals())
        inputs = {"x": x, "y": y}
        out = helper.create_variable_for_type_inference(dtype=paddle.float32)
        outputs = {"out": out}
        helper.append_op(type=op_type, inputs=inputs, outputs=outputs)
    return out


def i0(x, name=None):
    r"""
    The function used to calculate modified bessel function of order 0.

    Equation:
        ..  math::

            I_0(x) = \sum^{\infty}_{k=0}\frac{(x^2/4)^k}{(k!)^2}

    Args:
        x (Tensor): The input tensor, it's data type should be float32, float64.
        name (str, optional): For details, please refer to :ref:`api_guide_Name`. Generally, no setting is required. Default: None.

    Returns:
        - out (Tensor), A Tensor. the value of the modified bessel function of order 0 at x.

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> x = paddle.to_tensor([0, 1, 2, 3, 4], dtype="float32")
            >>> paddle.i0(x)
            Tensor(shape=[5], dtype=float32, place=Place(cpu), stop_gradient=True,
            [0.99999994 , 1.26606596 , 2.27958512 , 4.88079262 , 11.30192089])
    """
    if in_dynamic_mode():
        return _C_ops.i0(x)
    else:
        check_variable_and_dtype(x, "x", ["float32", "float64"], "i0")

        helper = LayerHelper("i0", **locals())
        out = helper.create_variable_for_type_inference(dtype=x.dtype)
        helper.append_op(type='i0', inputs={'x': x}, outputs={'out': out})
    return out


@inplace_apis_in_dygraph_only
def i0_(x, name=None):
    r"""
    Inplace version of ``i0`` API, the output Tensor will be inplaced with input ``x``.
    Please refer to :ref:`api_paddle_i0`.
    """

    if in_dynamic_mode():
        return _C_ops.i0_(x)


def i0e(x, name=None):
    r"""
    The function used to calculate exponentially scaled modified Bessel function of order 0.

    Equation:
        ..  math::

            I_0(x) = \sum^{\infty}_{k=0}\frac{(x^2/4)^k}{(k!)^2} \\
            I_{0e}(x) = e^{-|x|}I_0(x)

    Args:
        x (Tensor): The input tensor, it's data type should be float32, float64.
        name (str, optional): For details, please refer to :ref:`api_guide_Name`. Generally, no setting is required. Default: None.

    Returns:
        - out (Tensor), A Tensor. the value of the exponentially scaled modified Bessel function of order 0 at x.

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> x = paddle.to_tensor([0, 1, 2, 3, 4], dtype="float32")
            >>> print(paddle.i0e(x))
            Tensor(shape=[5], dtype=float32, place=Place(cpu), stop_gradient=True,
            [0.99999994, 0.46575963, 0.30850831, 0.24300036, 0.20700191])
    """
    if in_dynamic_mode():
        return _C_ops.i0e(x)
    else:
        check_variable_and_dtype(x, "x", ["float32", "float64"], "i0e")

        helper = LayerHelper("i0e", **locals())
        out = helper.create_variable_for_type_inference(dtype=x.dtype)
        helper.append_op(type='i0e', inputs={'x': x}, outputs={'out': out})
    return out


def i1(x, name=None):
    """
    The function is used to calculate modified bessel function of order 1.

    Args:
        x (Tensor): The input tensor, it's data type should be float32, float64.
        name (str, optional): For details, please refer to :ref:`api_guide_Name`. Generally, no setting is required. Default: None.

    Returns:
        - out (Tensor), A Tensor. the value of the modified bessel function of order 1 at x.

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> x = paddle.to_tensor([0, 1, 2, 3, 4], dtype="float32")
            >>> print(paddle.i1(x))
            Tensor(shape=[5], dtype=float32, place=Place(cpu), stop_gradient=True,
            [0.        , 0.56515908, 1.59063685, 3.95337057, 9.75946712])
    """
    if in_dynamic_mode():
        return _C_ops.i1(x)
    else:
        check_variable_and_dtype(x, "x", ["float32", "float64"], "i1")

        helper = LayerHelper("i1", **locals())
        out = helper.create_variable_for_type_inference(dtype=x.dtype)
        helper.append_op(
            type='i1', inputs={'x': x}, outputs={'out': out}, attrs={}
        )
    return out


def i1e(x, name=None):
    """
    The function is used to calculate exponentially scaled modified Bessel function of order 1.

    Args:

        x (Tensor): The input tensor, it's data type should be float32, float64.
        name (str, optional): For details, please refer to :ref:`api_guide_Name`. Generally, no setting is required. Default: None.

    Returns:
        - out (Tensor), A Tensor. the value of the exponentially scaled modified Bessel function of order 1 at x.

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> x = paddle.to_tensor([0, 1, 2, 3, 4], dtype="float32")
            >>> print(paddle.i1e(x))
            Tensor(shape=[5], dtype=float32, place=Place(cpu), stop_gradient=True,
            [0.        , 0.20791042, 0.21526928, 0.19682673, 0.17875087])
    """
    if in_dynamic_mode():
        return _C_ops.i1e(x)
    else:
        check_variable_and_dtype(x, "x", ["float32", "float64"], "i1e")

        helper = LayerHelper("i1e", **locals())
        out = helper.create_variable_for_type_inference(dtype=x.dtype)
        helper.append_op(
            type='i1e', inputs={'x': x}, outputs={'out': out}, attrs={}
        )
    return out


def polygamma(x, n, name=None):
    r"""
    Calculates the polygamma of the given input tensor, element-wise.

    The equation is:

    .. math::
        \Phi^n(x) = \frac{d^n}{dx^n} [\ln(\Gamma(x))]

    Args:
        x (Tensor): Input Tensor. Must be one of the following types: float32, float64.
        n (int): Order of the derivative. Must be integral.
        name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        - out (Tensor), A Tensor. the polygamma of the input Tensor, the shape and data type is the same with input.

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> data = paddle.to_tensor([2, 3, 25.5], dtype='float32')
            >>> res = paddle.polygamma(data, 1)
            >>> print(res)
            Tensor(shape=[3], dtype=float32, place=Place(cpu), stop_gradient=True,
            [0.64493412,  0.39493406,  0.03999467])
    """
    if not isinstance(n, int):
        raise TypeError(
            "The input of n must be int type, but received: %s " % (type(n))
        )
    if n < 0:
        raise ValueError(
            "The input of n must be greater than or equal to 0. But received n = %s"
            % (n)
        )
    if n == 0:
        return digamma(x)
    else:
        if in_dynamic_mode():
            return _C_ops.polygamma(x, n)
        else:
            check_variable_and_dtype(
                x, "x", ["float32", "float64"], "polygamma"
            )

            helper = LayerHelper("polygamma", **locals())
            out = helper.create_variable_for_type_inference(dtype=x.dtype)
            helper.append_op(
                type='polygamma',
                inputs={'x': x},
                outputs={'out': out},
                attrs={'n': n},
            )
        return out


@inplace_apis_in_dygraph_only
def polygamma_(x, n, name=None):
    r"""
    Inplace version of ``polygamma`` API, the output Tensor will be inplaced with input ``x``.
    Please refer to :ref:`api_paddle_polygamma`.
    """
    if not isinstance(n, int):
        raise TypeError(
            "The input of n must be int type, but received: %s " % (type(n))
        )
    if n < 0:
        raise ValueError(
            "The input of n must be greater than or equal to 0. But received n = %s"
            % (n)
        )
    if n == 0:
        return digamma_(x)
    else:
        if in_dynamic_mode():
            return _C_ops.polygamma_(x, n)


def ldexp(x, y, name=None):
    """
    Compute the result of multiplying x by 2 to the power of y. The equation is:

    .. math::
        out = x * 2^{y}

    Args:
        x (Tensor): The input Tensor, the data type is float32, float64, int32 or int64.
        y (Tensor):  A Tensor of exponents, typically integers.
        name (str, optional): Name for the operation (optional, default is None).For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        out (Tensor): An N-D Tensor. If x, y have different shapes and are "broadcastable", the resulting tensor shape is the shape of x and y after broadcasting. If x, y have the same shape, its shape is the same as x and y. And the data type is float32 or float64.

    Examples:

        .. code-block:: python

            >>> import paddle

            >>> # example1
            >>> x = paddle.to_tensor([1, 2, 3], dtype='float32')
            >>> y = paddle.to_tensor([2, 3, 4], dtype='int32')
            >>> res = paddle.ldexp(x, y)
            >>> print(res)
            Tensor(shape=[3], dtype=float32, place=Place(cpu), stop_gradient=True,
            [4. , 16., 48.])

            >>> # example2
            >>> x = paddle.to_tensor([1, 2, 3], dtype='float32')
            >>> y = paddle.to_tensor([2], dtype='int32')
            >>> res = paddle.ldexp(x, y)
            >>> print(res)
            Tensor(shape=[3], dtype=float32, place=Place(cpu), stop_gradient=True,
            [4. , 8. , 12.])

    """
    if not isinstance(x, (paddle.Tensor, Variable)):
        raise TypeError(f"x must be tensor type, but got {type(x)}")
    if not isinstance(y, (paddle.Tensor, Variable)):
        raise TypeError(f"y must be tensor type, but got {type(y)}")
    if x.dtype == paddle.float64 or y.dtype == paddle.float64:
        out_dtype = paddle.float64
    else:
        out_dtype = paddle.get_default_dtype()
    x = paddle.cast(x, dtype=out_dtype)
    y = paddle.cast(y, dtype=out_dtype)
    two = paddle.to_tensor(2, dtype=out_dtype)
    return paddle.multiply(x, paddle.pow(two, y))


def ldexp_(x, y, name=None):
    r"""
    Inplace version of ``polygamma`` API, the output Tensor will be inplaced with input ``x``.
    Please refer to :ref:`api_paddle_polygamma`.
    """
    if not isinstance(x, (paddle.Tensor, Variable)):
        raise TypeError(f"x must be tensor type, but got {type(x)}")
    if not isinstance(y, (paddle.Tensor, Variable)):
        raise TypeError(f"y must be tensor type, but got {type(y)}")
    if x.dtype == paddle.float64 or y.dtype == paddle.float64:
        out_dtype = paddle.float64
    else:
        out_dtype = paddle.get_default_dtype()
    x = paddle.cast_(x, dtype=out_dtype)
    y = paddle.cast(y, dtype=out_dtype)
    two = paddle.to_tensor(2, dtype=out_dtype)
    return paddle.multiply_(x, paddle.pow(two, y))
