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


from paddle.utils.inplace_utils import inplace_apis_in_dygraph_only

from .. import _C_ops
from ..fluid.data_feeder import check_variable_and_dtype
from ..framework import LayerHelper, in_dynamic_mode
from .layer_function_generator import (
    add_sample_code,
    generate_activation_fn,
    generate_inplace_fn,
    generate_layer_fn,
)

__deprecated_func_name__ = {
    'tanh_shrink': 'tanhshrink',
    'logsigmoid': 'log_sigmoid',
}

__activations_noattr__ = [
    'silu',
    'logsigmoid',
    'tanh_shrink',
    'softplus',
    'softsign',
    'tanh',
]

__unary_func__ = ['abs']

__inplace_unary_func__ = [
    'exp_',
    'sqrt_',
    'rsqrt_',
    'ceil_',
    'floor_',
    'round_',
    'reciprocal_',
    'sigmoid_',
    'abs_',
    'sin_',
    'sinh_',
    'asin_',
    'asinh_',
    'cos_',
    'cosh_',
    'acos_',
    'acosh_',
    'tan_',
    'atan_',
    'atanh_',
    'expm1_',
    'erf_',
    'square_',
]

__all__ = []

# It is a hot fix in some unittest using:
#   paddle.scale(x=x, scale=10.0, out=out_var)
# e.g.: test_program_code.py, test_dist_train.py
globals()['_scale'] = generate_layer_fn('scale')

globals()['_elementwise_div'] = generate_layer_fn('elementwise_div')

for _OP in set(__activations_noattr__):
    _new_OP = _OP
    if _OP in __deprecated_func_name__:
        _new_OP = __deprecated_func_name__[_OP]
    _func = generate_activation_fn(_OP)
    globals()[_OP] = _func

for _OP in set(__unary_func__):
    _new_OP = _OP
    if _OP in __deprecated_func_name__:
        _new_OP = __deprecated_func_name__[_OP]
    _func = generate_activation_fn(_OP)
    globals()[_OP] = _func

for _OP in set(__inplace_unary_func__):
    _new_OP = _OP
    if _OP in __deprecated_func_name__:
        _new_OP = __deprecated_func_name__[_OP]
    func = generate_inplace_fn(_OP)
    func.__module__ = __name__
    _func = inplace_apis_in_dygraph_only(func)
    globals()[_OP] = _func

add_sample_code(
    globals()["silu"],
    r"""
Examples:
    .. code-block:: python

        >>> import paddle
        >>> import paddle.nn.functional as F

        >>> x = paddle.to_tensor([1.0, 2.0, 3.0, 4.0])
        >>> out = F.silu(x)
        >>> print(out)
        Tensor(shape=[4], dtype=float32, place=Place(cpu), stop_gradient=True,
        [0.73105860, 1.76159406, 2.85772228, 3.92805505])
""",
)

add_sample_code(
    globals()["logsigmoid"],
    r"""
Examples:
    .. code-block:: python

        >>> import paddle
        >>> import paddle.nn.functional as F

        >>> x = paddle.to_tensor([-0.4, -0.2, 0.1, 0.3])
        >>> out = F.log_sigmoid(x)
        >>> print(out)
        Tensor(shape=[4], dtype=float32, place=Place(cpu), stop_gradient=True,
        [-0.91301525, -0.79813892, -0.64439666, -0.55435526])
""",
)

add_sample_code(
    globals()["tanh"],
    r"""
Examples:
    .. code-block:: python

        >>> import paddle

        >>> x = paddle.to_tensor([-0.4, -0.2, 0.1, 0.3])
        >>> out = paddle.tanh(x)
        >>> print(out)
        Tensor(shape=[4], dtype=float32, place=Place(cpu), stop_gradient=True,
        [-0.37994900, -0.19737528,  0.09966799,  0.29131261])
""",
)

add_sample_code(
    globals()["tanh_shrink"],
    r"""
Examples:
    .. code-block:: python

        >>> import paddle
        >>> import paddle.nn.functional as F

        >>> x = paddle.to_tensor([-0.4, -0.2, 0.1, 0.3])
        >>> out = F.tanhshrink(x)
        >>> print(out)
        Tensor(shape=[4], dtype=float32, place=Place(cpu), stop_gradient=True,
        [-0.02005100, -0.00262472,  0.00033201,  0.00868741])
""",
)

add_sample_code(
    globals()["abs"],
    r"""
Examples:
    .. code-block:: python

        >>> import paddle

        >>> x = paddle.to_tensor([-0.4, -0.2, 0.1, 0.3])
        >>> out = paddle.abs(x)
        >>> print(out)
        Tensor(shape=[4], dtype=float32, place=Place(cpu), stop_gradient=True,
        [0.40000001, 0.20000000, 0.10000000, 0.30000001])
""",
)

add_sample_code(
    globals()["softplus"],
    r"""
Examples:
    .. code-block:: python

        >>> import paddle
        >>> import paddle.nn.functional as F

        >>> x = paddle.to_tensor([-0.4, -0.2, 0.1, 0.3])
        >>> out = F.softplus(x)
        >>> print(out)
        Tensor(shape=[4], dtype=float32, place=Place(cpu), stop_gradient=True,
        [0.51301527, 0.59813893, 0.74439669, 0.85435522])
""",
)

add_sample_code(
    globals()["softsign"],
    r"""
Examples:
    .. code-block:: python

        >>> import paddle
        >>> import paddle.nn.functional as F

        >>> x = paddle.to_tensor([-0.4, -0.2, 0.1, 0.3])
        >>> out = F.softsign(x)
        >>> print(out)
        Tensor(shape=[4], dtype=float32, place=Place(cpu), stop_gradient=True,
        [-0.28571430, -0.16666666,  0.09090909,  0.23076925])
""",
)


def acos(x, name=None):
    """
    Acos Activation Operator.

    .. math::
        out = cos^{-1}(x)

    Args:
        x (Tensor): Input of Acos operator, an N-D Tensor, with data type float32, float64 or float16.
        name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        Tensor. Output of Acos operator, a Tensor with shape same as input.

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> x = paddle.to_tensor([-0.4, -0.2, 0.1, 0.3])
            >>> out = paddle.acos(x)
            >>> print(out)
            Tensor(shape=[4], dtype=float32, place=Place(cpu), stop_gradient=True,
            [1.98231316, 1.77215421, 1.47062886, 1.26610363])
    """
    if in_dynamic_mode():
        return _C_ops.acos(x)
    else:
        check_variable_and_dtype(
            x, 'x', ['float16', 'uint16', 'float32', 'float64'], 'acos'
        )
        helper = LayerHelper('acos', **locals())
        out = helper.create_variable_for_type_inference(dtype=x.dtype)
        helper.append_op(type='acos', inputs={"X": x}, outputs={"Out": out})
        return out


def acosh(x, name=None):
    """
    Acosh Activation Operator.

    .. math::
       out = acosh(x)

    Args:
        x (Tensor): Input of Acosh operator, an N-D Tensor, with data type float32, float64 or float16.
        name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        Tensor. Output of Acosh operator, a Tensor with shape same as input.

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> x = paddle.to_tensor([1., 3., 4., 5.])
            >>> out = paddle.acosh(x)
            >>> print(out)
            Tensor(shape=[4], dtype=float32, place=Place(cpu), stop_gradient=True,
            [0.        , 1.76274717, 2.06343699, 2.29243159])
    """
    if in_dynamic_mode():
        return _C_ops.acosh(x)
    else:
        check_variable_and_dtype(
            x, 'x', ['float16', 'uint16', 'float32', 'float64'], 'acosh'
        )
        helper = LayerHelper('acosh', **locals())
        out = helper.create_variable_for_type_inference(dtype=x.dtype)
        helper.append_op(type='acosh', inputs={"X": x}, outputs={"Out": out})
        return out


def asin(x, name=None):
    """
    Arcsine Operator.

    .. math::
       out = sin^{-1}(x)

    Args:
        x (Tensor): Input of Asin operator, an N-D Tensor, with data type float32, float64 or float16.
        name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        Tensor. Same shape and dtype as input.

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> x = paddle.to_tensor([-0.4, -0.2, 0.1, 0.3])
            >>> out = paddle.asin(x)
            >>> print(out)
            Tensor(shape=[4], dtype=float32, place=Place(cpu), stop_gradient=True,
            [-0.41151685, -0.20135793,  0.10016742,  0.30469266])
    """
    if in_dynamic_mode():
        return _C_ops.asin(x)
    else:
        check_variable_and_dtype(
            x, 'x', ['float16', 'uint16', 'float32', 'float64'], 'asin'
        )
        helper = LayerHelper('asin', **locals())
        out = helper.create_variable_for_type_inference(dtype=x.dtype)
        helper.append_op(type='asin', inputs={"X": x}, outputs={"Out": out})
        return out


def asinh(x, name=None):
    """
    Asinh Activation Operator.

    .. math::
       out = asinh(x)

    Args:
        x (Tensor): Input of Asinh operator, an N-D Tensor, with data type float32, float64 or float16.
        name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        Tensor. Output of Asinh operator, a Tensor with shape same as input.

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> x = paddle.to_tensor([-0.4, -0.2, 0.1, 0.3])
            >>> out = paddle.asinh(x)
            >>> print(out)
            Tensor(shape=[4], dtype=float32, place=Place(cpu), stop_gradient=True,
            [-0.39003533, -0.19869010,  0.09983408,  0.29567307])
    """
    if in_dynamic_mode():
        return _C_ops.asinh(x)
    else:
        check_variable_and_dtype(
            x, 'x', ['float16', 'uint16', 'float32', 'float64'], 'asinh'
        )
        helper = LayerHelper('asinh', **locals())
        out = helper.create_variable_for_type_inference(dtype=x.dtype)
        helper.append_op(type='asinh', inputs={"X": x}, outputs={"Out": out})
        return out


def atan(x, name=None):
    """
    Arctangent Operator.

    .. math::
       out = tan^{-1}(x)

    Args:
        x (Tensor): Input of Atan operator, an N-D Tensor, with data type float32, float64 or float16.
        name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        Tensor. Same shape and dtype as input x.

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> x = paddle.to_tensor([-0.4, -0.2, 0.1, 0.3])
            >>> out = paddle.atan(x)
            >>> print(out)
            Tensor(shape=[4], dtype=float32, place=Place(cpu), stop_gradient=True,
            [-0.38050640, -0.19739556,  0.09966865,  0.29145682])
    """
    if in_dynamic_mode():
        return _C_ops.atan(x)
    else:
        check_variable_and_dtype(
            x, 'x', ['float16', 'uint16', 'float32', 'float64'], 'atan'
        )
        helper = LayerHelper('atan', **locals())
        out = helper.create_variable_for_type_inference(dtype=x.dtype)
        helper.append_op(type='atan', inputs={"X": x}, outputs={"Out": out})
        return out


def atanh(x, name=None):
    """
    Atanh Activation Operator.

    .. math::
       out = atanh(x)

    Args:
        x (Tensor): Input of Atan operator, an N-D Tensor, with data type float32, float64 or float16.
        name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        Tensor. Output of Atanh operator, a Tensor with shape same as input.

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> x = paddle.to_tensor([-0.4, -0.2, 0.1, 0.3])
            >>> out = paddle.atanh(x)
            >>> print(out)
            Tensor(shape=[4], dtype=float32, place=Place(cpu), stop_gradient=True,
            [-0.42364895, -0.20273255,  0.10033534,  0.30951962])
    """
    if in_dynamic_mode():
        return _C_ops.atanh(x)
    else:
        check_variable_and_dtype(
            x, 'x', ['float16', 'uint16', 'float32', 'float64'], 'atanh'
        )
        helper = LayerHelper('atanh', **locals())
        out = helper.create_variable_for_type_inference(dtype=x.dtype)
        helper.append_op(type='atanh', inputs={"X": x}, outputs={"Out": out})
        return out


def ceil(x, name=None):
    """

    Ceil Operator. Computes ceil of x element-wise.

    .. math::
        out = \\left \\lceil x \\right \\rceil

    Args:
        x (Tensor): Input of Ceil operator, an N-D Tensor, with data type float32, float64 or float16.
        name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        Tensor. Output of Ceil operator, a Tensor with shape same as input.

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> x = paddle.to_tensor([-0.4, -0.2, 0.1, 0.3])
            >>> out = paddle.ceil(x)
            >>> print(out)
            Tensor(shape=[4], dtype=float32, place=Place(cpu), stop_gradient=True,
            [-0., -0., 1. , 1. ])
    """
    if in_dynamic_mode():
        return _C_ops.ceil(x)
    else:
        check_variable_and_dtype(
            x, 'x', ['float16', 'uint16', 'float32', 'float64'], 'ceil'
        )
        helper = LayerHelper('ceil', **locals())
        out = helper.create_variable_for_type_inference(dtype=x.dtype)
        helper.append_op(type='ceil', inputs={"X": x}, outputs={"Out": out})
        return out


def cos(x, name=None):
    """
    Cosine Operator. Computes cosine of x element-wise.

    Input range is `(-inf, inf)` and output range is `[-1,1]`.

    .. math::
       out = cos(x)

    Args:
        x (Tensor): Input of Cos operator, an N-D Tensor, with data type float32, float64 or float16.
        name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        Tensor. Output of Cos operator, a Tensor with shape same as input.

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> x = paddle.to_tensor([-0.4, -0.2, 0.1, 0.3])
            >>> out = paddle.cos(x)
            >>> print(out)
            Tensor(shape=[4], dtype=float32, place=Place(cpu), stop_gradient=True,
            [0.92106098, 0.98006660, 0.99500418, 0.95533651])
    """
    if in_dynamic_mode():
        return _C_ops.cos(x)
    else:
        check_variable_and_dtype(
            x,
            'x',
            ['float16', 'float32', 'float64', 'complex64', 'complex128'],
            'cos',
        )
        helper = LayerHelper('cos', **locals())
        out = helper.create_variable_for_type_inference(dtype=x.dtype)
        helper.append_op(type='cos', inputs={"X": x}, outputs={"Out": out})
        return out


def cosh(x, name=None):
    """
    Cosh Activation Operator.

    Input range `(-inf, inf)`, output range `(1, inf)`.

    .. math::
       out = \\frac{exp(x)+exp(-x)}{2}

    Args:
        x (Tensor): Input of Cosh operator, an N-D Tensor, with data type float32, float64 or float16.
        name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        Tensor. Output of Cosh operator, a Tensor with shape same as input.

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> x = paddle.to_tensor([-0.4, -0.2, 0.1, 0.3])
            >>> out = paddle.cosh(x)
            >>> print(out)
            Tensor(shape=[4], dtype=float32, place=Place(cpu), stop_gradient=True,
            [1.08107233, 1.02006674, 1.00500417, 1.04533851])
    """
    if in_dynamic_mode():
        return _C_ops.cosh(x)
    else:
        check_variable_and_dtype(
            x, 'x', ['float16', 'uint16', 'float32', 'float64'], 'cosh'
        )
        helper = LayerHelper('cosh', **locals())
        out = helper.create_variable_for_type_inference(dtype=x.dtype)
        helper.append_op(type='cosh', inputs={"X": x}, outputs={"Out": out})
        return out


def exp(x, name=None):
    """

    Computes exp of x element-wise with a natural number `e` as the base.

    .. math::
        out = e^x

    Args:
        x (Tensor): Input of Exp operator, an N-D Tensor, with data type int32, int64, float32, float64 or float16.
        name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        Tensor. Output of Exp operator, a Tensor with shape same as input.

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> x = paddle.to_tensor([-0.4, -0.2, 0.1, 0.3])
            >>> out = paddle.exp(x)
            >>> print(out)
            Tensor(shape=[4], dtype=float32, place=Place(cpu), stop_gradient=True,
            [0.67032003, 0.81873077, 1.10517097, 1.34985888])
    """
    if in_dynamic_mode():
        return _C_ops.exp(x)
    else:
        check_variable_and_dtype(
            x,
            'x',
            [
                'int32',
                'int64',
                'uint16',
                'float16',
                'float32',
                'float64',
                'complex64',
                'complex128',
            ],
            'exp',
        )
        helper = LayerHelper('exp', **locals())
        out = helper.create_variable_for_type_inference(dtype=x.dtype)
        helper.append_op(type='exp', inputs={"X": x}, outputs={"Out": out})
        return out


def expm1(x, name=None):
    """

    Expm1 Operator. Computes expm1 of x element-wise with a natural number :math:`e` as the base.

    .. math::
        out = e^x - 1

    Args:
        x (Tensor): Input of Expm1 operator, an N-D Tensor, with data type int32, int64, float32, float64 or float16.
        name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        Tensor. Output of Expm1 operator, a Tensor with shape same as input.

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> x = paddle.to_tensor([-0.4, -0.2, 0.1, 0.3])
            >>> out = paddle.expm1(x)
            >>> print(out)
            Tensor(shape=[4], dtype=float32, place=Place(cpu), stop_gradient=True,
            [-0.32967997, -0.18126924,  0.10517092,  0.34985882])
    """
    if in_dynamic_mode():
        return _C_ops.expm1(x)
    else:
        check_variable_and_dtype(
            x,
            'x',
            ['float16', 'uint16', 'float32', 'float64', 'int32', 'int64'],
            'expm1',
        )
        helper = LayerHelper('expm1', **locals())
        out = helper.create_variable_for_type_inference(dtype=x.dtype)
        helper.append_op(type='expm1', inputs={"X": x}, outputs={"Out": out})
        return out


def floor(x, name=None):
    """

    Floor Activation Operator. Computes floor of x element-wise.

    .. math::
        out = \\lfloor x \\rfloor

    Args:
        x (Tensor): Input of Floor operator, an N-D Tensor, with data type float32, float64 or float16.
        name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        Tensor. Output of Floor operator, a Tensor with shape same as input.

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> x = paddle.to_tensor([-0.4, -0.2, 0.1, 0.3])
            >>> out = paddle.floor(x)
            >>> print(out)
            Tensor(shape=[4], dtype=float32, place=Place(cpu), stop_gradient=True,
            [-1., -1.,  0.,  0.])
    """
    if in_dynamic_mode():
        return _C_ops.floor(x)
    else:
        check_variable_and_dtype(
            x, 'x', ['float16', 'uint16', 'float32', 'float64'], 'floor'
        )
        helper = LayerHelper('floor', **locals())
        out = helper.create_variable_for_type_inference(dtype=x.dtype)
        helper.append_op(type='floor', inputs={"X": x}, outputs={"Out": out})
        return out


def reciprocal(x, name=None):
    """

    Reciprocal Activation Operator.

    .. math::
        out = \\frac{1}{x}

    Args:
        x (Tensor): Input of Reciprocal operator, an N-D Tensor, with data type float32, float64 or float16.
        name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        Tensor. Output of Reciprocal operator, a Tensor with shape same as input.

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> x = paddle.to_tensor([-0.4, -0.2, 0.1, 0.3])
            >>> out = paddle.reciprocal(x)
            >>> print(out)
            Tensor(shape=[4], dtype=float32, place=Place(cpu), stop_gradient=True,
            [-2.50000000, -5.        ,  10.       ,  3.33333325])
    """
    if in_dynamic_mode():
        return _C_ops.reciprocal(x)
    else:
        check_variable_and_dtype(
            x, 'x', ['float16', 'uint16', 'float32', 'float64'], 'reciprocal'
        )
        helper = LayerHelper('reciprocal', **locals())
        out = helper.create_variable_for_type_inference(dtype=x.dtype)
        helper.append_op(
            type='reciprocal', inputs={"X": x}, outputs={"Out": out}
        )
        return out


def round(x, name=None):
    """

    Round the values in the input to the nearest integer value.

    .. code-block:: text

        input:
          x.shape = [4]
          x.data = [1.2, -0.9, 3.4, 0.9]

        output:
          out.shape = [4]
          out.data = [1., -1., 3., 1.]

    Args:
        x (Tensor): Input of Round operator, an N-D Tensor, with data type float32, float64 or float16.
        name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        Tensor. Output of Round operator, a Tensor with shape same as input.

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> x = paddle.to_tensor([-0.5, -0.2, 0.6, 1.5])
            >>> out = paddle.round(x)
            >>> print(out)
            Tensor(shape=[4], dtype=float32, place=Place(cpu), stop_gradient=True,
            [-1., -0.,  1.,  2.])
    """
    if in_dynamic_mode():
        return _C_ops.round(x)
    else:
        check_variable_and_dtype(
            x, 'x', ['float16', 'uint16', 'float32', 'float64'], 'round'
        )
        helper = LayerHelper('round', **locals())
        out = helper.create_variable_for_type_inference(dtype=x.dtype)
        helper.append_op(type='round', inputs={"X": x}, outputs={"Out": out})
        return out


def rsqrt(x, name=None):
    """
    Rsqrt Activation Operator.

    Please make sure input is legal in case of numeric errors.

    .. math::
       out = \\frac{1}{\\sqrt{x}}

    Args:
        x (Tensor): Input of Rsqrt operator, an N-D Tensor, with data type float32, float64 or float16.
        name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        Tensor. Output of Rsqrt operator, a Tensor with shape same as input.

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> x = paddle.to_tensor([0.1, 0.2, 0.3, 0.4])
            >>> out = paddle.rsqrt(x)
            >>> print(out)
            Tensor(shape=[4], dtype=float32, place=Place(cpu), stop_gradient=True,
            [3.16227770, 2.23606801, 1.82574177, 1.58113885])
    """
    if in_dynamic_mode():
        return _C_ops.rsqrt(x)
    else:
        check_variable_and_dtype(
            x, 'x', ['float16', 'uint16', 'float32', 'float64'], 'rsqrt'
        )
        helper = LayerHelper('rsqrt', **locals())
        out = helper.create_variable_for_type_inference(dtype=x.dtype)
        helper.append_op(type='rsqrt', inputs={"X": x}, outputs={"Out": out})
        return out


def sigmoid(x, name=None):
    """
    Sigmoid Activation.

    .. math::
       out = \\frac{1}{1 + e^{-x}}

    Args:
        x (Tensor): Input of Sigmoid operator, an N-D Tensor, with data type float32, float64 or float16.
        name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        Tensor. Output of Sigmoid operator, a Tensor with shape same as input.

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> import paddle.nn.functional as F

            >>> x = paddle.to_tensor([-0.4, -0.2, 0.1, 0.3])
            >>> out = F.sigmoid(x)
            >>> print(out)
            Tensor(shape=[4], dtype=float32, place=Place(cpu), stop_gradient=True,
            [0.40131235, 0.45016602, 0.52497917, 0.57444251])
    """
    if in_dynamic_mode():
        return _C_ops.sigmoid(x)
    else:
        check_variable_and_dtype(
            x, 'x', ['float16', 'float32', 'float64', 'uint16'], 'sigmoid'
        )
        helper = LayerHelper('sigmoid', **locals())
        out = helper.create_variable_for_type_inference(dtype=x.dtype)
        helper.append_op(type='sigmoid', inputs={"X": x}, outputs={"Out": out})
        return out


def sin(x, name=None):
    """
    Sine Activation Operator.

    .. math::
       out = sin(x)

    Args:
        x (Tensor): Input of Sin operator, an N-D Tensor, with data type float32, float64 or float16.
        name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        Tensor. Output of Sin operator, a Tensor with shape same as input.

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> x = paddle.to_tensor([-0.4, -0.2, 0.1, 0.3])
            >>> out = paddle.sin(x)
            >>> print(out)
            Tensor(shape=[4], dtype=float32, place=Place(cpu), stop_gradient=True,
            [-0.38941833, -0.19866933,  0.09983342,  0.29552022])
    """
    if in_dynamic_mode():
        return _C_ops.sin(x)
    else:
        check_variable_and_dtype(
            x,
            'x',
            [
                'float16',
                'uint16',
                'float32',
                'float64',
                'complex64',
                'complex128',
            ],
            'sin',
        )
        helper = LayerHelper('sin', **locals())
        out = helper.create_variable_for_type_inference(dtype=x.dtype)
        helper.append_op(type='sin', inputs={"X": x}, outputs={"Out": out})
        return out


def sinh(x, name=None):
    """
    Sinh Activation Operator.

    .. math::
       out = sinh(x)

    Args:
        x (Tensor): Input of Sinh operator, an N-D Tensor, with data type float32, float64 or float16.
        name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        Tensor. Output of Sinh operator, a Tensor with shape same as input.

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> x = paddle.to_tensor([-0.4, -0.2, 0.1, 0.3])
            >>> out = paddle.sinh(x)
            >>> print(out)
            Tensor(shape=[4], dtype=float32, place=Place(cpu), stop_gradient=True,
            [-0.41075233, -0.20133601,  0.10016675,  0.30452031])
    """
    if in_dynamic_mode():
        return _C_ops.sinh(x)
    else:
        check_variable_and_dtype(
            x, 'x', ['float16', 'uint16', 'float32', 'float64'], 'sinh'
        )
        helper = LayerHelper('sinh', **locals())
        out = helper.create_variable_for_type_inference(dtype=x.dtype)
        helper.append_op(type='sinh', inputs={"X": x}, outputs={"Out": out})
        return out


def sqrt(x, name=None):
    """
    Sqrt Activation Operator.

    .. math::
       out=\\sqrt{x}=x^{1/2}

    Args:
        x (Tensor): Input of Sqrt operator, an N-D Tensor, with data type float32, float64 or float16.
        name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        Tensor. Output of Sqrt operator, a Tensor with shape same as input.

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> x = paddle.to_tensor([0.1, 0.2, 0.3, 0.4])
            >>> out = paddle.sqrt(x)
            >>> print(out)
            Tensor(shape=[4], dtype=float32, place=Place(cpu), stop_gradient=True,
            [0.31622776, 0.44721359, 0.54772258, 0.63245553])
    """
    if in_dynamic_mode():
        return _C_ops.sqrt(x)
    else:
        check_variable_and_dtype(
            x,
            'x',
            ['float16', 'uint16', 'float32', 'float64'],
            'sqrt',
        )
        helper = LayerHelper('sqrt', **locals())
        out = helper.create_variable_for_type_inference(dtype=x.dtype)
        helper.append_op(type='sqrt', inputs={"X": x}, outputs={"Out": out})
        return out


def square(x, name=None):
    """
    Square each elements of the inputs.

    .. math::
       out = x^2

    Args:
        x (Tensor): Input of Square operator, an N-D Tensor, with data type float32, float64 or float16.
        name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        Tensor. Output of Square operator, a Tensor with shape same as input.

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> x = paddle.to_tensor([-0.4, -0.2, 0.1, 0.3])
            >>> out = paddle.square(x)
            >>> print(out)
            Tensor(shape=[4], dtype=float32, place=Place(cpu), stop_gradient=True,
            [0.16000001, 0.04000000, 0.01000000, 0.09000000])
    """
    if in_dynamic_mode():
        return _C_ops.square(x)
    else:
        check_variable_and_dtype(
            x,
            'x',
            [
                'int32',
                'int64',
                'float16',
                'float32',
                'float64',
                'complex64',
                'complex128',
            ],
            'square',
        )
        helper = LayerHelper('square', **locals())
        out = helper.create_variable_for_type_inference(dtype=x.dtype)
        helper.append_op(type='square', inputs={"X": x}, outputs={"Out": out})
        return out


def tan(x, name=None):
    """
    Tangent Operator. Computes tangent of x element-wise.

    Input range is `(k*pi-pi/2, k*pi+pi/2)` and output range is `(-inf, inf)`.

    .. math::
       out = tan(x)

    Args:
        x (Tensor): Input of Tan operator, an N-D Tensor, with data type float32, float64 or float16.
        name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        Tensor. Output of Tan operator, a Tensor with shape same as input.

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> x = paddle.to_tensor([-0.4, -0.2, 0.1, 0.3])
            >>> out = paddle.tan(x)
            >>> print(out)
            Tensor(shape=[4], dtype=float32, place=Place(cpu), stop_gradient=True,
            [-0.42279324, -0.20271003,  0.10033467,  0.30933627])
    """
    if in_dynamic_mode():
        return _C_ops.tan(x)
    else:
        check_variable_and_dtype(
            x,
            'x',
            [
                'float16',
                'uint16',
                'float32',
                'float64',
                'complex64',
                'complex128',
            ],
            'tan',
        )
        helper = LayerHelper('tan', **locals())
        out = helper.create_variable_for_type_inference(dtype=x.dtype)
        helper.append_op(type='tan', inputs={"X": x}, outputs={"Out": out})
        return out


_erf_ = generate_layer_fn('erf')


def erf(x, name=None):
    if in_dynamic_mode():
        return _C_ops.erf(x)

    locals_var = locals().copy()
    kwargs = {}
    for name, val in locals_var.items():
        if val is not None:
            kwargs[name] = val
    return _erf_(**kwargs)


erf.__doc__ = r"""
:strong:`Erf Operator`
For more details, see `Error function <https://en.wikipedia.org/wiki/Error_function>`_.

Equation:
    ..  math::
        out = \frac{2}{\sqrt{\pi}} \int_{0}^{x}e^{- \eta^{2}}d\eta

Args:

    x (Tensor): The input tensor, it's data type should be float32, float64.
    name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

Returns:

    Tensor: The output of Erf, dtype: float32 or float64, the same as the input, shape: the same as the input.

Examples:

    .. code-block:: python

        >>> import paddle

        >>> x = paddle.to_tensor([-0.4, -0.2, 0.1, 0.3])
        >>> out = paddle.erf(x)
        >>> print(out)
        Tensor(shape=[4], dtype=float32, place=Place(cpu), stop_gradient=True,
        [-0.42839241, -0.22270259,  0.11246292,  0.32862678])
"""
