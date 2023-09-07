#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

import paddle
from paddle import _C_ops, _legacy_C_ops, in_dynamic_mode
from paddle.framework import core
from paddle.utils.inplace_utils import inplace_apis_in_dygraph_only

from ...fluid.data_feeder import check_dtype, check_variable_and_dtype
from ...fluid.framework import convert_np_dtype_to_dtype_
from ...fluid.layer_helper import LayerHelper
from ...tensor.manipulation import chunk
from ...tensor.math import tanh  # noqa: F401
from ...tensor.math import tanh_  # noqa: F401
from ...tensor.ops import sigmoid  # noqa: F401

__all__ = []


def celu(x, alpha=1.0, name=None):
    r"""
    celu activation.

    Apply the following operation to each element of the input Tensor accroding to the `Continuously Differentiable Exponential Linear Units <https://arxiv.org/abs/1704.07483>`_.

    .. math::

        \operatorname{celu}(x) = \max(0, x) + \min(0, \alpha * (\mathrm{e}^{x/\alpha}-1))

    Parameters:
        x (Tensor): The input Tensor with data type float16, float32, or float64.
        alpha (float, optional): The 'alpha' value of the CELU formula. Default is 1.0.
        name (str, optional): For details, please refer to :ref:`api_guide_Name`. Generally, no setting is required. Default: None.

    Returns:
        A ``Tensor`` with the same data type and shape as ``x`` .

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> import paddle.nn.functional as F

            >>> x = paddle.to_tensor([[-1., 6.], [1., 15.6]])
            >>> out = F.celu(x, alpha=0.2)
            >>> print(out)
            Tensor(shape=[2, 2], dtype=float32, place=Place(cpu), stop_gradient=True,
            [[-0.19865242,  6.        ],
             [ 1.        , 15.60000038]])
    """
    if alpha == 0:
        raise ZeroDivisionError("alpha cannot be 0 for celu")
    if in_dynamic_mode():
        return _C_ops.celu(x, alpha)
    else:
        check_variable_and_dtype(
            x, 'x', ['float16', 'uint16', 'float32', 'float64'], 'celu'
        )
        helper = LayerHelper("celu", **locals())
        out = helper.create_variable_for_type_inference(x.dtype)
        helper.append_op(
            type='celu',
            inputs={'X': x},
            outputs={'Out': out},
            attrs={'alpha': alpha},
        )
        return out


def elu(x, alpha=1.0, name=None):
    r"""
    elu activation.

    .. math::

        elu(x)=
            \left\{
                \begin{array}{lcl}
                x,& &\text{if } \ x > 0 \\
                alpha * (e^{x} - 1),& &\text{if } \ x <= 0
                \end{array}
            \right.

    Parameters:
        x (Tensor): The input Tensor with data type float32, float64.
        alpha (float, optional): The 'alpha' value of the ELU formulation. Default is 1.0.
        name (str, optional): For details, please refer to :ref:`api_guide_Name`. Generally, no setting is required. Default: None.

    Returns:
        A Tensor with the same data type and shape as ``x`` .

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> import paddle.nn.functional as F

            >>> x = paddle.to_tensor([[-1., 6.], [1., 15.6]])
            >>> out = F.elu(x, alpha=0.2)
            >>> print(out)
            Tensor(shape=[2, 2], dtype=float32, place=Place(cpu), stop_gradient=True,
            [[-0.12642412,  6.        ],
             [ 1.        , 15.60000038]])
    """

    if in_dynamic_mode():
        return _C_ops.elu(x, alpha)

    else:
        check_variable_and_dtype(
            x, 'x', ['float16', 'uint16', 'float32', 'float64'], 'elu'
        )
        helper = LayerHelper("elu", **locals())
        out = helper.create_variable_for_type_inference(x.dtype)
        helper.append_op(
            type='elu',
            inputs={'X': x},
            outputs={'Out': out},
            attrs={'alpha': alpha},
        )
        return out


@inplace_apis_in_dygraph_only
def elu_(x, alpha=1.0, name=None):
    r"""
    Inplace version of ``elu`` API, the output Tensor will be inplaced with input ``x``.
    Please refer to :ref:`api_nn_cn_elu`.
    """
    assert alpha >= 0.0, "elu_ only support alpha >= 0, please use elu instead."
    if in_dynamic_mode():
        return _C_ops.elu_(x, alpha)
    return _legacy_C_ops.elu_(x, 'alpha', alpha)


def gelu(x, approximate=False, name=None):
    r"""
    gelu activation.

    The activation function of Gelu is calculated element by element. More information refers to :ref: `Gaussian Error Linear Units`.

    if approximate is True

    .. math::

        gelu(x) = 0.5 * x * (1 + tanh(\sqrt{\frac{2}{\pi}} * (x + 0.044715x^{3})))

    else

    .. math::

        gelu(x) = 0.5 * x * (1 + erf(\frac{x}{\sqrt{2}}))

    Parameters:
        x (Tensor): The input Tensor with data type float32, float64.
        approximate (bool, optional): Whether to enable approximation. Default is False.
        name (str, optional): For details, please refer to :ref:`api_guide_Name`. Generally, no setting is required. Default: None.

    Returns:
        A Tensor with the same data type and shape as ``x`` .

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> import paddle.nn.functional as F

            >>> x = paddle.to_tensor([[-1, 0.5], [1, 1.5]])
            >>> out1 = F.gelu(x)
            >>> print(out1)
            Tensor(shape=[2, 2], dtype=float32, place=Place(cpu), stop_gradient=True,
            [[-0.15865529,  0.34573123],
             [ 0.84134471,  1.39978933]])
            >>> out2 = F.gelu(x, True)
            >>> print(out2)
            Tensor(shape=[2, 2], dtype=float32, place=Place(cpu), stop_gradient=True,
            [[-0.15880796,  0.34571400],
             [ 0.84119201,  1.39957154]])
    """

    if in_dynamic_mode():
        return _C_ops.gelu(x, approximate)
    else:
        check_variable_and_dtype(
            x, 'x', ['float16', 'uint16', 'float32', 'float64'], 'gelu'
        )
        helper = LayerHelper("gelu", **locals())
        out = helper.create_variable_for_type_inference(x.dtype)
        helper.append_op(
            type='gelu',
            inputs={'X': x},
            outputs={'Out': out},
            attrs={'approximate': approximate},
        )
        return out


def hardshrink(x, threshold=0.5, name=None):
    r"""
    hard shrinkage activation

    .. math::

        hardshrink(x)=
            \left\{
                \begin{array}{rcl}
                x,&  &if \ {x > threshold}  \\
                x,&  &if \ {x < -threshold}   \\
                0,&  &if \ {others} &
                \end{array}
            \right.

    Args:
        x (Tensor): The input Tensor with data type float32, float64.
        threshold (float, optional): The value of threshold for hardthrink. Default is 0.5.
        name (str, optional): For details, please refer to :ref:`api_guide_Name`. Generally, no setting is required. Default: None.

    Returns:
        A Tensor with the same data type and shape as ``x`` .

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> import paddle.nn.functional as F

            >>> x = paddle.to_tensor([-1, 0.3, 2.5])
            >>> out = F.hardshrink(x)
            >>> print(out)
            Tensor(shape=[3], dtype=float32, place=Place(cpu), stop_gradient=True,
            [-1.       ,  0.       , 2.50000000])


    """
    if in_dynamic_mode():
        return _C_ops.hardshrink(x, threshold)
    else:
        check_variable_and_dtype(
            x, 'x', ['float16', 'uint16', 'float32', 'float64'], 'hardshrink'
        )
        helper = LayerHelper('hardshrink', **locals())
        out = helper.create_variable_for_type_inference(x.dtype)
        helper.append_op(
            type='hard_shrink',
            inputs={'X': x},
            outputs={'Out': out},
            attrs={'threshold': threshold},
        )
        return out


def hardtanh(x, min=-1.0, max=1.0, name=None):
    r"""
    hardtanh activation. Calculate the `hardtanh` of input `x`.

    .. math::

        hardtanh(x)=
            \left\{
                \begin{array}{cll}
                    max,& & \text{if } x > max \\
                    min,& & \text{if } x < min \\
                    x,& & \text{otherwise}
                \end{array}
            \right.

    Parameters:
        x (Tensor): The input Tensor with data type float32, float64.
        min (float, optional): The minimum value of the linear region range. Default is -1.
        max (float, optional): The maximum value of the linear region range. Default is 1.
        name (str, optional): For details, please refer to :ref:`api_guide_Name`. Generally, no setting is required. Default: None.

    Returns:
        A Tensor with the same data type and shape as ``x`` .

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> import paddle.nn.functional as F

            >>> x = paddle.to_tensor([-1.5, 0.3, 2.5])
            >>> out = F.hardtanh(x)
            >>> print(out)
            Tensor(shape=[3], dtype=float32, place=Place(cpu), stop_gradient=True,
            [-1.       , 0.30000001,  1.       ])
    """

    if in_dynamic_mode():
        return _C_ops.hardtanh(x, min, max)
    else:
        check_variable_and_dtype(
            x, 'x', ['float16', 'float32', 'float64'], 'hardtanh'
        )

        helper = LayerHelper('hardtanh', **locals())
        out = helper.create_variable_for_type_inference(dtype=x.dtype)
        helper.append_op(
            type='brelu',
            inputs={'X': x},
            outputs={'Out': out},
            attrs={'t_min': min, 't_max': max},
        )
        return out


@inplace_apis_in_dygraph_only
def hardtanh_(x, min=-1.0, max=1.0, name=None):
    r"""
    Inplace version of ``hardtanh`` API, the output Tensor will be inplaced with input ``x``.
    Please refer to :ref:`paddle_nn_functional_hardtanh`.
    """
    if in_dynamic_mode():
        return _C_ops.hardtanh_(x, min, max)


def hardsigmoid(x, slope=0.1666667, offset=0.5, name=None):
    r"""
    hardsigmoid activation. Calculate the `hardsigmoid` of input `x`.
    A 3-part piecewise linear approximation of sigmoid(https://arxiv.org/abs/1603.00391),
    which is much faster than sigmoid.

    .. math::

        hardsigmoid(x)=
            \left\{
                \begin{array}{lcl}
                0, & &\text{if } \ x \leq -3 \\
                1, & &\text{if } \ x \geq 3 \\
                slope * x + offset, & &\text{otherwise}
                \end{array}
            \right.

    Parameters:
        x (Tensor): The input Tensor with data type float32, float64.
        slope (float, optional): The slope of hardsigmoid function. Default is 0.1666667.
        offset (float, optional): The offset of hardsigmoid function. Default is 0.5.
        name (str, optional): For details, please refer to :ref:`api_guide_Name`. Generally, no setting is required. Default: None.

    Returns:
        A Tensor with the same data type and shape as ``x`` .

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> import paddle.nn.functional as F

            >>> x = paddle.to_tensor([-4., 5., 1.])
            >>> out = F.hardsigmoid(x)
            >>> print(out)
            Tensor(shape=[3], dtype=float32, place=Place(cpu), stop_gradient=True,
            [0.        , 1.        , 0.66666669])
    """

    if in_dynamic_mode():
        return _C_ops.hardsigmoid(x, slope, offset)
    else:
        check_variable_and_dtype(
            x, 'x', ['float16', 'uint16', 'float32', 'float64'], 'hardsigmoid'
        )

        helper = LayerHelper('hardsigmoid', **locals())
        out = helper.create_variable_for_type_inference(x.dtype)
        helper.append_op(
            type='hard_sigmoid',
            inputs={'X': x},
            outputs={'Out': out},
            attrs={'slope': slope, 'offset': offset},
        )
        return out


def hardswish(x, name=None):
    r"""
    hardswish activation. hardswish is proposed in MobileNetV3, and performs
    better in computational stability and efficiency compared to swish function.
    For more details please refer to: https://arxiv.org/pdf/1905.02244.pdf

    .. math::

        hardswish(x)=
            \left\{
                \begin{array}{cll}
                0 &, & \text{if } x \leq -3 \\
                x &, & \text{if } x \geq 3 \\
                \frac{x(x+3)}{6} &, & \text{otherwise}
                \end{array}
            \right.

    Parameters:
        x (Tensor): The input Tensor with data type float32, float64.
        name (str, optional): For details, please refer to :ref:`api_guide_Name`. Generally, no setting is required. Default: None.

    Returns:
        A Tensor with the same data type and shape as ``x`` .

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> import paddle.nn.functional as F

            >>> x = paddle.to_tensor([-4., 5., 1.])
            >>> out = F.hardswish(x)
            >>> print(out)
            Tensor(shape=[3], dtype=float32, place=Place(cpu), stop_gradient=True,
            [-0.       , 5.        , 0.66666669])
    """
    if in_dynamic_mode():
        return _C_ops.hardswish(x)
    else:
        check_variable_and_dtype(
            x, 'x', ['float16', 'uint16', 'float32', 'float64'], 'hardswish'
        )

        threshold = 6.0
        scale = 6.0
        offset = 3.0
        helper = LayerHelper('hardswish', **locals())
        out = helper.create_variable_for_type_inference(x.dtype)
        helper.append_op(
            type='hard_swish',
            inputs={'X': x},
            outputs={'Out': out},
            attrs={'threshold': threshold, 'scale': scale, 'offset': offset},
        )
        return out


def leaky_relu(x, negative_slope=0.01, name=None):
    r"""
    leaky_relu activation. The calculation formula is:

    .. math::
        leaky\_relu(x)=
        \left\{
            \begin{array}{rcl}
                x, & & if \ x >= 0 \\
                negative\_slope * x, & & otherwise \\
            \end{array}
        \right.

    Args:
        x (Tensor): The input Tensor with data type float32, float64.
        negative_slope (float, optional): Slope of the activation function at
            :math:`x < 0` . Default is 0.01.
        name (str, optional): For details, please refer to :ref:`api_guide_Name`. Generally, no setting is required. Default: None.

    Returns:
        A Tensor with the same data type and shape as ``x`` .

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> import paddle.nn.functional as F

            >>> x = paddle.to_tensor([-2., 0., 1.])
            >>> out = F.leaky_relu(x)
            >>> print(out)
            Tensor(shape=[3], dtype=float32, place=Place(cpu), stop_gradient=True,
            [-0.02000000,  0.        ,  1.        ])

    """
    if in_dynamic_mode():
        return _C_ops.leaky_relu(x, negative_slope)
    else:
        check_variable_and_dtype(
            x, 'x', ['float16', 'uint16', 'float32', 'float64'], 'leaky_relu'
        )
        helper = LayerHelper('leaky_relu', **locals())
        out = helper.create_variable_for_type_inference(dtype=x.dtype)
        helper.append_op(
            type='leaky_relu',
            inputs={'X': x},
            outputs={'Out': out},
            attrs={'alpha': negative_slope},
        )
        return out


@inplace_apis_in_dygraph_only
def leaky_relu_(x, negative_slope=0.01, name=None):
    r"""
    Inplace version of ``leaky_relu`` API, the output Tensor will be inplaced with input ``x``.
    Please refer to :ref:`paddle_nn_functional_leaky_relu`.
    """
    if in_dynamic_mode():
        return _C_ops.leaky_relu_(x, negative_slope)


def prelu(x, weight, data_format="NCHW", name=None):
    """
    prelu activation. The calculation formula is follows:

    .. math::

        prelu(x) = max(0, x) + weight * min(0, x)

    x and weight is input Tensor.

    Parameters:
        x (Tensor): The input Tensor with data type float32, float64.
        weight (Tensor): The learnable parameter with data type same as ``x``.
            The weight shape is [], [1] or [in], where `in` is the input channel of ``x``.
        name (str, optional): For details, please refer to :ref:`api_guide_Name`. Generally, no setting is required. Default: None.
        data_format(str, optional): Data format that specifies the layout of input.
            It may be "NC", "NCL", "NCHW", "NCDHW", "NLC", "NHWC" or "NDHWC". Default: "NCHW".

    Returns:
        A Tensor with the same data type and shape as ``x`` .

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> import paddle.nn.functional as F

            >>> data = paddle.to_tensor([[[[-2.0,  3.0, -4.0,  5.0],
            ...                            [ 3.0, -4.0,  5.0, -6.0],
            ...                            [-7.0, -8.0,  8.0,  9.0]],
            ...                           [[ 1.0, -2.0, -3.0,  4.0],
            ...                            [-5.0,  6.0,  7.0, -8.0],
            ...                            [ 6.0,  7.0,  8.0,  9.0]]]], dtype='float32')

            >>> w = paddle.to_tensor([0.25], dtype='float32')
            >>> out = F.prelu(data, w)
            >>> print(out)
            Tensor(shape=[1, 2, 3, 4], dtype=float32, place=Place(cpu), stop_gradient=True,
            [[[[-0.50000000,  3.        , -1.        ,  5.        ],
               [ 3.        , -1.        ,  5.        , -1.50000000],
               [-1.75000000, -2.        ,  8.        ,  9.        ]],
              [[ 1.        , -0.50000000, -0.75000000,  4.        ],
               [-1.25000000,  6.        ,  7.        , -2.        ],
               [ 6.        ,  7.        ,  8.        ,  9.        ]]]])
    """
    assert (
        len(weight.shape) == 0 or len(weight.shape) == 1
    ), "The dim count of weight shape should be 0 or 1 in prelu()."

    mode = 'all'
    if len(weight.shape) == 1 and weight.shape[0] > 1:
        true_data_format = [
            'NC',
            'NCL',
            'NCHW',
            'NCDHW',
            'NLC',
            'NHWC',
            'NDHWC',
        ]
        if data_format not in true_data_format:
            raise ValueError(
                "data_format must be one of 'NC', 'NCL', 'NCHW', 'NCDHW', "
                "'NLC', 'NHWC', 'NDHWC' but receive {}".format(data_format)
            )

        data_format = 'NCHW' if data_format[1] == 'C' else 'NHWC'

        assert (
            len(x.shape) > 1
        ), "The dim count of x should be equal or larger than 2 in prelu() when weight shape is not [1]."

        # NOTE(GuoxiaWang): support NHWC data format
        if data_format == 'NHWC':
            assert (
                weight.shape[0] == x.shape[-1]
            ), "The weight size should be equal to x input channel in prelu() when weight shape is not [1]."
        else:
            assert (
                weight.shape[0] == x.shape[1]
            ), "The weight size should be equal to x input channel in prelu() when weight shape is not [1]."
        mode = 'channel'

    if in_dynamic_mode():
        return _C_ops.prelu(x, weight, data_format, mode)
    else:
        check_variable_and_dtype(
            x, 'x', ['float16', 'float32', 'float64', 'uint16'], 'prelu'
        )
        check_variable_and_dtype(
            weight,
            'weight',
            ['float16', 'float32', 'float64', 'uint16'],
            'prelu',
        )
        helper = LayerHelper('prelu', **locals())
        out = helper.create_variable_for_type_inference(x.dtype)
        helper.append_op(
            type="prelu",
            inputs={"X": x, "Alpha": weight},
            outputs={"Out": out},
            attrs={"mode": mode, "data_format": data_format},
        )
        return out


def rrelu(x, lower=1.0 / 8.0, upper=1.0 / 3.0, training=True, name=None):
    r"""
    rrelu activation.

    Applies the randomized leaky rectified liner unit function to improve generalization performance,
    as described in the paper:
    `Empirical Evaluation of Rectified Activations in Convolutional Network <https://arxiv.org/abs/1505.00853>`_

    During training, randomly samples the negative slope for activation values as described below:

    .. math::

        rrelu(x)=
            \left\{
                \begin{array}{rcl}
                    x, & & if \ x >= 0 \\
                    a * x, & & otherwise \\
                \end{array}
            \right.

    where :math:`x` is the input tensor,
    :math:`a` is randomly sampled from uniform distribution in range (:math:`lower`, :math:`upper`),

    In the test phase, the negative slope will take the average value of :math:`lower` and :math:`upper`:

    .. math::

        rrelu(x)=
            \left\{
                \begin{array}{rcl}
                    x, & & if \ x >= 0 \\
                    (lower + upper) * 0.5 * x, & & otherwise \\
                \end{array}
            \right.

    where :math:`x` is the input tensor,
    :math:`lower` and :math:`upper` are the bounds of uniform distribution.

    Parameters:
        x (Tensor): The input Tensor with data type float16, float32, float64.
        lower (float, optional): The lower bound of uniform distribution. Default: 0.125.
        upper (float, optional): The upper bound of uniform distribution. Default: 0.3333333333333333.
        training (bool, optional): Current mode is in training or others.  Default is True.
        name (str, optional): For details, please refer to :ref:`api_guide_Name`. Generally, no setting is required. Default: None.

    Returns:
        A Tensor with the same data type and shape as ``x`` .

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> import paddle.nn.functional as F
            >>> paddle.seed(1)
            >>> input_tensor = paddle.to_tensor([[[[-2.0,  3.0, -4.0,  5.0],
            ...                                    [ 3.0, -4.0,  5.0, -6.0],
            ...                                    [-7.0, -8.0,  8.0,  9.0]],
            ...                                   [[ 1.0, -2.0, -3.0,  4.0],
            ...                                    [-5.0,  6.0,  7.0, -8.0],
            ...                                    [ 6.0,  7.0,  8.0,  9.0]]]], dtype='float32')
            >>> out = F.rrelu(input_tensor, 0.1, 0.3)
            >>> print(out)
            Tensor(shape=[1, 2, 3, 4], dtype=float32, place=Place(cpu), stop_gradient=True,
            [[[[-0.20715050,  3.        , -1.01193857,  5.        ],
               [ 3.        , -0.94084597,  5.        , -0.65544695],
               [-1.24268556, -2.34339547,  8.        ,  9.        ]],
              [[ 1.        , -0.44942653, -0.68969047,  4.        ],
               [-1.03736508,  6.        ,  7.        , -0.95799232],
               [ 6.        ,  7.        ,  8.        ,  9.        ]]]])
    """
    if not isinstance(lower, float) or not isinstance(upper, float):
        raise TypeError(
            "The lower and upper values must be float type. Received: lower {}, upper {}.".format(
                lower, upper
            )
        )

    if lower < 0 or lower > 1:
        raise ValueError(
            "The lower value must be no less than zero or greater than one. Received: {}.".format(
                lower
            )
        )

    if upper < lower:
        raise ValueError(
            "The upper value must be greater than lower value. Received: lower {}, upper {}.".format(
                lower, upper
            )
        )

    if upper > 1:
        raise ValueError(
            "The upper value must be no greater than one. Received: {}.".format(
                upper
            )
        )

    is_test = not training

    if in_dynamic_mode():
        return _C_ops.rrelu(x, lower, upper, is_test)
    else:
        check_variable_and_dtype(
            x, 'X', ['float16', 'uint16', 'float32', 'float64'], 'rrelu'
        )
        helper = LayerHelper('rrelu', **locals())
        out = helper.create_variable_for_type_inference(x.dtype)
        noise = helper.create_variable_for_type_inference(dtype=x.dtype)
        attrs = {'lower': lower, 'upper': upper, 'is_test': is_test}
        helper.append_op(
            type='rrelu',
            inputs={"X": x},
            outputs={"Out": out, "Noise": noise},
            attrs=attrs,
        )
        return out


def relu(x, name=None):
    """
    relu activation. The calculation formula is follows:

    .. math::

        out = max(x, 0)

    x is input Tensor.

    Parameters:
        x (Tensor): The input Tensor with data type float32, float64.
        name (str, optional): For details, please refer to :ref:`api_guide_Name`. Generally, no setting is required. Default: None.

    Returns:
        A Tensor with the same data type and shape as ``x`` .

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> import paddle.nn.functional as F

            >>> x = paddle.to_tensor([-2, 0, 1], dtype='float32')
            >>> out = F.relu(x)
            >>> print(out)
            Tensor(shape=[3], dtype=float32, place=Place(cpu), stop_gradient=True,
            [0., 0., 1.])
    """

    if in_dynamic_mode():
        return _C_ops.relu(x)
    else:
        if paddle.framework.in_dynamic_or_new_ir_mode():
            # Below code will be removed after we can generate IR api automatically
            return paddle._ir_ops.relu(x)

        check_variable_and_dtype(
            x, 'x', ['float16', 'uint16', 'float32', 'float64'], 'relu'
        )
        helper = LayerHelper('relu', **locals())
        out = helper.create_variable_for_type_inference(x.dtype)
        helper.append_op(type='relu', inputs={'X': x}, outputs={'Out': out})
        return out


@inplace_apis_in_dygraph_only
def relu_(x, name=None):
    """
    Inplace version of ``relu`` API, the output Tensor will be inplaced with input ``x``.
    Please refer to :ref:`api_nn_cn_relu`.
    """
    return _C_ops.relu_(x)


def log_sigmoid(x, name=None):
    r"""
    log_sigmoid activation.

    .. math::

        log\_sigmoid(x) = log \frac{1}{1 + e^{-x}}

    Parameters:
        x (Tensor): The input Tensor with data type float32, float64, complex64, complex128.
        name (str, optional): For details, please refer to :ref:`api_guide_Name`. Generally, no setting is required. Default: None.

    Returns:
        A Tensor with the same data type and shape as ``x`` .

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> import paddle.nn.functional as F

            >>> x = paddle.to_tensor([1.0, 2.0, 3.0, 4.0])
            >>> out = F.log_sigmoid(x)
            >>> print(out)
            Tensor(shape=[4], dtype=float32, place=Place(cpu), stop_gradient=True,
            [-0.31326166, -0.12692805, -0.04858733, -0.01814996])
    """

    if in_dynamic_mode():
        return _C_ops.logsigmoid(x)
    else:
        check_variable_and_dtype(
            x,
            'x',
            ['float16', 'float32', 'float64', 'complex64', 'complex128'],
            'log_sigmoid',
        )
        helper = LayerHelper("log_sigmoid", **locals())
        out = helper.create_variable_for_type_inference(x.dtype)
        helper.append_op(
            type='logsigmoid', inputs={'X': x}, outputs={'Out': out}
        )
        return out


def maxout(x, groups, axis=1, name=None):
    r"""
    maxout activation.

    Assumed the input shape is (N, Ci, H, W).
    The output shape is (N, Co, H, W).
    Then Co = Ci/groups and the operator formula is as follows:

    .. math::

        \begin{array}{l}
        &out_{si+j} = \max_{k} x_{gsi + sk + j} \\
        &g = groups \\
        &s = \frac{input.size}{num\_channels} \\
        &0 \le i < \frac{num\_channels}{groups} \\
        &0 \le j < s \\
        &0 \le k < groups
        \end{array}


    Parameters:
        x (Tensor): The input is 4-D Tensor with shape [N, C, H, W] or [N, H, W, C], the data type
            of input is float16, float32 or float64.
        groups (int): The groups number of maxout. `groups` specifies the
            index of channel dimension where maxout will be performed. This must be
            a factor of number of features.
        axis (int, optional): The axis along which to perform maxout calculations.
            It should be 1 when data format is NCHW, be -1 or 3 when data format
            is NHWC. If ``axis`` < 0, it works the same way as :math:`axis + D` ,
            where D is the dimensions of ``x`` . ``axis`` only supports 1, 3 or -1.
            Default is 1.
        name (str, optional): For details, please refer to :ref:`api_guide_Name`. Generally, no setting is required. Default: None.

    Returns:
        A Tensor with the same data type as ``x`` .

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> import paddle.nn.functional as F

            >>> paddle.seed(2023)
            >>> x = paddle.rand([1, 2, 3, 4])
            >>> print(x)
            Tensor(shape=[1, 2, 3, 4], dtype=float32, place=Place(cpu), stop_gradient=True,
            [[[[0.86583614, 0.52014720, 0.25960937, 0.90525323],
               [0.42400089, 0.40641287, 0.97020894, 0.74437362],
               [0.51785129, 0.73292869, 0.97786582, 0.04315904]],
              [[0.42639419, 0.71958369, 0.20811461, 0.19731510],
               [0.38424349, 0.14603184, 0.22713774, 0.44607511],
               [0.21657862, 0.67685395, 0.46460176, 0.92382854]]]])
            >>> out = F.maxout(x, groups=2)
            >>> print(out)
            Tensor(shape=[1, 1, 3, 4], dtype=float32, place=Place(cpu), stop_gradient=True,
            [[[[0.86583614, 0.71958369, 0.25960937, 0.90525323],
               [0.42400089, 0.40641287, 0.97020894, 0.74437362],
               [0.51785129, 0.73292869, 0.97786582, 0.92382854]]]])
    """
    if in_dynamic_mode():
        return _C_ops.maxout(x, groups, axis)
    else:
        check_variable_and_dtype(
            x, 'x', ['float16', 'float32', 'float64'], 'maxout'
        )
        if axis not in [1, -1, 3]:
            raise ValueError(
                "Attr(axis) should be 1 when data format is NCHW, -1 or 3 when data format is NHWC. Received "
                "Attr(axis): %s." % str(axis)
            )
        if axis == -1:
            axis = 3

        helper = LayerHelper('maxout', **locals())
        out = helper.create_variable_for_type_inference(x.dtype)
        helper.append_op(
            type='maxout',
            inputs={'X': x},
            outputs={'Out': out},
            attrs={'groups': groups, 'axis': axis},
        )
        return out


def relu6(x, name=None):
    """
    relu6 activation

    .. math::

        relu6(x) = min(max(0,x), 6)

    Parameters:
        x (Tensor): The input Tensor with data type float32, float64.
        name (str, optional): For details, please refer to :ref:`api_guide_Name`. Generally, no setting is required. Default: None.

    Returns:
        A Tensor with the same data type and shape as ``x`` .

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> import paddle.nn.functional as F

            >>> x = paddle.to_tensor([-1, 0.3, 6.5])
            >>> out = F.relu6(x)
            >>> print(out)
            Tensor(shape=[3], dtype=float32, place=Place(cpu), stop_gradient=True,
            [0.        , 0.30000001, 6.        ])
    """
    threshold = 6.0
    if in_dynamic_mode():
        return _C_ops.relu6(x)

    check_variable_and_dtype(
        x, 'x', ['float16', 'uint16', 'float32', 'float64'], 'relu6'
    )
    helper = LayerHelper('relu6', **locals())
    out = helper.create_variable_for_type_inference(x.dtype)
    helper.append_op(
        type='relu6',
        inputs={'X': x},
        outputs={'Out': out},
        attrs={'threshold': threshold},
    )
    return out


def selu(
    x,
    scale=1.0507009873554804934193349852946,
    alpha=1.6732632423543772848170429916717,
    name=None,
):
    r"""
    selu activation

    .. math::

        selu(x)= scale *
            \left\{
                \begin{array}{lcl}
                x,& &\text{if } \ x > 0 \\
                alpha * e^{x} - alpha,& &\text{if } \ x <= 0
                \end{array}
            \right.

    Parameters:
        x (Tensor): The input Tensor with data type float32, float64.
        scale (float, optional): The value of scale(must be greater than 1.0) for selu. Default is 1.0507009873554804934193349852946.
        alpha (float, optional): The value of alpha(must be no less than zero) for selu. Default is 1.6732632423543772848170429916717.
        name (str, optional): For details, please refer to :ref:`api_guide_Name`. Generally, no setting is required. Default: None.

    Returns:
        A Tensor with the same data type and shape as ``x`` .

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> import paddle.nn.functional as F

            >>> x = paddle.to_tensor([[0.0, 1.0],[2.0, 3.0]])
            >>> out = F.selu(x)
            >>> print(out)
            Tensor(shape=[2, 2], dtype=float32, place=Place(cpu), stop_gradient=True,
            [[0.        , 1.05070102],
             [2.10140204, 3.15210295]])
    """
    if scale <= 1.0:
        raise ValueError(
            f"The scale must be greater than 1.0. Received: {scale}."
        )

    if alpha < 0:
        raise ValueError(
            f"The alpha must be no less than zero. Received: {alpha}."
        )

    if in_dynamic_mode():
        return _C_ops.selu(x, scale, alpha)
    else:
        check_variable_and_dtype(
            x, 'x', ['float16', 'float32', 'float64'], 'selu'
        )
        helper = LayerHelper('selu', **locals())
        out = helper.create_variable_for_type_inference(x.dtype)
        helper.append_op(
            type='selu',
            inputs={'X': x},
            outputs={'Out': out},
            attrs={'scale': scale, 'alpha': alpha},
        )
        return out


def silu(x, name=None):
    r"""
    silu activation

    .. math::

        silu(x) = \frac{x}{1 + e^{-x}}

    Where :math:`x` is the input Tensor.

    Parameters:
        x (Tensor): The input Tensor with data type bfloat16, float16, float32, float64, complex64, complex128.
        name (str, optional): For details, please refer to :ref:`api_guide_Name`. Generally, no setting is required. Default: None.

    Returns:
        A Tensor with the same data type and shape as :attr:`x`.

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> import paddle.nn.functional as F

            >>> x = paddle.to_tensor([1.0, 2.0, 3.0, 4.0])
            >>> out = F.silu(x)
            >>> print(out)
            Tensor(shape=[4], dtype=float32, place=Place(cpu), stop_gradient=True,
            [0.73105860, 1.76159406, 2.85772228, 3.92805505])
    """

    if in_dynamic_mode():
        return _C_ops.silu(x)
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
            'silu',
        )
        helper = LayerHelper("silu", **locals())
        out = helper.create_variable_for_type_inference(x.dtype)
        helper.append_op(type='silu', inputs={'X': x}, outputs={'Out': out})
        return out


def softmax(x, axis=-1, dtype=None, name=None):
    r"""
    This operator implements the softmax layer. The calculation process is as follows:

    1. The dimension :attr:`axis` of ``x`` will be permuted to the last.

    2. Then ``x`` will be logically flattened to a 2-D matrix. The matrix's second
    dimension(row length) is the same as the dimension :attr:`axis` of ``x``,
    and the first dimension(column length) is the product of all other dimensions
    of ``x``. For each row of the matrix, the softmax operator squashes the
    K-dimensional(K is the width of the matrix, which is also the size of ``x``'s
    dimension :attr:`axis`) vector of arbitrary real values to a K-dimensional
    vector of real values in the range [0, 1] that add up to 1.

    3. After the softmax operation is completed, the inverse operations of steps 1 and 2
    are performed to restore the two-dimensional matrix to the same dimension as the ``x`` .

    It computes the exponential of the given dimension and the sum of exponential
    values of all the other dimensions in the K-dimensional vector input.
    Then the ratio of the exponential of the given dimension and the sum of
    exponential values of all the other dimensions is the output of the softmax
    operator.

    For each row :math:`i` and each column :math:`j` in the matrix, we have:

    .. math::

        softmax[i, j] = \frac{\exp(x[i, j])}{\sum_j(exp(x[i, j])}

    Example:

    .. code-block:: text

        Case 1:
          Input:
            x.shape = [2, 3, 4]
            x.data = [[[2.0, 3.0, 4.0, 5.0],
                       [3.0, 4.0, 5.0, 6.0],
                       [7.0, 8.0, 8.0, 9.0]],
                      [[1.0, 2.0, 3.0, 4.0],
                       [5.0, 6.0, 7.0, 8.0],
                       [6.0, 7.0, 8.0, 9.0]]]

          Attrs:
            axis = -1

          Output:
            out.shape = [2, 3, 4]
            out.data = [[[0.0320586 , 0.08714432, 0.23688282, 0.64391426],
                         [0.0320586 , 0.08714432, 0.23688282, 0.64391426],
                         [0.07232949, 0.19661193, 0.19661193, 0.53444665]],
                        [[0.0320586 , 0.08714432, 0.23688282, 0.64391426],
                         [0.0320586 , 0.08714432, 0.23688282, 0.64391426],
                         [0.0320586 , 0.08714432, 0.23688282, 0.64391426]]]

        Case 2:
          Input:
            x.shape = [2, 3, 4]
            x.data = [[[2.0, 3.0, 4.0, 5.0],
                       [3.0, 4.0, 5.0, 6.0],
                       [7.0, 8.0, 8.0, 9.0]],
                      [[1.0, 2.0, 3.0, 4.0],
                       [5.0, 6.0, 7.0, 8.0],
                       [6.0, 7.0, 8.0, 9.0]]]
          Attrs:
            axis = 1

          Output:
            out.shape = [2, 3, 4]
            out.data = [[[0.00657326, 0.00657326, 0.01714783, 0.01714783],
                         [0.01786798, 0.01786798, 0.04661262, 0.04661262],
                         [0.97555875, 0.97555875, 0.93623955, 0.93623955]],
                        [[0.00490169, 0.00490169, 0.00490169, 0.00490169],
                         [0.26762315, 0.26762315, 0.26762315, 0.26762315],
                         [0.72747516, 0.72747516, 0.72747516, 0.72747516]]]

    Parameters:
        x (Tensor): The input Tensor with data type bfloat16, float16, float32, float64.
        axis (int, optional): The axis along which to perform softmax
            calculations. It should be in range [-D, D), where D is the
            rank of ``x`` . If ``axis`` < 0, it works the same way as
            :math:`axis + D` . Default is -1.
        dtype (str, optional): The data type of the output tensor, can be bfloat16, float16, float32, float64.
        name (str, optional): For details, please refer to :ref:`api_guide_Name`. Generally, no setting is required. Default: None.

    Returns:
        A Tensor with the same shape and data type (use ``dtype`` if it is
        specified) as x.

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> import paddle.nn.functional as F

            >>> x = paddle.to_tensor([[[2.0, 3.0, 4.0, 5.0],
            ...                        [3.0, 4.0, 5.0, 6.0],
            ...                        [7.0, 8.0, 8.0, 9.0]],
            ...                       [[1.0, 2.0, 3.0, 4.0],
            ...                        [5.0, 6.0, 7.0, 8.0],
            ...                        [6.0, 7.0, 8.0, 9.0]]],dtype='float32')
            >>> out1 = F.softmax(x)
            >>> out2 = F.softmax(x, dtype='float64')
            >>> #out1's data type is float32; out2's data type is float64
            >>> #out1 and out2's value is as follows:
            >>> print(out1)
            >>> print(out2)
            Tensor(shape=[2, 3, 4], dtype=float32, place=Place(cpu), stop_gradient=True,
            [[[0.03205860, 0.08714432, 0.23688284, 0.64391428],
              [0.03205860, 0.08714432, 0.23688284, 0.64391428],
              [0.07232949, 0.19661194, 0.19661194, 0.53444666]],
             [[0.03205860, 0.08714432, 0.23688284, 0.64391428],
              [0.03205860, 0.08714432, 0.23688284, 0.64391428],
              [0.03205860, 0.08714432, 0.23688284, 0.64391428]]])
            Tensor(shape=[2, 3, 4], dtype=float64, place=Place(cpu), stop_gradient=True,
            [[[0.03205860, 0.08714432, 0.23688282, 0.64391426],
              [0.03205860, 0.08714432, 0.23688282, 0.64391426],
              [0.07232949, 0.19661193, 0.19661193, 0.53444665]],
             [[0.03205860, 0.08714432, 0.23688282, 0.64391426],
              [0.03205860, 0.08714432, 0.23688282, 0.64391426],
              [0.03205860, 0.08714432, 0.23688282, 0.64391426]]])
    """

    if (dtype is not None) and (not isinstance(dtype, core.VarDesc.VarType)):
        dtype = convert_np_dtype_to_dtype_(dtype)
    if in_dynamic_mode():
        outs_cast = x if dtype is None else _C_ops.cast(x, dtype)
        return _C_ops.softmax(outs_cast, axis)
    else:
        use_cudnn = True
        if dtype is None:
            check_variable_and_dtype(
                x, 'x', ['uint16', 'float16', 'float32', 'float64'], 'softmax'
            )
        else:
            check_dtype(
                dtype,
                'dtype',
                ['uint16', 'float16', 'float32', 'float64'],
                'softmax',
                'If dtype is not None, it only support uint16, float16, float32 or float64.',
            )

        helper = LayerHelper("softmax", **locals())
        outs_cast = x
        if dtype is not None:
            outs_cast = helper.create_variable_for_type_inference(dtype)
            helper.append_op(
                type='cast',
                inputs={'X': x},
                outputs={'Out': outs_cast},
                attrs={'in_dtype': x.dtype, 'out_dtype': dtype},
            )

        outs_softmax = helper.create_variable_for_type_inference(
            outs_cast.dtype
        )
        helper.append_op(
            type='softmax',
            inputs={'X': outs_cast},
            outputs={'Out': outs_softmax},
            attrs={'axis': axis, 'use_cudnn': use_cudnn},
        )

        return outs_softmax


@inplace_apis_in_dygraph_only
def softmax_(x, axis=-1, dtype=None, name=None):
    r"""
    Inplace version of ``softmax`` API, the output Tensor will be inplaced with input ``x``.
    Please refer to :ref:`api_nn_cn_softmax`.
    """
    if (dtype is not None) and (not isinstance(dtype, core.VarDesc.VarType)):
        dtype = convert_np_dtype_to_dtype_(dtype)
    outs_cast = (
        x
        if dtype is None
        else _legacy_C_ops.cast(x, 'in_dtype', x.dtype, 'out_dtype', dtype)
    )
    return _C_ops.softmax_(outs_cast, axis)


def softplus(x, beta=1, threshold=20, name=None):
    r"""
    softplus activation

    .. math::
        softplus(x)=\begin{cases}
                \frac{1}{\beta} * \log(1 + e^{\beta * x}),&x\leqslant\frac{\varepsilon}{\beta};\\
                x,&x>\frac{\varepsilon}{\beta}.
            \end{cases}

    Parameters:
        x (Tensor): The input Tensor with data type float32, float64.
        beta (float, optional): The value of :math:`\beta` for softplus. Default is 1
        threshold (float, optional): The value of :math:`\varepsilon` for softplus. Default is 20
        name (str, optional): For details, please refer to :ref:`api_guide_Name`. Generally, no setting is required. Default: None.

    Returns:
        A Tensor with the same data type and shape as ``x`` .

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> import paddle.nn.functional as F

            >>> x = paddle.to_tensor([-0.4, -0.2, 0.1, 0.3], dtype='float32')
            >>> out = F.softplus(x)
            >>> print(out)
            Tensor(shape=[4], dtype=float32, place=Place(cpu), stop_gradient=True,
            [0.51301527, 0.59813893, 0.74439669, 0.85435522])
    """

    if in_dynamic_mode():
        return _C_ops.softplus(x, beta, threshold)
    else:
        check_variable_and_dtype(
            x, 'x', ['float16', 'uint16', 'float32', 'float64'], 'softplus'
        )
        helper = LayerHelper('softplus', **locals())
        out = helper.create_variable_for_type_inference(x.dtype)
        helper.append_op(
            type='softplus',
            inputs={'X': x},
            outputs={'Out': out},
            attrs={'beta': beta, 'threshold': threshold},
        )
        return out


def softshrink(x, threshold=0.5, name=None):
    r"""
    softshrink activation

    .. math::

        softshrink(x)=
            \left\{
                \begin{array}{rcl}
                x - threshold,& & \text{if } x > threshold \\
                x + threshold,& & \text{if } x < -threshold \\
                0,& &  \text{otherwise}
            \end{array}
            \right.

    Parameters:
        x (Tensor): The input Tensor with data type float32, float64.
        threshold (float, optional): The value of threshold(must be no less than zero) for softplus. Default is 0.5
        name (str, optional): For details, please refer to :ref:`api_guide_Name`. Generally, no setting is required. Default: None.

    Returns:
        A Tensor with the same data type and shape as ``x`` .

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> import paddle.nn.functional as F

            >>> x = paddle.to_tensor([-0.9, -0.2, 0.1, 0.8])
            >>> out = F.softshrink(x)
            >>> print(out)
            Tensor(shape=[4], dtype=float32, place=Place(cpu), stop_gradient=True,
            [-0.39999998,  0.        ,  0.        ,  0.30000001])
    """
    if threshold < 0:
        raise ValueError(
            "The threshold must be no less than zero. Received: {}.".format(
                threshold
            )
        )

    if in_dynamic_mode():
        return _C_ops.softshrink(x, threshold)
    else:
        check_variable_and_dtype(
            x, 'x', ['float16', 'uint16', 'float32', 'float64'], 'softshrink'
        )
        helper = LayerHelper('softshrink', **locals())
        out = helper.create_variable_for_type_inference(x.dtype)
        helper.append_op(
            type='softshrink',
            inputs={'X': x},
            outputs={'Out': out},
            attrs={'lambda': threshold},
        )
        return out


def softsign(x, name=None):
    r"""
    softsign activation

    .. math::

        softsign(x) = \frac{x}{1 + |x|}

    Parameters:
        x (Tensor): The input Tensor with data type float32, float64.
        name (str, optional): For details, please refer to :ref:`api_guide_Name`. Generally, no setting is required. Default: None.

    Returns:
        A Tensor with the same data type and shape as ``x`` .

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> import paddle.nn.functional as F

            >>> x = paddle.to_tensor([-0.4, -0.2, 0.1, 0.3])
            >>> out = F.softsign(x)
            >>> print(out)
            Tensor(shape=[4], dtype=float32, place=Place(cpu), stop_gradient=True,
            [-0.28571430, -0.16666666,  0.09090909,  0.23076925])
    """
    if in_dynamic_mode():
        return _C_ops.softsign(x)

    check_variable_and_dtype(
        x, 'x', ['float16', 'uint16', 'float32', 'float64'], 'softsign'
    )
    helper = LayerHelper('softsign', **locals())
    out = helper.create_variable_for_type_inference(x.dtype)
    helper.append_op(type='softsign', inputs={'X': x}, outputs={'Out': out})
    return out


def swish(x, name=None):
    r"""
    swish activation.

    .. math::

        swish(x) = \frac{x}{1 + e^{-x}}

    Parameters:
        x (Tensor): The input Tensor with data type float32, float64.
        name (str, optional): For details, please refer to :ref:`api_guide_Name`. Generally, no setting is required. Default: None.

    Returns:
        A Tensor with the same data type and shape as ``x`` .

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> import paddle.nn.functional as F

            >>> x = paddle.to_tensor([-2., 0., 1.])
            >>> out = F.swish(x)
            >>> print(out)
            Tensor(shape=[3], dtype=float32, place=Place(cpu), stop_gradient=True,
            [-0.23840584,  0.        ,  0.73105860])
    """
    if in_dynamic_mode():
        return _C_ops.swish(x)
    else:
        check_variable_and_dtype(
            x, 'x', ['float16', 'uint16', 'float32', 'float64'], 'swish'
        )
        helper = LayerHelper('swish', **locals())
        out = helper.create_variable_for_type_inference(x.dtype)
        helper.append_op(
            type='swish',
            inputs={'X': x},
            outputs={'Out': out},
            attrs={'beta': 1.0},
        )
        return out


def mish(x, name=None):
    r"""
    mish activation.

    ..  math::

        softplus(x) = \begin{cases}
                x, \text{if } x > \text{threshold} \\
                \ln(1 + e^{x}),  \text{otherwise}
            \end{cases}

        mish(x) = x * \tanh(softplus(x))

    Parameters:
        x (Tensor): The input Tensor with data type float32, float64.
        name (str, optional): For details, please refer to :ref:`api_guide_Name`. Generally, no setting is required. Default: None.

    Returns:
        A Tensor with the same data type and shape as ``x`` .

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> import paddle.nn.functional as F

            >>> x = paddle.to_tensor([-5., 0., 5.])
            >>> out = F.mish(x)
            >>> print(out)
            Tensor(shape=[3], dtype=float32, place=Place(cpu), stop_gradient=True,
            [-0.03357624,  0.        ,  4.99955177])
    """
    if in_dynamic_mode():
        return _C_ops.mish(x, 20)
    else:
        check_variable_and_dtype(
            x, 'x', ['float16', 'uint16', 'float32', 'float64'], 'mish'
        )
        helper = LayerHelper('mish', **locals())
        out = helper.create_variable_for_type_inference(x.dtype)
        helper.append_op(type='mish', inputs={'X': x}, outputs={'Out': out})
        return out


def tanhshrink(x, name=None):
    """
    tanhshrink activation

    .. math::

        tanhshrink(x) = x - tanh(x)

    Args:
        x (Tensor): The input Tensor with data type float32, float64.
        name (str, optional): For details, please refer to :ref:`api_guide_Name`. Generally, no setting is required. Default: None.

    Returns:
        A Tensor with the same data type and shape as ``x`` .

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> import paddle.nn.functional as F

            >>> x = paddle.to_tensor([-0.4, -0.2, 0.1, 0.3])
            >>> out = F.tanhshrink(x)
            >>> print(out)
            Tensor(shape=[4], dtype=float32, place=Place(cpu), stop_gradient=True,
            [-0.02005100, -0.00262472,  0.00033201,  0.00868741])
    """
    if in_dynamic_mode():
        return _C_ops.tanh_shrink(x)
    else:
        check_variable_and_dtype(
            x, 'x', ['float16', 'uint16', 'float32', 'float64'], 'tanhshrink'
        )
        helper = LayerHelper('tanh_shrink', **locals())
        out = helper.create_variable_for_type_inference(x.dtype)
        helper.append_op(
            type='tanh_shrink', inputs={'X': x}, outputs={'Out': out}
        )
        return out


def thresholded_relu(x, threshold=1.0, name=None):
    r"""
    thresholded relu activation.

    .. math::

        thresholded\_relu(x) =
            \left\{
                \begin{array}{rl}
                x,& \text{if } \ x > threshold \\
                0,& \text{otherwise}
                \end{array}
            \right.


    Parameters:
        x (Tensor): The input Tensor with data type float32, float64.
        threshold (float, optional): The value of threshold for thresholded_relu. Default is 1.0
        name (str, optional): For details, please refer to :ref:`api_guide_Name`. Generally, no setting is required. Default: None.

    Returns:
        A Tensor with the same data type and shape as ``x`` .

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> import paddle.nn.functional as F

            >>> x = paddle.to_tensor([2., 0., 1.])
            >>> out = F.thresholded_relu(x)
            >>> print(out)
            Tensor(shape=[3], dtype=float32, place=Place(cpu), stop_gradient=True,
            [2., 0., 0.])
    """

    if in_dynamic_mode():
        return _C_ops.thresholded_relu(x, threshold)
    else:
        check_variable_and_dtype(
            x,
            'x',
            ['float16', 'uint16', 'float32', 'float64'],
            'thresholded_relu',
        )
        helper = LayerHelper('thresholded_relu', **locals())
        out = helper.create_variable_for_type_inference(x.dtype)
        helper.append_op(
            type='thresholded_relu',
            inputs={'X': x},
            outputs={'Out': out},
            attrs={'threshold': threshold},
        )
        return out


@inplace_apis_in_dygraph_only
def thresholded_relu_(x, threshold=1.0, name=None):
    r"""
    Inplace version of ``thresholded_relu`` API, the output Tensor will be inplaced with input ``x``.
    Please refer to :ref:`paddle_nn_functional_thresholded_relu`.
    """
    if in_dynamic_mode():
        return _C_ops.thresholded_relu_(x, threshold)


def log_softmax(x, axis=-1, dtype=None, name=None):
    r"""
    This operator implements the log_softmax layer. The calculation process is
    as follows:

    .. math::

        \begin{aligned}
        log\_softmax[i, j] &= log(softmax(x)) \\
        &= log(\frac{\exp(X[i, j])}{\sum_j(\exp(X[i, j])})
        \end{aligned}

    Parameters:
        x (Tensor): The input Tensor with data type float32, float64.
        axis (int, optional): The axis along which to perform log_softmax
            calculations. It should be in range [-D, D), where D is the
            dimensions of ``x`` . If ``axis`` < 0, it works the same way as
            :math:`axis + D` . Default is -1.
        dtype (str|np.dtype|core.VarDesc.VarType, optional): The desired data
            type of the output tensor. If dtype is specified, ``x`` is casted
            to ``dtype`` before the operation is performed. This is useful for
            preventing data type overflows. Supported dtype: float32, float64.
            If ``dtype`` is None, the output Tensor has the same dtype as x.
            Default is None.
        name (str, optional): For details, please refer to :ref:`api_guide_Name`. Generally, no setting is required. Default: None.

    Returns:
        A Tensor with the same shape and data type (use ``dtype`` if it is
        specified) as x.

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> import paddle.nn.functional as F
            >>> x = [[[-2.0, 3.0, -4.0, 5.0],
            ...       [3.0, -4.0, 5.0, -6.0],
            ...       [-7.0, -8.0, 8.0, 9.0]],
            ...      [[1.0, -2.0, -3.0, 4.0],
            ...       [-5.0, 6.0, 7.0, -8.0],
            ...       [6.0, 7.0, 8.0, 9.0]]]
            >>> x = paddle.to_tensor(x)
            >>> out1 = F.log_softmax(x)
            >>> out2 = F.log_softmax(x, dtype='float64')
            >>> #out1's data type is float32; out2's data type is float64
            >>> #out1 and out2's value is as follows:
            >>> print(out1)
            Tensor(shape=[2, 3, 4], dtype=float32, place=Place(cpu), stop_gradient=True,
            [[[-7.12783957 , -2.12783957 , -9.12783909 , -0.12783945 ],
              [-2.12705135 , -9.12705135 , -0.12705141 , -11.12705135],
              [-16.31326103, -17.31326103, -1.31326187 , -0.31326184 ]],
             [[-3.05181193 , -6.05181217 , -7.05181217 , -0.05181199 ],
              [-12.31326675, -1.31326652 , -0.31326646 , -15.31326675],
              [-3.44018984 , -2.44018984 , -1.44018972 , -0.44018975 ]]])
            >>> print(out2)
            Tensor(shape=[2, 3, 4], dtype=float64, place=Place(cpu), stop_gradient=True,
            [[[-7.12783948 , -2.12783948 , -9.12783948 , -0.12783948 ],
              [-2.12705141 , -9.12705141 , -0.12705141 , -11.12705141],
              [-16.31326180, -17.31326180, -1.31326180 , -0.31326180 ]],
             [[-3.05181198 , -6.05181198 , -7.05181198 , -0.05181198 ],
              [-12.31326640, -1.31326640 , -0.31326640 , -15.31326640],
              [-3.44018970 , -2.44018970 , -1.44018970 , -0.44018970 ]]])
    """

    if (dtype is not None) and (not isinstance(dtype, core.VarDesc.VarType)):
        dtype = convert_np_dtype_to_dtype_(dtype)

    if in_dynamic_mode():
        if dtype is not None:
            x = _C_ops.cast(x, dtype)
        return _C_ops.log_softmax(x, axis)
    else:
        if dtype is None:
            check_variable_and_dtype(
                x,
                'x',
                ['float16', 'uint16', 'float32', 'float64'],
                'log_softmax',
            )
        else:
            check_dtype(
                dtype,
                'dtype',
                ['float32', 'float64'],
                'log_softmax',
                'If dtype is not None, it only support float32 or float64.',
            )

        helper = LayerHelper("log_softmax", **locals())
        out_cast = x
        if dtype is not None:
            out_cast = helper.create_variable_for_type_inference(dtype)
            helper.append_op(
                type='cast',
                inputs={'X': x},
                outputs={'Out': out_cast},
                attrs={'in_dtype': x.dtype, 'out_dtype': dtype},
            )

        out = helper.create_variable_for_type_inference(out_cast.dtype)
        helper.append_op(
            type='log_softmax',
            inputs={'X': out_cast},
            outputs={'Out': out},
            attrs={'axis': axis},
        )

        return out


def glu(x, axis=-1, name=None):
    r"""
    The gated linear unit. The input is evenly splited into 2 parts along a
    given axis. The first part is used as the content, and the second part is
    passed through a sigmoid function then used as the gate. The output is a
    elementwise multiplication of the content and the gate.

    .. math::

        \mathrm{GLU}(a, b) = a \otimes \sigma(b)

    Parameters:
        x (Tensor): The input Tensor with data type float32, float64.
        axis (int, optional): The axis along which split the input tensor. It
            should be in range [-D, D), where D is the dimensions of ``x`` .
            If ``axis`` < 0, it works the same way as :math:`axis + D` .
            Default is -1.
        name (str, optional): For details, please refer to :ref:`api_guide_Name`. Generally, no setting is required. Default: None.

    Returns:
        A Tensor with the same data type as x. The size of the given aixs is
        halved.

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> from paddle.nn import functional as F
            >>> x = paddle.to_tensor(
            ...     [[-0.22014759, -1.76358426,  0.80566144,  0.04241343],
            ...         [-1.94900405, -1.89956081,  0.17134808, -1.11280477]]
            ... )
            >>> print(F.glu(x))
            Tensor(shape=[2, 2], dtype=float32, place=Place(cpu), stop_gradient=True,
            [[-0.15216254, -0.90048921],
            [-1.05778778, -0.46985325]])

    """
    check_variable_and_dtype(
        x, 'input', ['float16', 'float32', 'float64'], "glu"
    )
    rank = len(x.shape)
    if not (-rank <= axis < rank):
        raise ValueError(
            "Expected value range of `axis` is [{}, {}), but received axis: {}".format(
                -rank, rank, axis
            )
        )
    a, b = chunk(x, 2, axis=axis, name=name)
    gate = sigmoid(b, name=name)
    out = paddle.multiply(a, gate, name=name)
    return out


def gumbel_softmax(x, temperature=1.0, hard=False, axis=-1, name=None):
    r"""
    Samples from the Gumbel-Softmax distribution and optionally discretizes.
    temperature is denoted by t. The calculation process is as follows:

    First, generate gumbel noise:

    .. math::

        G_i = -log(-log(U_i)), U_i \sim U(0,1)

    Second, add noise to ``x``:

    .. math::

        v = [x_1 + G_1,...,x_n + G_n]

    Finally, calculate gumbel_softmax and generate samples:

    .. math::
        gumbel\_softmax(v_i)=\frac{e^{v_i/t}}{\sum_{j=1}^n{e^{v_j/t}}},i=1,2,3...n

    Parameters:
        x (Tensor): An N-D Tensor, the first N - 1 dimensions index into a batch
            of independent distributions and the last dimension represents
            a vector of probabilities with datatype float16, float32, float64.
        temperature (float, optional): non-negative scalar temperature.
            Default is 1.0.
        hard (bool, optional): if True, the returned samples will be discretized as
            one-hot vectors, but will be differentiated as if it is the soft sample
            in autograd. Default is False.
        axis (int, optional): The axis along will be calculated softmax value.
            Default is -1.
        name (str, optional): For details, please refer to :ref:`api_guide_Name`. Generally, no setting is required. Default: None.

    Returns:
        Sampled tensor of same shape as ``x`` from the Gumbel-Softmax distribution.
        If ``hard = True``, the returned samples will be one-hot, otherwise they will be
        probability distributions that sum to 1 across ``axis``.

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> import paddle.nn.functional as F

            >>> paddle.seed(2023)
            >>> logits = paddle.randn([4, 6])
            >>> temperature = 0.01
            >>> gumbel_softmax = F.gumbel_softmax(logits, temperature)
            >>> print(gumbel_softmax)
            Tensor(shape=[4, 6], dtype=float32, place=Place(cpu), stop_gradient=True,
            [[0.00000000, 1.        , 0.00000000, 0.00000000, 0.00000000, 0.00000000],
             [0.00000000, 0.00000000, 1.        , 0.00000000, 0.00000000, 0.00000000],
             [0.00000000, 0.00000004, 0.00000000, 0.00000000, 1.        , 0.00000000],
             [0.00000000, 1.        , 0.00000000, 0.00000000, 0.00000000, 0.00000000]])

    """
    if in_dynamic_mode():
        return _C_ops.gumbel_softmax(x, temperature, hard, axis)

    helper = LayerHelper("gumbel_softmax", **locals())
    check_variable_and_dtype(
        x, 'x', ['float16', 'float32', 'float64'], 'gumbel_softmax'
    )
    out = helper.create_variable_for_type_inference(x.dtype)
    helper.append_op(
        type='gumbel_softmax',
        inputs={'X': x},
        outputs={'Out': out},
        attrs={'temperature': temperature, 'hard': hard, 'axis': axis},
    )
    return out
