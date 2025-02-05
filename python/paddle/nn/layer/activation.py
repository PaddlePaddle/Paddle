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


from __future__ import annotations

from typing import TYPE_CHECKING

from paddle.framework import get_default_dtype

from .. import functional as F
from ..initializer import Constant
from .layers import Layer

if TYPE_CHECKING:
    from paddle import Tensor
    from paddle._typing import DataLayoutND, ParamAttrLike
__all__ = []


class CELU(Layer):
    r"""
    CELU Activation.

    .. math::

        CELU(x) = max(0, x) + min(0, \alpha * (e^{x/\alpha}-1))

    Parameters:
        alpha (float, optional): The 'alpha' value of the CELU formulation. Default is 1.0.
        name (str|None, optional): Name for the operation (optional, default is None).
            For more information, please refer to :ref:`api_guide_Name`.

    Shape:
        - input: Tensor with any shape.
        - output: Tensor with the same shape as input.

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> x = paddle.to_tensor([[-1. ,6.], [1., 15.6]])
            >>> m = paddle.nn.CELU(0.2)
            >>> out = m(x)
            >>> print(out)
            Tensor(shape=[2, 2], dtype=float32, place=Place(cpu), stop_gradient=True,
            [[-0.19865242,  6.        ],
             [ 1.        , 15.60000038]])
    """

    def __init__(self, alpha: float = 1.0, name: str | None = None) -> None:
        super().__init__()
        self._alpha = alpha
        self._name = name

    def forward(self, x: Tensor) -> Tensor:
        return F.celu(x, self._alpha, self._name)

    def extra_repr(self) -> str:
        name_str = f', name={self._name}' if self._name else ''
        return f'alpha={self._alpha}{name_str}'


class ELU(Layer):
    r"""
    ELU Activation.

    .. math::

        ELU(x)=
            \left\{
                \begin{array}{lcl}
                x,& &\text{if } \ x > 0 \\
                alpha * (e^{x} - 1),& &\text{if } \ x <= 0
                \end{array}
            \right.

    Parameters:
        alpha (float, optional): The 'alpha' value of the ELU formulation. Default is 1.0.
        name (str|None, optional): Name for the operation (optional, default is None).
            For more information, please refer to :ref:`api_guide_Name`.

    Shape:
        - input: Tensor with any shape.
        - output: Tensor with the same shape as input.

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> x = paddle.to_tensor([[-1. ,6.], [1., 15.6]])
            >>> m = paddle.nn.ELU(0.2)
            >>> out = m(x)
            >>> print(out)
            Tensor(shape=[2, 2], dtype=float32, place=Place(cpu), stop_gradient=True,
            [[-0.12642412,  6.        ],
             [ 1.        , 15.60000038]])
    """

    def __init__(self, alpha: float = 1.0, name: str | None = None) -> None:
        super().__init__()
        self._alpha = alpha
        self._name = name

    def forward(self, x: Tensor) -> Tensor:
        return F.elu(x, self._alpha, self._name)

    def extra_repr(self) -> str:
        name_str = f', name={self._name}' if self._name else ''
        return f'alpha={self._alpha}{name_str}'


class GLU(Layer):
    r"""
    GLU Activation.

    .. math::

        GLU(a, b) = a \otimes \sigma(b) where :math:`a` is the first half of the input matrices and :math:`b` is the second half.

    Parameters:
        axis (int, optional): The axis along which split the input tensor. It
            should be in range [-D, D), where D is the dimensions of ``x`` .
            If ``axis`` < 0, it works the same way as :math:`axis + D` .
            Default is -1.
        name (str|None, optional): Name for the operation (optional, default is None).
            For more information, please refer to :ref:`api_guide_Name`.

    Shape:
        - input: Tensor which the size of the given axis is even.
        - output: Tensor which the size of the given axis is halved.

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> x = paddle.to_tensor(
            ...     [[-0.22014759, -1.76358426,  0.80566144,  0.04241343],
            ...         [-1.94900405, -1.89956081,  0.17134808, -1.11280477]]
            ... )
            >>> m = paddle.nn.GLU()
            >>> out = m(x)
            >>> print(out)
            Tensor(shape=[2, 2], dtype=float32, place=Place(cpu), stop_gradient=True,
            [[-0.15216254, -0.90048921],
            [-1.05778778, -0.46985325]])
    """

    def __init__(self, axis: int = -1, name: str | None = None) -> None:
        super().__init__()
        self._axis = axis
        self._name = name

    def forward(self, x: Tensor) -> Tensor:
        return F.glu(x, self._axis, self._name)

    def extra_repr(self) -> str:
        name_str = f', name={self._name}' if self._name else ''
        return f'axis={self._axis}{name_str}'


class GELU(Layer):
    r"""
    GELU Activation.

    If approximate is True

    .. math::

        GELU(x) = 0.5 * x * (1 + tanh(\sqrt{\frac{2}{\pi}} * (x + 0.044715x^{3})))

    else

    .. math::

        GELU(x) = 0.5 * x * (1 + erf(\frac{x}{\sqrt{2}}))

    Parameters:
        approximate (bool, optional): Whether to enable approximation. Default is False.
        name (str|None, optional): Name for the operation (optional, default is None).
            For more information, please refer to :ref:`api_guide_Name`.

    Shape:
        - input: Tensor with any shape.
        - output: Tensor with the same shape as input.

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> x = paddle.to_tensor([[-1, 0.5],[1, 1.5]])
            >>> m = paddle.nn.GELU()
            >>> out = m(x)
            >>> print(out)
            Tensor(shape=[2, 2], dtype=float32, place=Place(cpu), stop_gradient=True,
            [[-0.15865529,  0.34573123],
             [ 0.84134471,  1.39978933]])
            >>> m = paddle.nn.GELU(True)
            >>> out = m(x)
            >>> print(out)
            Tensor(shape=[2, 2], dtype=float32, place=Place(cpu), stop_gradient=True,
            [[-0.15880796,  0.34571400],
             [ 0.84119201,  1.39957154]])
    """

    def __init__(
        self, approximate: bool = False, name: str | None = None
    ) -> None:
        super().__init__()
        self._approximate = approximate
        self._name = name

    def forward(self, x: Tensor) -> Tensor:
        return F.gelu(x, self._approximate, self._name)

    def extra_repr(self) -> str:
        name_str = f', name={self._name}' if self._name else ''
        return f'approximate={self._approximate}{name_str}'


class Hardshrink(Layer):
    r"""
    Hardshrink Activation

    .. math::

        hardshrink(x)=
            \left\{
                \begin{array}{rcl}
                    x, & & if \ x > threshold \\
                    x, & & if \ x < -threshold \\
                    0, & & if \ others
            \end{array}
            \right.

    Parameters:
        threshold (float, optional): The value of threshold for hardthrink. Default is 0.5
        name (str|None, optional): Name for the operation (optional, default is None).
            For more information, please refer to :ref:`api_guide_Name`.

    Shape:
        - input: Tensor with any shape.
        - output: Tensor with the same shape as input.

    Examples:

        .. code-block:: python

            >>> import paddle

            >>> x = paddle.to_tensor([-1, 0.3, 2.5])
            >>> m = paddle.nn.Hardshrink()
            >>> out = m(x)
            >>> print(out)
            Tensor(shape=[3], dtype=float32, place=Place(cpu), stop_gradient=True,
            [-1.       ,  0.       , 2.50000000])
    """

    def __init__(self, threshold: float = 0.5, name: str | None = None) -> None:
        super().__init__()
        self._threshold = threshold
        self._name = name

    def forward(self, x: Tensor) -> Tensor:
        return F.hardshrink(x, self._threshold, self._name)

    def extra_repr(self) -> str:
        name_str = f', name={self._name}' if self._name else ''
        return f'threshold={self._threshold}{name_str}'


class Hardswish(Layer):
    r"""
    Hardswish activation. Create a callable object of `Hardswish`. Hardswish
    is proposed in MobileNetV3, and performs better in computational stability
    and efficiency compared to swish function. For more details please refer
    to: https://arxiv.org/pdf/1905.02244.pdf

    .. math::

        Hardswish(x)=
            \left\{
                \begin{array}{cll}
                0 &, & \text{if } x \leq -3 \\
                x &, & \text{if } x \geq 3 \\
                \frac{x(x+3)}{6} &, & \text{otherwise}
                \end{array}
            \right.


    Parameters:
        name (str|None, optional): Name for the operation (optional, default is None).
            For more information, please refer to :ref:`api_guide_Name`.

    Shape:
        - input: Tensor with any shape.
        - output: Tensor with the same shape as input.

    Examples:

        .. code-block:: python

            >>> import paddle

            >>> x = paddle.to_tensor([-4., 5., 1.])
            >>> m = paddle.nn.Hardswish()
            >>> out = m(x)
            >>> print(out)
            Tensor(shape=[3], dtype=float32, place=Place(cpu), stop_gradient=True,
            [-0.       , 5.        , 0.66666669])
    """

    def __init__(self, name: str | None = None) -> None:
        super().__init__()
        self._name = name

    def forward(self, x: Tensor) -> Tensor:
        return F.hardswish(x, self._name)

    def extra_repr(self) -> str:
        name_str = f'name={self._name}' if self._name else ''
        return name_str


class Tanh(Layer):
    r"""
    Tanh Activation.

    .. math::
        Tanh(x) = \frac{e^{x} - e^{-x}}{e^{x} + e^{-x}}

    Parameters:
        name (str|None, optional): Name for the operation (optional, default is None).
            For more information, please refer to :ref:`api_guide_Name`.

    Shape:
        - input: Tensor with any shape.
        - output: Tensor with the same shape as input.

    Examples:

        .. code-block:: python

            >>> import paddle

            >>> x = paddle.to_tensor([-0.4, -0.2, 0.1, 0.3])
            >>> m = paddle.nn.Tanh()
            >>> out = m(x)
            >>> print(out)
            Tensor(shape=[4], dtype=float32, place=Place(cpu), stop_gradient=True,
            [-0.37994900, -0.19737528,  0.09966799,  0.29131261])
    """

    def __init__(self, name: str | None = None) -> None:
        super().__init__()
        self._name = name

    def forward(self, x: Tensor) -> Tensor:
        return F.tanh(x, self._name)

    def extra_repr(self) -> str:
        name_str = f'name={self._name}' if self._name else ''
        return name_str


class Hardtanh(Layer):
    r"""
    Hardtanh Activation. Create a callable object of `Hardtanh`.

    .. math::

        Hardtanh(x)=
            \left\{
                \begin{array}{cll}
                    max,& & \text{if } x > max \\
                    min,& & \text{if } x < min \\
                    x,& & \text{otherwise}
                \end{array}
            \right.


    Parameters:
        min (float, optional): The value of min for Hardtanh. Default is -1.
        max (float, optional): The value of max for Hardtanh. Default is 1.
        name (str|None, optional): Name for the operation (optional, default is None).
            For more information, please refer to :ref:`api_guide_Name`.

    Shape:
        - input: Tensor with any shape.
        - output: Tensor with the same shape as input.

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> x = paddle.to_tensor([-1.5, 0.3, 2.5])
            >>> m = paddle.nn.Hardtanh()
            >>> out = m(x)
            >>> print(out)
            Tensor(shape=[3], dtype=float32, place=Place(cpu), stop_gradient=True,
            [-1.       , 0.30000001,  1.       ])
    """

    def __init__(
        self, min: float = -1.0, max: float = 1.0, name: str | None = None
    ) -> None:
        super().__init__()
        self._min = min
        self._max = max
        self._name = name

    def forward(self, x: Tensor) -> Tensor:
        return F.hardtanh(x, self._min, self._max, self._name)

    def extra_repr(self) -> str:
        name_str = f', name={self._name}' if self._name else ''
        return f'min={self._min}, max={self._max}{name_str}'


class PReLU(Layer):
    """
    PReLU Activation. The calculation formula is follows:

    If approximate calculation is used:

    .. math::

        PReLU(x) = max(0, x) + weight * min(0, x)

    x is input Tensor.

    Parameters:
        num_parameters (int, optional): Number of `weight` to learn. The supported values are:
            1 - a single parameter `alpha` is used for all input channels;
            Number of channels - a separate `alpha` is used for each input channel.
            Default is 1.
        init (float, optional): Init value of learnable `weight`. Default is 0.25.
        weight_attr(ParamAttr, optional): The parameter attribute for the learnable `weight`.
            Default is None. For more information, please refer to :ref:`api_paddle_ParamAttr`.
        data_format(str, optional): Data format that specifies the layout of input.
            It may be "NC", "NCL", "NCHW", "NCDHW", "NLC", "NHWC" or "NDHWC". Default: "NCHW".
        name (str|None, optional): Name for the operation (optional, default is None).
            For more information, please refer to :ref:`api_guide_Name`.

    Shape:
        - input: Tensor with any shape. Default dtype is float32.
        - output: Tensor with the same shape as input.

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> data = paddle.to_tensor([[[[-2.0,  3.0, -4.0,  5.0],
            ...                            [ 3.0, -4.0,  5.0, -6.0],
            ...                            [-7.0, -8.0,  8.0,  9.0]],
            ...                           [[ 1.0, -2.0, -3.0,  4.0],
            ...                            [-5.0,  6.0,  7.0, -8.0],
            ...                            [ 6.0,  7.0,  8.0,  9.0]]]])
            ...
            >>> m = paddle.nn.PReLU(1, 0.25)
            >>> out = m(data)
            >>> print(out)
            Tensor(shape=[1, 2, 3, 4], dtype=float32, place=Place(cpu), stop_gradient=False,
            [[[[-0.50000000,  3.        , -1.        ,  5.        ],
               [ 3.        , -1.        ,  5.        , -1.50000000],
               [-1.75000000, -2.        ,  8.        ,  9.        ]],
              [[ 1.        , -0.50000000, -0.75000000,  4.        ],
               [-1.25000000,  6.        ,  7.        , -2.        ],
               [ 6.        ,  7.        ,  8.        ,  9.        ]]]])
    """

    def __init__(
        self,
        num_parameters: int = 1,
        init: float = 0.25,
        weight_attr: ParamAttrLike | None = None,
        data_format: DataLayoutND = "NCHW",
        name: str | None = None,
    ) -> None:
        super().__init__()
        self._num_parameters = num_parameters
        self._init = init
        self._weight_attr = weight_attr
        self._name = name
        self._data_format = data_format

        self._weight = self.create_parameter(
            attr=self._weight_attr,
            shape=[self._num_parameters],
            dtype=get_default_dtype(),
            is_bias=False,
            default_initializer=Constant(self._init),
        )

    def forward(self, x: Tensor) -> Tensor:
        return F.prelu(x, self._weight, data_format=self._data_format)

    def extra_repr(self) -> str:
        name_str = f', name={self._name}' if self._name else ''
        return f'num_parameters={self._num_parameters}, data_format={self._data_format}, init={self._init}, dtype={self._dtype}{name_str}'


class RReLU(Layer):
    r"""
    RReLU activation layer.

    Applies the randomized leaky rectified liner unit function to improve generalization performance,
    as described in the paper:
    `Empirical Evaluation of Rectified Activations in Convolutional Network <https://arxiv.org/abs/1505.00853>`_

    During training, randomly samples the negative slope for activation values as described below:

    .. math::

        RReLU(x)=
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

        RReLU(x)=
            \left\{
                \begin{array}{rcl}
                    x, & & if \ x >= 0 \\
                    (lower + upper) * 0.5 * x, & & otherwise \\
                \end{array}
            \right.

    where :math:`x` is the input tensor,
    :math:`lower` and :math:`upper` are the bounds of uniform distribution.

    Parameters:
        lower (float, optional): The lower bound of uniform distribution. Default: 1.0/8.0.
        upper (float, optional): The upper bound of uniform distribution. Default: 1.0/3.0.
        name (str|None, optional): Name for the operation (optional, default is None).
            For more information, please refer to :ref:`api_guide_Name`.

    Shape:
        - input: Tensor with any shape. Default dtype is float32.
        - output: Tensor with the same shape as input.

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> paddle.seed(2023)

            >>> input_tensor = paddle.to_tensor([[[[-2.0,  3.0, -4.0,  5.0],
            ...                                    [ 3.0, -4.0,  5.0, -6.0],
            ...                                    [-7.0, -8.0,  8.0,  9.0]],
            ...                                   [[ 1.0, -2.0, -3.0,  4.0],
            ...                                    [-5.0,  6.0,  7.0, -8.0],
            ...                                    [ 6.0,  7.0,  8.0,  9.0]]]], dtype='float32')
            ...
            >>> rrelu_layer = paddle.nn.RReLU(0.1, 0.3)
            >>> out = rrelu_layer(input_tensor)
            >>> print(out)
            Tensor(shape=[1, 2, 3, 4], dtype=float32, place=Place(cpu), stop_gradient=True,
            [[[[-0.54633451,  3.        , -0.81611776,  5.        ],
               [ 3.        , -0.60768753,  5.        , -1.68630385],
               [-1.29360127, -1.45026064,  8.        ,  9.        ]],
              [[ 1.        , -0.58808362, -0.74662417,  4.        ],
               [-1.01785135,  6.        ,  7.        , -1.97268605],
               [ 6.        ,  7.        ,  8.        ,  9.        ]]]])
    """

    def __init__(
        self,
        lower: float = 1.0 / 8.0,
        upper: float = 1.0 / 3.0,
        name: str | None = None,
    ) -> None:
        super().__init__()
        self._lower = lower
        self._upper = upper
        self._name = name

    def forward(self, x: Tensor) -> Tensor:
        return F.rrelu(
            x, lower=self._lower, upper=self._upper, training=self.training
        )

    def extra_repr(self) -> str:
        name_str = f', name={self._name}' if self._name else ''
        return f'lower={self._lower}, upper={self._upper}, training={self.training}, dtype={self._dtype}{name_str}'


class ReLU(Layer):
    """
    ReLU Activation. The calculation formula is follows:

    .. math::

        ReLU(x) = max(x, 0)

    x is input Tensor.

    Parameters:
        name (str|None, optional): Name for the operation (optional, default is None).
            For more information, please refer to :ref:`api_guide_Name`.

    Shape:
        - input: Tensor with any shape.
        - output: Tensor with the same shape as input.

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> x = paddle.to_tensor([-2., 0., 1.])
            >>> m = paddle.nn.ReLU()
            >>> out = m(x)
            >>> print(out)
            Tensor(shape=[3], dtype=float32, place=Place(cpu), stop_gradient=True,
            [0., 0., 1.])
    """

    def __init__(self, name: str | None = None) -> None:
        super().__init__()
        self._name = name

    def forward(self, x: Tensor) -> Tensor:
        return F.relu(x, self._name)

    def extra_repr(self) -> str:
        name_str = f'name={self._name}' if self._name else ''
        return name_str


class ReLU6(Layer):
    """
    ReLU6 Activation

    .. math::

        ReLU6(x) = min(max(0,x), 6)

    x is input Tensor.

    Parameters:
        name (str|None, optional): Name for the operation (optional, default is None).
            For more information, please refer to :ref:`api_guide_Name`.

    Shape:
        - input: Tensor with any shape.
        - output: Tensor with the same shape as input.

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> x = paddle.to_tensor([-1., 0.3, 6.5])
            >>> m = paddle.nn.ReLU6()
            >>> out = m(x)
            >>> print(out)
            Tensor(shape=[3], dtype=float32, place=Place(cpu), stop_gradient=True,
            [0.        , 0.30000000, 6.        ])
    """

    def __init__(self, name: str | None = None) -> None:
        super().__init__()
        self._name = name

    def forward(self, x: Tensor) -> Tensor:
        return F.relu6(x, self._name)

    def extra_repr(self) -> str:
        name_str = f'name={self._name}' if self._name else ''
        return name_str


class SELU(Layer):
    r"""
    SELU Activation

    .. math::

        SELU(x)= scale *
            \left\{
                \begin{array}{lcl}
                x,& &\text{if } \ x > 0 \\
                alpha * e^{x} - alpha,& &\text{if } \ x <= 0
                \end{array}
            \right.

    Parameters:
        scale (float, optional): The value of scale(must be greater than 1.0) for SELU. Default is 1.0507009873554804934193349852946.
        alpha (float, optional): The value of alpha(must be no less than zero) for SELU. Default is 1.6732632423543772848170429916717.
        name (str|None, optional): Name for the operation (optional, default is None).
            For more information, please refer to :ref:`api_guide_Name`.

    Shape:
        - input: Tensor with any shape.
        - output: Tensor with the same shape as input.

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> x = paddle.to_tensor([[0.0, 1.0],[2.0, 3.0]])
            >>> m = paddle.nn.SELU()
            >>> out = m(x)
            >>> print(out)
            Tensor(shape=[2, 2], dtype=float32, place=Place(cpu), stop_gradient=True,
            [[0.        , 1.05070102],
             [2.10140204, 3.15210295]])
    """

    def __init__(
        self,
        scale: float = 1.0507009873554804934193349852946,
        alpha: float = 1.6732632423543772848170429916717,
        name: str | None = None,
    ) -> None:
        super().__init__()
        self._scale = scale
        self._alpha = alpha
        self._name = name

    def forward(self, x: Tensor) -> Tensor:
        return F.selu(x, self._scale, self._alpha, self._name)

    def extra_repr(self) -> str:
        name_str = f', name={self._name}' if self._name else ''
        return f'scale={self._scale:.16f}, alpha={self._alpha:.16f}{name_str}'


class LeakyReLU(Layer):
    r"""
    Leaky ReLU Activation. Create a callable object of `LeakyReLU` to calculate
    the `LeakyReLU` of input `x`.

    .. math::

        LeakyReLU(x)=
            \left\{
                \begin{array}{rcl}
                    x, & & if \ x >= 0 \\
                    negative\_slope * x, & & otherwise \\
                \end{array}
            \right.


    Parameters:
        negative_slope (float, optional): Slope of the activation function at
            :math:`x < 0` . Default is 0.01.
        name (str|None, optional): Name for the operation (optional, default is None).
            For more information, please refer to :ref:`api_guide_Name`.

    Shape:
        - input: Tensor with any shape.
        - output: Tensor with the same shape as input.

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> m = paddle.nn.LeakyReLU()
            >>> x = paddle.to_tensor([-2.0, 0, 1])
            >>> out = m(x)
            >>> print(out)
            Tensor(shape=[3], dtype=float32, place=Place(cpu), stop_gradient=True,
            [-0.02000000,  0.        ,  1.        ])
    """

    def __init__(
        self, negative_slope: float = 0.01, name: str | None = None
    ) -> None:
        super().__init__()
        self._negative_slope = negative_slope
        self._name = name

    def forward(self, x: Tensor) -> Tensor:
        return F.leaky_relu(x, self._negative_slope, self._name)

    def extra_repr(self) -> str:
        name_str = f', name={self._name}' if self._name else ''
        return f'negative_slope={self._negative_slope}{name_str}'


class Sigmoid(Layer):
    r"""
    this interface is used to construct a callable object of the ``Sigmoid`` class. This layer calculate the `sigmoid` of input x.

    .. math::

        sigmoid(x) = \frac{1}{1 + e^{-x}}

    Parameters:
        name (str|None, optional): For details, please refer to :ref:`api_guide_Name`. Generally, no setting is required. Default: None.

    Shape:
        x: N-D tensor, available dtype is float16, float32, float64.

    Returns:
        A callable object of Sigmoid.

    Examples:

        .. code-block:: python

            >>> import paddle

            >>> m = paddle.nn.Sigmoid()
            >>> x = paddle.to_tensor([1.0, 2.0, 3.0, 4.0])
            >>> out = m(x)
            >>> print(out)
            Tensor(shape=[4], dtype=float32, place=Place(cpu), stop_gradient=True,
            [0.73105860, 0.88079703, 0.95257413, 0.98201376])
    """

    def __init__(self, name: str | None = None) -> None:
        super().__init__()
        self.name = name

    def forward(self, x: Tensor) -> Tensor:
        return F.sigmoid(x, self.name)

    def extra_repr(self) -> str:
        name_str = f'name={self.name}' if self.name else ''
        return name_str


class Hardsigmoid(Layer):
    r"""
    ``Hardsigmoid`` Activation Layers, Construct a callable object of
    the ``Hardsigmoid`` class. This layer calculate the `hardsigmoid` of input x.

    A 3-part piecewise linear approximation of sigmoid(https://arxiv.org/abs/1603.00391),
    which is much faster than sigmoid.

    .. math::

        Hardsigmoid(x)=
            \left\{
                \begin{array}{rcl}
            0, & & \text{if } \ x \leq -3 \\
            1, & & \text{if } \ x \geq 3 \\
            x/6 + 1/2, & & \text{otherwise}
                \end{array}
            \right.

    Parameters:
        name (str|None, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Shape:
        x: N-D tensor, available dtype is float32, float64.

    Returns:
        A callable object of Hardsigmoid.

    Examples:

        .. code-block:: python

            >>> import paddle

            >>> m = paddle.nn.Hardsigmoid()
            >>> x = paddle.to_tensor([-4., 5., 1.])
            >>> out = m(x)
            >>> print(out)
            Tensor(shape=[3], dtype=float32, place=Place(cpu), stop_gradient=True,
            [0.        , 1.        , 0.66666669])
    """

    def __init__(self, name: str | None = None) -> None:
        super().__init__()
        self.name = name

    def forward(self, x: Tensor) -> Tensor:
        return F.hardsigmoid(x, name=self.name)

    def extra_repr(self) -> str:
        name_str = f'name={self.name}' if self.name else ''
        return name_str


class Softplus(Layer):
    r"""
    Softplus Activation

    .. math::
        softplus(x)=\begin{cases}
                \frac{1}{\beta} * \log(1 + e^{\beta * x}),&x\leqslant\frac{\varepsilon}{\beta};\\
                x,&x>\frac{\varepsilon}{\beta}.
            \end{cases}

    Parameters:
        beta (float, optional): The value of :math:`\beta` for Softplus. Default is 1
        threshold (float, optional): The value of :math:`\varepsilon` for Softplus. Default is 20
        name (str|None, optional): For details, please refer to :ref:`api_guide_Name`. Generally, no setting is required. Default: None.

    Shape:
        - input: Tensor with any shape.
        - output: Tensor with the same shape as input.

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> x = paddle.to_tensor([-0.4, -0.2, 0.1, 0.3], dtype='float32')
            >>> m = paddle.nn.Softplus()
            >>> out = m(x)
            >>> print(out)
            Tensor(shape=[4], dtype=float32, place=Place(cpu), stop_gradient=True,
            [0.51301527, 0.59813893, 0.74439669, 0.85435522])
    """

    def __init__(
        self, beta: float = 1, threshold: float = 20, name: str | None = None
    ) -> None:
        super().__init__()
        self._beta = beta
        self._threshold = threshold
        self._name = name

    def forward(self, x: Tensor) -> Tensor:
        return F.softplus(x, self._beta, self._threshold, self._name)

    def extra_repr(self) -> str:
        name_str = f', name={self._name}' if self._name else ''
        return f'beta={self._beta}, threshold={self._threshold}{name_str}'


class Softshrink(Layer):
    r"""
    Softshrink Activation

    .. math::

        Softshrink(x)=
            \left\{
                \begin{array}{rcl}
                x - threshold,& & \text{if } x > threshold \\
                x + threshold,& & \text{if } x < -threshold \\
                0,& &  \text{otherwise}
            \end{array}
            \right.


    Parameters:
        threshold (float, optional): The value of threshold(must be no less than zero) for softplus. Default is 0.5
        name (str|None, optional): Name for the operation (optional, default is None).
            For more information, please refer to :ref:`api_guide_Name`.

    Shape:
        - input: Tensor with any shape.
        - output: Tensor with the same shape as input.

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> x = paddle.to_tensor([-0.9, -0.2, 0.1, 0.8])
            >>> m = paddle.nn.Softshrink()
            >>> out = m(x)
            >>> print(out)
            Tensor(shape=[4], dtype=float32, place=Place(cpu), stop_gradient=True,
            [-0.39999998,  0.        ,  0.        ,  0.30000001])
    """

    def __init__(self, threshold: float = 0.5, name: str | None = None) -> None:
        super().__init__()
        self._threshold = threshold
        self._name = name

    def forward(self, x: Tensor) -> Tensor:
        return F.softshrink(x, self._threshold, self._name)

    def extra_repr(self) -> str:
        name_str = f', name={self._name}' if self._name else ''
        return f'threshold={self._threshold}{name_str}'


class Softsign(Layer):
    r"""
    Softsign Activation

    .. math::

        Softsign(x) = \frac{x}{1 + |x|}

    Parameters:
        name (str|None, optional): Name for the operation (optional, default is None).
            For more information, please refer to :ref:`api_guide_Name`.

    Shape:
        - input: Tensor with any shape.
        - output: Tensor with the same shape as input.

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> x = paddle.to_tensor([-0.4, -0.2, 0.1, 0.3])
            >>> m = paddle.nn.Softsign()
            >>> out = m(x)
            >>> print(out)
            Tensor(shape=[4], dtype=float32, place=Place(cpu), stop_gradient=True,
            [-0.28571430, -0.16666666,  0.09090909,  0.23076925])
    """

    def __init__(self, name: str | None = None) -> None:
        super().__init__()
        self._name = name

    def forward(self, x: Tensor) -> Tensor:
        return F.softsign(x, self._name)

    def extra_repr(self) -> str:
        name_str = f'name={self._name}' if self._name else ''
        return name_str


class Swish(Layer):
    r"""
    Swish Activation.

    .. math::

        Swish(x) = \frac{x}{1 + e^{-x}}

    Parameters:
        name (str|None, optional): Name for the operation (optional, default is None).
            For more information, please refer to :ref:`api_guide_Name`.

    Shape:
        - input: Tensor with any shape.
        - output: Tensor with the same shape as input.

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> x = paddle.to_tensor([-2., 0., 1.])
            >>> m = paddle.nn.Swish()
            >>> out = m(x)
            >>> print(out)
            Tensor(shape=[3], dtype=float32, place=Place(cpu), stop_gradient=True,
            [-0.23840584,  0.        ,  0.73105860])
    """

    def __init__(self, name: str | None = None) -> None:
        super().__init__()
        self._name = name

    def forward(self, x: Tensor) -> Tensor:
        return F.swish(x, self._name)

    def extra_repr(self) -> str:
        name_str = f'name={self._name}' if self._name else ''
        return name_str


class Mish(Layer):
    r"""
    Mish Activation.

    ..  math::

        softplus(x) = \begin{cases}
                x, \text{if } x > \text{threshold} \\
                \ln(1 + e^{x}),  \text{otherwise}
            \end{cases}

        Mish(x) = x * \tanh(softplus(x))

    Parameters:
        name (str|None, optional): Name for the operation (optional, default is None).
            For more information, please refer to :ref:`api_guide_Name`.

    Shape:
        - input: Tensor with any shape.
        - output: Tensor with the same shape as input.

    Examples:

        .. code-block:: python

            >>> import paddle

            >>> x = paddle.to_tensor([-5., 0., 5.])
            >>> m = paddle.nn.Mish()
            >>> out = m(x)
            >>> print(out)
            Tensor(shape=[3], dtype=float32, place=Place(cpu), stop_gradient=True,
            [-0.03357624,  0.        ,  4.99955177])

    """

    def __init__(self, name: str | None = None) -> None:
        super().__init__()
        self._name = name

    def forward(self, x: Tensor) -> Tensor:
        return F.mish(x, self._name)

    def extra_repr(self) -> str:
        name_str = f'name={self._name}' if self._name else ''
        return name_str


class Tanhshrink(Layer):
    """
    Tanhshrink Activation

    .. math::

        Tanhshrink(x) = x - tanh(x)

    Parameters:
        name (str, optional): Name for the operation (optional, default is None).
            For more information, please refer to :ref:`api_guide_Name`.

    Shape:
        - input: Tensor with any shape.
        - output: Tensor with the same shape as input.

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> x = paddle.to_tensor([-0.4, -0.2, 0.1, 0.3])
            >>> m = paddle.nn.Tanhshrink()
            >>> out = m(x)
            >>> print(out)
            Tensor(shape=[4], dtype=float32, place=Place(cpu), stop_gradient=True,
            [-0.02005100, -0.00262472,  0.00033201,  0.00868741])
    """

    def __init__(self, name: str | None = None) -> None:
        super().__init__()
        self._name = name

    def forward(self, x: Tensor) -> Tensor:
        return F.tanhshrink(x, self._name)

    def extra_repr(self) -> str:
        name_str = f'name={self._name}' if self._name else ''
        return name_str


class ThresholdedReLU(Layer):
    r"""
    Thresholded ReLU Activation

    .. math::

        ThresholdedReLU(x) =
            \left\{
                \begin{array}{rl}
                x,& \text{if } \ x > threshold \\
                value,& \text{otherwise}
                \end{array}
            \right.


    Parameters:
        threshold (float, optional): The value of threshold for ThresholdedReLU. Default is 1.0
        value (float, optional): The value to replace with when x is less than threshold. Default is 0.0
        name (str|None, optional): Name for the operation (optional, default is None).
            For more information, please refer to :ref:`api_guide_Name`.

    Shape:
        - input: Tensor with any shape.
        - output: Tensor with the same shape as input.

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> x = paddle.to_tensor([2., 0., 1.])
            >>> m = paddle.nn.ThresholdedReLU()
            >>> out = m(x)
            >>> print(out)
            Tensor(shape=[3], dtype=float32, place=Place(cpu), stop_gradient=True,
            [2., 0., 0.])
    """

    def __init__(
        self,
        threshold: float = 1.0,
        value: float = 0.0,
        name: str | None = None,
    ) -> None:
        super().__init__()
        self._threshold = threshold
        self._value = value
        self._name = name

    def forward(self, x: Tensor) -> Tensor:
        return F.thresholded_relu(x, self._threshold, self._value, self._name)

    def extra_repr(self) -> str:
        name_str = f', name={self._name}' if self._name else ''
        return f'threshold={self._threshold}, value={self._value}{name_str}'


class Silu(Layer):
    r"""
    Silu Activation

    .. math::

        silu(x) = \frac{x}{1 + \mathrm{e}^{-x}}

    Where :math:`x` is the input Tensor.

    Parameters:
        name (str|None, optional): For details, please refer to :ref:`api_guide_Name`. Generally, no setting is required. Default: None.

    Shape:
        - input: Tensor with any shape.
        - output: Tensor with the same shape as input.

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> x = paddle.to_tensor([1.0, 2.0, 3.0, 4.0])
            >>> m = paddle.nn.Silu()
            >>> out = m(x)
            >>> print(out)
            Tensor(shape=[4], dtype=float32, place=Place(cpu), stop_gradient=True,
            [0.73105860, 1.76159406, 2.85772228, 3.92805505])
    """

    def __init__(self, name: str | None = None) -> str:
        super().__init__()
        self._name = name

    def forward(self, x: Tensor) -> Tensor:
        return F.silu(x, self._name)

    def extra_repr(self) -> str:
        name_str = f'name={self._name}' if self._name else ''
        return name_str


class LogSigmoid(Layer):
    r"""
    LogSigmoid Activation.

    .. math::

        LogSigmoid(x) = log \frac{1}{1 + e^{-x}}

    Parameters:
        x (Tensor): The input Tensor with data type float32, or float64.
        name (str|None, optional): Name for the operation (optional, default is None).
            For more information, please refer to :ref:`api_guide_Name`.

    Shape:
        - input: Tensor with any shape.
        - output: Tensor with the same shape as input.

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> x = paddle.to_tensor([1.0, 2.0, 3.0, 4.0])
            >>> m = paddle.nn.LogSigmoid()
            >>> out = m(x)
            >>> print(out)
            Tensor(shape=[4], dtype=float32, place=Place(cpu), stop_gradient=True,
            [-0.31326166, -0.12692805, -0.04858733, -0.01814996])
    """

    def __init__(self, name: str | None = None) -> None:
        super().__init__()
        self._name = name

    def forward(self, x: Tensor) -> Tensor:
        return F.log_sigmoid(x, self._name)

    def extra_repr(self) -> str:
        name_str = f'name={self._name}' if self._name else ''
        return name_str


class Softmax(Layer):
    r"""
    Softmax Activation.

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

        Softmax[i, j] = \frac{\exp(x[i, j])}{\sum_j(exp(x[i, j])}

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
        axis (int, optional): The axis along which to perform log_softmax
            calculations. It should be in range [-D, D), where D is the
            dimensions of ``x`` . If ``axis`` < 0, it works the same way as
            :math:`axis + D` . Default is -1.
        name (str|None, optional): Name for the operation (optional, default is None).
            For more information, please refer to :ref:`api_guide_Name`.

    Shape:
        - input: Tensor with any shape.
        - output: Tensor with the same shape as input.

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> x = paddle.to_tensor([[[2.0, 3.0, 4.0, 5.0],
            ...                        [3.0, 4.0, 5.0, 6.0],
            ...                        [7.0, 8.0, 8.0, 9.0]],
            ...                       [[1.0, 2.0, 3.0, 4.0],
            ...                        [5.0, 6.0, 7.0, 8.0],
            ...                        [6.0, 7.0, 8.0, 9.0]]], dtype='float32')
            >>> m = paddle.nn.Softmax()
            >>> out = m(x)
            >>> print(out)
            Tensor(shape=[2, 3, 4], dtype=float32, place=Place(cpu), stop_gradient=True,
            [[[0.03205860, 0.08714432, 0.23688284, 0.64391428],
              [0.03205860, 0.08714432, 0.23688284, 0.64391428],
              [0.07232949, 0.19661194, 0.19661194, 0.53444666]],
             [[0.03205860, 0.08714432, 0.23688284, 0.64391428],
              [0.03205860, 0.08714432, 0.23688284, 0.64391428],
              [0.03205860, 0.08714432, 0.23688284, 0.64391428]]])

    """

    def __init__(self, axis: int = -1, name: str | None = None) -> None:
        super().__init__()
        self._axis = axis
        self._dtype = None
        self._name = name

    def forward(self, x: Tensor) -> Tensor:
        return F.softmax(x, self._axis, name=self._name)

    def extra_repr(self) -> str:
        name_str = f', name={self._name}' if self._name else ''
        return f'axis={self._axis}{name_str}'


class LogSoftmax(Layer):
    r"""
    This operator implements the log_softmax layer. The calculation process is as follows:

    .. math::

        \begin{array} {rcl}
            Out[i, j] &= &log(softmax(x)) \\
            &= &log(\frac{\exp(X[i, j])}{\sum_j(\exp(X[i, j])})
        \end{array}

    Parameters:
        axis (int, optional): The axis along which to perform log_softmax
            calculations. It should be in range [-D, D), where D is the
            dimensions of the input Tensor . If ``axis`` < 0, it works the
            same way as :math:`axis + D` . Default is -1.
        name (str|None, optional): Name for the operation (optional, default is None).
            For more information, please refer to :ref:`api_guide_Name`.

    Shape:
        - input: Tensor with any shape.
        - output: Tensor with the same shape as input.

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> x = [[[-2.0,  3.0, -4.0,  5.0],
            ...       [ 3.0, -4.0,  5.0, -6.0],
            ...       [-7.0, -8.0,  8.0,  9.0]],
            ...      [[ 1.0, -2.0, -3.0,  4.0],
            ...       [-5.0,  6.0,  7.0, -8.0],
            ...       [ 6.0,  7.0,  8.0,  9.0]]]
            >>> m = paddle.nn.LogSoftmax()
            >>> x = paddle.to_tensor(x)
            >>> out = m(x)
            >>> print(out)
            Tensor(shape=[2, 3, 4], dtype=float32, place=Place(cpu), stop_gradient=True,
            [[[-7.12783957 , -2.12783957 , -9.12783909 , -0.12783945 ],
              [-2.12705135 , -9.12705135 , -0.12705141 , -11.12705135],
              [-16.31326103, -17.31326103, -1.31326187 , -0.31326184 ]],
             [[-3.05181193 , -6.05181217 , -7.05181217 , -0.05181199 ],
              [-12.31326675, -1.31326652 , -0.31326646 , -15.31326675],
              [-3.44018984 , -2.44018984 , -1.44018972 , -0.44018975 ]]])

    """

    def __init__(self, axis: int = -1, name: str | None = None) -> None:
        super().__init__()
        self._axis = axis
        self._name = name

    def forward(self, x: Tensor) -> Tensor:
        return F.log_softmax(x, self._axis)

    def extra_repr(self) -> str:
        name_str = f', name={self._name}' if self._name else ''
        return f'axis={self._axis}{name_str}'


class Maxout(Layer):
    r"""
    Maxout Activation. Create a callable object of `Maxout`.

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
        groups (int): The groups number of maxout. `groups` specifies the
            index of channel dimension where maxout will be performed. This must be
            a factor of number of features. Default is 1.
        axis (int, optional): The axis along which to perform maxout calculations.
            It should be 1 when data format is NCHW, be -1 or 3 when data format
            is NHWC. If ``axis`` < 0, it works the same way as :math:`axis + D` ,
            where D is the dimensions of ``x`` . Default is 1.
        name (str|None, optional): Name for the operation (optional, default is None).
            For more information, please refer to :ref:`api_guide_Name`.

    Shape:
        - input: :math:`(N, C_{in}, H_{in}, W_{in})`
        - output: :math:`(N, C_{out}, H_{out}, W_{out})`

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> paddle.seed(100)

            >>> x = paddle.rand([1, 2, 3, 4])
            >>> m = paddle.nn.Maxout(groups=2)
            >>> out = m(x)
            >>> print(out)
            Tensor(shape=[1, 1, 3, 4], dtype=float32, place=Place(cpu), stop_gradient=True,
            [[[[0.85139430, 0.95717543, 0.43864486, 0.51577556],
               [0.84765935, 0.45680618, 0.39412445, 0.72039396],
               [0.59444654, 0.78120756, 0.78364515, 0.90572405]]]])
    """

    def __init__(
        self, groups: int, axis: int = 1, name: str | None = None
    ) -> None:
        super().__init__()
        self._groups = groups
        self._axis = axis
        self._name = name

    def forward(self, x: Tensor) -> Tensor:
        return F.maxout(x, self._groups, self._axis, self._name)

    def extra_repr(self) -> str:
        name_str = f', name={self._name}' if self._name else ''
        return f'groups={self._groups}, axis={self._axis}{name_str}'


class Softmax2D(Layer):
    r"""

    Softmax2D Activation.
    Given a Tensor with shape (B, C, H, W) or (C, H, W), it will apply Softmax to each location (C, h_i, w_j).
    The sum of result in each location (C, H_i, W_j) will be one.

    Shape:
        - Input: :math:`(B, C, H, W)` or :math:`(C, H, W)`
        - Output: :math:`(B, C, H, W)` or :math:`(C, H, W)` (same as input)

    Returns:
        A Tensor of the same shape and dtype as input with value in range [0, 1].

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> paddle.seed(100)

            >>> x = paddle.rand([1, 2, 3, 4])
            >>> m = paddle.nn.Softmax2D()
            >>> out = m(x)
            >>> print(out)
            Tensor(shape=[1, 2, 3, 4], dtype=float32, place=Place(cpu), stop_gradient=True,
            [[[[0.42608523, 0.32081410, 0.39483935, 0.55642301],
               [0.38131708, 0.45118359, 0.44891062, 0.46053308],
               [0.35746980, 0.60766530, 0.38638926, 0.70425135]],
              [[0.57391477, 0.67918587, 0.60516071, 0.44357699],
               [0.61868292, 0.54881644, 0.55108935, 0.53946698],
               [0.64253020, 0.39233473, 0.61361068, 0.29574865]]]])

    """

    def __init__(self, name: str | None = None) -> None:
        super().__init__()
        self._dtype = None
        self._name = name

    def forward(self, x: Tensor) -> Tensor:
        assert (
            x.ndim == 3 or x.ndim == 4
        ), f"Softmax2D requires a 3D or 4D tensor as input. Received: {x.ndim}D."
        return F.softmax(x, axis=-3, dtype=self._dtype, name=self._name)

    def extra_repr(self) -> str:
        name_str = f'name={self._name}' if self._name else ''
        return name_str
