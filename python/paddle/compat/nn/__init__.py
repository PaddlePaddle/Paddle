# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

import collections
from itertools import repeat
from math import sqrt
from typing import TYPE_CHECKING

import paddle
from paddle import nn
from paddle.utils.decorator_utils import ForbidKeywordsDecorator

from . import functional
from .transformer import MultiheadAttention

if TYPE_CHECKING:
    from paddle import Tensor
    from paddle._typing import (
        DTypeLike,
        PlaceLike,
        Size1,
        Size2,
        Size3,
    )


__all__ = [
    'Unfold',
    'Linear',
    'Softmax',
    'AvgPool1D',
    'AvgPool2D',
    'AvgPool3D',
    'AvgPool1d',
    'AvgPool2d',
    'AvgPool3d',
    'MultiheadAttention',
]


def _ntuple(n, name="parse"):
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return tuple(x)
        return tuple(repeat(x, n))

    parse.__name__ = name
    return parse


_single = _ntuple(1, "_single")


class AvgPool1D(nn.Layer):
    r"""
    This operation applies a 1D average pooling over an input signal composed
    of several input planes, based on the input, output_size, return_mask parameters.
    Input(X) and output(Out) are in NCL format, where N is batch
    size, C is the number of channels, L is the length of the feature.
    The output tensor shape will be [N, C, output_size].

    The output value of the layer with input size (N, C, L),
    output (N, C, :math:`L_{out}`) and kernel_size ksize can be precisely described as
    For average pool1d:

    ..  math::

        Output(N_i, C_i, l) = \frac{Input[N_i, C_i, stride \times l:stride \times l+k]}{ksize}

    Parameters:
        kernel_size(int|list|tuple): The pool kernel size. If pool kernel size is a tuple or list,
            it must contain an integer.
        stride(int|list|tuple|None, optional): The pool stride size. If pool stride size is a tuple or list,
            it must contain an integer. Default None, then stride will be equal to the kernel_size.
        padding(str|int|list|tuple, optional): The padding size. Padding could be in one of the following forms.
            1. A string in ['valid', 'same'].
            2. An int, which means the feature map is zero padded by size of `padding` on every sides.
            3. A list[int] or tuple(int) whose length is 1, which means the feature map is zero padded by the size of `padding[0]` on every sides.
            4. A list[int] or tuple(int) whose length is 2. It has the form [pad_before, pad_after].
            5. A list or tuple of pairs of integers. It has the form [[pad_before, pad_after], [pad_before, pad_after], ...]. Note that, the batch dimension and channel dimension should be [0,0] or (0,0).
            The default value is 0.
        ceil_mode(bool, optional): ${ceil_mode_comment}Whether to use the ceil function to calculate output height
            and width. If it is set to False, the floor function will be used. The default value is False.
        count_include_pad(bool, optional): Whether to include padding points in average pooling mode, default is `False`.

    Shape:
        - x(Tensor): The input tensor of avg pool1d operator, which is a 3-D tensor.
          The data type can be float32, float64.
        - output(Tensor): The output tensor of avg pool1d  operator, which is a 3-D tensor.
          The data type is same as input x.

    Returns:
        A callable object of AvgPool1D.

    Examples:

        .. code-block:: pycon

            >>> import paddle
            >>> import paddle.compat.nn as nn

            >>> data = paddle.uniform([1, 3, 32], dtype="float32", min=-1, max=1)
            >>> AvgPool1D = nn.AvgPool1D(kernel_size=2, stride=2, padding=0)
            >>> pool_out = AvgPool1D(data)
            >>> print(pool_out.shape)
            paddle.Size([1, 3, 16])

    """

    __constants__ = [
        "kernel_size",
        "stride",
        "padding",
        "ceil_mode",
        "count_include_pad",
    ]

    kernel_size: Size1
    stride: Size1
    padding: Size1
    ceil_mode: bool
    count_include_pad: bool

    @ForbidKeywordsDecorator(
        illegal_keys={"exclusive", "name"},
        func_name="paddle.compat.nn.AvgPool1D",
        correct_name="paddle.nn.AvgPool1D",
    )
    def __init__(
        self,
        kernel_size: Size1,
        stride: Size1 | None = None,
        padding: Size1 = 0,
        ceil_mode: bool = False,
        count_include_pad: bool = True,
    ) -> None:
        super().__init__()
        self.kernel_size = _single(kernel_size)
        self.stride = _single(stride if stride is not None else kernel_size)
        self.padding = _single(padding)
        self.ceil_mode = ceil_mode
        self.count_include_pad = count_include_pad

    def forward(self, input: Tensor) -> Tensor:
        return nn.functional.avg_pool1d(
            input,
            self.kernel_size,
            self.stride,
            self.padding,
            not self.count_include_pad,
            self.ceil_mode,
        )

    def extra_repr(self) -> str:
        return f"kernel_size={self.kernel_size}, stride={self.stride}, padding={self.padding}"


class AvgPool2D(nn.Layer):
    r"""
    This operation applies 2D average pooling over input features based on the input,
    and kernel_size, stride, padding parameters. Input(X) and Output(Out) are
    in NCHW format, where N is batch size, C is the number of channels,
    H is the height of the feature, and W is the width of the feature.

    Example:
        Input:
            X shape: :math:`(N, C, :math:`H_{in}`, :math:`W_{in}`)`
        Attr:
            kernel_size: ksize

        Output:
            Out shape: :math:`(N, C, :math:`H_{out}`, :math:`W_{out}`)`

        ..  math::

            Output(N_i, C_j, h, w)  = \frac{\sum_{m=0}^{ksize[0]-1} \sum_{n=0}^{ksize[1]-1}
                Input(N_i, C_j, stride[0] \times h + m, stride[1] \times w + n)}{ksize[0] * ksize[1]}


    Parameters:
        kernel_size(int|list|tuple): The pool kernel size. If pool kernel size is a tuple or list,
            it must contain two integers, (pool_size_Height, pool_size_Width).
            Otherwise, the pool kernel size will be a square of an int.
        stride(int|list|tuple|None, optional): The pool stride size. If pool stride size is a tuple or list,
            it must contain two integers, (pool_stride_Height, pool_stride_Width).
            Otherwise, the pool stride size will be a square of an int.
            Default None, then stride will be equal to the kernel_size.
        padding(str|int|list|tuple, optional): The padding size. Padding could be in one of the following forms.
            1. A string in ['valid', 'same'].
            2. An int, which means the feature map is zero padded by size of `padding` on every sides.
            3. A list[int] or tuple(int) whose length is 2, [pad_height, pad_weight] whose value means the padding size of each dimension.
            4. A list[int] or tuple(int) whose length is 4. [pad_height_top, pad_height_bottom, pad_width_left, pad_width_right] whose value means the padding size of each side.
            5. A list or tuple of pairs of integers. It has the form [[pad_before, pad_after], [pad_before, pad_after], ...]. Note that, the batch dimension and channel dimension should be [0,0] or (0,0).
            The default value is 0.
        ceil_mode(bool, optional): When True, will use `ceil` instead of `floor` to compute the output shape.
        count_include_pad(bool, optional): Whether to include padding points in average pooling
            mode, default is `False`.
        divisor_override(float, optional): If specified, it will be used as divisor, otherwise kernel_size will be
            used. Default None.

    Shape:
        - x(Tensor): The input tensor of avg pool2d operator, which is a 4-D tensor.
          The data type can be float32, float64.
        - output(Tensor): The output tensor of avg pool2d  operator, which is a 4-D tensor.
          The data type is same as input x.

    Returns:
        A callable object of AvgPool2D.

    Examples:
        .. code-block:: pycon

            >>> import paddle
            >>> import paddle.compat.nn as nn

            >>> # max pool2d
            >>> input = paddle.uniform([1, 3, 32, 32], dtype="float32", min=-1, max=1)
            >>> AvgPool2D = nn.AvgPool2D(kernel_size=2, stride=2, padding=0)
            >>> output = AvgPool2D(input)
            >>> print(output.shape)
            paddle.Size([1, 3, 16, 16])

    """

    __constants__ = [
        "kernel_size",
        "stride",
        "padding",
        "ceil_mode",
        "count_include_pad",
        "divisor_override",
    ]

    kernel_size: Size2
    stride: Size2
    padding: Size2
    ceil_mode: bool
    count_include_pad: bool
    divisor_override: int | None

    @ForbidKeywordsDecorator(
        illegal_keys={"exclusive", "data_format", "name"},
        func_name="paddle.compat.nn.AvgPool2D",
        correct_name="paddle.nn.AvgPool2D",
    )
    def __init__(
        self,
        kernel_size: Size2,
        stride: Size2 | None = None,
        padding: Size2 = 0,
        ceil_mode: bool = False,
        count_include_pad: bool = True,
        divisor_override: int | None = None,
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride if (stride is not None) else kernel_size
        self.padding = padding
        self.ceil_mode = ceil_mode
        self.count_include_pad = count_include_pad
        self.divisor_override = divisor_override

    def forward(self, input: Tensor) -> Tensor:
        return nn.functional.avg_pool2d(
            input,
            self.kernel_size,
            self.stride,
            self.padding,
            self.ceil_mode,
            not self.count_include_pad,
            self.divisor_override,
        )

    def extra_repr(self) -> str:
        return f"kernel_size={self.kernel_size}, stride={self.stride}, padding={self.padding}"


class AvgPool3D(nn.Layer):
    """

    This operation applies 3D max pooling over input features based on the input,
    and kernel_size, stride, padding parameters. Input(X) and Output(Out) are
    in NCDHW format, where N is batch size, C is the number of channels,
    H is the height of the feature,  D is the depth of the feature, and W is the width of the feature.

    Parameters:
        kernel_size(int|list|tuple): The pool kernel size. If pool kernel size
            is a tuple or list, it must contain three integers,
            (kernel_size_Depth, kernel_size_Height, kernel_size_Width).
            Otherwise, the pool kernel size will be the cube of an int.
        stride(int|list|tuple|None, optional): The pool stride size. If pool stride size is a tuple or list,
            it must contain three integers, [stride_Depth, stride_Height, stride_Width).
            Otherwise, the pool stride size will be a cube of an int.
            Default None, then stride will be equal to the kernel_size.
        padding(str|int|list|tuple, optional): The padding size. Padding could be in one of the following forms.

            1. A string in ['valid', 'same'].
            2. An int, which means the feature map is zero padded by size of `padding` on every sides.
            3. A list[int] or tuple(int) whose length is 3, [pad_depth, pad_height, pad_weight] whose value means the padding size of each dimension.
            4. A list[int] or tuple(int) whose length is 6. [pad_depth_front, pad_depth_back, pad_height_top, pad_height_bottom, pad_width_left, pad_width_right] whose value means the padding size of each side.
            5. A list or tuple of pairs of integers. It has the form [[pad_before, pad_after], [pad_before, pad_after], ...]. Note that, the batch dimension and channel dimension should be [0,0] or (0,0).

            The default value is 0.
        ceil_mode(bool, optional): ${ceil_mode_comment}
        count_include_pad(bool, optional): Whether to include padding points in average pooling mode, default is True.
        divisor_override(int|float, optional): if specified, it will be used as divisor, otherwise kernel_size will
            be used. Default None.

    Returns:
        A callable object of AvgPool3D.

    Shape:
        - x(Tensor): The input tensor of avg pool3d operator, which is a 5-D tensor.
          The data type can be float16, float32, float64.
        - output(Tensor): The output tensor of avg pool3d  operator, which is a 5-D tensor.
          The data type is same as input x.

    Examples:
        .. code-block:: pycon

            >>> import paddle
            >>> import paddle.compat.nn as nn

            >>> # avg pool3d
            >>> input = paddle.uniform([1, 2, 3, 32, 32], dtype="float32", min=-1, max=1)
            >>> AvgPool3D = nn.AvgPool3D(kernel_size=2, stride=2, padding=0)
            >>> output = AvgPool3D(input)
            >>> print(output.shape)
            paddle.Size([1, 2, 1, 16, 16])

    """

    __constants__ = [
        "kernel_size",
        "stride",
        "padding",
        "ceil_mode",
        "count_include_pad",
        "divisor_override",
    ]
    kernel_size: Size3
    stride: Size3
    padding: Size3
    ceil_mode: bool
    count_include_pad: bool
    divisor_override: int | None

    @ForbidKeywordsDecorator(
        illegal_keys={"exclusive", "data_format", "name"},
        func_name="paddle.compat.nn.AvgPool3D",
        correct_name="paddle.nn.AvgPool3D",
    )
    def __init__(
        self,
        kernel_size: Size3,
        stride: Size3 | None = None,
        padding: Size3 = 0,
        ceil_mode: bool = False,
        count_include_pad: bool = True,
        divisor_override: int | None = None,
    ) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride if (stride is not None) else kernel_size
        self.padding = padding
        self.ceil_mode = ceil_mode
        self.count_include_pad = count_include_pad
        self.divisor_override = divisor_override

    def forward(self, input: Tensor) -> Tensor:
        return nn.functional.avg_pool3d(
            input,
            self.kernel_size,
            self.stride,
            self.padding,
            self.ceil_mode,
            not self.count_include_pad,
            self.divisor_override,
        )

    def extra_repr(self) -> str:
        return f"kernel_size={self.kernel_size}, stride={self.stride}, padding={self.padding}"

    def __setstate__(self, state):
        super().__setstate__(state)
        self.__dict__.setdefault("padding", 0)
        self.__dict__.setdefault("ceil_mode", False)
        self.__dict__.setdefault("count_include_pad", True)


class Unfold(nn.Unfold):
    """
    A compatible version of paddle.nn.Unfold:

    The keyword arguments are in non-plural forms, example: `kernel_size` instead of `kernel_sizes`. `padding` restricts the size of the input to be 1(int) or 2, Size4 is not allowed.

    All the input parameters allow `Tensor` or `pir.Value` as inputs, and will be converted to lists. Other aspects are the same. To use a more input-flexible version of Unfold, please refer to `paddle.nn.Unfold`.

    Args:
        kernel_size(int|list|tuple|Tensor): The size of convolution kernel, should be [k_h, k_w]
            or an integer k treated as [k, k].
        stride(int|list|tuple|Tensor, optional): The strides, should be [stride_h, stride_w]
            or an integer stride treated as [sride, stride]. For default, strides will be [1, 1].
        padding(int|list|tuple|Tensor, optional): The paddings of each dimension, should be
            a single integer or [padding_h, padding_w]. If [padding_h, padding_w] was given, it will expanded to
            [padding_h, padding_w, padding_h, padding_w]. If an integer padding was given,
            [padding, padding, padding, padding] will be used. By default, paddings will be 0.
        dilation(int|list|tuple|Tensor, optional): The dilations of convolution kernel, should be
            [dilation_h, dilation_w], or an integer dilation treated as [dilation, dilation].
            For default, it will be [1, 1].

    Examples:
        .. code-block:: pycon

            >>> import paddle
            >>> x = paddle.randn((100, 3, 224, 224))
            >>> unfold = paddle.compat.nn.Unfold(kernel_size=[3, 3])
            >>> result = unfold(x)
            >>> print(result.shape)
            paddle.Size([100, 27, 49284])
    """

    kernel_sizes: Size2
    dilations: Size2
    paddings: Size2
    strides: Size2

    @ForbidKeywordsDecorator(
        illegal_keys={"kernel_sizes", "dilations", "paddings", "strides"},
        func_name="paddle.compat.nn.Unfold",
        correct_name="paddle.nn.Unfold",
    )
    def __init__(
        self,
        kernel_size: Size2,
        dilation: Size2 = 1,
        padding: Size2 = 0,
        stride: Size2 = 1,
    ) -> None:
        super().__init__(kernel_size, dilation, padding, stride)

    def forward(self, input: Tensor) -> Tensor:
        def to_list_if_necessary(x):
            if isinstance(x, (paddle.pir.Value, paddle.Tensor)):
                x = x.tolist()
            return x

        return nn.functional.unfold(
            input,
            kernel_sizes=to_list_if_necessary(self.kernel_sizes),
            strides=to_list_if_necessary(self.strides),
            paddings=to_list_if_necessary(self.paddings),
            dilations=to_list_if_necessary(self.dilations),
        )


class Linear(nn.Layer):
    r"""

    Python compatible fully-connected linear transformation layer. For each input :math:`X` ,
    the equation is:

    .. math::

        Out = XW^T + b

    where :math:`W` is the weight and :math:`b` is the bias.

    Linear layer takes only one multi-dimensional tensor as input with the
    shape :math:`[*, in\_features]` , where :math:`*` means any
    number of additional dimensions. It multiplies input tensor with the transpose
    of weight (a 2-D tensor of shape :math:`[out\_features, in\_features]` ) and
    produces an output tensor of shape :math:`[*, out\_features]` .
    If ``bias`` is not False, the bias (a 1-D tensor of
    shape :math:`[out\_features]` ) will be created and added to the output. At the
    end of the initialization, ``reset_parameters`` will be called to initialize
    the ``weight`` and ``bias`` (if available) randomly.

    Parameters:
        in_features (int):
            The number of input units.
        out_features (int):
            The number of output units.
        bias (bool): If True, the bias (a 1-D tensor of shape :math:`[out\_features]` ) will be created and
            added to the output. Default: True.
        device (PlaceLike): The device of the parameters created. Default: None,
            representing the default paddle device.
        dtype (DTypeLike): The dtype of the parameters created. Default: None, and is set by
            the default dtype of Linear (float32).

    Variables:
        weight (paddle.Tensor): learnable parameters of the module of shape :math:`[out\_features, in\_features]`.
            The values are initialized from :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})`, where :math:`k` is :math:`\frac{1}{in\_features}`.
        bias (paddle.Tensor): learnable parameters of the module of shape :math:`[out\_features]`. If ``bias`` is True,
            the values are initialized from :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})`, where :math:`k` is :math:`\frac{1}{in\_features}`.

    Shape:
        - input: Multi-dimensional tensor with shape :math:`[*, in\_features]` . Its data types are float16, float32, float64 ,The default is float32 .
        - output: Multi-dimensional tensor with shape :math:`[*, out\_features]` . The data type is the same as the input .

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> paddle.seed(100)

            >>> # Define the linear layer.
            >>> linear = paddle.compat.nn.Linear(2, 4, bias=True)
            >>> print(linear.weight)
            Parameter containing:
            Tensor(shape=[4, 2], dtype=float32, place=Place(cpu), stop_gradient=False,
                   [[-0.49191639,  0.28120756],
                    [-0.17887023,  0.40572405],
                    [ 0.35139430,  0.45717543],
                    [-0.06135514, -0.21088189]])

            >>> print(linear.bias)
            Parameter containing:
            Tensor(shape=[4], dtype=float32, place=Place(cpu), stop_gradient=False,
                   [ 0.49166456, -0.06108528, -0.14973064,  0.31168410])

            >>> x = paddle.arange(6, dtype="float32").reshape([3, 2])
            >>> y = linear(x)
            >>> print(y)
            Tensor(shape=[3, 4], dtype=float32, place=Place(cpu), stop_gradient=False,
                   [[ 0.77287209,  0.34463876,  0.30744481,  0.10080221],
                    [ 0.35145447,  0.79834640,  1.92458415, -0.44367185],
                    [-0.06996319,  1.25205410,  3.54172373, -0.98814595]])
    """

    __constants__ = ["in_features", "out_features"]
    in_features: int
    out_features: int
    weight: Tensor

    @ForbidKeywordsDecorator(
        illegal_keys={"weight_attr", "bias_attr", "name"},
        func_name="paddle.compat.nn.Linear",
        correct_name="paddle.nn.Linear",
    )
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device: PlaceLike | None = None,
        dtype: DTypeLike | None = None,
    ) -> None:
        super().__init__()
        self._dtype = (
            self._helper.get_default_dtype() if dtype is None else dtype
        )
        self.in_features = in_features
        self.out_features = out_features
        self.weight = self.create_parameter(
            shape=[out_features, in_features],
            attr=None,
            dtype=self._dtype,
            is_bias=False,
            device=device,
        )
        self.bias = None
        if bias:
            self.bias = self.create_parameter(
                shape=[out_features],
                attr=None,
                dtype=self._dtype,
                is_bias=True,
                device=device,
            )
        # The same parameter initialization as PyTorch
        self.reset_parameters()

    def forward(self, input: Tensor) -> Tensor:
        return functional.linear.__wrapped__(  # bypass ForbidKeywordsDecorator
            input=input, weight=self.weight, bias=self.bias
        )

    def extra_repr(self) -> str:
        """
        Return the extra representation of the module.
        """
        return f"in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}"

    def reset_parameters(self) -> None:
        """
        Resets parameters based on their initialization used in ``__init__``.
        """

        nn.init.kaiming_uniform_(self.weight, a=sqrt(5))
        if self.bias is not None:
            # nn.init._calculate_fan_in_and_fan_out(self.weight) for 2D array
            # is equivalent to returning (weight.shape[1], weight.shape[0])
            # TODO(heqianyue): use _calculate_fan_in_and_fan_out when available
            fan_in = self.weight.shape[1]
            bound = 1 / sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)


class Softmax(nn.Layer):
    r"""
    Softmax Activation.

    This operator implements the softmax layer. The calculation process is as follows:

    1. The dimension :attr:`dim` of ``input`` will be permuted to the last.

    2. Then ``input`` will be logically flattened to a 2-D matrix. The matrix's second
    dimension(row length) is the same as the dimension :attr:`dim` of ``input``,
    and the first dimension(column length) is the product of all other dimensions
    of ``input``. For each row of the matrix, the softmax operator squashes the
    K-dimensional(K is the width of the matrix, which is also the size of ``input``'s
    dimension :attr:`dim`) vector of arbitrary real values to a K-dimensional
    vector of real values in the range [0, 1] that add up to 1.

    3. After the softmax operation is completed, the inverse operations of steps 1 and 2
    are performed to restore the two-dimensional matrix to the same dimension as the ``input`` .

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
            dim = -1

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
            dim = 1

          Output:
            out.shape = [2, 3, 4]
            out.data = [[[0.00657326, 0.00657326, 0.01714783, 0.01714783],
                         [0.01786798, 0.01786798, 0.04661262, 0.04661262],
                         [0.97555875, 0.97555875, 0.93623955, 0.93623955]],
                        [[0.00490169, 0.00490169, 0.00490169, 0.00490169],
                         [0.26762315, 0.26762315, 0.26762315, 0.26762315],
                         [0.72747516, 0.72747516, 0.72747516, 0.72747516]]]

    Parameters:
        dim (int, optional): The dim along which to perform log_softmax
            calculations. It should be in range [-D, D), where D is the
            dimensions of ``input`` . If ``dim`` < 0, it works the same way as
            :math:`dim + D` . Default is None.

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
            >>> m = paddle.compat.nn.Softmax()
            >>> out = m(x)
            >>> print(out)
            Tensor(shape=[2, 3, 4], dtype=float32, place=Place(cpu), stop_gradient=True,
            [[[0.73105854, 0.73105854, 0.73105854, 0.73105854],
              [0.11920292, 0.11920292, 0.11920292, 0.11920292],
              [0.73105854, 0.73105854, 0.50000000, 0.50000000]],
             [[0.26894143, 0.26894143, 0.26894143, 0.26894143],
              [0.88079703, 0.88079703, 0.88079703, 0.88079703],
              [0.26894143, 0.26894143, 0.50000000, 0.50000000]]])

    """

    @ForbidKeywordsDecorator(
        illegal_keys={"axis"},
        func_name="paddle.compat.nn.Softmax",
        correct_name="paddle.nn.Softmax",
    )
    def __init__(self, dim: int | None = None) -> None:
        super().__init__()
        self._dim = dim
        self._dtype = None

    def forward(self, input: Tensor) -> Tensor:
        return functional.softmax(input, self._dim)

    def extra_repr(self) -> str:
        return f"dim={self.dim}"


AvgPool1d = AvgPool1D
AvgPool2d = AvgPool2D
AvgPool3d = AvgPool3D
