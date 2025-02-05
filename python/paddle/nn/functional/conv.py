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

from paddle import _C_ops, _legacy_C_ops, get_flags, in_dynamic_mode, pir
from paddle.base.framework import _global_flags, in_dynamic_or_pir_mode
from paddle.device import (
    get_all_custom_device_type,
    is_compiled_with_cuda,
    is_compiled_with_rocm,
)
from paddle.tensor.manipulation import reshape
from paddle.tensor.math import _add_with_axis

from ...base.data_feeder import check_dtype, check_variable_and_dtype
from ...base.layer_helper import LayerHelper
from ...common_ops_import import Variable
from ...device import get_cudnn_version
from ...framework import no_grad
from ...tensor.manipulation import squeeze, unsqueeze
from ...utils import (
    _contain_var,
    _convert_to_tensor_list,
    _is_symmetric_padding,
    convert_to_list,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from paddle import Tensor
    from paddle._typing import (
        DataLayout1D,
        DataLayout2D,
        DataLayout3D,
        DataLayoutND,
        Size1,
        Size2,
        Size3,
        Size4,
        Size6,
    )

    from .common import _PaddingSizeMode


__all__ = []


def _is_list_or_tuple(input):
    return isinstance(input, (list, tuple))


def _zero_padding_in_batch_and_channel(padding, channel_last):
    if channel_last:
        return list(padding[0]) == [0, 0] and list(padding[-1]) == [0, 0]
    else:
        return list(padding[0]) == [0, 0] and list(padding[1]) == [0, 0]


def _exclude_padding_in_batch_and_channel(padding, channel_last):
    padding_ = padding[1:-1] if channel_last else padding[2:]
    padding_ = [elem for pad_a_dim in padding_ for elem in pad_a_dim]
    return padding_


def _update_padding_nd(padding, channel_last, num_dims):
    if isinstance(padding, str):
        padding = padding.upper()
        if padding not in ["SAME", "VALID"]:
            raise ValueError(
                f"Unknown padding: '{padding}'. It can only be 'SAME' or 'VALID'."
            )
        if padding == "VALID":
            padding_algorithm = "VALID"
            padding = [0] * num_dims
        else:
            padding_algorithm = "SAME"
            padding = [0] * num_dims
    elif _is_list_or_tuple(padding):
        # for padding like
        # [(pad_before, pad_after), (pad_before, pad_after), ...]
        # padding for batch_dim and channel_dim included
        if len(padding) == 2 + num_dims and _is_list_or_tuple(padding[0]):
            if not _zero_padding_in_batch_and_channel(padding, channel_last):
                raise ValueError(
                    f"Non-zero padding({padding}) in the batch or channel dimensions "
                    "is not supported."
                )
            padding_algorithm = "EXPLICIT"
            padding = _exclude_padding_in_batch_and_channel(
                padding, channel_last
            )
            if _is_symmetric_padding(padding, num_dims):
                padding = padding[0::2]
        # for padding like [pad_before, pad_after, pad_before, pad_after, ...]
        elif len(padding) == 2 * num_dims and isinstance(padding[0], int):
            padding_algorithm = "EXPLICIT"
            padding = convert_to_list(padding, 2 * num_dims, 'padding')
            if _is_symmetric_padding(padding, num_dims):
                padding = padding[0::2]
        # for padding like [pad_d1, pad_d2, ...]
        elif len(padding) == num_dims and isinstance(padding[0], int):
            padding_algorithm = "EXPLICIT"
            padding = convert_to_list(padding, num_dims, 'padding')
        else:
            raise ValueError(f"In valid padding: {padding}")
    # for integer padding
    else:
        padding_algorithm = "EXPLICIT"
        padding = convert_to_list(padding, num_dims, 'padding')
    if not all(p >= 0 for p in padding):
        raise ValueError(
            f"Invalid padding, all value should be larger than or equal to 0, but received: {padding}"
        )
    return padding, padding_algorithm


def _conv_nd(
    x: Tensor,
    weight: Tensor,
    bias: Tensor | None = None,
    stride: int | Sequence[int] = 1,
    padding: _PaddingSizeMode | int | Sequence[int] | Sequence[Size2] = 0,
    padding_algorithm=None,
    dilation: int | Sequence[int] = 1,
    groups: int = 1,
    data_format: DataLayoutND = "NCHW",
    channel_dim: int = 1,
    op_type: str = "conv2d",
    use_cudnn: bool = True,
    name: str | None = None,
) -> Tensor:
    # Due to the poor performance of NHWC, we transpose the input to NCHW.
    if in_dynamic_or_pir_mode() and op_type == "conv2d":
        pre_bias = _C_ops.conv2d(
            x,
            weight,
            stride,
            padding,
            padding_algorithm,
            dilation,
            groups,
            data_format,
        )
        if bias is not None:
            new_shape = [1] * len(x.shape)
            new_shape[channel_dim] = -1
            bias = bias.reshape(new_shape)
            # TODO(qili93): temporary for ascend npu performance to be removed along with npu_identity op
            if (
                _global_flags()['FLAGS_npu_storage_format']
                and 'npu' in get_all_custom_device_type()
            ):
                with no_grad():
                    bias_storage = _C_ops.npu_identity(
                        bias, 3
                    )  # ACL_FORMAT_NC1HWC0 = 3
                    bias_storage._share_underline_tensor_to(bias)
            return _C_ops.add(pre_bias, bias)
        else:
            return pre_bias

    if in_dynamic_or_pir_mode() and op_type == "depthwise_conv2d":
        pre_bias = _C_ops.depthwise_conv2d(
            x,
            weight,
            stride,
            padding,
            padding_algorithm,
            groups,
            dilation,
            data_format,
        )
        if bias is not None:
            new_shape = [1] * len(x.shape)
            new_shape[channel_dim] = -1
            bias = bias.reshape(new_shape)
            return _C_ops.add(pre_bias, bias)
        else:
            return pre_bias

    if in_dynamic_or_pir_mode() and op_type == "conv3d":
        pre_bias = _C_ops.conv3d(
            x,
            weight,
            stride,
            padding,
            padding_algorithm,
            groups,
            dilation,
            data_format,
        )
        if bias is not None:
            new_shape = [1] * len(x.shape)
            new_shape[channel_dim] = -1
            bias = bias.reshape(new_shape)
            return _C_ops.add(pre_bias, bias)
        else:
            return pre_bias

    if in_dynamic_mode():
        attrs = (
            'strides',
            stride,
            'paddings',
            padding,
            'dilations',
            dilation,
            'groups',
            groups,
            'use_cudnn',
            use_cudnn,
            'fuse_relu_before_depthwise_conv',
            False,
            "padding_algorithm",
            padding_algorithm,
            "data_format",
            data_format,
        )
        pre_bias = getattr(_legacy_C_ops, op_type)(x, weight, *attrs)
        if bias is not None:
            out = _add_with_axis(pre_bias, bias, axis=channel_dim)
        else:
            out = pre_bias
    else:
        inputs = {'Input': [x], 'Filter': [weight]}
        attrs = {
            'strides': stride,
            'paddings': padding,
            'dilations': dilation,
            'groups': groups,
            'use_cudnn': use_cudnn,
            'fuse_relu_before_depthwise_conv': False,
            "padding_algorithm": padding_algorithm,
            "data_format": data_format,
        }
        check_variable_and_dtype(
            x, 'x', ['float16', 'uint16', 'float32', 'float64'], op_type
        )
        helper = LayerHelper(op_type, **locals())
        dtype = helper.input_dtype(input_param_name='x')
        pre_bias = helper.create_variable_for_type_inference(dtype)
        outputs = {"Output": [pre_bias]}
        helper.append_op(
            type=op_type, inputs=inputs, outputs=outputs, attrs=attrs
        )
        if bias is not None:
            out = helper.create_variable_for_type_inference(dtype)
            x_shape = list(pre_bias.shape)
            y_shape = list(bias.shape)
            if channel_dim == -1 or len(x_shape) == len(y_shape):
                helper.append_op(
                    type='elementwise_add',
                    inputs={'X': [pre_bias], 'Y': [bias]},
                    outputs={'Out': [out]},
                    attrs={'axis': -1},
                )
            else:
                assert len(x_shape) > len(
                    y_shape
                ), 'The length of pre_bias must greater than the length of bias'
                padding = len(x_shape) - len(y_shape) - channel_dim
                bias = reshape(
                    bias, [1] * channel_dim + y_shape + [1] * padding
                )

                helper.append_op(
                    type='elementwise_add',
                    inputs={'X': [pre_bias], 'Y': [bias]},
                    outputs={'Out': [out]},
                    attrs={'axis': -1},
                )
        else:
            out = pre_bias
    return out


def conv1d(
    x: Tensor,
    weight: Tensor,
    bias: Tensor | None = None,
    stride: Size1 = 1,
    padding: _PaddingSizeMode | Size1 | Size2 | Sequence[Size2] = 0,
    dilation: Size1 = 1,
    groups: int = 1,
    data_format: DataLayout1D = 'NCL',
    name: str | None = None,
) -> Tensor:
    r"""
    The convolution1D layer calculates the output based on the input, filter
    and strides, paddings, dilations, groups parameters. Input and
    Output are in NCL format, where N is batch size, C is the number of
    channels, L is the length of the feature.
    Filter is in MCK format, where M is the number of output image channels,
    C is the number of input image channels, K is the size of the kernel.
    If the groups is greater than 1, C will equal the number of input image
    channels divided by the groups. If bias attribution and activation type
    are provided, bias is added to the output of the convolution, and the
    corresponding activation function is applied to the final result.

    For each input :math:`X`, the equation is:

    .. math::

        Out = \sigma (W \ast X + b)

    Where:

    * :math:`X`: Input value, a tensor with NCL format.
    * :math:`W`: Kernel value, a tensor with MCK format.
    * :math:`\\ast`: Convolution operation.
    * :math:`b`: Bias value, a 2-D tensor with shape [M, 1].
    * :math:`\\sigma`: Activation function.
    * :math:`Out`: Output value, the shape of :math:`Out` and :math:`X` may be different.

    Example:

        - Input:

          Input shape: :math:`(N, C_{in}, L_{in})`

          Filter shape: :math:`(C_{out}, C_{in}, L_f)`

        - Output:

          Output shape: :math:`(N, C_{out}, L_{out})`

        Where

        .. math::

            L_{out} = \frac{(L_{in} + 2 * padding - (dilation * (L_f - 1) + 1))}{stride} + 1

    Args:
        x (Tensor): The input is 3-D Tensor with shape [N, C, L], the data type
            of input is float16 or float32 or float64.
        weight (Tensor): The convolution kernel with shape [M, C/g, K], where M is
            the number of output channels, g is the number of groups, K is the kernel's size.
        bias (Tensor, optional): The bias with shape [M,]. Default: None.
        stride (int|list|tuple, optional): The stride size. If stride is a list/tuple, it must
            contain one integers, (stride_size). Default: 1.
        padding (int|str|tuple|list, optional): The padding size. Padding could be in one of the following forms.
            1. a string in ['valid', 'same'].
            2. an int, which means the feature map is zero paded by size of `padding` on both sides.
            3. a list[int] or tuple[int] whose length is 1, which means the feature map is zero paded by size of `padding[0]` on both sides.
            4. a list[int] or tuple[int] whose length is 2. It has the form  [pad_before, pad_after].
            5. a list or tuple of pairs of ints. It has the form [[pad_before, pad_after], [pad_before, pad_after], ...]. Note that, the batch dimension and channel dimension are also included. Each pair of integers correspond to the amount of padding for a dimension of the input. Padding in batch dimension and channel dimension should be [0, 0] or (0, 0).
            The default value is 0.
        dilation (int|list|tuple, optional): The dilation size. If dilation is a list/tuple, it must
            contain one integer, (dilation_size). Default: 1.
        groups (int, optional): The groups number of the conv1d function. According to grouped
            convolution in Alex Krizhevsky's Deep CNN paper: when group=2,
            the first half of the filters is only connected to the first half
            of the input channels, while the second half of the filters is only
            connected to the second half of the input channels. Default: 1.
        data_format (str, optional): Specify the data format of the input, and the data format of the output
            will be consistent with that of the input. An optional string from: `"NCL"`, `"NLC"`.
            The default is `"NCL"`. When it is `"NCL"`, the data is stored in the order of:
            `[batch_size, input_channels, feature_length]`.
        name(str, optional): For detailed information, please refer
           to :ref:`api_guide_Name`. Usually name is no need to set and
           None by default.

    Returns:
        A tensor representing the conv1d, whose data type is the
        same with input.

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> import paddle.nn.functional as F

            >>> x = paddle.to_tensor([[[4, 8, 1, 9],
            ...                        [7, 2, 0, 9],
            ...                        [6, 9, 2, 6]]], dtype="float32")
            >>> w = paddle.to_tensor([[[9, 3, 4],
            ...                        [0, 0, 7],
            ...                        [2, 5, 6]],
            ...                       [[0, 3, 4],
            ...                        [2, 9, 7],
            ...                        [5, 6, 8]]], dtype="float32")

            >>> y = F.conv1d(x, w)
            >>> print(y)
            Tensor(shape=[1, 2, 2], dtype=float32, place=Place(cpu), stop_gradient=True,
            [[[133., 238.],
            [160., 211.]]])
    """
    cudnn_version = get_cudnn_version()
    if cudnn_version is not None:
        use_cudnn = True
    else:
        use_cudnn = False

    if data_format not in ["NCL", "NLC"]:
        raise ValueError(
            "Attr(data_format) should be 'NCL' or 'NLC'. "
            f"Received Attr(data_format): {data_format}."
        )

    channel_last = data_format == "NLC"
    channel_dim = -1 if channel_last else 1
    conv2d_data_format = "NHWC" if channel_last else "NCHW"
    if len(x.shape) != 3:
        raise ValueError(
            f"Input x should be 3D tensor, but received x with the shape of {x.shape}"
        )
    num_channels = x.shape[channel_dim]
    num_filters = weight.shape[0]

    if groups == 0:
        raise ValueError("The groups of conv1d should not be zero")
    if num_channels % groups != 0 and (
        in_dynamic_mode() or (num_channels != -1 and groups != -1)
    ):
        raise ValueError(
            "The channel of input must be divisible by groups,"
            f"received: the channel of input is {num_channels}, the shape of input is {x.shape}"
            f", the groups is {groups}"
        )
    if num_filters % groups != 0 and (
        in_dynamic_mode() or (num_filters != -1 and groups != -1)
    ):
        raise ValueError(
            "The number of filters must be divisible by groups,"
            f"received: the number of filters is {num_filters}, the shape of weight is {weight.shape}"
            f", the groups is {groups}"
        )

    # update attrs
    padding, padding_algorithm = _update_padding_nd(padding, channel_last, 1)

    if len(padding) == 2:
        padding = [0, 0, *padding]
    elif len(padding) == 1:
        padding = [0, *padding]
    else:
        raise ValueError(
            f"The size of padding's dimension should be 1 or 2. But got padding={padding}"
        )
    stride = [1, *convert_to_list(stride, 1, "stride")]
    dilation = [1, *convert_to_list(dilation, 1, "dilation")]
    from ...tensor.creation import assign as paddle_assign

    weight = paddle_assign(weight)
    weight = unsqueeze(weight, axis=[-2])

    l_type = "conv2d"

    # When "groups==num_channels and num_filters% num_channels == 0" using depthwise_conv2d has better performance
    if (
        is_compiled_with_cuda()
        and num_channels == groups
        and num_channels != 1
        and num_filters % num_channels == 0
    ):
        l_type = 'depthwise_conv2d'
        use_cudnn = False

    squeeze_axis = -3 if channel_last else -2
    x = unsqueeze(x, axis=[squeeze_axis])

    if in_dynamic_or_pir_mode():
        if l_type == 'conv2d':
            out = _C_ops.conv2d(
                x,
                weight,
                stride,
                padding,
                padding_algorithm,
                dilation,
                groups,
                conv2d_data_format,
            )
        else:
            out = _C_ops.depthwise_conv2d(
                x,
                weight,
                stride,
                padding,
                padding_algorithm,
                groups,
                dilation,
                conv2d_data_format,
                False,
                -1,
                False,
                False,
            )
        if bias is not None:
            out = _add_with_axis(out, bias, axis=channel_dim)
    else:
        inputs = {'Input': [x], 'Filter': [weight]}
        attrs = {
            'strides': stride,
            'paddings': padding,
            'dilations': dilation,
            'groups': groups,
            'use_cudnn': use_cudnn,
            'fuse_relu_before_depthwise_conv': False,
            "padding_algorithm": padding_algorithm,
            "data_format": conv2d_data_format,
        }
        check_variable_and_dtype(
            x, 'input', ['float16', 'float32', 'float64'], 'conv2d'
        )
        helper = LayerHelper(l_type, **locals())
        dtype = helper.input_dtype(input_param_name='x')
        out = helper.create_variable_for_type_inference(dtype)
        outputs = {"Output": [out]}
        helper.append_op(
            type=l_type, inputs=inputs, outputs=outputs, attrs=attrs
        )
        if bias is not None:
            out = _add_with_axis(out, bias, axis=channel_dim)
    out = squeeze(out, axis=[squeeze_axis])
    return out


def conv2d(
    x: Tensor,
    weight: Tensor,
    bias: Tensor | None = None,
    stride: Size2 = 1,
    padding: _PaddingSizeMode | Size2 | Size4 | Sequence[Size2] = 0,
    dilation: Size2 = 1,
    groups: int = 1,
    data_format: DataLayout2D = "NCHW",
    name: str | None = None,
) -> Tensor:
    r"""

    The convolution2D layer calculates the output based on the input, filter
    and strides, paddings, dilations, groups parameters. Input and
    Output are in NCHW or NHWC format, where N is batch size, C is the number of
    channels, H is the height of the feature, and W is the width of the feature.
    Filter is in MCHW format, where M is the number of output image channels,
    C is the number of input image channels, H is the height of the filter,
    and W is the width of the filter. If the groups is greater than 1,
    C will equal the number of input image channels divided by the groups.
    Please refer to UFLDL's `convolution
    <http://ufldl.stanford.edu/tutorial/supervised/FeatureExtractionUsingConvolution/>`_
    for more details.
    If bias attribution and activation type are provided, bias is added to the
    output of the convolution, and the corresponding activation function is
    applied to the final result.

    For each input :math:`X`, the equation is:

    ..  math::

        Out = \sigma (W \ast X + b)

    Where:

    * :math:`X`: Input value, a tensor with NCHW or NHWC format.
    * :math:`W`: Filter value, a tensor with MCHW format.
    * :math:`\\ast`: Convolution operation.
    * :math:`b`: Bias value, a 2-D tensor with shape [M, 1].
    * :math:`\\sigma`: Activation function.
    * :math:`Out`: Output value, the shape of :math:`Out` and :math:`X` may be different.

    Example:

        - Input:

          Input shape: :math:`(N, C_{in}, H_{in}, W_{in})`

          Filter shape: :math:`(C_{out}, C_{in}, H_f, W_f)`

        - Output:

          Output shape: :math:`(N, C_{out}, H_{out}, W_{out})`

        Where

        ..  math::

            H_{out}&= \frac{(H_{in} + 2 * paddings[0] - (dilations[0] * (H_f - 1) + 1))}{strides[0]} + 1 \\\\
            W_{out}&= \frac{(W_{in} + 2 * paddings[1] - (dilations[1] * (W_f - 1) + 1))}{strides[1]} + 1

    Args:
        x (Tensor): The input is 4-D Tensor with shape [N, C, H, W], the data type
            of input is float16 or float32 or float64.
        weight (Tensor): The convolution kernel with shape [M, C/g, kH, kW], where M is
            the number of output channels, g is the number of groups, kH is the filter's
            height, kW is the filter's width.
        bias (Tensor, optional): The bias with shape [M,].
        stride (int|list|tuple, optional): The stride size. It means the stride in convolution.
            If stride is a list/tuple, it must contain two integers, (stride_height, stride_width).
            Otherwise, stride_height = stride_width = stride. Default: stride = 1.
        padding (string|int|list|tuple, optional): The padding size. It means the number of zero-paddings
            on both sides for each dimension.If `padding` is a string, either 'VALID' or
            'SAME' which is the padding algorithm. If padding size is a tuple or list,
            it could be in three forms: `[pad_height, pad_width]` or
            `[pad_height_top, pad_height_bottom, pad_width_left, pad_width_right]`, and when
            `data_format` is `"NCHW"`, `padding` can be in the form `[[0,0], [0,0],
            [pad_height_top, pad_height_bottom], [pad_width_left, pad_width_right]]`.
            when `data_format` is `"NHWC"`, `padding` can be in the form
            `[[0,0], [pad_height_top, pad_height_bottom], [pad_width_left, pad_width_right], [0,0]]`.
            Default: padding = 0.
        dilation (int|list|tuple, optional): The dilation size. It means the spacing between the kernel
            points. If dilation is a list/tuple, it must contain two integers, (dilation_height,
            dilation_width). Otherwise, dilation_height = dilation_width = dilation.
            Default: dilation = 1.
        groups (int, optional): The groups number of the Conv2D Layer. According to grouped
            convolution in Alex Krizhevsky's Deep CNN paper: when group=2,
            the first half of the filters is only connected to the first half
            of the input channels, while the second half of the filters is only
            connected to the second half of the input channels. Default: groups=1.
        data_format (str, optional): Specify the data format of the input, and the data format of the output
            will be consistent with that of the input. An optional string from: `"NCHW"`, `"NHWC"`.
            The default is `"NCHW"`. When it is `"NCHW"`, the data is stored in the order of:
            `[batch_size, input_channels, input_height, input_width]`.
        name(str, optional): For detailed information, please refer
           to :ref:`api_guide_Name`. Usually name is no need to set and
           None by default.

    Returns:
        A Tensor representing the conv2d result, whose data type is the same with input.

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> import paddle.nn.functional as F

            >>> x_var = paddle.randn((2, 3, 8, 8), dtype='float32')
            >>> w_var = paddle.randn((6, 3, 3, 3), dtype='float32')

            >>> y_var = F.conv2d(x_var, w_var)

            >>> print(y_var.shape)
            [2, 6, 6, 6]
    """
    # entry checks
    if data_format not in ["NCHW", "NHWC"]:
        raise ValueError(
            "Attr(data_format) should be 'NCHW' or 'NHWC'. "
            f"Received Attr(data_format): {data_format}."
        )

    channel_last = data_format == "NHWC"
    channel_dim = -1 if channel_last else 1
    if len(x.shape) != 4:
        raise ValueError(
            f"Input x should be 4D tensor, but received x with the shape of {x.shape}"
        )
    num_channels = x.shape[channel_dim]
    num_filters = weight.shape[0]

    if groups == 0:
        raise ValueError("The groups of conv2d should be not be zero.")

    if num_channels % groups != 0 and (
        in_dynamic_mode() or (num_channels != -1 and groups != -1)
    ):
        raise ValueError(
            "The channel of input must be divisible by groups,"
            f"received: the channel of input is {num_channels}, the shape of input is {x.shape}"
            f", the groups is {groups}"
        )
    if num_filters % groups != 0 and (
        in_dynamic_mode() or (num_filters != -1 and groups != -1)
    ):
        raise ValueError(
            "The number of filters must be divisible by groups,"
            f"received: the number of filters is {num_filters}, the shape of weight is {weight.shape}"
            f", the groups is {groups}"
        )

    cudnn_version = get_cudnn_version()

    use_cudnn = (
        True
        if (is_compiled_with_cuda() and cudnn_version is not None)
        else False
    )

    # update attrs
    padding, padding_algorithm = _update_padding_nd(padding, channel_last, 2)
    stride = convert_to_list(stride, 2, 'stride')
    dilation = convert_to_list(dilation, 2, 'dilation')

    l_type = "conv2d"
    if (
        num_channels == groups
        and num_channels != 1
        and num_filters % num_channels == 0
    ):
        l_type = 'depthwise_conv2d'
        if is_compiled_with_rocm():
            use_cudnn = True
        else:
            use_cudnn = False
    else:
        if in_dynamic_mode():
            pre_bias = _C_ops.conv2d(
                x,
                weight,
                stride,
                padding,
                padding_algorithm,
                dilation,
                groups,
                data_format,
            )
            if bias is not None:
                channel_dim = (
                    channel_dim + len(x.shape)
                    if channel_dim < 0
                    else channel_dim
                )
                if len(bias.shape) < len(x.shape):
                    bias = _C_ops.reshape(
                        bias,
                        [1 for i in range(channel_dim)]
                        + bias.shape
                        + [1 for i in range(len(x.shape) - channel_dim - 1)],
                    )
                # TODO(qili93): temporary for ascend npu performance to be removed along with npu_identity op
                if (
                    _global_flags()['FLAGS_npu_storage_format']
                    and 'npu' in get_all_custom_device_type()
                ):
                    with no_grad():
                        bias_storage = _C_ops.npu_identity(
                            bias, 3
                        )  # ACL_FORMAT_NC1HWC0 = 3
                        bias_storage._share_underline_tensor_to(bias)
                return _C_ops.add(pre_bias, bias)
            else:
                return pre_bias

    if (
        is_compiled_with_cuda()
        and get_flags("FLAGS_conv2d_disable_cudnn")[
            "FLAGS_conv2d_disable_cudnn"
        ]
    ):
        use_cudnn = False

    return _conv_nd(
        x,
        weight,
        bias,
        stride,
        padding,
        padding_algorithm,
        dilation,
        groups,
        data_format,
        channel_dim,
        l_type,
        use_cudnn,
        name,
    )


def conv1d_transpose(
    x: Tensor,
    weight: Tensor,
    bias: Tensor | None = None,
    stride: Size1 = 1,
    padding: _PaddingSizeMode | Size1 | Size2 | Sequence[Size2] = 0,
    output_padding: Size1 = 0,
    groups: int = 1,
    dilation: Size1 = 1,
    output_size: Size1 | None = None,
    data_format: DataLayout1D = "NCL",
    name: str | None = None,
) -> Tensor:
    r"""
    The 1-D convolution transpose layer calculates the output based on the input,
    filter, and dilation, stride, padding. Input(Input) and output(Output)
    are in 'NCL' format or 'NLC' where N is batch size, C is the number of channels,
    L is the length of the feature. The details of convolution transpose
    layer, please refer to the following explanation and references
    `therein <https://arxiv.org/pdf/1603.07285.pdf>`_.
    If bias attribution and activation type are provided, bias is added to
    the output of the convolution, and the corresponding activation function
    is applied to the final result.

    For each input :math:`X`, the equation is:

    .. math::

        Out = \sigma (W \ast X + b)

    Where:

    * :math:`X`: Input value, a 3-D Tensor with 'NCL' format or 'NLC' format.
    * :math:`W`: Filter value, a 3-D Tensor with 'MCK' format.
    * :math:`\\ast`: Convolution operation.
    * :math:`b`: Bias value, a 2-D Tensor with shape [M, 1].
    * :math:`\\sigma`: Activation function.
    * :math:`Out`: Output value, a 3-D Tensor with data format 'NCL' or 'NLC', the shape of :math:`Out` and :math:`X` may be different.

    Example:

        - Input:

          Input shape: :math:`(N, C_{in}, L_{in})`

          Filter shape: :math:`(C_{in}, C_{out}, L_f)`

        - Output:

          Output shape: :math:`(N, C_{out}, L_{out})`

        Where

        .. math::

           L^\prime_{out} &= (L_{in} - 1) * stride - 2 * padding + dilation * (L_f - 1) + 1 \\
           L_{out} &\in [ L^\prime_{out}, L^\prime_{out} + stride ]

    Note:
          The conv1d_transpose can be seen as the backward of the conv1d. For conv1d,
          when stride > 1, conv1d maps multiple input shape to the same output shape,
          so for conv1d_transpose, when stride > 1, input shape maps multiple output shape.
          If output_size is None, :math:`L_{out} = L^\prime_{out}`;
          else, the :math:`L_{out}` of the output size must between :math:`L^\prime_{out}`
          and :math:`L^\prime_{out} + stride`.

    Args:
        x(Tensor): 3-D tensor with [N, C, L] or [N, L, C] format,
                         its data type is float32 or float64.
        weight(Tensor): The convolution kernel, a Tensor with shape [C, M/g, K],
            where M is the number of output channels(filters), g is the number of groups,
            K is the size of the kernel.
        bias(Tensor, optional): The bias, a Tensor with shape [M, ].
        stride(int|tuple|list, optional): The stride size. It means the stride in transposed convolution.
            If stride is a list/tuple, it must contain one integer, `(stride_size)`.
            Default: stride = 1.
        padding(int|list|str|tuple, optional): The padding size. The padding argument effectively adds
             `dilation * (kernel - 1)` amount of zero-padding on both sides of input. If `padding` is a
             string, either 'VALID' or 'SAME' supported, which is the padding algorithm.
             If `padding` is a tuple or list, it could be in two forms:
             `[pad]` or `[pad_left, pad_right]`. Default: padding = 0.
        output_padding(int|list|tuple, optional): The count of zeros to be added to tail of each dimension.
             If it is a list/tuple, it must contain one integer. Default: 0.
        groups(int, optional): The groups number of the conv1d transpose function. Inspired by
            grouped convolution in Alex Krizhevsky's Deep CNN paper, in which
            when group=2, the first half of the filters is only connected to the
            first half of the input channels, while the second half of the
            filters is only connected to the second half of the input channels.
            Default: groups = 1.
        dilation(int|tuple|list, optional): The dilation size. It means the spacing between the kernel points.
            If dilation is a list/tuple, it must contain one integer, `(dilation_size)`.
            Default: dilation = 1.
        output_size(int|tuple|list, optional): The output image size. If output size is a
            tuple/list, it must contain one integer, `(feature_length)`. None if use
            filter_size(shape of weight), padding, and stride to calculate output_size.
        data_format (str, optional): Specify the data format of the input, and the data format of the output
            will be consistent with that of the input. An optional string from: `"NCL"`, `"NLC"`.
            The default is `"NCL"`. When it is `"NCL"`, the data is stored in the order of:
            `[batch_size, input_channels, input_length]`.
        name(str, optional): For detailed information, please refer
           to :ref:`api_guide_Name`. Usually name is no need to set and
           None by default.

    Returns:
        A  tensor representing the result of 1-D transpose convolution, whose
        data type is the same with input. And its shape is (num_batches, channels, length)
        when data_format is `"NCL"` and (num_batches, length, channels) when data_format is
        `"NLC"`.

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> import paddle.nn.functional as F

            >>> # shape: (1, 2, 4)
            >>> x = paddle.to_tensor([[[4, 0, 9, 7],
            >>>                       [8, 0, 9, 2,]]], dtype="float32")
            >>> # shape: (2, 1, 2)
            >>> w = paddle.to_tensor([[[7, 0]],
            >>>                       [[4, 2]]], dtype="float32")

            >>> y = F.conv1d_transpose(x, w)
            >>> print(y)
            Tensor(shape=[1, 1, 5], dtype=float32, place=Place(cpu), stop_gradient=True,
            [[[60., 16., 99., 75., 4. ]]])
    """
    cudnn_version = get_cudnn_version()
    if cudnn_version is not None:
        use_cudnn = True
    else:
        use_cudnn = False

    if data_format not in ['NCL', 'NLC']:
        raise ValueError(
            "Attr(data_format) of conv2d_transpose got wrong value: "
            f"received {data_format}, but only 'NCL' or 'NLC' are supported."
        )
    channel_last = data_format == "NLC"
    channel_dim = -1 if channel_last else 1
    if len(x.shape) != 3:
        raise ValueError(
            f"Input x should be 3D tensor, but received x with the shape of {x.shape}"
        )

    num_channels = x.shape[channel_dim]

    if groups == 0:
        raise ValueError("The groups of conv1d_transpose should not be zero.")
    if num_channels % groups != 0 and (
        in_dynamic_mode() or (num_channels != -1 and groups != -1)
    ):
        raise ValueError(
            "the channel of input must be divisible by groups,"
            f"received: the channel of input is {num_channels}, the shape of input is {x.shape}"
            f", the groups is {groups}"
        )

    # update attrs
    padding, padding_algorithm = _update_padding_nd(padding, channel_last, 1)

    if len(padding) == 2:
        padding = [*padding, 0, 0]
    elif len(padding) == 1:
        padding = [*padding, 0]
    else:
        raise ValueError(
            f"The size of padding's dimension should 1 or 2. But got padding={padding}"
        )

    stride = [*convert_to_list(stride, 1, "stride"), 1]
    dilation = [*convert_to_list(dilation, 1, "dilation"), 1]

    if output_size is None:
        output_size = []
    else:
        if output_padding != 0:
            raise ValueError(
                'output_padding option is mutually exclusive with '
                'output_size'
            )
        if isinstance(output_size, (list, tuple, int)):
            output_size = [*convert_to_list(output_size, 1, 'output_size'), 1]
        else:
            raise ValueError(
                "output_size should be int, or list, tuple of ints"
            )

    if output_padding == 0:
        output_padding = []
    else:
        output_padding = [
            *convert_to_list(output_padding, 1, 'output_padding'),
            0,
        ]

    if len(output_padding) > 0 and output_padding[0] > stride[0]:
        raise ValueError(
            "The size of output_padding should not be greater than stride."
            f"But got output_padding={output_padding[0]} and stride={stride[0]}"
        )

    if len(weight.shape) != 3:
        raise ValueError(
            f'Input weight should be 3D tensor, but received weight with the shape of {weight.shape}'
        )

    op_type = 'conv2d_transpose'
    num_filters = weight.shape[1]
    if (
        num_channels == groups
        and num_channels != 1
        and num_filters == 1
        and not use_cudnn
    ):
        op_type = 'depthwise_conv2d_transpose'
        use_cudnn = False

    squeeze_axis = -2 if channel_last else -1
    conv2d_data_format = "NHWC" if channel_last else "NCHW"

    x = unsqueeze(x, axis=[squeeze_axis])
    weight = unsqueeze(weight, axis=[-1])

    if in_dynamic_or_pir_mode():
        out = getattr(_C_ops, op_type)(
            x,
            weight,
            stride,
            padding,
            output_padding,
            output_size,
            padding_algorithm,
            groups,
            dilation,
            conv2d_data_format,
        )
        if bias is not None:
            out = _add_with_axis(out, bias, axis=channel_dim)
    else:
        inputs = {'Input': [x], 'Filter': [weight]}
        attrs = {
            'output_padding': output_padding,
            'output_size': output_size,
            'strides': stride,
            'paddings': padding,
            'padding_algorithm': padding_algorithm,
            'dilations': dilation,
            'groups': groups,
            'use_cudnn': use_cudnn,
            'data_format': conv2d_data_format,
        }
        check_variable_and_dtype(
            x, 'input', ['float16', 'float32', 'float64'], 'conv2d_transpose'
        )
        helper = LayerHelper(op_type, **locals())
        dtype = helper.input_dtype(input_param_name='x')
        out = helper.create_variable_for_type_inference(dtype)
        outputs = {"Output": [out]}
        helper.append_op(
            type=op_type, inputs=inputs, outputs=outputs, attrs=attrs
        )
        if bias is not None:
            out = _add_with_axis(out, bias, axis=channel_dim)

    out = squeeze(out, axis=[squeeze_axis])
    return out


def conv2d_transpose(
    x: Tensor,
    weight: Tensor,
    bias: Tensor | None = None,
    stride: Size2 = 1,
    padding: _PaddingSizeMode | Size2 | Size4 | Sequence[Size2] = 0,
    output_padding: Size2 = 0,
    dilation: Size2 = 1,
    groups: int = 1,
    output_size: Size2 | None = None,
    data_format: DataLayout2D = 'NCHW',
    name: str | None = None,
) -> Tensor:
    r"""

    The convolution2D transpose layer calculates the output based on the input,
    filter, and dilations, strides, paddings. Input(Input) and output(Output)
    are in NCHW or NHWC format. Where N is batch size, C is the number of channels,
    H is the height of the feature, and W is the width of the feature.
    Parameters(dilations, strides, paddings) are two elements. These two elements
    represent height and width, respectively. The details of convolution transpose
    layer, please refer to the following explanation and references
    `therein <https://arxiv.org/pdf/1603.07285.pdf>`_.
    If bias attribution and activation type are provided, bias is added to
    the output of the convolution, and the corresponding activation function
    is applied to the final result.
    See more detail in :ref:`api_paddle_nn_Conv2DTranspose` .

    For each input :math:`X`, the equation is:

    ..  math::

        Out = \sigma (W \ast X + b)

    Where:

    * :math:`X`: Input value, a 4-D Tensor with NCHW or NHWC format.
    * :math:`W`: Filter value, a 4-D Tensor with MCHW format.
    * :math:`\\ast`: Convolution operation.
    * :math:`b`: Bias value, a 2-D Tensor with shape [M, 1].
    * :math:`\\sigma`: Activation function.
    * :math:`Out`: Output value, a 4-D Tensor with data format 'NCHW' or 'NHWC', the shape of :math:`Out` and :math:`X` may be different.

    Example:

        - Input:

          Input shape: :math:`(N, C_{in}, H_{in}, W_{in})`

          Filter shape: :math:`(C_{in}, C_{out}, H_f, W_f)`

        - Output:

          Output shape: :math:`(N, C_{out}, H_{out}, W_{out})`

        Where

        ..  math::

           H^\prime_{out} &= (H_{in} - 1) * strides[0] - 2 * paddings[0] + dilations[0] * (H_f - 1) + 1 \\
           W^\prime_{out} &= (W_{in} - 1) * strides[1] - 2 * paddings[1] + dilations[1] * (W_f - 1) + 1 \\
           H_{out} &\in [ H^\prime_{out}, H^\prime_{out} + strides[0] ] \\
           W_{out} &\in [ W^\prime_{out}, W^\prime_{out} + strides[1] ]

    Note:
          The conv2d_transpose can be seen as the backward of the conv2d. For conv2d,
          when stride > 1, conv2d maps multiple input shape to the same output shape,
          so for conv2d_transpose, when stride > 1, input shape maps multiple output shape.
          If output_size is None, :math:`H_{out} = H^\prime_{out}, W_{out} = W^\prime_{out}`;
          else, the :math:`H_{out}` of the output size must between :math:`H^\prime_{out}`
          and :math:`H^\prime_{out} + strides[0]`, and the :math:`W_{out}` of the output size must
          between :math:`W^\prime_{out}` and :math:`W^\prime_{out} + strides[1]`.

    Args:
        x(Tensor): 4-D Tensor with [N, C, H, W] or [N, H, W, C] format,
            whose data type is float32 or float64.
        weight(Tensor): The convolution kernel, a Tensor with shape [C, M/g, kH, kW],
            where M is the number of output channels(filters), g is the number of groups,
            kH is the height of the kernel, and kW is the width of the kernel.
        bias(Tensor, optional): The bias, a Tensor with shape [M, ].
        stride(int|list|tuple, optional): The stride size. It means the stride in transposed convolution.
            If stride is a list/tuple, it must contain two integers, (stride_height, stride_width).
            Otherwise, stride_height = stride_width = stride. Default: stride = 1.
        padding(str|int|list|tuple, optional): The padding size. It means the number of zero-paddings
            on both sides for each dimension. If `padding` is a string, either 'VALID' or
            'SAME' which is the padding algorithm. If padding size is a tuple or list,
            it could be in three forms: `[pad_height, pad_width]` or
            `[pad_height_top, pad_height_bottom, pad_width_left, pad_width_right]`,
            and when `data_format` is `"NCHW"`, `padding` can be in the form
            `[[0,0], [0,0], [pad_height_top, pad_height_bottom], [pad_width_left, pad_width_right]]`.
            when `data_format` is `"NHWC"`, `padding` can be in the form
            `[[0,0], [pad_height_top, pad_height_bottom], [pad_width_left, pad_width_right], [0,0]]`.
            Default: padding = 0.
        output_padding(int|list|tuple, optional): Additional size added to one side
            of each dimension in the output shape. Default: 0.
        groups(int, optional): The groups number of the Conv2D transpose layer. Inspired by
            grouped convolution in Alex Krizhevsky's Deep CNN paper, in which
            when group=2, the first half of the filters is only connected to the
            first half of the input channels, while the second half of the
            filters is only connected to the second half of the input channels.
            Default: groups = 1.
        dilation(int|list|tuple, optional): The dilation size. It means the spacing between the kernel points.
            If dilation is a list/tuple, it must contain two integers, (dilation_height, dilation_width).
            Otherwise, dilation_height = dilation_width = dilation. Default: dilation = 1.
        output_size(int|tuple|list, optional): The output image size. If output size is a
            tuple/list, it must contain two integers, (image_height, image_width). None if use
            filter_size(shape of weight), padding, and stride to calculate output_size.
        data_format (str, optional): Specify the data format of the input, and the data format of the output
            will be consistent with that of the input. An optional string from: `"NCHW"`, `"NHWC"`.
            The default is `"NCHW"`. When it is `"NCHW"`, the data is stored in the order of:
            `[batch_size, input_channels, input_height, input_width]`.
        name(str, optional): For detailed information, please refer
           to :ref:`api_guide_Name`. Usually name is no need to set and
           None by default.

    Returns:
        A Tensor representing the conv2d_transpose, whose
        data type is the same with input and shape is (num_batches, channels, out_h,
        out_w) or (num_batches, out_h, out_w, channels). The tensor variable storing
        transposed convolution result.

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> import paddle.nn.functional as F

            >>> x_var = paddle.randn((2, 3, 8, 8), dtype='float32')
            >>> w_var = paddle.randn((3, 6, 3, 3), dtype='float32')

            >>> y_var = F.conv2d_transpose(x_var, w_var)

            >>> print(y_var.shape)
            [2, 6, 10, 10]
    """

    if data_format not in ['NCHW', 'NHWC']:
        raise ValueError(
            "Attr(data_format) of conv2d_transpose got wrong value: "
            f"received {data_format}, but only 'NCHW' or 'NHWC' are supported."
        )
    channel_last = data_format == "NHWC"
    channel_dim = -1 if channel_last else 1
    if len(x.shape) != 4:
        raise ValueError(
            f"Input x should be 4D tensor, but received x with the shape of {x.shape}"
        )
    if len(weight.shape) != 4:
        raise ValueError(
            f"Input weight should be 4D tensor, but received weight with the shape of {weight.shape}"
        )
    num_channels = x.shape[channel_dim]

    if groups == 0:
        raise ValueError("The groups of conv2d_transpose should not be zero.")
    if num_channels % groups != 0 and (
        in_dynamic_mode() or (num_channels != -1 and groups != -1)
    ):
        raise ValueError(
            "the channel of input must be divisible by groups,"
            f"received: the channel of input is {num_channels}, the shape of input is {x.shape}"
            f", the groups is {groups}"
        )

    cudnn_version = get_cudnn_version()

    use_cudnn = (
        True
        if (is_compiled_with_cuda() and cudnn_version is not None)
        else False
    )

    # update attrs
    padding, padding_algorithm = _update_padding_nd(padding, channel_last, 2)
    stride = convert_to_list(stride, 2, 'stride')
    dilation = convert_to_list(dilation, 2, 'dilation')

    if output_size is None:
        output_size = []
    else:
        if output_padding != 0:
            raise ValueError(
                'output_padding option is mutually exclusive with '
                'output_size'
            )
        if isinstance(output_size, (list, tuple)):
            if _contain_var(output_size):
                output_size = _convert_to_tensor_list(output_size)
            else:
                output_size = convert_to_list(output_size, 2, 'output_size')
        elif isinstance(output_size, int):
            output_size = convert_to_list(output_size, 2, 'output_size')
        elif isinstance(output_size, (Variable, pir.Value)):
            check_dtype(
                output_size.dtype,
                'output_size',
                ['int32', 'int64'],
                'conv2d_transpose',
            )
            if len(output_size.shape) == 1 and (
                output_size.shape[0] == 1 or output_size.shape[0] == 2
            ):
                if output_size.shape[0] == 1:
                    output_size = [output_size, output_size]
            else:
                raise ValueError(
                    "output_size must contain one or two integers."
                )
        else:
            raise ValueError(
                "output_size should be int or Tensor or list, tuple of ints or Tensor"
            )

    if output_padding == 0:
        output_padding = []
    else:
        output_padding = convert_to_list(output_padding, 2, 'output_padding')

    op_type = 'conv2d_transpose'
    num_filters = weight.shape[1]
    if num_channels == groups and num_channels != 1 and num_filters == 1:
        op_type = 'depthwise_conv2d_transpose'
        use_cudnn = False

    if in_dynamic_or_pir_mode():
        op = (
            _C_ops.conv2d_transpose
            if op_type == 'conv2d_transpose'
            else _C_ops.depthwise_conv2d_transpose
        )
        pre_bias = op(
            x,
            weight,
            stride,
            padding,
            output_padding,
            output_size,
            padding_algorithm,
            groups,
            dilation,
            data_format,
        )
        if bias is not None:
            return _add_with_axis(pre_bias, bias, axis=channel_dim)
        else:
            return pre_bias
    else:
        inputs = {'Input': [x], 'Filter': [weight]}
        attrs = {
            'output_padding': output_padding,
            'output_size': output_size,
            'strides': stride,
            'paddings': padding,
            'padding_algorithm': padding_algorithm,
            'dilations': dilation,
            'groups': groups,
            'use_cudnn': use_cudnn,
            'data_format': data_format,
        }
        check_variable_and_dtype(
            x,
            'x',
            ['float16', 'uint16', 'float32', 'float64'],
            'conv2d_transpose',
        )
        helper = LayerHelper(op_type, **locals())
        pre_bias = helper.create_variable_for_type_inference(x.dtype)
        outputs = {"Output": [pre_bias]}
        helper.append_op(
            type=op_type, inputs=inputs, outputs=outputs, attrs=attrs
        )

        if bias is not None:
            out = helper.create_variable_for_type_inference(x.dtype)
            x_shape = list(pre_bias.shape)
            y_shape = list(bias.shape)
            if channel_dim == -1 or len(x_shape) == len(y_shape):
                helper.append_op(
                    type='elementwise_add',
                    inputs={'X': [pre_bias], 'Y': [bias]},
                    outputs={'Out': [out]},
                    attrs={'axis': -1},
                )
            else:
                assert len(x_shape) > len(
                    y_shape
                ), 'The length of pre_bias must greater than the length of bias'
                padding = len(x_shape) - len(y_shape) - channel_dim
                bias = reshape(
                    bias, [1] * channel_dim + y_shape + [1] * padding
                )
                helper.append_op(
                    type='elementwise_add',
                    inputs={'X': [pre_bias], 'Y': [bias]},
                    outputs={'Out': [out]},
                    attrs={'axis': -1},
                )
        else:
            out = pre_bias

    return out


def conv3d(
    x: Tensor,
    weight: Tensor,
    bias: Tensor | None = None,
    stride: Size3 = 1,
    padding: _PaddingSizeMode | Size3 | Size6 | Sequence[Size2] = 0,
    dilation: Size3 = 1,
    groups: int = 1,
    data_format: DataLayout3D = "NCDHW",
    name: str | None = None,
) -> Tensor:
    r"""

    The convolution3D layer calculates the output based on the input, filter
    and strides, paddings, dilations, groups parameters. Input(Input) and
    Output(Output) are in NCDHW or NDHWC format. Where N is batch size C is the number of
    channels, D is the depth of the feature, H is the height of the feature,
    and W is the width of the feature. Convolution3D is similar with Convolution2D
    but adds one dimension(depth). If bias attribution and activation type are
    provided, bias is added to the output of the convolution, and the
    corresponding activation function is applied to the final result.

    For each input :math:`X`, the equation is:

    ..  math::

        Out = \sigma (W \ast X + b)

    In the above equation:

    * :math:`X`: Input value, a tensor with NCDHW or NDHWC format.
    * :math:`W`: Filter value, a tensor with MCDHW format.
    * :math:`\\ast`: Convolution operation.
    * :math:`b`: Bias value, a 2-D tensor with shape [M, 1].
    * :math:`\\sigma`: Activation function.
    * :math:`Out`: Output value, the shape of :math:`Out` and :math:`X` may be different.

    Example:

        - Input:

          Input shape: :math:`(N, C_{in}, D_{in}, H_{in}, W_{in})`

          Filter shape: :math:`(C_{out}, C_{in}, D_f, H_f, W_f)`

        - Output:
          Output shape: :math:`(N, C_{out}, D_{out}, H_{out}, W_{out})`

        Where

        ..  math::

            D_{out}&= \frac{(D_{in} + 2 * paddings[0] - (dilations[0] * (D_f - 1) + 1))}{strides[0]} + 1 \\
            H_{out}&= \frac{(H_{in} + 2 * paddings[1] - (dilations[1] * (H_f - 1) + 1))}{strides[1]} + 1 \\
            W_{out}&= \frac{(W_{in} + 2 * paddings[2] - (dilations[2] * (W_f - 1) + 1))}{strides[2]} + 1

    Args:
        x (Tensor): The input is 5-D Tensor with shape [N, C, D, H, W], the data
            type of input is float16 or float32 or float64.
        weight (Tensor): The convolution kernel, a Tensor with shape [M, C/g, kD, kH, kW],
            where M is the number of filters(output channels), g is the number of groups,
            kD, kH, kW are the filter's depth, height and width respectively.
        bias (Tensor, optional): The bias, a Tensor of shape [M, ].
        stride (int|list|tuple, optional): The stride size. It means the stride in convolution. If stride is a
            list/tuple, it must contain three integers, (stride_depth, stride_height, stride_width).
            Otherwise, stride_depth = stride_height = stride_width = stride. Default: stride = 1.
        padding (string|int|list|tuple, optional): The padding size. It means the number of zero-paddings
            on both sides for each dimension. If `padding` is a string, either 'VALID' or
            'SAME' which is the padding algorithm. If padding size is a tuple or list,
            it could be in three forms: `[pad_depth, pad_height, pad_width]` or
            `[pad_depth_front, pad_depth_back, pad_height_top, pad_height_bottom, pad_width_left, pad_width_right]`,
            and when `data_format` is `"NCDHW"`, `padding` can be in the form
            `[[0,0], [0,0], [pad_depth_front, pad_depth_back], [pad_height_top, pad_height_bottom], [pad_width_left, pad_width_right]]`.
            when `data_format` is `"NDHWC"`, `padding` can be in the form
            `[[0,0], [pad_depth_front, pad_depth_back], [pad_height_top, pad_height_bottom], [pad_width_left, pad_width_right], [0,0]]`.
            Default: padding = 0.
        dilation (int|list|tuple, optional): The dilation size. It means the spacing between the kernel points.
            If dilation is a list/tuple, it must contain three integers, (dilation_depth, dilation_height,
            dilation_width). Otherwise, dilation_depth = dilation_height = dilation_width = dilation.
            Default: dilation = 1.
        groups (int, optional): The groups number of the Conv3D Layer. According to grouped
            convolution in Alex Krizhevsky's Deep CNN paper: when group=2,
            the first half of the filters is only connected to the first half
            of the input channels, while the second half of the filters is only
            connected to the second half of the input channels. Default: groups=1
        data_format (str, optional): Specify the data format of the input, and the data format of the output
            will be consistent with that of the input. An optional string from: `"NCDHW"`, `"NDHWC"`.
            The default is `"NCDHW"`. When it is `"NCDHW"`, the data is stored in the order of:
            `[batch_size, input_channels, input_depth, input_height, input_width]`.
        name(str|None, optional): For detailed information, please refer
           to :ref:`api_guide_Name`. Usually name is no need to set and
           None by default.

    Returns:
        A Tensor representing the conv3d, whose data type is
        the same with input. If act is None, the tensor storing the
        convolution result, and if act is not None, the tensor storing
        convolution and non-linearity activation result.

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> import paddle.nn.functional as F

            >>> x_var = paddle.randn((2, 3, 8, 8, 8), dtype='float32')
            >>> w_var = paddle.randn((6, 3, 3, 3, 3), dtype='float32')

            >>> y_var = F.conv3d(x_var, w_var)

            >>> print(y_var.shape)
            [2, 6, 6, 6, 6]
    """
    # entry check
    if data_format not in ["NCDHW", "NDHWC"]:
        raise ValueError(
            "Attr(data_format) should be 'NCDHW' or 'NDHWC'. Received "
            f"Attr(data_format): {data_format}."
        )

    channel_last = data_format == "NDHWC"
    channel_dim = -1 if channel_last else 1
    if len(x.shape) != 5:
        raise ValueError(
            f"Input x should be 5D tensor, but received x with the shape of {x.shape}"
        )
    num_channels = x.shape[channel_dim]
    num_filters = weight.shape[0]

    if groups == 0:
        raise ValueError("The groups of conv3d should not be 0.")
    if num_channels % groups != 0 and (
        in_dynamic_mode() or (num_channels != -1 and groups != -1)
    ):
        raise ValueError(
            "The number of input channels must be divisible by Attr(groups). "
            f"Received: number of channels({num_channels}), groups({groups})."
        )
    if num_filters % groups != 0 and (
        in_dynamic_mode() or (num_filters != -1 and groups != -1)
    ):
        raise ValueError(
            "The number of filters must be divisible by Attr(groups). "
            f"Received: number of filters({num_filters}), groups({groups})."
        )

    cudnn_version = get_cudnn_version()
    use_cudnn = (
        True
        if (is_compiled_with_cuda() and cudnn_version is not None)
        else False
    )

    padding, padding_algorithm = _update_padding_nd(padding, channel_last, 3)
    stride = convert_to_list(stride, 3, 'stride')
    dilation = convert_to_list(dilation, 3, 'dilation')
    op_type = "conv3d"

    return _conv_nd(
        x,
        weight,
        bias,
        stride,
        padding,
        padding_algorithm,
        dilation,
        groups,
        data_format,
        channel_dim,
        op_type,
        use_cudnn,
        name,
    )


def conv3d_transpose(
    x: Tensor,
    weight: Tensor,
    bias: Tensor | None = None,
    stride: Size3 = 1,
    padding: _PaddingSizeMode | Size3 | Size6 | Sequence[Size2] = 0,
    output_padding: Size3 = 0,
    groups: int = 1,
    dilation: Size3 = 1,
    output_size: Size3 | None = None,
    data_format: DataLayout3D = 'NCDHW',
    name: str | None = None,
) -> Tensor:
    r"""
    The convolution3d transpose layer calculates the output based on the input,
    filter, and dilations, strides, paddings. Input(Input) and output(Output)
    are in NCDHW or NDHWC format. Where N is batch size, C is the number of channels,
    D is the depth of the feature, H is the height of the feature, and W
    is the width of the feature. Parameters(dilations, strides, paddings) are
    two elements. These two elements represent height and width, respectively.
    The details of convolution transpose layer, please refer to the following
    explanation and references `therein <https://arxiv.org/pdf/1603.07285.pdf>`_.
    If bias attribution and activation type are provided, bias is added to
    the output of the convolution, and the corresponding activation function
    is applied to the final result.
    See more detail in :ref:`api_paddle_nn_Conv3DTranspose` .

    For each input :math:`X`, the equation is:

    ..  math::

        Out = \sigma (W \ast X + b)

    In the above equation:

    * :math:`X`: Input value, a Tensor with NCDHW or NDHWC format.
    * :math:`W`: Filter value, a Tensor with NCDHW format.
    * :math:`\ast`: Convolution operation.
    * :math:`b`: Bias value, a 2-D Tensor with shape [M, 1].
    * :math:`\sigma`: Activation function.
    * :math:`Out`: Output value, the shape of :math:`Out` and :math:`X` may be different.

    Example:

        - Input:

          Input shape: :math:`(N, C_{in}, D_{in}, H_{in}, W_{in})`

          Filter shape: :math:`(C_{in}, C_{out}, D_f, H_f, W_f)`

        - Output:

          Output shape: :math:`(N, C_{out}, D_{out}, H_{out}, W_{out})`

        Where

        ..  math::

           D^\prime_{out} &= (D_{in} - 1) * strides[0] - 2 * paddings[0] + dilations[0] * (D_f - 1) + 1 \\
           H^\prime_{out} &= (H_{in} - 1) * strides[1] - 2 * paddings[1] + dilations[1] * (H_f - 1) + 1 \\
           W^\prime_{out} &= (W_{in} - 1) * strides[2] - 2 * paddings[2] + dilations[2] * (W_f - 1) + 1 \\
           D_{out} &\in [ D^\prime_{out}, D^\prime_{out} + strides[0] ] \\
           H_{out} &\in [ H^\prime_{out}, H^\prime_{out} + strides[1] ] \\
           W_{out} &\in [ W^\prime_{out}, W^\prime_{out} + strides[2] ]

    Note:
        The conv3d_transpose can be seen as the backward of the conv3d. For conv3d,
        when stride > 1, conv3d maps multiple input shape to the same output shape,
        so for conv3d_transpose, when stride > 1, input shape maps multiple output shape.
        If output_size is None, :math:`H_{out} = H^\prime_{out}, W_{out} = W^\prime_{out}`;
        else, the :math:`D_{out}` of the output size must between :math:`D^\prime_{out}` and
        :math:`D^\prime_{out} + strides[0]`, the :math:`H_{out}` of the output size must
        between :math:`H^\prime_{out}` and :math:`H^\prime_{out} + strides[1]`, and the
        :math:`W_{out}` of the output size must between :math:`W^\prime_{out}` and
        :math:`W^\prime_{out} + strides[2]`.

    Args:
        x (Tensor): The input is 5-D Tensor with shape [N, C, D, H, W] or [N, D, H, W, C], the data type
            of input is float32 or float64.
        weight (Tensor): The convolution kernel, a Tensor with shape [C, M/g, kD, kH, kW],
            where M is the number of filters (output channels), g is the number of groups,
            kD, kH, kW are the filter's depth, height and width respectively.
        bias (Tensor, optional): The bias, a Tensor of shape [M, ]. Default: None.
        stride (int|list|tuple, optional): The stride size. It means the stride in transposed convolution.
            If stride is a list/tuple, it must contain three integers, (stride_depth, stride_height,
            stride_width). Otherwise, stride_depth = stride_height = stride_width = stride.
            Default: 1.
        padding (str|int|list|tuple, optional): The padding size. It means the number of zero-paddings
            on both sides for each dimension. If `padding` is a string, either 'VALID' or
            'SAME' which is the padding algorithm. If padding size is a tuple or list,
            it could be in three forms: `[pad_depth, pad_height, pad_width]` or
            `[pad_depth_front, pad_depth_back, pad_height_top, pad_height_bottom, pad_width_left, pad_width_right]`,
            and when `data_format` is `"NCDHW"`, `padding` can be in the form
            `[[0,0], [0,0], [pad_depth_front, pad_depth_back], [pad_height_top, pad_height_bottom], [pad_width_left, pad_width_right]]`.
            when `data_format` is `"NDHWC"`, `padding` can be in the form
            `[[0,0], [pad_depth_front, pad_depth_back], [pad_height_top, pad_height_bottom], [pad_width_left, pad_width_right], [0,0]]`.
            Default: 0.
        output_padding (int|list|tuple, optional): Additional size added to one side
            of each dimension in the output shape. Default: 0.
        groups (int, optional): The groups number of the Conv3D transpose layer. Inspired by
            grouped convolution in `Alex Krizhevsky's Deep CNN paper <https://papers.nips.cc/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf>`_, in which
            when groups = 2, the first half of the filters is only connected to the
            first half of the input channels, while the second half of the
            filters is only connected to the second half of the input channels.
            Default: 1.
        dilation (int|list|tuple, optional): The dilation size. It means the spacing between the kernel points.
            If dilation is a list/tuple, it must contain three integers, (dilation_depth, dilation_height,
            dilation_width). Otherwise, dilation_depth = dilation_height = dilation_width = dilation.
            Default: 1.
        output_size (int|list|tuple, optional): The output image size. If output size is a
            list/tuple, it must contain three integers, (image_depth, image_height, image_width).
            None if use filter_size(shape of weight), padding, and stride to calculate output_size.
        data_format (str, optional): Specify the data format of the input, and the data format of the output
            will be consistent with that of the input. An optional string from: `"NCHW"`, `"NHWC"`.
            When it is `"NCHW"`, the data is stored in the order of: `[batch_size, input_channels, input_height, input_width]`.
            Default: `"NCHW"`.
        name (str, optional): For detailed information, please refer
           to :ref:`api_guide_Name`. Usually name is no need to set.
           Default: None.

    Returns:
        A Tensor representing the conv3d_transpose, whose data
        type is the same with input and shape is (num_batches, channels, out_d, out_h,
        out_w) or (num_batches, out_d, out_h, out_w, channels). If act is None, the tensor
        variable storing the transposed convolution result, and if act is not None, the tensor
        variable storing transposed convolution and non-linearity activation result.

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> import paddle.nn.functional as F

            >>> x_var = paddle.randn((2, 3, 8, 8, 8), dtype='float32')
            >>> w_var = paddle.randn((3, 6, 3, 3, 3), dtype='float32')

            >>> y_var = F.conv3d_transpose(x_var, w_var)

            >>> print(y_var.shape)
            [2, 6, 10, 10, 10]
    """
    # entry checks
    if data_format not in ["NCDHW", "NDHWC"]:
        raise ValueError(
            "Attr(data_format) should be 'NCDHW' or 'NDHWC'. Received "
            f"Attr(data_format): {data_format}."
        )

    channel_last = data_format == "NDHWC"
    channel_dim = -1 if channel_last else 1
    if len(x.shape) != 5:
        raise ValueError(
            f"Input x should be 5D tensor, but received x with the shape of {x.shape}"
        )
    if len(weight.shape) != 5:
        raise ValueError(
            f"Input weight should be 5D tensor, but received weight with the shape of {weight.shape}"
        )
    num_channels = x.shape[channel_dim]

    if groups == 0:
        raise ValueError("The groups of conv3d_transpose should not be zero.")
    if num_channels % groups != 0 and (
        in_dynamic_mode() or (num_channels != -1 and groups != -1)
    ):
        raise ValueError(
            "The number of input channels must be divisible by Attr(groups). "
            f"Received: number of channels({num_channels}), groups({groups})."
        )

    padding, padding_algorithm = _update_padding_nd(padding, channel_last, 3)
    stride = convert_to_list(stride, 3, 'stride')
    dilation = convert_to_list(dilation, 3, 'dilation')
    if output_size is None:
        output_size = []
    else:
        if output_padding != 0:
            raise ValueError(
                'output_padding option is mutually exclusive with '
                'output_size'
            )
        if isinstance(output_size, (list, tuple, int)):
            output_size = convert_to_list(output_size, 3, 'output_size')
        else:
            raise ValueError(
                "output_size should be int, or list, tuple of ints"
            )

    if output_padding == 0:
        output_padding = []
    else:
        output_padding = convert_to_list(output_padding, 3, 'output_padding')

    cudnn_version = get_cudnn_version()

    # TODO(LielinJiang): whether to use cudnn according to the version of cudnn
    use_cudnn = (
        True
        if (is_compiled_with_cuda() and cudnn_version is not None)
        else False
    )

    op_type = 'conv3d_transpose'
    data_format_ = "NHWC" if channel_last else "NCHW"

    if in_dynamic_or_pir_mode():
        pre_bias = _C_ops.conv3d_transpose(
            x,
            weight,
            stride,
            padding,
            output_padding,
            output_size,
            padding_algorithm,
            groups,
            dilation,
            data_format_,
        )
        if bias is not None:
            return _add_with_axis(pre_bias, bias, axis=channel_dim)
        else:
            return pre_bias
    else:
        inputs = {'Input': [x], 'Filter': [weight]}
        attrs = {
            'output_padding': output_padding,
            'output_size': output_size,
            'paddings': padding,
            "padding_algorithm": padding_algorithm,
            'strides': stride,
            'dilations': dilation,
            'groups': groups,
            'use_cudnn': use_cudnn,
            "data_format": data_format_,
        }
        helper = LayerHelper(op_type, **locals())
        check_variable_and_dtype(
            x, 'x', ['float16', 'float32', 'float64'], 'conv3d'
        )

        pre_bias = helper.create_variable_for_type_inference(x.dtype)
        outputs = {"Output": [pre_bias]}

        helper.append_op(
            type=op_type, inputs=inputs, outputs=outputs, attrs=attrs
        )
        if bias is not None:
            out = _add_with_axis(pre_bias, bias, axis=channel_dim)
        else:
            out = pre_bias

    return out
