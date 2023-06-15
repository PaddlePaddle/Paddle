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

__all__ = []

import paddle
from paddle import _C_ops, in_dynamic_mode
from paddle.fluid.layer_helper import LayerHelper
from paddle.nn.functional.conv import _update_padding_nd
from paddle.utils import convert_to_list

from ...binary import add
from ...unary import reshape


def _conv3d(
    x,
    weight,
    bias=None,
    stride=1,
    padding=0,
    dilation=1,
    groups=1,
    subm=False,
    key=None,
    data_format="NDHWC",
    name=None,
):
    assert groups == 1, "Currently, only support groups=1"

    dims = 3

    # Currently, only support 'NDHWC'
    if data_format not in ["NDHWC"]:
        raise ValueError(
            "Attr(data_format) should be 'NDHWC'. Received "
            "Attr(data_format): {}.".format(data_format)
        )
    if len(x.shape) != 5:
        raise ValueError(
            "Input x should be 5D tensor, but received x with the shape of {}".format(
                x.shape
            )
        )

    channel_last = data_format == "NDHWC"
    channel_dim = -1 if channel_last else 1
    if len(x.shape) != 5:
        raise ValueError(
            "Input x should be 5D tensor, but received x with the shape of {}".format(
                x.shape
            )
        )
    num_channels = x.shape[channel_dim]
    if num_channels < 0:
        raise ValueError(
            "The channel dimension of the input({}) should be defined. "
            "Received: {}.".format(x.shape, num_channels)
        )

    padding, padding_algorithm = _update_padding_nd(padding, channel_last, dims)
    stride = convert_to_list(stride, dims, 'stride')
    dilation = convert_to_list(dilation, dims, 'dilation')

    if in_dynamic_mode():
        pre_bias = _C_ops.sparse_conv3d(
            x,
            weight,
            padding,
            dilation,
            stride,
            groups,
            subm,
            key if key is not None else "",
        )
        if bias is not None:
            return add(pre_bias, bias)
        else:
            return pre_bias
    else:
        inputs = {'x': x, 'kernel': weight}
        attrs = {
            'paddings': padding,
            'dilations': dilation,
            'strides': stride,
            'groups': groups,
            'subm': subm,
            'key': key,
        }
        op_type = 'sparse_conv3d'
        helper = LayerHelper(op_type, **locals())
        rulebook = helper.create_variable_for_type_inference(
            dtype='int32', stop_gradient=True
        )
        counter = helper.create_variable_for_type_inference(
            dtype='int32', stop_gradient=True
        )
        pre_bias = helper.create_sparse_variable_for_type_inference(x.dtype)
        outputs = {"out": pre_bias, "rulebook": rulebook, "counter": counter}
        helper.append_op(
            type=op_type, inputs=inputs, outputs=outputs, attrs=attrs
        )
        if bias is not None:
            return add(pre_bias, bias)
        else:
            return pre_bias


def _conv2d(
    x,
    weight,
    bias=None,
    stride=1,
    padding=0,
    dilation=1,
    groups=1,
    subm=False,
    key=None,
    data_format="NHWC",
    name=None,
):
    assert groups == 1, "Currently, only support groups=1"

    dims = 2

    # Currently, only support 'NHWC'
    if data_format not in ["NHWC"]:
        raise ValueError(
            "Attr(data_format) should be 'NHWC'. Received "
            "Attr(data_format): {}.".format(data_format)
        )
    if len(x.shape) != 4:
        raise ValueError(
            "Input x should be 4D tensor, but received x with the shape of {}".format(
                x.shape
            )
        )

    channel_last = data_format == "NHWC"
    n_dim = 0
    channel_dim = -1 if channel_last else 1
    h_dim = 1 if channel_last else 2
    w_dim = 2 if channel_last else -1
    if len(x.shape) != 4:
        raise ValueError(
            "Input x should be 4D tensor, but received x with the shape of {}".format(
                x.shape
            )
        )
    n = x.shape[n_dim]
    d = 1
    h = x.shape[h_dim]
    w = x.shape[w_dim]
    num_channels = x.shape[channel_dim]
    if num_channels < 0:
        raise ValueError(
            "The channel dimension of the input({}) should be defined. "
            "Received: {}.".format(x.shape, num_channels)
        )

    padding, padding_algorithm = _update_padding_nd(padding, channel_last, dims)
    stride = convert_to_list(stride, dims, 'stride')
    dilation = convert_to_list(dilation, dims, 'dilation')

    padding.insert(0, 0)
    stride.insert(0, 1)
    dilation.insert(0, 1)

    x = reshape(x, [n, d, h, w, num_channels])
    h_filter = weight.shape[0]
    w_filter = weight.shape[1]
    c_filter = weight.shape[2]
    m_filter = weight.shape[3]
    weight = paddle.reshape(weight, [d, h_filter, w_filter, c_filter, m_filter])
    if in_dynamic_mode():
        pre_bias = _C_ops.sparse_conv3d(
            x,
            weight,
            padding,
            dilation,
            stride,
            groups,
            subm,
            key if key is not None else "",
        )
        x = reshape(x, [n, h, w, -1])
        weight = paddle.reshape(
            weight, [h_filter, w_filter, c_filter, m_filter]
        )
        n_out = pre_bias.shape[0]
        h_out = pre_bias.shape[2]
        w_out = pre_bias.shape[3]
        channels_out = pre_bias.shape[4]
        pre_bias = reshape(pre_bias, [n_out, h_out, w_out, channels_out])
        if bias is not None:
            return add(pre_bias, bias)
        else:
            return pre_bias
    else:
        raise ValueError("Only support dynamic_mode now.")


def conv3d(
    x,
    weight,
    bias=None,
    stride=1,
    padding=0,
    dilation=1,
    groups=1,
    data_format="NDHWC",
    name=None,
):
    r"""

    The sparse convolution3d functional calculates the output based on the input, filter
    and strides, paddings, dilations, groups parameters. Input(Input) and
    Output(Output) are multidimensional SparseCooTensors with a shape of
    :math:`[N, D, H, W, C]` . Where N is batch size, C is the number of
    channels, D is the depth of the feature, H is the height of the feature,
    and W is the width of the feature. If bias attribution is provided,
    bias is added to the output of the convolution.

    For each input :math:`X`, the equation is:

    ..  math::

        Out = \sigma (W \ast X + b)

    In the above equation:

    * :math:`X`: Input value, a tensor with NCDHW or NDHWC format.
    * :math:`W`: Filter value, a tensor with MCDHW format.
    * :math:`\\ast`: Convolution operation.
    * :math:`b`: Bias value, a 1-D tensor with shape [M].
    * :math:`Out`: Output value, the shape of :math:`Out` and :math:`X` may be different.

    Args:
        x (Tensor): The input is 5-D SparseCooTensor with shape [N, D, H, W, C], the data
            type of input is float16 or float32 or float64.
        weight (Tensor): The convolution kernel, a Tensor with shape [kD, kH, kW, C/g, M],
            where M is the number of filters(output channels), g is the number of groups,
            kD, kH, kW are the filter's depth, height and width respectively.
        bias (Tensor, optional): The bias, a Tensor of shape [M].
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
            connected to the second half of the input channels. Default: groups=1. Currently, only support groups=1.
        data_format (str, optional): Specify the data format of the input, and the data format of the output
            will be consistent with that of the input. An optional string from: `"NCDHW"`, `"NDHWC"`.
            The default is `"NDHWC"`. When it is `"NDHWC"`, the data is stored in the order of:
            `[batch_size, input_depth, input_height, input_width, input_channels]`.
        name(str, optional): For detailed information, please refer
           to :ref:`api_guide_Name`. Usually name is no need to set and
           None by default.

    Returns:
        A SparseCooTensor representing the conv3d, whose data type is the same with input.

    Examples:
        .. code-block:: python

            import paddle

            indices = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 1, 2], [1, 3, 2, 3]]
            values = [[1], [2], [3], [4]]
            indices = paddle.to_tensor(indices, dtype='int32')
            values = paddle.to_tensor(values, dtype='float32')
            dense_shape = [1, 1, 3, 4, 1]
            sparse_x = paddle.sparse.sparse_coo_tensor(indices, values, dense_shape, stop_gradient=True)
            weight = paddle.randn((1, 3, 3, 1, 1), dtype='float32')
            y = paddle.sparse.nn.functional.conv3d(sparse_x, weight)
            print(y.shape)
            # (1, 1, 1, 2, 1)
    """
    return _conv3d(
        x,
        weight,
        bias,
        stride,
        padding,
        dilation,
        groups,
        False,
        None,
        data_format,
        name,
    )


def subm_conv3d(
    x,
    weight,
    bias=None,
    stride=1,
    padding=0,
    dilation=1,
    groups=1,
    data_format="NDHWC",
    key=None,
    name=None,
):
    r"""

    The sparse submanifold convolution3d functional calculates the output based on the input, filter
    and strides, paddings, dilations, groups parameters. Input(Input) and
    Output(Output) are multidimensional SparseCooTensors with a shape of
    :math:`[N, D, H, W, C]` . Where N is batch size, C is the number of
    channels, D is the depth of the feature, H is the height of the feature,
    and W is the width of the feature. If bias attribution is provided,
    bias is added to the output of the convolution.

    For each input :math:`X`, the equation is:

    ..  math::

        Out = W \ast X + b

    In the above equation:

    * :math:`X`: Input value, a tensor with NCDHW or NDHWC format.
    * :math:`W`: Filter value, a tensor with DHWCM format.
    * :math:`\\ast`: Submanifold Convolution operation, refer to the paper: https://arxiv.org/abs/1706.01307.
    * :math:`b`: Bias value, a 1-D tensor with shape [M].
    * :math:`Out`: Output value, the shape of :math:`Out` and :math:`X` may be different.

    Args:
        x (Tensor): The input is 5-D SparseCooTensor with shape [N, D, H, W, C], the data
            type of input is float16 or float32 or float64.
        weight (Tensor): The convolution kernel, a Tensor with shape [kD, kH, kW, C/g, M],
            where M is the number of filters(output channels), g is the number of groups,
            kD, kH, kW are the filter's depth, height and width respectively.
        bias (Tensor, optional): The bias, a Tensor of shape [M].
        stride (int|list|tuple, optional): The stride size. It means the stride in convolution. If stride is a
            list/tuple, it must contain three integers, (stride_depth, stride_height, stride_width).
            Otherwise, stride_depth = stride_height = stride_width = stride. Default: stride = 1.
        padding (string|int|list|tuple): The padding size. It means the number of zero-paddings
            on both sides for each dimension. If `padding` is a string, either 'VALID' or
            'SAME' which is the padding algorithm. If padding size is a tuple or list,
            it could be in three forms: `[pad_depth, pad_height, pad_width]` or
            `[pad_depth_front, pad_depth_back, pad_height_top, pad_height_bottom, pad_width_left, pad_width_right]`,
            and when `data_format` is `"NCDHW"`, `padding` can be in the form
            `[[0,0], [0,0], [pad_depth_front, pad_depth_back], [pad_height_top, pad_height_bottom], [pad_width_left, pad_width_right]]`.
            when `data_format` is `"NHWC"`, `padding` can be in the form
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
            connected to the second half of the input channels. Currently, only support groups=1.
        data_format (str, optional): Specify the data format of the input, and the data format of the output
            will be consistent with that of the input. An optional string from: `"NCDHW"`, `"NDHWC"`.
            The default is `"NDHWC"`. When it is `"NDHWC"`, the data is stored in the order of:
            `[batch_size, input_depth, input_height, input_width, input_channels]`.
        key(str, optional): the key is used to save or use the same rulebook,
            the definition and role of rulebook refers to
            https://pdfs.semanticscholar.org/5125/a16039cabc6320c908a4764f32596e018ad3.pdf. The
            default value is None.
        name(str, optional): For detailed information, please refer
           to :ref:`api_guide_Name`. Usually name is no need to set and
           None by default.

    Returns:
        A SparseCooTensor representing the conv3d, whose data type is
        the same with input.

    Examples:
        .. code-block:: python

            import paddle

            indices = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 1, 2], [1, 3, 2, 3]]
            values = [[1], [2], [3], [4]]
            indices = paddle.to_tensor(indices, dtype='int32')
            values = paddle.to_tensor(values, dtype='float32')
            dense_shape = [1, 1, 3, 4, 1]
            sparse_x = paddle.sparse.sparse_coo_tensor(indices, values, dense_shape, stop_gradient=True)
            weight = paddle.randn((1, 3, 3, 1, 1), dtype='float32')
            y = paddle.sparse.nn.functional.subm_conv3d(sparse_x, weight)
            print(y.shape)
            #(1, 1, 3, 4, 1)
    """
    return _conv3d(
        x,
        weight,
        bias,
        stride,
        padding,
        dilation,
        groups,
        True,
        key,
        data_format,
        name,
    )


def conv2d(
    x,
    weight,
    bias=None,
    stride=1,
    padding=0,
    dilation=1,
    groups=1,
    data_format="NHWC",
    name=None,
):
    r"""

    The sparse convolution2d functional calculates the output based on the input, filter
    and strides, paddings, dilations, groups parameters. Input(Input) and
    Output(Output) are multidimensional SparseCooTensors with a shape of
    :math:`[N, H, W, C]` . Where N is batch size, C is the number of
    channels, H is the height of the feature,
    and W is the width of the feature. If bias attribution is provided,
    bias is added to the output of the convolution.

    For each input :math:`X`, the equation is:

    ..  math::

        Out = \sigma (W \ast X + b)

    In the above equation:

    * :math:`X`: Input value, a tensor with NHWC format.
    * :math:`W`: Filter value, a tensor with HWCM format.
    * :math:`\\ast`: Convolution operation.
    * :math:`b`: Bias value, a 1-D tensor with shape [M].
    * :math:`Out`: Output value, the shape of :math:`Out` and :math:`X` may be different.

    Args:
        x (Tensor): The input is 4-D SparseCooTensor with shape [N, H, W, C], the data
            type of input is float16 or float32 or float64.
        weight (Tensor): The convolution kernel, a Tensor with shape [kH, kW, C/g, M],
            where M is the number of filters(output channels), g is the number of groups,
            kD, kH, kW are the filter's height and width respectively.
        bias (Tensor, optional): The bias, a Tensor of shape [M].
        stride (int|list|tuple, optional): The stride size. It means the stride in convolution. If stride is a
            list/tuple, it must contain two integers, (stride_height, stride_width).
            Otherwise, stride_height = stride_width = stride. Default: stride = 1.
        padding (string|int|list|tuple, optional): The padding size. It means the number of zero-paddings
            on both sides for each dimension. If `padding` is a string, either 'VALID' or
            'SAME' which is the padding algorithm. If padding size is a tuple or list,
            it could be in three forms: `[pad_height, pad_width]` or
            `[pad_height_top, pad_height_bottom, pad_width_left, pad_width_right]`,
            when `data_format` is `"NHWC"`, `padding` can be in the form
            `[[0,0], [pad_height_top, pad_height_bottom], [pad_width_left, pad_width_right], [0,0]]`.
            Default: padding = 0.
        dilation (int|list|tuple, optional): The dilation size. It means the spacing between the kernel points.
            If dilation is a list/tuple, it must contain two integers, (dilation_height,
            dilation_width). Otherwise, dilation_height = dilation_width = dilation.
            Default: dilation = 1.
        groups (int, optional): The groups number of the Conv2D Layer. According to grouped
            convolution in Alex Krizhevsky's Deep CNN paper: when group=2,
            the first half of the filters is only connected to the first half
            of the input channels, while the second half of the filters is only
            connected to the second half of the input channels. Default: groups=1. Currently, only support groups=1.
        data_format (str, optional): Specify the data format of the input, and the data format of the output
            will be consistent with that of the input. An optional string from: `"NHWC"`.
            The default is `"NHWC"`. When it is `"NHWC"`, the data is stored in the order of:
            `[batch_size, input_height, input_width, input_channels]`.
        name(str, optional): For detailed information, please refer
           to :ref:`api_guide_Name`. Usually name is no need to set and
           None by default.

    Returns:
        A SparseCooTensor representing the conv2d, whose data type is the same with input.

    Examples:
        .. code-block:: python

            import paddle

            indices = [[0, 0, 0, 0], [0, 0, 1, 2], [1, 3, 2, 3]]
            values = [[1], [2], [3], [4]]
            indices = paddle.to_tensor(indices, dtype='int32')
            values = paddle.to_tensor(values, dtype='float32')
            dense_shape = [1, 3, 4, 1]
            sparse_x = paddle.sparse.sparse_coo_tensor(indices, values, dense_shape, stop_gradient=True)
            weight = paddle.randn((3, 3, 1, 1), dtype='float32')
            y = paddle.sparse.nn.functional.conv2d(sparse_x, weight)
            print(y.shape)
            # (1, 1, 2, 1)
    """
    return _conv2d(
        x,
        weight,
        bias,
        stride,
        padding,
        dilation,
        groups,
        False,
        None,
        data_format,
        name,
    )


def subm_conv2d(
    x,
    weight,
    bias=None,
    stride=1,
    padding=0,
    dilation=1,
    groups=1,
    data_format="NHWC",
    key=None,
    name=None,
):
    r"""

    The sparse submanifold convolution2d functional calculates the output based on the input, filter
    and strides, paddings, dilations, groups parameters. Input(Input) and
    Output(Output) are multidimensional SparseCooTensors with a shape of
    :math:`[N, H, W, C]` . Where N is batch size, C is the number of
    channels, H is the height of the feature,
    and W is the width of the feature. If bias attribution is provided,
    bias is added to the output of the convolution.

    For each input :math:`X`, the equation is:

    ..  math::

        Out = \sigma (W \ast X + b)

    In the above equation:

    * :math:`X`: Input value, a tensor with NHWC format.
    * :math:`W`: Filter value, a tensor with HWCM format.
    * :math:`\\ast`: Submanifold Convolution operation, refer to the paper: https://arxiv.org/abs/1706.01307.
    * :math:`b`: Bias value, a 1-D tensor with shape [M].
    * :math:`Out`: Output value, the shape of :math:`Out` and :math:`X` may be different.

    Args:
        x (Tensor): The input is 4-D SparseCooTensor with shape [N, H, W, C], the data
            type of input is float16 or float32 or float64.
        weight (Tensor): The convolution kernel, a Tensor with shape [kH, kW, C/g, M],
            where M is the number of filters(output channels), g is the number of groups,
            kD, kH, kW are the filter's height and width respectively.
        bias (Tensor, optional): The bias, a Tensor of shape [M].
        stride (int|list|tuple, optional): The stride size. It means the stride in convolution. If stride is a
            list/tuple, it must contain two integers, (stride_height, stride_width).
            Otherwise, stride_height = stride_width = stride. Default: stride = 1.
        padding (string|int|list|tuple, optional): The padding size. It means the number of zero-paddings
            on both sides for each dimension. If `padding` is a string, either 'VALID' or
            'SAME' which is the padding algorithm. If padding size is a tuple or list,
            it could be in three forms: `[pad_height, pad_width]` or
            `[pad_height_top, pad_height_bottom, pad_width_left, pad_width_right]`,
            when `data_format` is `"NHWC"`, `padding` can be in the form
            `[[0,0], [pad_height_top, pad_height_bottom], [pad_width_left, pad_width_right], [0,0]]`.
            Default: padding = 0.
        dilation (int|list|tuple, optional): The dilation size. It means the spacing between the kernel points.
            If dilation is a list/tuple, it must contain two integers, (dilation_height,
            dilation_width). Otherwise, dilation_height = dilation_width = dilation.
            Default: dilation = 1.
        groups (int, optional): The groups number of the Conv2D Layer. According to grouped
            convolution in Alex Krizhevsky's Deep CNN paper: when group=2,
            the first half of the filters is only connected to the first half
            of the input channels, while the second half of the filters is only
            connected to the second half of the input channels. Default: groups=1. Currently, only support groups=1.
        data_format (str, optional): Specify the data format of the input, and the data format of the output
            will be consistent with that of the input. An optional string from: `"NHWC"`.
            The default is `"NHWC"`. When it is `"NHWC"`, the data is stored in the order of:
            `[batch_size, input_height, input_width, input_channels]`.
        key(str, optional): the key is used to save or use the same rulebook,
            the definition and role of rulebook refers to
            https://pdfs.semanticscholar.org/5125/a16039cabc6320c908a4764f32596e018ad3.pdf. The
            default value is None.
        name(str, optional): For detailed information, please refer
           to :ref:`api_guide_Name`. Usually name is no need to set and
           None by default.

    Returns:
        A SparseCooTensor representing the conv2d, whose data type is the same with input.

    Examples:
        .. code-block:: python

            import paddle

            indices = [[0, 0, 0, 0], [0, 0, 1, 2], [1, 3, 2, 3]]
            values = [[1], [2], [3], [4]]
            indices = paddle.to_tensor(indices, dtype='int32')
            values = paddle.to_tensor(values, dtype='float32')
            dense_shape = [1, 3, 4, 1]
            sparse_x = paddle.sparse.sparse_coo_tensor(indices, values, dense_shape, stop_gradient=True)
            weight = paddle.randn((3, 3, 1, 1), dtype='float32')
            y = paddle.sparse.nn.functional.subm_conv2d(sparse_x, weight)
            print(y.shape)
            # (1, 3, 4, 1)
    """
    return _conv2d(
        x,
        weight,
        bias,
        stride,
        padding,
        dilation,
        groups,
        True,
        key,
        data_format,
        name,
    )
