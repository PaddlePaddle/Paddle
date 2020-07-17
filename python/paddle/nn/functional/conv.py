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
from __future__ import print_function

__all__ = ['conv2d', 'conv2d_transpose', 'conv3d', 'conv3d_transpose']

import numpy as np
from ...fluid.framework import Variable, in_dygraph_mode
from ...fluid import core, dygraph_utils
from ...fluid.layers import nn, utils
from ...fluid.data_feeder import check_variable_and_dtype
from ...fluid.param_attr import ParamAttr
from ...fluid.layer_helper import LayerHelper


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
                "Unknown padding: '{}'. It can only be 'SAME' or 'VALID'.".
                format(padding))
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
                    "Non-zero padding({}) in the batch or channel dimensions "
                    "is not supported.".format(padding))
            padding_algorithm = "EXPLICIT"
            padding = _exclude_padding_in_batch_and_channel(padding,
                                                            channel_last)
            if utils._is_symmetric_padding(padding, num_dims):
                padding = padding[0::2]
        # for padding like [pad_before, pad_after, pad_before, pad_after, ...]
        elif len(padding) == 2 * num_dims and isinstance(padding[0], int):
            padding_algorithm = "EXPLICIT"
            padding = utils.convert_to_list(padding, 2 * num_dims, 'padding')
            if utils._is_symmetric_padding(padding, num_dims):
                padding = padding[0::2]
        # for padding like [pad_d1, pad_d2, ...]
        elif len(padding) == num_dims and isinstance(padding[0], int):
            padding_algorithm = "EXPLICIT"
            padding = utils.convert_to_list(padding, num_dims, 'padding')
        else:
            raise ValueError("In valid padding: {}".format(padding))
    # for integer padding
    else:
        padding_algorithm = "EXPLICIT"
        padding = utils.convert_to_list(padding, num_dims, 'padding')
    return padding, padding_algorithm


def conv2d(input,
           weight,
           bias=None,
           padding=0,
           stride=1,
           dilation=1,
           groups=1,
           use_cudnn=True,
           act=None,
           data_format="NCHW",
           name=None):
    """
	:alias_main: paddle.nn.functional.conv2d
	:alias: paddle.nn.functional.conv2d,paddle.nn.functional.conv.conv2d

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

    .. math::

        Out = \sigma (W \\ast X + b)

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

        .. math::

            H_{out}&= \\frac{(H_{in} + 2 * paddings[0] - (dilations[0] * (H_f - 1) + 1))}{strides[0]} + 1 \\\\
            W_{out}&= \\frac{(W_{in} + 2 * paddings[1] - (dilations[1] * (W_f - 1) + 1))}{strides[1]} + 1

    Args:
        input (Variable): The input is 4-D Tensor with shape [N, C, H, W], the data type 
            of input is float16 or float32 or float64.
        weight (Variable): The convolution kernel with shape [M, C/g, kH, kW], where M is
            the number of output channels, g is the number of groups, kH is the filter's
            height, kW is the filter's width. 
        bias (Variable, optional): The bias with shape [M,].
        padding (string|int|list|tuple): The padding size. It means the number of zero-paddings
            on both sides for each dimension.If `padding` is a string, either 'VALID' or
            'SAME' which is the padding algorithm. If padding size is a tuple or list,
            it could be in three forms: `[pad_height, pad_width]` or
            `[pad_height_top, pad_height_bottom, pad_width_left, pad_width_right]`, and when 
            `data_format` is `"NCHW"`, `padding` can be in the form `[[0,0], [0,0], 
            [pad_height_top, pad_height_bottom], [pad_width_left, pad_width_right]]`.
            when `data_format` is `"NHWC"`, `pool_padding` can be in the form
            `[[0,0], [pad_height_top, pad_height_bottom], [pad_width_left, pad_width_right], [0,0]]`.
            Default: padding = 0.
        stride (int|tuple): The stride size. It means the stride in convolution. 
            If stride is a tuple, it must contain two integers, (stride_height, stride_width). 
            Otherwise, stride_height = stride_width = stride. Default: stride = 1.
        dilation (int|tuple): The dilation size. It means the spacing between the kernel
            points. If dilation is a tuple, it must contain two integers, (dilation_height, 
            dilation_width). Otherwise, dilation_height = dilation_width = dilation. 
            Default: dilation = 1.
        groups (int): The groups number of the Conv2d Layer. According to grouped
            convolution in Alex Krizhevsky's Deep CNN paper: when group=2,
            the first half of the filters is only connected to the first half
            of the input channels, while the second half of the filters is only
            connected to the second half of the input channels. Default: groups=1.
        use_cudnn (bool): Use cudnn kernel or not, it is valid only when the cudnn
            library is installed. Default: True
        act (str): Activation type, if it is set to None, activation is not appended.
            Default: None
        data_format (str, optional): Specify the data format of the input, and the data format of the output 
            will be consistent with that of the input. An optional string from: `"NCHW"`, `"NHWC"`.
            The default is `"NCHW"`. When it is `"NCHW"`, the data is stored in the order of:
            `[batch_size, input_channels, input_height, input_width]`.
        name(str, optional): For detailed information, please refer 
           to :ref:`api_guide_Name`. Usually name is no need to set and 
           None by default.

    Returns:
        A Variable holding Tensor representing the conv2d, whose data type is the 
        same with input. If act is None, the tensor variable storing the convolution 
        result, and if act is not None, the tensor variable storing convolution 
        and non-linearity activation result.

    Raises:
        ValueError: If the type of `use_cudnn` is not bool.
        ValueError: If `data_format` is not "NCHW" or "NHWC".
        ValueError: If the channel dimmention of the input is less than or equal to zero.
        ValueError: If `padding` is a string, but not "SAME" or "VALID".
        ValueError: If `padding` is a tuple, but the element corresponding to the input's batch size is not 0 
            or the element corresponding to the input's channel is not 0.
        ShapeError: If the input is not 4-D Tensor.
        ShapeError: If the input's dimension size and filter's dimension size not equal.
        ShapeError: If the dimension size of input minus the size of `stride` is not 2.
        ShapeError: If the number of input channels is not equal to filter's channels * groups.
        ShapeError: If the number of output channels is not be divided by groups.

    Examples:
        .. code-block:: python

          from paddle import fluid
          import paddle.nn.functional as F
          import paddle.fluid.dygraph as dg
          import numpy as np

          x = np.random.randn(2, 3, 8, 8).astype(np.float32)
          w = np.random.randn(6, 3, 3, 3).astype(np.float32)

          place = fluid.CPUPlace()
          with dg.guard(place):
              x_var = dg.to_variable(x)
              w_var = dg.to_variable(w)
              y_var = F.conv2d(x_var, w_var, act="relu")
              y_np = y_var.numpy()
          print(y_np.shape)

          # (2, 6, 6, 6)
    """
    # entry checks
    if not isinstance(use_cudnn, bool):
        raise ValueError("Attr(use_cudnn) should be True or False. "
                         "Received Attr(use_cudnn): {}.".format(use_cudnn))
    if data_format not in ["NCHW", "NHWC"]:
        raise ValueError("Attr(data_format) should be 'NCHW' or 'NHWC'. "
                         "Received Attr(data_format): {}.".format(data_format))

    channel_last = (data_format == "NHWC")
    channel_dim = -1 if channel_last else 1
    num_channels = input.shape[channel_dim]
    num_filters = weight.shape[0]
    if num_channels < 0:
        raise ValueError("The channel dimmention of the input({}) "
                         "should be defined. Received: {}.".format(
                             input.shape, num_channels))
    if num_channels % groups != 0:
        raise ValueError(
            "the channel of input must be divisible by groups,"
            "received: the channel of input is {}, the shape of input is {}"
            ", the groups is {}".format(num_channels, input.shape, groups))
    if num_filters % groups != 0:
        raise ValueError(
            "the number of filters must be divisible by groups,"
            "received: the number of filters is {}, the shape of weight is {}"
            ", the groups is {}".format(num_filters, weight.shape, groups))

    # update attrs
    padding, padding_algorithm = _update_padding_nd(padding, channel_last, 2)
    stride = utils.convert_to_list(stride, 2, 'stride')
    dilation = utils.convert_to_list(dilation, 2, 'dilation')

    l_type = "conv2d"
    if (num_channels == groups and num_filters % num_channels == 0 and
            not use_cudnn):
        l_type = 'depthwise_conv2d'

    inputs = {'Input': [input], 'Filter': [weight]}
    attrs = {
        'strides': stride,
        'paddings': padding,
        'dilations': dilation,
        'groups': groups,
        'use_cudnn': use_cudnn,
        'use_mkldnn': False,
        'fuse_relu_before_depthwise_conv': False,
        "padding_algorithm": padding_algorithm,
        "data_format": data_format
    }

    if in_dygraph_mode():
        attrs = ('strides', stride, 'paddings', padding, 'dilations', dilation,
                 'groups', groups, 'use_cudnn', use_cudnn, 'use_mkldnn', False,
                 'fuse_relu_before_depthwise_conv', False, "padding_algorithm",
                 padding_algorithm, "data_format", data_format)
        pre_bias = getattr(core.ops, l_type)(input, weight, *attrs)
        if bias is not None:
            pre_act = nn.elementwise_add(pre_bias, bias, axis=channel_dim)
        else:
            pre_act = pre_bias
        out = dygraph_utils._append_activation_in_dygraph(
            pre_act, act, use_cudnn=use_cudnn)
    else:
        inputs = {'Input': [input], 'Filter': [weight]}
        attrs = {
            'strides': stride,
            'paddings': padding,
            'dilations': dilation,
            'groups': groups,
            'use_cudnn': use_cudnn,
            'use_mkldnn': False,
            'fuse_relu_before_depthwise_conv': False,
            "padding_algorithm": padding_algorithm,
            "data_format": data_format
        }
        check_variable_and_dtype(input, 'input',
                                 ['float16', 'float32', 'float64'], 'conv2d')
        helper = LayerHelper(l_type, **locals())
        dtype = helper.input_dtype()
        pre_bias = helper.create_variable_for_type_inference(dtype)
        outputs = {"Output": [pre_bias]}
        helper.append_op(
            type=l_type, inputs=inputs, outputs=outputs, attrs=attrs)
        if bias is not None:
            pre_act = nn.elementwise_add(pre_bias, bias, axis=channel_dim)
        else:
            pre_act = pre_bias
        out = helper.append_activation(pre_act)
    return out


def conv2d_transpose(input,
                     weight,
                     bias=None,
                     output_size=None,
                     padding=0,
                     stride=1,
                     dilation=1,
                     groups=1,
                     use_cudnn=True,
                     act=None,
                     data_format='NCHW',
                     name=None):
    """
	:alias_main: paddle.nn.functional.conv2d_transpose
	:alias: paddle.nn.functional.conv2d_transpose,paddle.nn.functional.conv.conv2d_transpose

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

    For each input :math:`X`, the equation is:

    .. math::

        Out = \sigma (W \\ast X + b)

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

        .. math::

           H^\prime_{out} &= (H_{in} - 1) * strides[0] - pad_height_top - pad_height_bottom + dilations[0] * (H_f - 1) + 1 \\\\
           W^\prime_{out} &= (W_{in} - 1) * strides[1] - pad_width_left - pad_width_right + dilations[1] * (W_f - 1) + 1 \\\\
           H_{out} &\in [ H^\prime_{out}, H^\prime_{out} + strides[0] ] \\\\
           W_{out} &\in [ W^\prime_{out}, W^\prime_{out} + strides[1] ]

    Note:
          The conv2d_transpose can be seen as the backward of the conv2d. For conv2d, 
          when stride > 1, conv2d maps multiple input shape to the same output shape, 
          so for conv2d_transpose, when stride > 1, input shape maps multiple output shape.
          If output_size is None, :math:`H_{out} = H^\prime_{out}, W_{out} = W^\prime_{out}`; 
          else, the :math:`H_{out}` of the output size must between :math:`H^\prime_{out}` 
          and :math:`H^\prime_{out} + strides[0]`, and the :math:`W_{out}` of the output size must 
          between :math:`W^\prime_{out}` and :math:`W^\prime_{out} + strides[1]`, 
          conv2d_transpose can compute the kernel size automatically.

    Args:
        input(Variable): 4-D Tensor with [N, C, H, W] or [N, H, W, C] format,
            whose data type is float32 or float64.
        weight(Variable): The convolution kernel, a Tensor with shape [C, M/g, kH, kW],
            where M is the number of output channels(filters), g is the number of groups,
            kH is the height of the kernel, and kW is the width of the kernel.
        bias(Variable, optional): The bias, a Tensor with shape [M, ].
        output_size(int|tuple|list, optional): The output image size. If output size is a
            tuple, it must contain two integers, (image_height, image_width). None if use
            filter_size, padding, and stride to calculate output_size.
            If output_size is specified, output_size and filter_size (weight)'s shape 
            should follow the formula above. Default: None. output_size and filter_size 
            should not be None at the same time.
        padding(int|list|str|tuple, optional): The padding size. The padding argument effectively adds
             `dilation * (kernel - 1)` amount of zero-padding on both sides of input. If `padding` is a
             string, either 'VALID' or 'SAME' supported, which is the padding algorithm.
             If `padding` is a tuple or list, it could be in three forms:
             `[pad_height, pad_width]` or
            `[pad_height_top, pad_height_bottom, pad_width_left, pad_width_right]`, and
            when `data_format` is `'NCHW'`,
            `padding` can be in the form `[[0,0], [0,0], [pad_height_top, pad_height_bottom], [pad_width_left, pad_width_right]]`.
            when `data_format` is `'NHWC'`, `padding` can be in the form
            `[[0,0], [pad_height_top, pad_height_bottom], [pad_width_left, pad_width_right], [0,0]]`.
            Default: padding = 0.
        stride(int|tuple, optional): The stride size. It means the stride in transposed convolution. 
            If stride is a tuple, it must contain two integers, (stride_height, stride_width). 
            Otherwise, stride_height = stride_width = stride. Default: stride = 1.
        dilation(int|tuple, optional): The dilation size. It means the spacing between the kernel points. 
            If dilation is a tuple, it must contain two integers, (dilation_height, dilation_width). 
            Otherwise, dilation_height = dilation_width = dilation. Default: dilation = 1.
        groups(int, optional): The groups number of the Conv2d transpose layer. Inspired by
            grouped convolution in Alex Krizhevsky's Deep CNN paper, in which
            when group=2, the first half of the filters is only connected to the
            first half of the input channels, while the second half of the
            filters is only connected to the second half of the input channels.
            Default: groups = 1.
        use_cudnn(bool, optional): Use cudnn kernel or not, it is valid only when the cudnn
            library is installed. Default: True.
        act (str, optional): Activation type, if it is set to None, activation is not appended.
            Default: None.
        data_format (str, optional): Specify the data format of the input, and the data format of the output 
            will be consistent with that of the input. An optional string from: `"NCHW"`, `"NHWC"`.
            The default is `"NCHW"`. When it is `"NCHW"`, the data is stored in the order of:
            `[batch_size, input_channels, input_height, input_width]`.
        name(str, optional): For detailed information, please refer 
           to :ref:`api_guide_Name`. Usually name is no need to set and 
           None by default.

    Returns:
        A Variable holding Tensor representing the conv2d_transpose, whose 
        data type is the same with input and shape is (num_batches, channels, out_h, 
        out_w) or (num_batches, out_h, out_w, channels). If act is None, the tensor variable 
        storing the transposed convolution result, and if act is not None, the 
        tensor variable storing transposed convolution and non-linearity activation 
        result.

    Raises:
        ValueError: If the type of `use_cudnn` is not bool.
        ValueError: If `data_format` is not "NCHW" or "NHWC".
        ValueError: If `padding` is a string, but not "SAME" or "VALID".
        ValueError: If `padding` is a tuple, but the element corresponding to the input's batch size is not 0 
            or the element corresponding to the input's channel is not 0.
        ValueError: If `output_size` and filter_size are None at the same time.
        ShapeError: If the input is not 4-D Tensor.
        ShapeError: If the input's dimension size and filter's dimension size not equal.
        ShapeError: If the dimension size of input minus the size of `stride` is not 2.
        ShapeError: If the number of input channels is not equal to filter's channels.
        ShapeError: If the size of `output_size` is not equal to that of `stride`.

    Examples:
        .. code-block:: python

          from paddle import fluid
          import paddle.nn.functional as F
          import paddle.fluid.dygraph as dg
          import numpy as np

          x = np.random.randn(2, 3, 8, 8).astype(np.float32)
          w = np.random.randn(3, 6, 3, 3).astype(np.float32)

          place = fluid.CPUPlace()
          with dg.guard(place):
              x_var = dg.to_variable(x)
              w_var = dg.to_variable(w)
              y_var = F.conv2d_transpose(x_var, w_var, act="relu")
              y_np = y_var.numpy()
          print(y_np.shape)

          # (2, 6, 10, 10)
    """

    if not isinstance(use_cudnn, bool):
        raise ValueError("Attr(use_cudnn) should be True or False. "
                         "Received Attr(use_cudnn): {}.".format(use_cudnn))
    if data_format not in ['NCHW', 'NHWC']:
        raise ValueError(
            "Attr(data_format) of conv2d_transpose got wrong value: "
            "received {}, but only 'NCHW' or 'NHWC' are supported.".format(
                data_format))
    channel_last = (data_format == "NHWC")
    channel_dim = -1 if channel_last else 1
    num_channels = input.shape[channel_dim]
    if num_channels < 0:
        raise ValueError("The channel dimmention of the input({}) "
                         "should be defined. Received: {}.".format(
                             input.shape, num_channels))
    if num_channels % groups != 0:
        raise ValueError(
            "the channel of input must be divisible by groups,"
            "received: the channel of input is {}, the shape of input is {}"
            ", the groups is {}".format(num_channels, input.shape, groups))

    # update attrs
    padding, padding_algorithm = _update_padding_nd(padding, channel_last, 2)
    stride = utils.convert_to_list(stride, 2, 'stride')
    dilation = utils.convert_to_list(dilation, 2, 'dilation')
    if output_size is None:
        output_size = []
    elif isinstance(output_size, (list, tuple, int)):
        output_size = utils.convert_to_list(output_size, 2, 'output_size')
    else:
        raise ValueError("output_size should be int, or list, tuple of ints")

    op_type = 'conv2d_transpose'
    num_filters = weight.shape[1]
    if (num_channels == groups and num_filters == 1 and not use_cudnn):
        op_type = 'depthwise_conv2d_transpose'

    if in_dygraph_mode():
        attrs = ('output_size', output_size, 'strides', stride, 'paddings',
                 padding, 'padding_algorithm', padding_algorithm, 'dilations',
                 dilation, 'groups', groups, 'use_cudnn', use_cudnn,
                 'data_format', data_format)
        pre_bias = getattr(core.ops, op_type)(input, weight, *attrs)
        if bias is not None:
            pre_act = nn.elementwise_add(pre_bias, bias, axis=channel_dim)
        else:
            pre_act = pre_bias
        out = dygraph_utils._append_activation_in_dygraph(
            pre_act, act, use_cudnn=use_cudnn)
    else:
        inputs = {'Input': [input], 'Filter': [weight]}
        attrs = {
            'output_size': output_size,
            'strides': stride,
            'paddings': padding,
            'padding_algorithm': padding_algorithm,
            'dilations': dilation,
            'groups': groups,
            'use_cudnn': use_cudnn,
            'data_format': data_format
        }
        check_variable_and_dtype(input, 'input',
                                 ['float16', 'float32', 'float64'],
                                 'conv2d_transpose')
        helper = LayerHelper(op_type, **locals())
        dtype = helper.input_dtype()
        pre_bias = helper.create_variable_for_type_inference(dtype)
        outputs = {"Output": [pre_bias]}
        helper.append_op(
            type=op_type, inputs=inputs, outputs=outputs, attrs=attrs)
        if bias is not None:
            pre_act = nn.elementwise_add(pre_bias, bias, axis=channel_dim)
        else:
            pre_act = pre_bias
        out = helper.append_activation(pre_act)
    return out


def conv3d(input,
           weight,
           bias=None,
           padding=0,
           stride=1,
           dilation=1,
           groups=1,
           use_cudnn=True,
           act=None,
           data_format="NCDHW",
           name=None):
    """
	:alias_main: paddle.nn.functional.conv3d
	:alias: paddle.nn.functional.conv3d,paddle.nn.functional.conv.conv3d

    The convolution3D layer calculates the output based on the input, filter
    and strides, paddings, dilations, groups parameters. Input(Input) and
    Output(Output) are in NCDHW or NDHWC format. Where N is batch size C is the number of
    channels, D is the depth of the feature, H is the height of the feature,
    and W is the width of the feature. Convlution3D is similar with Convlution2D
    but adds one dimension(depth). If bias attribution and activation type are
    provided, bias is added to the output of the convolution, and the
    corresponding activation function is applied to the final result.

    For each input :math:`X`, the equation is:

    .. math::

        Out = \sigma (W \\ast X + b)

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

        .. math::

            D_{out}&= \\frac{(D_{in} + 2 * paddings[0] - (dilations[0] * (D_f - 1) + 1))}{strides[0]} + 1 \\\\
            H_{out}&= \\frac{(H_{in} + 2 * paddings[1] - (dilations[1] * (H_f - 1) + 1))}{strides[1]} + 1 \\\\
            W_{out}&= \\frac{(W_{in} + 2 * paddings[2] - (dilations[2] * (W_f - 1) + 1))}{strides[2]} + 1

    Args:
        input (Variable): The input is 5-D Tensor with shape [N, C, D, H, W], the data 
            type of input is float16 or float32 or float64.
        weight (Variable): The convolution kernel, a Tensor with shape [M, C/g, kD, kH, kW],
            where M is the number of filters(output channels), g is the number of groups,
            kD, kH, kW are the filter's depth, height and width respectively.
        bias (Variable, optional): The bias, a Tensor of shape [M, ].
        padding (string|int|list|tuple): The padding size. It means the number of zero-paddings 
            on both sides for each dimension. If `padding` is a string, either 'VALID' or
            'SAME' which is the padding algorithm. If padding size is a tuple or list,
            it could be in three forms: `[pad_depth, pad_height, pad_width]` or
            `[pad_depth_front, pad_depth_back, pad_height_top, pad_height_bottom, pad_width_left, pad_width_right]`,
            and when `data_format` is `"NCDHW"`, `pool_padding` can be in the form
            `[[0,0], [0,0], [pad_depth_front, pad_depth_back], [pad_height_top, pad_height_bottom], [pad_width_left, pad_width_right]]`.
            when `data_format` is `"NDHWC"`, `pool_padding` can be in the form
            `[[0,0], [pad_depth_front, pad_depth_back], [pad_height_top, pad_height_bottom], [pad_width_left, pad_width_right], [0,0]]`.
            Default: padding = 0.
        stride (int|tuple): The stride size. It means the stride in convolution. If stride is a 
            tuple, it must contain three integers, (stride_depth, stride_height, stride_width). 
            Otherwise, stride_depth = stride_height = stride_width = stride. Default: stride = 1.
        dilation (int|tuple): The dilation size. It means the spacing between the kernel points. 
            If dilation is a tuple, it must contain three integers, (dilation_depth, dilation_height,
            dilation_width). Otherwise, dilation_depth = dilation_height = dilation_width = dilation. 
            Default: dilation = 1.
        groups (int): The groups number of the Conv3d Layer. According to grouped
            convolution in Alex Krizhevsky's Deep CNN paper: when group=2,
            the first half of the filters is only connected to the first half
            of the input channels, while the second half of the filters is only
            connected to the second half of the input channels. Default: groups=1
        use_cudnn (bool): Use cudnn kernel or not, it is valid only when the cudnn
            library is installed. Default: True
        act (str): Activation type, if it is set to None, activation is not appended.
            Default: None.
        data_format (str, optional): Specify the data format of the input, and the data format of the output 
            will be consistent with that of the input. An optional string from: `"NCHW"`, `"NHWC"`.
            The default is `"NCHW"`. When it is `"NCHW"`, the data is stored in the order of:
            `[batch_size, input_channels, input_height, input_width]`.
        name(str|None): For detailed information, please refer 
           to :ref:`api_guide_Name`. Usually name is no need to set and 
           None by default.

    Returns:
        A Variable holding Tensor representing the conv3d, whose data type is 
        the same with input. If act is None, the tensor variable storing the 
        convolution result, and if act is not None, the tensor variable storing 
        convolution and non-linearity activation result.

    Raises:
        ValueError: If the type of `use_cudnn` is not bool.
        ValueError: If `data_format` is not "NCDHW" or "NDHWC".
        ValueError: If the channel dimmention of the input is less than or equal to zero.
        ValueError: If `padding` is a string, but not "SAME" or "VALID".
        ValueError: If `padding` is a tuple, but the element corresponding to the input's batch size is not 0 
            or the element corresponding to the input's channel is not 0.
        ShapeError: If the input is not 5-D Tensor.
        ShapeError: If the input's dimension size and filter's dimension size not equal.
        ShapeError: If the dimension size of input minus the size of `stride` is not 2.
        ShapeError: If the number of input channels is not equal to filter's channels * groups.
        ShapeError: If the number of output channels is not be divided by groups.

    Examples:
        .. code-block:: python

            from paddle import fluid
            import paddle.nn.functional as F
            import paddle.fluid.dygraph as dg
            import numpy as np

            x = np.random.randn(2, 3, 8, 8, 8).astype(np.float32)
            w = np.random.randn(6, 3, 3, 3, 3).astype(np.float32)

            place = fluid.CPUPlace()
            with dg.guard(place):
                x_var = dg.to_variable(x)
                w_var = dg.to_variable(w)
                y_var = F.conv3d(x_var, w_var, act="relu")
                y_np = y_var.numpy()
            print(y_np.shape)

            # (2, 6, 6, 6, 6)
    """
    # entry check
    if not isinstance(use_cudnn, bool):
        raise ValueError("Attr(use_cudnn) should be True or False. Received "
                         "Attr(use_cudnn): {}. ".format(use_cudnn))

    if data_format not in ["NCDHW", "NDHWC"]:
        raise ValueError(
            "Attr(data_format) should be 'NCDHW' or 'NDHWC'. Received "
            "Attr(data_format): {}.".format(data_format))

    channel_last = (data_format == "NDHWC")
    channel_dim = -1 if channel_last else 1
    num_channels = input.shape[channel_dim]
    num_filters = weight.shape[0]
    if num_channels < 0:
        raise ValueError(
            "The channel dimmention of the input({}) should be defined. "
            "Received: {}.".format(input.shape, num_channels))
    if num_channels % groups != 0:
        raise ValueError(
            "The number of input channels must be divisible by Attr(groups). "
            "Received: number of channels({}), groups({}).".format(num_channels,
                                                                   groups))
    if num_filters % groups != 0:
        raise ValueError(
            "The number of filters must be divisible by Attr(groups). "
            "Received: number of filters({}), groups({}).".format(num_filters,
                                                                  groups))

    padding, padding_algorithm = _update_padding_nd(padding, channel_last, 3)
    stride = utils.convert_to_list(stride, 3, 'stride')
    dilation = utils.convert_to_list(dilation, 3, 'dilation')
    op_type = "conv3d"

    if in_dygraph_mode():
        attrs = ('strides', stride, 'paddings', padding, 'dilations', dilation,
                 'groups', groups, 'use_cudnn', use_cudnn, 'use_mkldnn', False,
                 "padding_algorithm", padding_algorithm, "data_format",
                 data_format)
        pre_bias = getattr(core.ops, op_type)(input, weight, *attrs)
        if bias is not None:
            pre_act = nn.elementwise_add(pre_bias, bias, axis=channel_dim)
        else:
            pre_act = pre_bias
        out = dygraph_utils._append_activation_in_dygraph(
            pre_act, act, use_cudnn=use_cudnn)
    else:
        inputs = {'Input': [input], 'Filter': [weight]}
        attrs = {
            'strides': stride,
            'paddings': padding,
            'dilations': dilation,
            'groups': groups,
            'use_cudnn': use_cudnn,
            'use_mkldnn': False,
            "padding_algorithm": padding_algorithm,
            "data_format": data_format
        }
        helper = LayerHelper(op_type, **locals())
        dtype = helper.input_dtype()
        check_variable_and_dtype(input, 'input',
                                 ['float16', 'float32', 'float64'], 'conv3d')

        pre_bias = helper.create_variable_for_type_inference(dtype)
        outputs = {"Output": [pre_bias]}

        helper.append_op(
            type=op_type, inputs=inputs, outputs=outputs, attrs=attrs)
        if bias is not None:
            pre_act = nn.elementwise_add(pre_bias, bias, axis=channel_dim)
        else:
            pre_act = pre_bias
        out = helper.append_activation(pre_act)

    return out


def conv3d_transpose(input,
                     weight,
                     bias=None,
                     output_size=None,
                     padding=0,
                     stride=1,
                     dilation=1,
                     groups=1,
                     use_cudnn=True,
                     act=None,
                     data_format='NCDHW',
                     name=None):
    """
	:alias_main: paddle.nn.functional.conv3d_transpose
	:alias: paddle.nn.functional.conv3d_transpose,paddle.nn.functional.conv.conv3d_transpose

    The convolution3D transpose layer calculates the output based on the input,
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

    For each input :math:`X`, the equation is:

    .. math::

        Out = \sigma (W \\ast X + b)

    In the above equation:

    * :math:`X`: Input value, a Tensor with NCDHW or NDHWC format.
    * :math:`W`: Filter value, a Tensor with MCDHW format.
    * :math:`\\ast`: Convolution operation.
    * :math:`b`: Bias value, a 2-D Tensor with shape [M, 1].
    * :math:`\\sigma`: Activation function.
    * :math:`Out`: Output value, the shape of :math:`Out` and :math:`X` may be different.

    Example:

        - Input:

          Input shape: :math:`(N, C_{in}, D_{in}, H_{in}, W_{in})`

          Filter shape: :math:`(C_{in}, C_{out}, D_f, H_f, W_f)`

        - Output:

          Output shape: :math:`(N, C_{out}, D_{out}, H_{out}, W_{out})`

        Where

        .. math::

           D^\prime_{out} &= (D_{in} - 1) * strides[0] - 2 * paddings[0] + dilations[0] * (D_f - 1) + 1 \\\\
           H^\prime_{out} &= (H_{in} - 1) * strides[1] - 2 * paddings[1] + dilations[1] * (H_f - 1) + 1 \\\\
           W^\prime_{out} &= (W_{in} - 1) * strides[2] - 2 * paddings[2] + dilations[2] * (W_f - 1) + 1 \\\\
           D_{out} &\in [ D^\prime_{out}, D^\prime_{out} + strides[0] ] \\\\
           H_{out} &\in [ H^\prime_{out}, H^\prime_{out} + strides[1] ] \\\\
           W_{out} &\in [ W^\prime_{out}, W^\prime_{out} + strides[2] ]

    Note:
          The conv3d_transpose can be seen as the backward of the conv3d. For conv3d, 
          when stride > 1, conv3d maps multiple input shape to the same output shape, 
          so for conv3d_transpose, when stride > 1, input shape maps multiple output shape.
          If output_size is None, :math:`H_{out} = H^\prime_{out}, :math:`H_{out} = \
          H^\prime_{out}, W_{out} = W^\prime_{out}`; else, the :math:`D_{out}` of the output 
          size must between :math:`D^\prime_{out}` and :math:`D^\prime_{out} + strides[0]`, 
          the :math:`H_{out}` of the output size must between :math:`H^\prime_{out}` 
          and :math:`H^\prime_{out} + strides[1]`, and the :math:`W_{out}` of the output size must 
          between :math:`W^\prime_{out}` and :math:`W^\prime_{out} + strides[2]`, 
          conv3d_transpose can compute the kernel size automatically.

    Args:
        input(Variable): The input is 5-D Tensor with shape [N, C, D, H, W] or [N, D, H, W, C], the data type 
            of input is float32 or float64.
        weight (Variable): The convolution kernel, a Tensor with shape [C, M/g, kD, kH, kW],
            where M is the number of filters(output channels), g is the number of groups,
            kD, kH, kW are the filter's depth, height and width respectively.
        bias (Variable, optional): The bias, a Tensor of shape [M, ].
        output_size(int|tuple, optional): The output image size. If output size is a
            tuple, it must contain three integers, (image_depth, image_height, image_width). This
            parameter only works when filter_size is None. If output_size and filter_size are 
            specified at the same time, They should follow the formula above. Default: None. 
            Output_size and filter_size should not be None at the same time.
        padding(int|list|str|tuple, optional): The padding size. The padding argument effectively
             adds `dilation * (kernel - 1)` amount of zero-padding on both sides of input. If `padding` is a string,
             either 'VALID' or 'SAME' supported, which is the padding algorithm. If `padding`
             is a tuple or list, it could be in three forms: `[pad_depth, pad_height, pad_width]` or
            `[pad_depth_front, pad_depth_back, pad_height_top, pad_height_bottom, pad_width_left, pad_width_right]`,
            and when `data_format` is `'NCDHW'`, `padding` can be in the form
            `[[0,0], [0,0], [pad_depth_front, pad_depth_back], [pad_height_top, pad_height_bottom], [pad_width_left, pad_width_right]]`.
            when `data_format` is `'NDHWC'`, `padding` can be in the form
            `[[0,0], [pad_depth_front, pad_depth_back], [pad_height_top, pad_height_bottom], [pad_width_left, pad_width_right], [0,0]]`.
            Default: padding = 0.
        stride(int|tuple, optional): The stride size. It means the stride in transposed convolution. 
            If stride is a tuple, it must contain three integers, (stride_depth, stride_height, 
            stride_width). Otherwise, stride_depth = stride_height = stride_width = stride. 
            Default: stride = 1.
        dilation(int|tuple, optional): The dilation size. It means the spacing between the kernel points. 
            If dilation is a tuple, it must contain three integers, (dilation_depth, dilation_height, 
            dilation_width). Otherwise, dilation_depth = dilation_height = dilation_width = dilation. 
            Default: dilation = 1.
        groups(int, optional): The groups number of the Conv3d transpose layer. Inspired by
            grouped convolution in Alex Krizhevsky's Deep CNN paper, in which
            when group=2, the first half of the filters is only connected to the
            first half of the input channels, while the second half of the
            filters is only connected to the second half of the input channels.
            Default: groups=1
        use_cudnn(bool, optional): Use cudnn kernel or not, it is valid only when the cudnn
            library is installed. Default: True
        act (str, optional): Activation type, if it is set to None, activation is not appended.
            Default: None.
        data_format (str, optional): Specify the data format of the input, and the data format of the output 
            will be consistent with that of the input. An optional string from: `"NCHW"`, `"NHWC"`.
            The default is `"NCHW"`. When it is `"NCHW"`, the data is stored in the order of:
            `[batch_size, input_channels, input_height, input_width]`.
        name(str, optional): For detailed information, please refer 
           to :ref:`api_guide_Name`. Usually name is no need to set and 
           None by default.

    Returns:
        A Variable holding Tensor representing the conv3d_transpose, whose data 
        type is the same with input and shape is (num_batches, channels, out_d, out_h, 
        out_w) or (num_batches, out_d, out_h, out_w, channels). If act is None, the tensor 
        variable storing the transposed convolution result, and if act is not None, the tensor 
        variable storing transposed convolution and non-linearity activation result.

    Raises:
        ValueError: If the type of `use_cudnn` is not bool.
        ValueError: If `data_format` is not "NCDHW" or "NDHWC".
        ValueError: If `padding` is a string, but not "SAME" or "VALID".
        ValueError: If `padding` is a tuple, but the element corresponding to the input's batch size is not 0 
            or the element corresponding to the input's channel is not 0.
        ValueError: If `output_size` and filter_size are None at the same time.
        ShapeError: If the input is not 5-D Tensor.
        ShapeError: If the input's dimension size and filter's dimension size not equal.
        ShapeError: If the dimension size of input minus the size of `stride` is not 2.
        ShapeError: If the number of input channels is not equal to filter's channels.
        ShapeError: If the size of `output_size` is not equal to that of `stride`.

    Examples:
       .. code-block:: python

          from paddle import fluid
          import paddle.nn.functional as F
          import paddle.fluid.dygraph as dg
          import numpy as np

          x = np.random.randn(2, 3, 8, 8, 8).astype(np.float32)
          w = np.random.randn(3, 6, 3, 3, 3).astype(np.float32)

          place = fluid.CPUPlace()
          with dg.guard(place):
              x_var = dg.to_variable(x)
              w_var = dg.to_variable(w)
              y_var = F.conv3d_transpose(x_var, w_var, act="relu")
              y_np = y_var.numpy()
          print(y_np.shape)

          # (2, 6, 10, 10, 10)
    """
    # entry checks
    if not isinstance(use_cudnn, bool):
        raise ValueError("Attr(use_cudnn) should be True or False. "
                         "Received Attr(use_cudnn): {}.".format(use_cudnn))
    if data_format not in ["NCDHW", "NDHWC"]:
        raise ValueError(
            "Attr(data_format) should be 'NCDHW' or 'NDHWC'. Received "
            "Attr(data_format): {}.".format(data_format))

    channel_last = (data_format == "NDHWC")
    channel_dim = -1 if channel_last else 1
    num_channels = input.shape[channel_dim]
    num_filters = weight.shape[1]
    if num_channels < 0:
        raise ValueError(
            "The channel dimmention of the input({}) should be defined. "
            "Received: {}.".format(input.shape, num_channels))
    if num_channels % groups != 0:
        raise ValueError(
            "The number of input channels must be divisible by Attr(groups). "
            "Received: number of channels({}), groups({}).".format(num_channels,
                                                                   groups))

    padding, padding_algorithm = _update_padding_nd(padding, channel_last, 3)
    stride = utils.convert_to_list(stride, 3, 'stride')
    dilation = utils.convert_to_list(dilation, 3, 'dilation')
    if output_size is None:
        output_size = []
    elif isinstance(output_size, (list, tuple, int)):
        output_size = utils.convert_to_list(output_size, 3, 'output_size')
    else:
        raise ValueError("output_size should be int, or list, tuple of ints")

    op_type = 'conv3d_transpose'
    data_format_ = "NHWC" if channel_last else "NCHW"

    if in_dygraph_mode():
        attrs = ('output_size', output_size, 'paddings', padding,
                 "padding_algorithm", padding_algorithm, 'strides', stride,
                 'dilations', dilation, 'groups', groups, 'use_cudnn',
                 use_cudnn, "data_format", data_format_)
        pre_bias = getattr(core.ops, op_type)(input, weight, *attrs)
        if bias is not None:
            pre_act = nn.elementwise_add(pre_bias, bias, axis=channel_dim)
        else:
            pre_act = pre_bias
        out = dygraph_utils._append_activation_in_dygraph(
            pre_act, act, use_cudnn=use_cudnn)
    else:
        inputs = {'Input': [input], 'Filter': [weight]}
        attrs = {
            'output_size': output_size,
            'paddings': padding,
            "padding_algorithm": padding_algorithm,
            'strides': stride,
            'dilations': dilation,
            'groups': groups,
            'use_cudnn': use_cudnn,
            "data_format": data_format_
        }
        helper = LayerHelper(op_type, **locals())
        dtype = helper.input_dtype()
        check_variable_and_dtype(input, 'input',
                                 ['float16', 'float32', 'float64'], 'conv3d')

        pre_bias = helper.create_variable_for_type_inference(dtype)
        outputs = {"Output": [pre_bias]}

        helper.append_op(
            type=op_type, inputs=inputs, outputs=outputs, attrs=attrs)
        if bias is not None:
            pre_act = nn.elementwise_add(pre_bias, bias, axis=channel_dim)
        else:
            pre_act = pre_bias
        out = helper.append_activation(pre_act)

    return out
