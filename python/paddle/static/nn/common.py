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

import paddle
from paddle.fluid.framework import static_only
from paddle.fluid.initializer import Normal

from paddle.common_ops_import import (
    check_type,
    check_variable_and_dtype,
    utils,
    LayerHelper,
)

__all__ = []


@static_only
def fc(
    x,
    size,
    num_flatten_dims=1,
    weight_attr=None,
    bias_attr=None,
    activation=None,
    name=None,
):
    r"""

    Fully-Connected layer can take a tensor or a list of tensor as its inputs.
    It creates a 2-D weight tensor for each input tensor, which represents its
    weight matrix from each input unit to each output unit. The fully connected
    layer multiplies each input tensor with its corresponding weight to produce
    an output tensor with shape :math:`[batch\_size, *, size]` , where :math:`*`
    means any number of additional dimensions. If a list of tensor is given,
    the results of multiple output tensors with shape :math:`[batch\_size, *, size]`
    will be summed up. If :attr:`bias_attr` is not False, a 1-D bias tensor will
    be created and added to the output. Finally, if :attr:`activation` is not None,
    it will be applied to the output as well.

    For a single input tensor :math:`X` , the equation is:

    .. math::

        Out = Act({XW + b})

    For a list of input tensor, the equation is:

    .. math::

        Out = Act({\sum_{i=0}^{N-1}X_iW_i + b})

    where:

    * :math:`N`: The number of the input tensors. :math:`N` equals to :math:`len(X)` if :math:`X` is list of tensor.
    * :math:`X_i`: The i-th input tensor.
    * :math:`W_i`: The i-th weight matrix corresponding i-th input tensor.
    * :math:`b`: The bias created by this layer (if needed).
    * :math:`Act`: The activation function.
    * :math:`Out`: The output tensor.

    .. code-block:: text

        # Case 1, input is a single tensor:
        x.data = [[[0.1, 0.2],
                   [0.3, 0.4]]]
        x.shape = (1, 2, 2) # 1 is batch_size

        out = paddle.static.nn.fc(x=x, size=1, num_flatten_dims=2)

        # Get the output:
        out.data = [[0.83234344], [0.34936576]]
        out.shape = (1, 2, 1)

        # Case 2, input is a list of tensor:
        x0.data = [[[0.1, 0.2],
                    [0.3, 0.4]]]
        x0.shape = (1, 2, 2) # 1 is batch_size

        x1.data = [[[0.1, 0.2, 0.3]]]
        x1.shape = (1, 1, 3)

        out = paddle.static.nn.fc(x=[x0, x1], size=2)

        # Get the output:
        out.data = [[0.18669507, 0.1893476]]
        out.shape = (1, 2)

    Args:
        x (Tensor|list[Tensor]|tuple[Tensor]): A tensor or a list/tuple of tensors. The number of dimensions
            of each tensor is at least 2. The data type should be float16, float32 or float64.
        size (int): The number of output units in this layer, which also means the feature
            size of output tensor.
        num_flatten_dims (int, optional): The fc layer can accept an input tensor with more than
            two dimensions. If this happens, the multi-dimensional tensor will first be flattened
            into a 2-D matrix. The parameter :attr:`num_flatten_dims` determines how the input
            tensor is flattened: the first :math:`num\_flatten\_dims` (inclusive, index starts from 1)
            dimensions will be flatten to form the first dimension of the final matrix (height of
            the matrix), and the rest :math:`rank(x) - num\_flatten\_dims` dimensions are
            flattened to form the second dimension of the final matrix (width of the matrix).
            For example, assuming that :attr:`x` is a 5-dimensional tensor with a shape
            :math:`[2, 3, 4, 5, 6]` , and :attr:`num_flatten_dims` = 3.
            Then, the flattened matrix will have a shape :math:`[2 * 3 * 4, 5 * 6] = [24, 30]` .
            Default: 1.
        weight_attr (ParamAttr, optional): The attribute for the learnable weight.
            The default value is None, and the weight will be initialized to zero.
            For detailed information, please refer to :attr:`paddle.ParamAttr`.
            Warning, if x is a list of tensor, weight_attr should also be a list of same length.
        bias_attr (ParamAttr|bool, optional): The attribute of the learnable bias.
            If it is set to False, no bias will be added to the output.
            If it is set to None or one kind of ParamAttr, a bias parameter will
            be created according to ParamAttr. For detailed information, please refer
            to :attr:`paddle.ParamAttr`. The default value is None and the bias will be
            initialized to zero.
        activation (str, optional): Activation to be applied to the output of
            this layer, such as tanh, softmax, sigmoid, relu. For more information,
            please refer to :ref:`api_guide_activations_en` . Default: None.
        name (str, optional): The default value is None. Normally there is no need for user to set
            it. For more information, please refer to :ref:`api_guide_Name` .

    Returns:
        Tensor, its shape is :math:`[batch\_size, *, size]` , and the data type is same with input.

    Examples:
        .. code-block:: python

          import paddle
          paddle.enable_static()

          # When input is a single tensor
          x = paddle.static.data(name="x", shape=[1, 2, 2], dtype="float32")
          # x: [[[0.1 0.2]
          #      [0.3 0.4]]]
          out = paddle.static.nn.fc(
              x=x,
              size=1,
              num_flatten_dims=2,
              weight_attr=paddle.ParamAttr(initializer=paddle.nn.initializer.Constant(value=0.5)),
              bias_attr=paddle.ParamAttr(initializer=paddle.nn.initializer.Constant(value=1.0)))
          # out: [[[1.15]
          #        [1.35]]]

          # When input is multiple tensors
          x0 = paddle.static.data(name="x0", shape=[1, 2, 2], dtype="float32")
          # x0: [[[0.1 0.2]
          #       [0.3 0.4]]]
          x1 = paddle.static.data(name="x1", shape=[1, 1, 3], dtype="float32")
          # x1: [[[0.1 0.2 0.3]]]
          out = paddle.static.nn.fc(
              x=[x0, x1],
              size=2,
              weight_attr=paddle.ParamAttr(initializer=paddle.nn.initializer.Constant(value=0.5)),
              bias_attr=paddle.ParamAttr(initializer=paddle.nn.initializer.Constant(value=1.0)))
          # out: [[1.8 1.8]]
    """
    return paddle.fluid.layers.fc(
        input=x,
        size=size,
        num_flatten_dims=num_flatten_dims,
        param_attr=weight_attr,
        bias_attr=bias_attr,
        act=activation,
        name=name,
    )


def conv3d(
    input,
    num_filters,
    filter_size,
    stride=1,
    padding=0,
    dilation=1,
    groups=None,
    param_attr=None,
    bias_attr=None,
    use_cudnn=True,
    act=None,
    name=None,
    data_format="NCDHW",
):
    r"""
    :api_attr: Static Graph

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
        input (Tensor): The input is 5-D Tensor with shape [N, C, D, H, W], the data
            type of input is float16 or float32 or float64.
        num_filters(int): The number of filter. It is as same as the output
            image channel.
        filter_size (int|tuple): The filter size. If filter_size is a tuple,
            it must contain three integers, (filter_size_depth, filter_size_height,
            filter_size_width). Otherwise, filter_size_depth = filter_size_height = \
            filter_size_width = filter_size.
        stride (int|tuple): The stride size. It means the stride in convolution. If stride is a
            tuple, it must contain three integers, (stride_depth, stride_height, stride_width).
            Otherwise, stride_depth = stride_height = stride_width = stride. Default: stride = 1.
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
        dilation (int|tuple): The dilation size. It means the spacing between the kernel points.
            If dilation is a tuple, it must contain three integers, (dilation_depth, dilation_height,
            dilation_width). Otherwise, dilation_depth = dilation_height = dilation_width = dilation.
            Default: dilation = 1.
        groups (int): The groups number of the Conv3d Layer. According to grouped
            convolution in Alex Krizhevsky's Deep CNN paper: when group=2,
            the first half of the filters is only connected to the first half
            of the input channels, while the second half of the filters is only
            connected to the second half of the input channels. Default: groups=1
        param_attr (ParamAttr|None): The parameter attribute for learnable parameters/weights
            of conv3d. If it is set to None or one attribute of ParamAttr, conv3d
            will create ParamAttr as param_attr. If it is set to None, the parameter
            is initialized with :math:`Normal(0.0, std)`, and the :math:`std` is
            :math:`(\\frac{2.0 }{filter\_elem\_num})^{0.5}`. Default: None.
        bias_attr (ParamAttr|bool|None): The parameter attribute for the bias of conv3d.
            If it is set to False, no bias will be added to the output units.
            If it is set to None or one attribute of ParamAttr, conv3d
            will create ParamAttr as bias_attr. If the Initializer of the bias_attr
            is not set, the bias is initialized zero. Default: None.
        use_cudnn (bool): Use cudnn kernel or not, it is valid only when the cudnn
            library is installed. Default: True
        act (str): Activation type, if it is set to None, activation is not appended.
            Default: None.
        name(str|None): For detailed information, please refer
           to :ref:`api_guide_Name`. Usually name is no need to set and
           None by default.
        data_format (str, optional): Specify the data format of the input, and the data format of the output
            will be consistent with that of the input. An optional string from: `"NCHW"`, `"NHWC"`.
            The default is `"NCHW"`. When it is `"NCHW"`, the data is stored in the order of:
            `[batch_size, input_channels, input_height, input_width]`.

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

          import paddle
          import numpy as np

          paddle.enable_static()
          data = paddle.static.data(name='data', shape=[None, 3, 12, 32, 32], dtype='float32')
          param_attr = paddle.framework.ParamAttr(name='conv3d.weight', initializer=paddle.nn.initializer.XavierNormal(), learning_rate=0.001)
          res = paddle.static.nn.conv3d(input=data, num_filters=2, filter_size=3, act="relu", param_attr=param_attr)
          place = paddle.CPUPlace()
          exe = paddle.static.Executor(place)
          exe.run(paddle.static.default_startup_program())
          x = np.random.rand(1, 3, 12, 32, 32).astype("float32")
          output = exe.run(feed={"data": x}, fetch_list=[res])
          print(output)
    """

    l_type = 'conv3d'
    assert param_attr is not False, "param_attr should not be False here."
    helper = LayerHelper(l_type, **locals())
    dtype = helper.input_dtype()

    if not isinstance(use_cudnn, bool):
        raise ValueError(
            "Attr(use_cudnn) should be True or False. Received "
            "Attr(use_cudnn): %s. " % str(use_cudnn)
        )

    if data_format not in ["NCDHW", "NDHWC"]:
        raise ValueError(
            "Attr(data_format) should be 'NCDHW' or 'NDHWC'. Received "
            "Attr(data_format): %s." % str(data_format)
        )

    channel_last = data_format == "NDHWC"
    if len(input.shape) != 5:
        raise ValueError(
            "Input should be 5D tensor, but received input with the shape of {}".format(
                input.shape
            )
        )
    num_channels = input.shape[4] if channel_last else input.shape[1]
    if num_channels < 0:
        raise ValueError(
            "The channel dimmention of the input(%s) should be defined. "
            "Received: %s." % (str(input.shape), str(num_channels))
        )

    if groups is None:
        num_filter_channels = num_channels
    elif groups <= 0:
        raise ValueError(
            "the groups of conv3d should be greater than 0. Received groups: {}".format(
                groups
            )
        )
    else:
        if num_channels % groups != 0:
            raise ValueError(
                "The number of input channels must be divisible by Attr(groups). "
                "Received: number of channels(%s), groups(%s)."
                % (str(num_channels), str(groups))
            )
        num_filter_channels = num_channels // groups

    filter_size = utils.convert_to_list(filter_size, 3, 'filter_size')
    stride = utils.convert_to_list(stride, 3, 'stride')
    dilation = utils.convert_to_list(dilation, 3, 'dilation')

    def _update_padding(padding, data_format):
        def is_list_or_tuple(ele):
            if isinstance(ele, list) or isinstance(ele, tuple):
                return True
            return False

        if is_list_or_tuple(padding) and len(padding) == 5:
            if is_list_or_tuple(padding[0]) and (data_format == "NCDHW"):
                if not (padding[0] == [0, 0] and padding[1] == [0, 0]):
                    raise ValueError(
                        "Non-zero padding(%s) in the batch or channel dimensions "
                        "is not supported." % str(padding)
                    )
                padding = padding[2:5]
                padding = [ele for a_list in padding for ele in a_list]
            elif is_list_or_tuple(padding[0]) and (data_format == "NDHWC"):
                if not (padding[0] == [0, 0] and padding[4] == [0, 0]):
                    raise ValueError(
                        "Non-zero padding(%s) in the batch or channel dimensions "
                        "is not supported." % str(padding)
                    )
                padding = padding[1:4]
                padding = [ele for a_list in padding for ele in a_list]
            padding = utils.convert_to_list(padding, 6, 'padding')
            if utils._is_symmetric_padding(padding, 3):
                padding = [padding[0], padding[2], padding[4]]
        elif is_list_or_tuple(padding) and len(padding) == 6:
            padding = utils.convert_to_list(padding, 6, 'padding')
            if utils._is_symmetric_padding(padding, 3):
                padding = [padding[0], padding[2], padding[4]]
        else:
            padding = utils.convert_to_list(padding, 3, 'padding')

        return padding

    padding_algorithm = "EXPLICIT"
    if isinstance(padding, str):
        padding = padding.upper()
        if padding not in ["SAME", "VALID"]:
            raise ValueError(
                "Unknown padding: '%s'. It can only be 'SAME' or 'VALID'."
                % str(padding)
            )
        if padding == "VALID":
            padding_algorithm = "VALID"
            padding = [0, 0, 0]
        elif padding == "SAME":
            padding_algorithm = "SAME"
            padding = [0, 0, 0]

    padding = _update_padding(padding, data_format)

    input_shape = input.shape
    filter_shape = [num_filters, num_filter_channels] + filter_size

    def _get_default_param_initializer():
        filter_elem_num = (
            filter_size[0] * filter_size[1] * filter_size[2] * num_channels
        )
        if filter_elem_num <= 0:
            raise ValueError(
                "Invalid filter number, excepted number is larger than 0, but"
                " received {}, please check the input shape and "
                "filter size.".format(filter_elem_num)
            )

        std = (2.0 / filter_elem_num) ** 0.5
        return Normal(0.0, std, 0)

    filter_param = helper.create_parameter(
        attr=helper.param_attr,
        shape=filter_shape,
        dtype=dtype,
        default_initializer=_get_default_param_initializer(),
    )

    pre_bias = helper.create_variable_for_type_inference(dtype)

    helper.append_op(
        type=l_type,
        inputs={
            'Input': input,
            'Filter': filter_param,
        },
        outputs={"Output": pre_bias},
        attrs={
            'strides': stride,
            'paddings': padding,
            'dilations': dilation,
            'groups': groups,
            'use_cudnn': use_cudnn,
            'use_mkldnn': False,
            "padding_algorithm": padding_algorithm,
            "data_format": data_format,
        },
    )

    if data_format == 'NCDHW':
        pre_act = helper.append_bias_op(pre_bias, dim_start=1, dim_end=2)
    else:
        pre_act = helper.append_bias_op(pre_bias, dim_start=4, dim_end=5)

    return helper.append_activation(pre_act)


def deformable_conv(
    input,
    offset,
    mask,
    num_filters,
    filter_size,
    stride=1,
    padding=0,
    dilation=1,
    groups=None,
    deformable_groups=None,
    im2col_step=None,
    param_attr=None,
    bias_attr=None,
    modulated=True,
    name=None,
):
    r"""

    **Deformable Convolution op**

    Compute 2-D deformable convolution on 4-D input.
    Given input image x, output feature map y, the deformable convolution operation can be expressed as follow:


    Deformable Convolution v2:

    .. math::

        y(p) = \sum_{k=1}^{K}{w_k * x(p + p_k + \Delta p_k) * \Delta m_k}

    Deformable Convolution v1:

    .. math::

        y(p) = \sum_{k=1}^{K}{w_k * x(p + p_k + \Delta p_k)}

    Where :math:`\Delta p_k` and :math:`\Delta m_k` are the learnable offset and modulation scalar for the k-th location,
    Which :math:`\Delta m_k` is one in deformable convolution v1. Please refer to `Deformable ConvNets v2: More Deformable, Better Results
    <https://arxiv.org/abs/1811.11168v2>`_ and `Deformable Convolutional Networks <https://arxiv.org/abs/1703.06211>`_.

    Example:
        - Input:

          Input shape: :math:`(N, C_{in}, H_{in}, W_{in})`

          Filter shape: :math:`(C_{out}, C_{in}, H_f, W_f)`

          Offset shape: :math:`(N, 2 * deformable\_groups * H_f * H_w, H_{in}, W_{in})`

          Mask shape: :math:`(N, deformable\_groups * H_f * H_w, H_{in}, W_{in})`

        - Output:

          Output shape: :math:`(N, C_{out}, H_{out}, W_{out})`

        Where

        .. math::

            H_{out}&= \\frac{(H_{in} + 2 * paddings[0] - (dilations[0] * (H_f - 1) + 1))}{strides[0]} + 1 \\\\
            W_{out}&= \\frac{(W_{in} + 2 * paddings[1] - (dilations[1] * (W_f - 1) + 1))}{strides[1]} + 1

    Args:
        input (Tensor): The input image with [N, C, H, W] format. A Tensor with type
            float32, float64.
        offset (Tensor): The input coordinate offset of deformable convolution layer.
            A Tensor with type float32, float64.
        Mask (Variable, Optional): The input mask of deformable convolution layer.
            A Tensor with type float32, float64. It should be None when you use
            deformable convolution v1.
        num_filters(int): The number of filter. It is as same as the output
            image channel.
        filter_size (int|tuple): The filter size. If filter_size is a tuple,
            it must contain two integers, (filter_size_H, filter_size_W).
            Otherwise, the filter will be a square.
        stride (int|tuple): The stride size. If stride is a tuple, it must
            contain two integers, (stride_H, stride_W). Otherwise, the
            stride_H = stride_W = stride. Default: stride = 1.
        padding (int|tuple): The padding size. If padding is a tuple, it must
            contain two integers, (padding_H, padding_W). Otherwise, the
            padding_H = padding_W = padding. Default: padding = 0.
        dilation (int|tuple): The dilation size. If dilation is a tuple, it must
            contain two integers, (dilation_H, dilation_W). Otherwise, the
            dilation_H = dilation_W = dilation. Default: dilation = 1.
        groups (int): The groups number of the deformable conv layer. According to
            grouped convolution in Alex Krizhevsky's Deep CNN paper: when group=2,
            the first half of the filters is only connected to the first half
            of the input channels, while the second half of the filters is only
            connected to the second half of the input channels. Default: groups=1.
        deformable_groups (int): The number of deformable group partitions.
            Default: deformable_groups = 1.
        im2col_step (int): Maximum number of images per im2col computation;
            The total batch size should be devisable by this value or smaller
            than this value; if you face out of memory problem, you can try
            to use a smaller value here.
            Default: im2col_step = 64.
        param_attr (ParamAttr, Optional): The parameter attribute for learnable parameters/weights
            of deformable conv. If it is set to None or one attribute of ParamAttr,
            deformable conv will create ParamAttr as param_attr.
            If the Initializer of the param_attr is not set, the parameter is
            initialized with :math:`Normal(0.0, std)`, and the
            :math:`std` is :math:`(\\frac{2.0 }{filter\_elem\_num})^{0.5}`. Default: None.
        bias_attr (ParamAttr|bool, Optional): The parameter attribute for the bias of
            deformable conv layer. If it is set to False, no bias will be added
            to the output units. If it is set to None or one attribute of ParamAttr, conv2d
            will create ParamAttr as bias_attr. If the Initializer of the bias_attr
            is not set, the bias is initialized zero. Default: None.
        modulated (bool): Make sure which version should be used between v1 and v2, where v2 is \
            used while True. Default: True.
        name(str, Optional): For details, please refer to :ref:`api_guide_Name`.
                        Generally, no setting is required. Default: None.
    Returns:
        Tensor: The tensor variable storing the deformable convolution \
                  result. A Tensor with type float32, float64.
    Examples:
        .. code-block:: python

          #deformable conv v2:

              import paddle
              paddle.enable_static()

              C_in, H_in, W_in = 3, 32, 32
              filter_size, deformable_groups = 3, 1
              data = paddle.static.data(name='data', shape=[None, C_in, H_in, W_in], dtype='float32')
              offset = paddle.static.data(name='offset', shape=[None, 2*deformable_groups*filter_size**2, H_in, W_in], dtype='float32')
              mask = paddle.static.data(name='mask', shape=[None, deformable_groups*filter_size**2, H_in, W_in], dtype='float32')
              out = paddle.static.layers.common.deformable_conv(input=data, offset=offset, mask=mask,
                                                 num_filters=2, filter_size=filter_size, padding=1, modulated=True)

              #deformable conv v1:

              import paddle
              C_in, H_in, W_in = 3, 32, 32
              filter_size, deformable_groups = 3, 1
              data = paddle.static.data(name='data', shape=[None, C_in, H_in, W_in], dtype='float32')
              offset = paddle.static.data(name='offset', shape=[None, 2*deformable_groups*filter_size**2, H_in, W_in], dtype='float32')
              out = paddle.static.layers.common.deformable_conv(input=data, offset=offset, mask=None,
                                                 num_filters=2, filter_size=filter_size, padding=1, modulated=False)
    """

    check_variable_and_dtype(
        input, "input", ['float32', 'float64'], 'deformable_conv'
    )
    check_variable_and_dtype(
        offset, "offset", ['float32', 'float64'], 'deformable_conv'
    )
    check_type(
        mask, 'mask', (paddle.static.Variable, type(None)), 'deformable_conv'
    )

    num_channels = input.shape[1]
    assert param_attr is not False, "param_attr should not be False here."

    helper = LayerHelper('deformable_conv', **locals())
    dtype = helper.input_dtype()

    if not isinstance(input, paddle.static.Variable):
        raise TypeError("Input of deformable_conv must be Variable")
    if not isinstance(offset, paddle.static.Variable):
        raise TypeError("Input Offset of deformable_conv must be Variable")

    if groups is None:
        num_filter_channels = num_channels
    else:
        if num_channels % groups != 0:
            raise ValueError("num_channels must be divisible by groups.")
        num_filter_channels = num_channels // groups

    filter_size = utils.convert_to_list(filter_size, 2, 'filter_size')
    stride = utils.convert_to_list(stride, 2, 'stride')
    padding = utils.convert_to_list(padding, 2, 'padding')
    dilation = utils.convert_to_list(dilation, 2, 'dilation')

    input_shape = input.shape
    filter_shape = [num_filters, int(num_filter_channels)] + filter_size

    def _get_default_param_initializer():
        filter_elem_num = filter_size[0] * filter_size[1] * num_channels
        if filter_elem_num <= 0:
            raise ValueError(
                "Invalid filter number, excepted number is larger than 0, but"
                " received {}, please check the input shape and "
                "filter size.".format(filter_elem_num)
            )
        std = (2.0 / filter_elem_num) ** 0.5
        return paddle.nn.initializer.normal.NormalInitializer(0.0, std, 0)

    filter_param = helper.create_parameter(
        attr=helper.param_attr,
        shape=filter_shape,
        dtype=dtype,
        default_initializer=_get_default_param_initializer(),
    )

    pre_bias = helper.create_variable_for_type_inference(dtype)

    if modulated:
        helper.append_op(
            type='deformable_conv',
            inputs={
                'Input': input,
                'Filter': filter_param,
                'Offset': offset,
                'Mask': mask,
            },
            outputs={"Output": pre_bias},
            attrs={
                'strides': stride,
                'paddings': padding,
                'dilations': dilation,
                'groups': groups,
                'deformable_groups': deformable_groups,
                'im2col_step': im2col_step,
            },
        )

    else:
        helper.append_op(
            type='deformable_conv_v1',
            inputs={
                'Input': input,
                'Filter': filter_param,
                'Offset': offset,
            },
            outputs={"Output": pre_bias},
            attrs={
                'strides': stride,
                'paddings': padding,
                'dilations': dilation,
                'groups': groups,
                'deformable_groups': deformable_groups,
                'im2col_step': im2col_step,
            },
        )

    output = helper.append_bias_op(pre_bias, dim_start=1, dim_end=2)
    return output


@static_only
def deform_conv2d(
    x,
    offset,
    mask,
    num_filters,
    filter_size,
    stride=1,
    padding=0,
    dilation=1,
    groups=1,
    deformable_groups=1,
    im2col_step=1,
    weight_attr=None,
    bias_attr=None,
    name=None,
):
    r"""

    Compute 2-D deformable convolution on 4-D input.
    Given input image x, output feature map y, the deformable convolution operation can be expressed as follow:


    Deformable Convolution v2:

    .. math::

        y(p) = \sum_{k=1}^{K}{w_k * x(p + p_k + \Delta p_k) * \Delta m_k}

    Deformable Convolution v1:

    .. math::

        y(p) = \sum_{k=1}^{K}{w_k * x(p + p_k + \Delta p_k)}

    Where :math:`\Delta p_k` and :math:`\Delta m_k` are the learnable offset and modulation scalar for the k-th location,
    Which :math:`\Delta m_k` is one in deformable convolution v1. Please refer to `Deformable ConvNets v2: More Deformable, Better Results
    <https://arxiv.org/abs/1811.11168v2>`_ and `Deformable Convolutional Networks <https://arxiv.org/abs/1703.06211>`_.

    Example:
        - Input:

          X shape: :math:`(N, C_{in}, H_{in}, W_{in})`

          Filter shape: :math:`(C_{out}, C_{in}, H_f, W_f)`

          Offset shape: :math:`(N, 2 * deformable\_groups * H_f * H_w, H_{in}, W_{in})`

          Mask shape: :math:`(N, deformable\_groups * H_f * H_w, H_{in}, W_{in})`

        - Output:

          Output shape: :math:`(N, C_{out}, H_{out}, W_{out})`

        Where

        .. math::

            H_{out}&= \\frac{(H_{in} + 2 * paddings[0] - (dilations[0] * (H_f - 1) + 1))}{strides[0]} + 1 \\\\
            W_{out}&= \\frac{(W_{in} + 2 * paddings[1] - (dilations[1] * (W_f - 1) + 1))}{strides[1]} + 1

    Args:
        x (Tensor): The input image with [N, C, H, W] format. A Tensor with type
            float32, float64.
        offset (Tensor): The input coordinate offset of deformable convolution layer.
            A Tensor with type float32, float64.
        mask (Tensor, Optional): The input mask of deformable convolution layer.
            A Tensor with type float32, float64. It should be None when you use
            deformable convolution v1.
        num_filters(int): The number of filter. It is as same as the output
            image channel.
        filter_size (int|list|tuple): The filter size. If filter_size is a list/tuple,
            it must contain two integers, (filter_size_H, filter_size_W).
            Otherwise, the filter will be a square.
        stride (int|list|tuple, Optional): The stride size. If stride is a list/tuple, it must
            contain two integers, (stride_H, stride_W). Otherwise, the
            stride_H = stride_W = stride. Default: stride = 1.
        padding (int|list|tuple, Optional): The padding size. If padding is a list/tuple, it must
            contain two integers, (padding_H, padding_W). Otherwise, the
            padding_H = padding_W = padding. Default: padding = 0.
        dilation (int|list|tuple, Optional): The dilation size. If dilation is a list/tuple, it must
            contain two integers, (dilation_H, dilation_W). Otherwise, the
            dilation_H = dilation_W = dilation. Default: dilation = 1.
        groups (int, Optional): The groups number of the deformable conv layer. According to
            grouped convolution in Alex Krizhevsky's Deep CNN paper: when group=2,
            the first half of the filters is only connected to the first half
            of the input channels, while the second half of the filters is only
            connected to the second half of the input channels. Default: groups=1.
        deformable_groups (int, Optional): The number of deformable group partitions.
            Default: deformable_groups = 1.
        im2col_step (int, Optional): Maximum number of images per im2col computation;
            The total batch size should be devisable by this value or smaller
            than this value; if you face out of memory problem, you can try
            to use a smaller value here.
            Default: im2col_step = 1.
        weight_attr (ParamAttr, Optional): The parameter attribute for learnable parameters/weights
            of deformable conv. If it is set to None or one attribute of ParamAttr,
            deformable conv will create ParamAttr as weight_attr.
            If the Initializer of the weight_attr is not set, the parameter is
            initialized with :math:`Normal(0.0, std)`, and the
            :math:`std` is :math:`(\\frac{2.0 }{filter\_elem\_num})^{0.5}`. Default: None.
        bias_attr (ParamAttr|bool, Optional): The parameter attribute for the bias of
            deformable conv layer. If it is set to False, no bias will be added
            to the output units. If it is set to None or one attribute of ParamAttr, conv2d
            will create ParamAttr as bias_attr. If the Initializer of the bias_attr
            is not set, the bias is initialized zero. Default: None.
        name(str, Optional): For details, please refer to :ref:`api_guide_Name`.
                        Generally, no setting is required. Default: None.
    Returns:
        Tensor: The tensor storing the deformable convolution \
                  result. A Tensor with type float32, float64.

    Examples:
        .. code-block:: python

          #deformable conv v2:

          import paddle
          paddle.enable_static()

          C_in, H_in, W_in = 3, 32, 32
          filter_size, deformable_groups = 3, 1
          data = paddle.static.data(name='data', shape=[None, C_in, H_in, W_in], dtype='float32')
          offset = paddle.static.data(name='offset', shape=[None, 2*deformable_groups*filter_size**2, H_in, W_in], dtype='float32')
          mask = paddle.static.data(name='mask', shape=[None, deformable_groups*filter_size**2, H_in, W_in], dtype='float32')
          out = paddle.static.nn.deform_conv2d(x=data, offset=offset, mask=mask,
                                             num_filters=2, filter_size=filter_size, padding=1)

          #deformable conv v1:

          import paddle
          paddle.enable_static()

          C_in, H_in, W_in = 3, 32, 32
          filter_size, deformable_groups = 3, 1
          data = paddle.static.data(name='data', shape=[None, C_in, H_in, W_in], dtype='float32')
          offset = paddle.static.data(name='offset', shape=[None, 2*deformable_groups*filter_size**2, H_in, W_in], dtype='float32')
          out = paddle.static.nn.deform_conv2d(x=data, offset=offset, mask=None,
                                             num_filters=2, filter_size=filter_size, padding=1)
    """

    if mask is None:
        return deformable_conv(
            input=x,
            offset=offset,
            mask=mask,
            num_filters=num_filters,
            filter_size=filter_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            deformable_groups=deformable_groups,
            im2col_step=im2col_step,
            param_attr=weight_attr,
            bias_attr=bias_attr,
            modulated=False,
            name=name,
        )
    else:
        return deformable_conv(
            input=x,
            offset=offset,
            mask=mask,
            num_filters=num_filters,
            filter_size=filter_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            deformable_groups=deformable_groups,
            im2col_step=im2col_step,
            param_attr=weight_attr,
            bias_attr=bias_attr,
            modulated=True,
            name=name,
        )
