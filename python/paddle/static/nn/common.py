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
from paddle.fluid.param_attr import ParamAttr
from paddle.fluid.initializer import Normal, Constant
from paddle.fluid.framework import static_only, Variable, _non_static_mode
from paddle.fluid.layers.layer_function_generator import templatedoc

from paddle.fluid.data_feeder import check_dtype

from paddle.common_ops_import import (
    check_type,
    check_variable_and_dtype,
    utils,
    LayerHelper,
)
from functools import reduce

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
    helper = LayerHelper("fc", **locals())
    check_type(input, 'input', (list, tuple, Variable), 'fc')
    if isinstance(input, (list, tuple)):
        for i, input_x in enumerate(input):
            check_type(input_x, 'input[' + str(i) + ']', Variable, 'fc')
    dtype = helper.input_dtype()
    check_dtype(
        dtype, 'input', ['float16', 'uint16', 'float32', 'float64'], 'fc'
    )
    mul_results = []
    for input_var, weight_attr in helper.iter_inputs_and_params():
        input_shape = input_var.shape
        if num_flatten_dims == -1:
            num_flatten_dims = len(input_shape) - 1
        param_shape = [
            reduce(lambda a, b: a * b, input_shape[num_flatten_dims:], 1)
        ] + [size]

        w = helper.create_parameter(
            attr=weight_attr, shape=param_shape, dtype=dtype, is_bias=False
        )
        tmp = helper.create_variable_for_type_inference(dtype)
        helper.append_op(
            type="mul",
            inputs={"X": input_var, "Y": w},
            outputs={"Out": tmp},
            attrs={"x_num_col_dims": num_flatten_dims, "y_num_col_dims": 1},
        )
        mul_results.append(tmp)

    if len(mul_results) == 1:
        pre_bias = mul_results[0]
    else:
        pre_bias = helper.create_variable_for_type_inference(dtype)
        helper.append_op(
            type="sum",
            inputs={"X": mul_results},
            outputs={"Out": pre_bias},
            attrs={"use_mkldnn": False},
        )
    # add bias
    pre_activation = helper.append_bias_op(pre_bias, dim_start=num_flatten_dims)
    # add activation
    return helper.append_activation(pre_activation)


def instance_norm(
    input, epsilon=1e-05, param_attr=None, bias_attr=None, name=None
):
    r"""
    :api_attr: Static Graph

    **Instance Normalization Layer**

    Can be used as a normalizer function for convolution or fully_connected operations.
    The required data format for this layer is one of the following:

    DataLayout: NCHW `[batch, in_channels, in_height, in_width]`

    Refer to `Instance Normalization: The Missing Ingredient for
    Fast Stylization <https://arxiv.org/pdf/1607.08022.pdf>`_
    for more details.

    :math:`input` is the input features over a mini-batch.

    ..  math::

        \\mu_{\\beta} &\\gets \\frac{1}{HW} \\sum_{i=1}^{HW} x_i \\qquad &//\\
        \\ mean\ of\ one\  feature\ map\ in\ mini-batch \\\\
        \\sigma_{\\beta}^{2} &\\gets \\frac{1}{HW} \\sum_{i=1}^{HW}(x_i - \\
        \\mu_{\\beta})^2 \\qquad &//\ variance\ of\ one\ feature\ map\ in\ mini-batch \\\\
        \\hat{x_i} &\\gets \\frac{x_i - \\mu_\\beta} {\\sqrt{\\
        \\sigma_{\\beta}^{2} + \\epsilon}} \\qquad &//\ normalize \\\\
        y_i &\\gets \\gamma \\hat{x_i} + \\beta \\qquad &//\ scale\ and\ shift

    Note:
        `H` means height of feature map, `W` means width of feature map.

    Args:
        input(Tensor): The rank of input tensor can be 2, 3, 4, 5.
            The data type is float32 or float64.
        epsilon(float, Default 1e-05): A value added to the denominator for
            numerical stability. Default is 1e-5.
        param_attr(ParamAttr|None|bool, optional): The parameter attribute for Parameter `scale`
             of instance_norm. If it is set to None or one attribute of ParamAttr, instance_norm
         will create ParamAttr as param_attr, the name of scale can be set in ParamAttr.
         If the Initializer of the param_attr is not set, the parameter is initialized
         with Xavier. If the param_attr is set to False, instance_norm will not create param_attr.
             Default: None.
        bias_attr(ParamAttr|None|bool, optional): The parameter attribute for the bias of instance_norm.
             If it is set to None or one attribute of ParamAttr, instance_norm
         will create ParamAttr as bias_attr, the name of bias can be set in ParamAttr.
         If the Initializer of the bias_attr is not set, the bias is initialized zero.
             If the bias_attr is set to False, instance_norm will not create bias_attr.
         Default: None.
        name(string, Default None): A name for this layer(optional). If set None, the layer
            will be named automatically.

    Returns:
        A Tensor which is the result after applying instance normalization on the input,
        has same shape and data type with input.

    Examples:

        .. code-block:: python

            import paddle
            paddle.enable_static()
            x = paddle.static.data(name='x', shape=[3, 7, 3, 7], dtype='float32')
            hidden1 = paddle.static.nn.fc(x, size=200)
            hidden2 = paddle.static.nn.instance_norm(hidden1)
    """
    check_variable_and_dtype(
        input, 'input', ['float32', 'float64'], 'instance_norm'
    )
    if param_attr is False:
        assert (
            bias_attr is False
        ), "param_attr and bias_attr must be set to False at the same time in instance_norm"

    helper = LayerHelper('instance_norm', **locals())
    dtype = helper.input_dtype()

    # use fp32 for in parameter
    if dtype == paddle.framework.core.VarDesc.VarType.FP16:
        dtype = paddle.framework.core.VarDesc.VarType.FP32

    input_shape = input.shape
    if len(input.shape) < 2 or len(input.shape) > 5:
        raise ValueError(
            'expected 2D or 3D or 4D or 5D input (got {}D input, input shape is: {})'.format(
                len(input.shape), input_shape
            )
        )
    channel_num = input_shape[1]

    param_shape = [channel_num]

    if param_attr and bias_attr:
        # create parameter
        scale = helper.create_parameter(
            attr=helper.param_attr,
            shape=param_shape,
            dtype=dtype,
            default_initializer=Constant(1.0),
        )
        bias = helper.create_parameter(
            attr=helper.bias_attr,
            shape=param_shape,
            dtype=dtype,
            is_bias=True,
            default_initializer=Constant(0.0),
        )

    # create output
    saved_mean = helper.create_variable_for_type_inference(
        dtype=dtype, stop_gradient=True
    )
    saved_variance = helper.create_variable_for_type_inference(
        dtype=dtype, stop_gradient=True
    )

    instance_norm_out = helper.create_variable_for_type_inference(dtype)

    inputs = {"X": input}
    if param_attr and bias_attr:
        inputs["Scale"] = scale
        inputs["Bias"] = bias

    helper.append_op(
        type="instance_norm",
        inputs=inputs,
        outputs={
            "Y": instance_norm_out,
            "SavedMean": saved_mean,
            "SavedVariance": saved_variance,
        },
        attrs={
            "epsilon": epsilon,
        },
    )

    return instance_norm_out


@static_only
def data_norm(
    input,
    act=None,
    epsilon=1e-05,
    param_attr=None,
    data_layout='NCHW',
    in_place=False,
    name=None,
    moving_mean_name=None,
    moving_variance_name=None,
    do_model_average_for_mean_and_var=True,
    slot_dim=-1,
    sync_stats=False,
    summary_decay_rate=0.9999999,
    enable_scale_and_shift=False,
):
    r"""
    :api_attr: Static Graph

    **Data Normalization Layer**

    This op can be used as a normalizer function for conv2d and fully_connected operations.
    The required data format for this layer is one of the following:

    1. NHWC `[batch, in_height, in_width, in_channels]`

    2. NCHW `[batch, in_channels, in_height, in_width]`

    :math:`input` is the input features over a mini-batch.

    ..  math::

        \\mu_{\\beta} &\\gets \\frac{1}{m} \\sum_{i=1}^{m} x_i \\qquad &//\\
        \ mini-batch\ mean \\\\
        \\sigma_{\\beta}^{2} &\\gets \\frac{1}{m} \\sum_{i=1}^{m}(x_i - \\
        \\mu_{\\beta})^2 \\qquad &//\ mini-batch\ variance \\\\
        \\hat{x_i} &\\gets \\frac{x_i - \\mu_\\beta} {\\sqrt{\\
        \\sigma_{\\beta}^{2} + \\epsilon}} \\qquad &//\ normalize \\\\
        y_i &\\gets \\gamma \\hat{x_i} + \\beta \\qquad &//\ scale\ and\ shift

    Args:
        input(Tensor): The input Tensor.
        act(string, Default None): Activation type, linear|relu|prelu|...
        epsilon(float, Default 1e-05):
        param_attr(ParamAttr): The parameter attribute for Parameter `scale`.
        data_layout (str, optional): Specify the data format of the input, and the data format of the output
            will be consistent with that of the input. An optional string from: `"NCHW"`, `"NHWC"`.
            The default is `"NCHW"`. When it is `"NCHW"`, the data is stored in the order of:
            `[batch_size, input_channels, input_height, input_width]`.
        in_place(bool, Default False): Make the input and output of batch norm reuse memory.
        name(string, Default None): A name for this layer(optional). If set None, the layer
            will be named automatically.
        moving_mean_name(string, Default None): The name of moving_mean which store the global Mean.
        moving_variance_name(string, Default None): The name of the moving_variance which store the global Variance.
        do_model_average_for_mean_and_var(bool, Default True): Whether parameter mean and variance
            should do model average when model average is enabled.
        slot_dim(int): The embedding dimension of one slot. Slot is a set of one specific feature. In pslib mode, we
            distinguish feature ids by slot and pull their embeddings from parameter server (pslib). The first
            place of the embedding is the historical show number (occurence time of this feature id with a label 0).
            If the input of this op is concated by slot-wise embeddings, and the show number is zero when this slot
            is new or empty, the normalization result may be impractical. To avoid this, we add slot_dim to locate
            the show number and judge if the show number is zero. If so, we choose to skip normalization on this
            embedding.
        sync_stats(bool, Default False): When running with multiple GPU cards, using allreduce to sync the
            summary messages.
        summary_decay_rate(float, Default 0.9999999): The decay rate when updating summary.
        enable_scale_and_shift(bool, Default False): do scale&shift after normalization.

    Returns:
        Tensor: A tensor which is the result after applying data normalization on the input.

    Examples:

        .. code-block:: python

            import paddle
            paddle.enable_static()

            x = paddle.randn(shape=[32,100])
            hidden2 = paddle.static.nn.data_norm(input=x)
    """
    helper = LayerHelper('data_norm', **locals())
    dtype = helper.input_dtype()

    input_shape = input.shape
    if data_layout == 'NCHW':
        channel_num = input_shape[1]
    else:
        if data_layout == 'NHWC':
            channel_num = input_shape[-1]
        else:
            raise ValueError("unsupported data layout:" + data_layout)

    param_shape = [channel_num]

    batch_size_default = 1e4
    batch_sum_default = 0.0
    batch_square_sum_default = 1e4
    scale_w_default = 1.0
    bias_default = 0.0

    if param_attr and isinstance(param_attr, dict):
        batch_size_default = param_attr.get("batch_size", 1e4)
        batch_sum_default = param_attr.get("batch_sum", 0.0)
        batch_square_sum_default = param_attr.get("batch_square", 1e4)
    if enable_scale_and_shift:
        scale_w_default = param_attr.get("scale_w", 1.0)
        bias_default = param_attr.get("bias", 0.0)

    # create scale and shift(bias) when enable_scale_and_shift is True
    if name is None:
        name = "dn"
    if enable_scale_and_shift:
        scale_w = helper.create_parameter(
            attr=ParamAttr(
                name=name + '.scale_w',
                initializer=Constant(value=float(scale_w_default)),
                trainable=True,
            ),
            shape=param_shape,
            dtype=input.dtype,
        )
        bias = helper.create_parameter(
            attr=ParamAttr(
                name=name + '.bias',
                initializer=Constant(value=float(bias_default)),
                trainable=True,
            ),
            shape=param_shape,
            dtype=input.dtype,
        )
    # create parameter
    batch_size = helper.create_parameter(
        attr=ParamAttr(
            name=name + '.batch_size',
            initializer=Constant(value=float(batch_size_default)),
            trainable=True,
        ),
        shape=param_shape,
        dtype=input.dtype,
    )

    batch_sum = helper.create_parameter(
        attr=ParamAttr(
            name=name + '.batch_sum',
            initializer=Constant(value=float(batch_sum_default)),
            trainable=True,
        ),
        shape=param_shape,
        dtype=input.dtype,
    )

    batch_square_sum = helper.create_parameter(
        attr=ParamAttr(
            name=name + '.batch_square_sum',
            initializer=Constant(value=float(batch_square_sum_default)),
            trainable=True,
        ),
        shape=param_shape,
        dtype=input.dtype,
    )

    means = helper.create_variable(dtype=dtype, stop_gradient=True)
    scales = helper.create_variable(dtype=dtype, stop_gradient=True)

    data_norm_out = input if in_place else helper.create_variable(dtype=dtype)

    inputs = {
        "X": input,
        "BatchSize": batch_size,
        "BatchSum": batch_sum,
        "BatchSquareSum": batch_square_sum,
    }
    attrs = {
        "epsilon": epsilon,
        "data_layout": data_layout,
        "sync_stats": sync_stats,
        "summary_decay_rate": summary_decay_rate,
    }
    if slot_dim > 0:
        attrs["slot_dim"] = slot_dim
    if enable_scale_and_shift:
        attrs["enable_scale_and_shift"] = enable_scale_and_shift
    if enable_scale_and_shift:
        inputs["scale_w"] = scale_w
        inputs["bias"] = bias
    helper.append_op(
        type="data_norm",
        inputs=inputs,
        outputs={
            "Y": data_norm_out,
            "Means": means,
            "Scales": scales,
            "BatchSize": batch_size,
            "BatchSum": batch_sum,
            "BatchSquareSum": batch_square_sum,
        },
        attrs=attrs,
    )

    return helper.append_activation(data_norm_out)


@templatedoc()
def group_norm(
    input,
    groups,
    epsilon=1e-05,
    param_attr=None,
    bias_attr=None,
    act=None,
    data_layout='NCHW',
    name=None,
):
    """
    :api_attr: Static Graph

    **Group Normalization Layer**

    Refer to `Group Normalization <https://arxiv.org/abs/1803.08494>`_ .

    Parameters:
        input(Tensor): Tensor with dimension greater than 1, the data type is float32 or float64.
        groups(int): The number of groups that divided from channels, the data type
            is int32.
        epsilon(float, optional): The small value added to the variance to prevent
            division by zero, the data type is float32. Default: 1e-05.
        param_attr(ParamAttr|bool, optional): ParamAttr object that specifies weight parameter
            attribute. If a bool type, only False is supported, which means there is no weight parameter.
            Default: None, the default weight parameter attribute is used. For more information, please
            refer to :ref:`api_guide_ParamAttr` .
        bias_attr(ParamAttr|bool, optional): ParamAttr object that specifies bias parameter
            attribute. If a bool type, only False is supported, which means there is no bias parameter.
            Default: None, the default bias parameter attribute is used. For more information, please
            refer to :ref:`api_guide_ParamAttr` .
        act(str, optional): Activation to be applied to the output of group normalization.
        data_layout(str, optional): Specify the data format of the input, and the data format of the output
            will be consistent with that of the input. An optional string from: `"NCHW"`, `"NHWC"`.
            The default is `"NCHW"`. When it is `"NCHW"`, the data is stored in the order of:
            `[batch_size, input_channels, *]`.
        name (str, optional): The default value is None. Normally there is no need for user to set this
            property. For more information, please refer to :ref:`api_guide_Name` .

    Returns:
        Tensor: A Tensor has same data type and data format with `input`.

    Examples:
       .. code-block:: python

            import paddle
            paddle.enable_static()

            data = paddle.static.data(name='data', shape=[2, 8, 32, 32], dtype='float32')
            x = paddle.static.nn.group_norm(input=data, groups=4)
            print(x.shape) # [2, 8, 32, 32]
    """
    helper = LayerHelper('group_norm', **locals())
    dtype = helper.input_dtype()
    check_variable_and_dtype(
        input, 'input', ['float32', 'float64'], 'group_norm'
    )
    # create intput and parameters
    inputs = {'X': input}
    input_shape = input.shape
    if len(input_shape) < 2:
        raise ValueError(
            f"The dimensions of Op(static.nn.group_norm)'s input should be more than 1. But received {len(input_shape)}"
        )
    if data_layout != 'NCHW' and data_layout != 'NHWC':
        raise ValueError(
            "Param(data_layout) of Op(static.nn.group_norm) got wrong value: received "
            + data_layout
            + " but only NCHW or NHWC supported."
        )
    channel_num = input_shape[1] if data_layout == 'NCHW' else input_shape[-1]
    param_shape = [channel_num]
    if param_attr:
        scale = helper.create_parameter(
            attr=helper.param_attr,
            shape=param_shape,
            dtype=dtype,
            default_initializer=Constant(1.0),
        )
        inputs['Scale'] = scale
    if bias_attr:
        bias = helper.create_parameter(
            attr=helper.bias_attr, shape=param_shape, dtype=dtype, is_bias=True
        )
        inputs['Bias'] = bias

    # create output
    mean_out = helper.create_variable(dtype=dtype, stop_gradient=True)
    variance_out = helper.create_variable(dtype=dtype, stop_gradient=True)
    group_norm_out = helper.create_variable(dtype=dtype)

    helper.append_op(
        type="group_norm",
        inputs=inputs,
        outputs={
            "Y": group_norm_out,
            "Mean": mean_out,
            "Variance": variance_out,
        },
        attrs={
            "epsilon": epsilon,
            "groups": groups,
            "data_layout": data_layout,
        },
    )

    return helper.append_activation(group_norm_out)


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


def conv2d_transpose(
    input,
    num_filters,
    output_size=None,
    filter_size=None,
    padding=0,
    stride=1,
    dilation=1,
    groups=None,
    param_attr=None,
    bias_attr=None,
    use_cudnn=True,
    act=None,
    name=None,
    data_format='NCHW',
):
    r"""
    :api_attr: Static Graph

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
        input(Tensor): 4-D Tensor with [N, C, H, W] or [N, H, W, C] format,
                         its data type is float32 or float64.
        num_filters(int): The number of the filter. It is as same as the output
            image channel.
        output_size(int|tuple, optional): The output image size. If output size is a
            tuple, it must contain two integers, (image_height, image_width). None if use
            filter_size, padding, and stride to calculate output_size.
            If output_size and filter_size are specified at the same time, They
            should follow the formula above. Default: None. output_size and filter_size
            should not be None at the same time.
        filter_size(int|tuple, optional): The filter size. If filter_size is a tuple,
            it must contain two integers, (filter_size_height, filter_size_width).
            Otherwise, filter_size_height = filter_size_width = filter_size. None if
            use output size to calculate filter_size. Default: None. filter_size and
            output_size should not be None at the same time.
        stride(int|tuple, optional): The stride size. It means the stride in transposed convolution.
            If stride is a tuple, it must contain two integers, (stride_height, stride_width).
            Otherwise, stride_height = stride_width = stride. Default: stride = 1.
        padding(str|int|list|tuple, optional): The padding size. It means the number of zero-paddings
            on both sides for each dimension. If `padding` is a string, either 'VALID' or
            'SAME' which is the padding algorithm. If `padding` is a tuple or list,
            it could be in three forms: `[pad_height, pad_width]` or
            `[pad_height_top, pad_height_bottom, pad_width_left, pad_width_right]`,
            and when `data_format` is `"NCHW"`, `padding` can be in the form
            `[[0,0], [0,0], [pad_height_top, pad_height_bottom], [pad_width_left, pad_width_right]]`.
            when `data_format` is `"NHWC"`, `padding` can be in the form
            `[[0,0], [pad_height_top, pad_height_bottom], [pad_width_left, pad_width_right], [0,0]]`.
            Default: padding = 0.
        dilation(int|tuple, optional): The dilation size. It means the spacing between the kernel points.
            If dilation is a tuple, it must contain two integers, (dilation_height, dilation_width).
            Otherwise, dilation_height = dilation_width = dilation. Default: dilation = 1.
        filter_size(int|tuple, optional): The filter size. If filter_size is a tuple,
            it must contain two integers, (filter_size_height, filter_size_width).
            Otherwise, filter_size_height = filter_size_width = filter_size. None if
            use output size to calculate filter_size. Default: None.
        groups(int, optional): The groups number of the Conv2d transpose layer. Inspired by
            grouped convolution in Alex Krizhevsky's Deep CNN paper, in which
            when group=2, the first half of the filters is only connected to the
            first half of the input channels, while the second half of the
            filters is only connected to the second half of the input channels.
            Default: groups = 1.
        param_attr (ParamAttr, optional): The parameter attribute for learnable parameters/weights
            of conv2d_transpose. If it is set to None or one attribute of ParamAttr, conv2d_transpose
            will create ParamAttr as param_attr. If the Initializer of the param_attr
            is not set, the parameter is initialized with Xavier. Default: None.
        bias_attr (ParamAttr|bool, optional): The parameter attribute for the bias of conv2d_transpose.
            If it is set to False, no bias will be added to the output units.
            If it is set to None or one attribute of ParamAttr, conv2d_transpose
            will create ParamAttr as bias_attr. If the Initializer of the bias_attr
            is not set, the bias is initialized zero. Default: None.
        use_cudnn(bool, optional): Use cudnn kernel or not, it is valid only when the cudnn
            library is installed. Default: True.
        act (str, optional): Activation type, if it is set to None, activation is not appended.
            Default: None.
        name(str, optional): For detailed information, please refer
           to :ref:`api_guide_Name`. Usually name is no need to set and
           None by default.
        data_format (str, optional): Specify the data format of the input, and the data format of the output
            will be consistent with that of the input. An optional string from: `"NCHW"`, `"NHWC"`.
            The default is `"NCHW"`. When it is `"NCHW"`, the data is stored in the order of:
            `[batch_size, input_channels, input_height, input_width]`.

    Returns:
        A Tensor representing the conv2d_transpose, whose
        data type is the same with input and shape is (num_batches, channels, out_h,
        out_w) or (num_batches, out_h, out_w, channels). If act is None, the tensor
        storing the transposed convolution result, and if act is not None, the
        tensor storing transposed convolution and non-linearity activation
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

          import paddle
          paddle.enable_static()

          data = paddle.static.data(name='data', shape=[None, 3, 32, 32], dtype='float32')
          conv2d_transpose = paddle.static.nn.conv2d_transpose(input=data, num_filters=2, filter_size=3)
          print(conv2d_transpose.shape) # [-1, 2, 34, 34]
    """
    assert (
        param_attr is not False
    ), "param_attr should not be False in conv2d_transpose."
    if len(input.shape) != 4:
        raise ValueError(
            "Input size should be 4, "
            "but received {}".format(len(input.shape))
        )

    if data_format not in ['NCHW', 'NHWC']:
        raise ValueError(
            "Attr(data_format) of Op(paddle.static.nn.layers.conv2d_transpose) got wrong value: received "
            + data_format
            + " but only NCHW or NHWC supported."
        )

    input_channel = input.shape[1] if data_format == 'NCHW' else input.shape[-1]
    op_type = 'conv2d_transpose'
    if (
        input_channel == groups
        and num_filters == input_channel
        and not use_cudnn
    ):
        op_type = 'depthwise_conv2d_transpose'

    helper = LayerHelper(op_type, **locals())
    if not isinstance(input, Variable):
        raise TypeError("Input of conv2d_transpose must be Variable")

    stride = utils.convert_to_list(stride, 2, 'stride')
    dilation = utils.convert_to_list(dilation, 2, 'dilation')

    if not isinstance(use_cudnn, bool):
        raise ValueError("use_cudnn should be True or False")

    def _update_padding(padding, data_format):
        def is_list_or_tuple(ele):
            if isinstance(ele, list) or isinstance(ele, tuple):
                return True
            return False

        if is_list_or_tuple(padding) and len(padding) == 4:
            if is_list_or_tuple(padding[0]) and (data_format == "NCHW"):
                if not (padding[0] == [0, 0] and padding[1] == [0, 0]):
                    raise ValueError(
                        "Non-zero padding(%s) in the batch or channel dimensions "
                        "is not supported." % str(padding)
                    )
                padding = padding[2:4]
                padding = [ele for a_list in padding for ele in a_list]
            elif is_list_or_tuple(padding[0]) and (data_format == "NHWC"):
                if not (padding[0] == [0, 0] and padding[3] == [0, 0]):
                    raise ValueError(
                        "Non-zero padding(%s) in the batch or channel dimensions "
                        "is not supported." % str(padding)
                    )
                padding = padding[1:3]
                padding = [ele for a_list in padding for ele in a_list]
            padding = utils.convert_to_list(padding, 4, 'padding')
        else:
            padding = utils.convert_to_list(padding, 2, 'padding')
            padding = [padding[0], padding[0], padding[1], padding[1]]
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
            padding = [0, 0, 0, 0]
        elif padding == "SAME":
            padding_algorithm = "SAME"
            padding = [0, 0, 0, 0]

    padding = _update_padding(padding, data_format)

    if output_size is None:
        output_size = []
    elif isinstance(output_size, (list, tuple)):
        if utils._contain_var(output_size):
            output_size = utils._convert_to_tensor_list(output_size)
        else:
            output_size = utils.convert_to_list(output_size, 2, 'output_size')
    elif isinstance(output_size, int):
        output_size = utils.convert_to_list(output_size, 2, 'output_size')
    elif isinstance(output_size, Variable):
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
            raise ValueError("output_size must contain one or two integers.")
    else:
        raise ValueError(
            "output_size should be int, list[int] or tuple[int] or Tensor"
        )

    if filter_size is None:
        if output_size is []:
            raise ValueError("output_size must be set when filter_size is None")
        if not _non_static_mode():
            if isinstance(output_size, Variable) or utils._contain_var(
                output_size
            ):
                raise ValueError(
                    "filter_size should not be None when output_size is Variable or contain Variable in static mode."
                )
        else:
            output_size = utils.convert_shape_to_list(output_size)
            if len(output_size) == 1:
                output_size = utils.convert_to_list(
                    output_size[0], 2, 'output_size'
                )

        h_in = input.shape[2] if data_format == 'NCHW' else input.shape[1]
        w_in = input.shape[3] if data_format == 'NCHW' else input.shape[2]

        filter_size_h = (
            output_size[0]
            - (h_in - 1) * stride[0]
            + padding[0]
            + padding[1]
            - 1
        ) // dilation[0] + 1
        filter_size_w = (
            output_size[1]
            - (w_in - 1) * stride[1]
            + padding[2]
            + padding[3]
            - 1
        ) // dilation[1] + 1
        filter_size = [filter_size_h, filter_size_w]
    else:
        filter_size = utils.convert_to_list(
            filter_size, 2, 'conv2d_transpose.filter_size'
        )

    if len(padding) == 4 and utils._is_symmetric_padding(padding, 2):
        padding = [padding[0], padding[2]]

    if groups is None:
        groups = 1
    elif groups <= 0:
        raise ValueError(
            "the groups of input must be greater than 0, "
            "but received the groups of input is {}".format(groups)
        )

    filter_shape = [input_channel, num_filters // groups] + filter_size

    img_filter = helper.create_parameter(
        dtype=input.dtype, shape=filter_shape, attr=helper.param_attr
    )

    pre_bias = helper.create_variable_for_type_inference(dtype=input.dtype)
    helper.append_op(
        type=op_type,
        inputs={'Input': [input], 'Filter': [img_filter]},
        outputs={'Output': pre_bias},
        attrs={
            'output_size': output_size,
            'strides': stride,
            'paddings': padding,
            'padding_algorithm': padding_algorithm,
            'dilations': dilation,
            'groups': groups,
            'use_cudnn': use_cudnn,
            'data_format': data_format,
        },
    )

    if data_format == 'NCHW':
        pre_act = helper.append_bias_op(pre_bias, dim_start=1, dim_end=2)
    else:
        pre_act = helper.append_bias_op(pre_bias, dim_start=3, dim_end=4)
    out = helper.append_activation(pre_act)
    return out


def conv3d_transpose(
    input,
    num_filters,
    output_size=None,
    filter_size=None,
    padding=0,
    stride=1,
    dilation=1,
    groups=None,
    param_attr=None,
    bias_attr=None,
    use_cudnn=True,
    act=None,
    name=None,
    data_format='NCDHW',
):
    r"""
    :api_attr: Static Graph

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

        Out = \sigma (W \ast X + b)

    In the above equation:

    * :math:`X`: Input value, a Tensor with NCDHW or NDHWC format.
    * :math:`W`: Filter value, a Tensor with MCDHW format.
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
        input(Tensor): The input is 5-D Tensor with shape [N, C, D, H, W] or [N, D, H, W, C], the data type
            of input is float32 or float64.
        num_filters(int): The number of the filter. It is as same as the output
            image channel.
        output_size(int|tuple, optional): The output image size. If output size is a
            tuple, it must contain three integers, (image_depth, image_height, image_width). This
            parameter only works when filter_size is None. If output_size and filter_size are
            specified at the same time, They should follow the formula above. Default: None.
            Output_size and filter_size should not be None at the same time.
        filter_size(int|tuple, optional): The filter size. If filter_size is a tuple,
            it must contain three integers, (filter_size_depth, filter_size_height,
            filter_size_width). Otherwise, filter_size_depth = filter_size_height = \
            filter_size_width = filter_size. None if use output size to
            calculate filter_size. Default: None. filter_size and output_size should not be
            None at the same time.
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
        param_attr (ParamAttr, optional): The parameter attribute for learnable parameters/weights
            of conv3d_transpose. If it is set to None or one attribute of ParamAttr, conv3d_transpose
            will create ParamAttr as param_attr. If the Initializer of the param_attr
            is not set, the parameter is initialized with Xavier. Default: None.
        bias_attr (ParamAttr|bool, optional): The parameter attribute for the bias of conv3d_transpose.
            If it is set to False, no bias will be added to the output units.
            If it is set to None or one attribute of ParamAttr, conv3d_transpose
            will create ParamAttr as bias_attr. If the Initializer of the bias_attr
            is not set, the bias is initialized zero. Default: None.
        use_cudnn(bool, optional): Use cudnn kernel or not, it is valid only when the cudnn
            library is installed. Default: True
        act (str, optional): Activation type, if it is set to None, activation is not appended.
            Default: None.
        name(str, optional): For detailed information, please refer
           to :ref:`api_guide_Name`. Usually name is no need to set and
           None by default.
        data_format (str, optional): Specify the data format of the input, and the data format of the output
            will be consistent with that of the input. An optional string from: `"NCHW"`, `"NHWC"`.
            The default is `"NCHW"`. When it is `"NCHW"`, the data is stored in the order of:
            `[batch_size, input_channels, input_height, input_width]`.

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

          import paddle
          import numpy as np

          paddle.enable_static()
          data = paddle.static.data(name='data', shape=[None, 3, 12, 32, 32], dtype='float32')
          param_attr = paddle.framework.ParamAttr(name='conv3d.weight', initializer=paddle.nn.initializer.XavierNormal(), learning_rate=0.001)
          res = paddle.static.nn.conv3d_transpose(input=data, num_filters=2, filter_size=3, act="relu", param_attr=param_attr)
          place = paddle.CPUPlace()
          exe = paddle.static.Executor(place)
          exe.run(paddle.static.default_startup_program())
          x = np.random.rand(1, 3, 12, 32, 32).astype("float32")
          output = exe.run(feed={"data": x}, fetch_list=[res])
          print(output)
    """
    assert (
        param_attr is not False
    ), "param_attr should not be False in conv3d_transpose."
    if data_format not in ['NCDHW', 'NDHWC']:
        raise ValueError(
            "Param(data_format) of Op(paddle.static.nn.conv3d_transpose) got wrong value: received "
            + data_format
            + " but only NCDHW or NDHWC supported."
        )

    l_type = "conv3d_transpose"
    helper = LayerHelper(l_type, **locals())
    if not isinstance(input, Variable):
        raise TypeError("Input of conv3d_transpose must be Variable")
    if len(input.shape) != 5:
        raise ValueError(
            "Input should be 5D tensor, but received input with the shape of {}".format(
                input.shape
            )
        )
    input_channel = (
        input.shape[1] if data_format == 'NCDHW' else input.shape[-1]
    )

    stride = utils.convert_to_list(stride, 3, 'stride')
    dilation = utils.convert_to_list(dilation, 3, 'dilation')

    if not isinstance(use_cudnn, bool):
        raise ValueError("use_cudnn should be True or False")

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

        elif is_list_or_tuple(padding) and len(padding) == 6:
            padding = utils.convert_to_list(padding, 6, 'padding')

        else:
            padding = utils.convert_to_list(padding, 3, 'padding')
            padding = [
                padding[0],
                padding[0],
                padding[1],
                padding[1],
                padding[2],
                padding[2],
            ]
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
            padding = [0, 0, 0, 0, 0, 0]
        elif padding == "SAME":
            padding_algorithm = "SAME"
            padding = [0, 0, 0, 0, 0, 0]

    padding = _update_padding(padding, data_format)

    if filter_size is None:
        if output_size is None:
            raise ValueError("output_size must be set when filter_size is None")
        if isinstance(output_size, int):
            output_size = [output_size, output_size, output_size]

        d_in = input.shape[2] if data_format == 'NCDHW' else input.shape[1]
        h_in = input.shape[3] if data_format == 'NCDHW' else input.shape[2]
        w_in = input.shape[4] if data_format == 'NCDHW' else input.shape[3]

        filter_size_d = (
            output_size[0]
            - (d_in - 1) * stride[0]
            + padding[0]
            + padding[1]
            - 1
        ) // dilation[0] + 1
        filter_size_h = (
            output_size[1]
            - (h_in - 1) * stride[1]
            + padding[2]
            + padding[3]
            - 1
        ) // dilation[1] + 1
        filter_size_w = (
            output_size[2]
            - (w_in - 1) * stride[2]
            + padding[4]
            + padding[5]
            - 1
        ) // dilation[2] + 1
        filter_size = [filter_size_d, filter_size_h, filter_size_w]
    else:
        filter_size = utils.convert_to_list(
            filter_size, 3, 'conv3d_transpose.filter_size'
        )

    if len(padding) == 6 and utils._is_symmetric_padding(padding, 3):
        padding = [padding[0], padding[2], padding[4]]

    if output_size is None:
        output_size = []
    elif isinstance(output_size, (list, tuple, int)):
        output_size = utils.convert_to_list(output_size, 3, 'output_size')
    else:
        raise ValueError("output_size should be int, list[int] or tuple[int]")

    groups = 1 if groups is None else groups
    if groups <= 0:
        raise ValueError(
            "the groups of conv3d_transpose should be greater than 0. Received groups: {}".format(
                groups
            )
        )
    if num_filters % groups != 0:
        raise ValueError(
            "Attr(num_filters) must be divisible by groups,"
            "Received: Attr(num_filters) is {}, the groups is {}".format(
                num_filters, groups
            )
        )

    filter_shape = [input_channel, num_filters // groups] + filter_size
    img_filter = helper.create_parameter(
        dtype=input.dtype, shape=filter_shape, attr=helper.param_attr
    )

    if data_format == 'NCDHW':
        data_format = 'NCHW'
    if data_format == 'NDHWC':
        data_format = 'NHWC'

    pre_bias = helper.create_variable_for_type_inference(dtype=input.dtype)
    helper.append_op(
        type=l_type,
        inputs={'Input': [input], 'Filter': [img_filter]},
        outputs={'Output': pre_bias},
        attrs={
            'output_size': output_size,
            'strides': stride,
            'paddings': padding,
            'padding_algorithm': padding_algorithm,
            'dilations': dilation,
            'groups': groups,
            'use_cudnn': use_cudnn,
            'data_format': data_format,
        },
    )

    if data_format == 'NCHW':
        pre_act = helper.append_bias_op(pre_bias, dim_start=1, dim_end=2)
    else:
        pre_act = helper.append_bias_op(pre_bias, dim_start=4, dim_end=5)
    out = helper.append_activation(pre_act)
    return out


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


@static_only
def prelu(x, mode, param_attr=None, data_format="NCHW", name=None):
    r"""

    prelu activation.

    .. math::
        prelu(x) = max(0, x) + \alpha * min(0, x)

    There are three modes for the activation:

    .. code-block:: text

        all: All elements share same alpha.
        channel: Elements in same channel share same alpha.
        element: All elements do not share alpha. Each element has its own alpha.

    Parameters:
        x (Tensor): The input Tensor or LoDTensor with data type float32.
        mode (str): The mode for weight sharing.
        param_attr (ParamAttr|None, optional): The parameter attribute for the learnable \
            weight (alpha), it can be create by ParamAttr. None by default. \
            For detailed information, please refer to :ref:`api_paddle_ParamAttr`.
        data_format(str, optional): Data format that specifies the layout of input.
            It may be "NC", "NCL", "NCHW", "NCDHW", "NLC", "NHWC" or "NDHWC". Default: "NCHW".
        name (str, optional): Name for the operation (optional, default is None). \
            For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        Tensor: A tensor with the same shape and data type as x.

    Examples:

        .. code-block:: python

            import paddle
            paddle.enable_static()

            x = paddle.static.data(name="x", shape=[None,5,10,10], dtype="float32")
            mode = 'channel'
            output = paddle.static.nn.prelu(
                x,mode,param_attr=paddle.ParamAttr(name='alpha'))

    """
    check_variable_and_dtype(x, 'x', ['float16', 'float32', 'float64'], 'prelu')

    helper = LayerHelper('prelu', **locals())
    if mode not in ['all', 'channel', 'element']:
        raise ValueError('mode should be one of all, channel, element.')

    alpha_shape = [1]
    if mode == 'channel':

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
            len(x.shape) >= 2
        ), "The size of input shape should be equal or larger than 2 in prelu() when mode is 'channel'"
        # NOTE(zhiqiu): The alpha_shape should be [1, channel] + [1] * len(x.shape[2:]).
        # To be consistent with Prelu, it is simplified.
        # NOTE(zhiqiu): Revert shape to [1, channel, 1, 1] for compatibility with saved model of old version.
        # NOTE(GuoxiaWang): support NHWC data format
        if data_format == 'NHWC':
            alpha_shape = [1, 1, 1, x.shape[-1]]
        else:
            alpha_shape = [1, x.shape[1], 1, 1]

    elif mode == 'element':
        assert (
            len(x.shape) >= 1
        ), "The size of input shape should be equal or larger than 1 in prelu() when mode is 'element'"
        alpha_shape = [1] + list(x.shape)[1:]
    dtype = helper.input_dtype(input_param_name='x')
    alpha = helper.create_parameter(
        attr=helper.param_attr,
        shape=alpha_shape,
        dtype=dtype,
        is_bias=False,
        default_initializer=paddle.nn.initializer.Constant(0.25),
    )

    out = helper.create_variable_for_type_inference(dtype)
    helper.append_op(
        type="prelu",
        inputs={"X": x, 'Alpha': alpha},
        attrs={"mode": mode, "data_format": data_format},
        outputs={"Out": out},
    )
    return out
