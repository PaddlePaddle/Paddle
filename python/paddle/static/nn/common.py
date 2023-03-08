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

import inspect
import warnings
from functools import reduce

import numpy as np

import paddle
from paddle.common_ops_import import (
    LayerHelper,
    check_type,
    check_variable_and_dtype,
    utils,
)
from paddle.fluid import core
from paddle.fluid.data_feeder import check_dtype
from paddle.fluid.framework import Variable, _non_static_mode, static_only
from paddle.fluid.layers.layer_function_generator import templatedoc
from paddle.fluid.param_attr import ParamAttr
from paddle.nn.initializer import Constant, Normal

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

    def fc_fluid(
        input,
        size,
        num_flatten_dims=1,
        param_attr=None,
        bias_attr=None,
        act=None,
        name=None,
    ):
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
        for input_var, param_attr in helper.iter_inputs_and_params():
            input_shape = input_var.shape
            if num_flatten_dims == -1:
                num_flatten_dims = len(input_shape) - 1
            param_shape = [
                reduce(lambda a, b: a * b, input_shape[num_flatten_dims:], 1)
            ] + [size]

            w = helper.create_parameter(
                attr=param_attr, shape=param_shape, dtype=dtype, is_bias=False
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
        pre_activation = helper.append_bias_op(
            pre_bias, dim_start=num_flatten_dims
        )
        # add activation
        return helper.append_activation(pre_activation)

    return fc_fluid(
        input=x,
        size=size,
        num_flatten_dims=num_flatten_dims,
        param_attr=weight_attr,
        bias_attr=bias_attr,
        act=activation,
        name=name,
    )


def instance_norm(
    input, epsilon=1e-05, param_attr=None, bias_attr=None, name=None
):
    r"""

    **Instance Normalization Layer**

    Can be used as a normalizer function for convolution or fully_connected operations.
    The required data format for this layer is one of the following:

    DataLayout: NCHW `[batch, in_channels, in_height, in_width]`

    Refer to `Instance Normalization: The Missing Ingredient for
    Fast Stylization <https://arxiv.org/pdf/1607.08022.pdf>`_
    for more details.

    :math:`input` is the input features over a mini-batch.

    ..  math::

        \mu_{\beta} &\gets \frac{1}{HW} \sum_{i=1}^{HW} x_i \qquad &//
        \ mean\ of\ one\ feature\ map\ in\ mini-batch \\
        \sigma_{\beta}^{2} &\gets \frac{1}{HW} \sum_{i=1}^{HW}(x_i -
        \mu_{\beta})^2 \qquad &//\ variance\ of\ one\ feature\ map\ in\ mini-batch \\
        \hat{x_i} &\gets \frac{x_i - \mu_\beta} {\sqrt{
        \sigma_{\beta}^{2} + \epsilon}} \qquad &//\ normalize \\
        y_i &\gets \gamma \hat{x_i} + \beta \qquad &//\ scale\ and\ shift

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
def continuous_value_model(input, cvm, use_cvm=True):
    r"""
    **continuous_value_model layers**
    Now, this OP is used in CTR project to remove or dispose show and click value in :attr:`input`.
    :attr:`input` is an embedding vector including show and click value, whose shape is :math:`[N, D]` (N is batch size. D is `2 + embedding dim` ).
    Show and click at first two dims of embedding vector D.
    If :attr:`use_cvm` is True, it will calculate :math:`log(show)` and :math:`log(click)` , and output shape is :math:`[N, D]` .
    If :attr:`use_cvm` is False, it will remove show and click from :attr:`input` , and output shape is :math:`[N, D - 2]` .
    :attr:`cvm` is show_click info, whose shape is :math:`[N, 2]` .
    Args:
        input (Variable): The input variable. A 2-D LoDTensor with shape :math:`[N, D]` , where N is the batch size, D is `2 + the embedding dim` . `lod level = 1` .
        A Tensor with type float32, float64.
        cvm (Variable): Show and click variable. A 2-D Tensor with shape :math:`[N, 2]` , where N is the batch size, 2 is show and click.
        A Tensor with type float32, float64.
        use_cvm  (bool):  Use show_click or not. if use, the output dim is the same as input.
                          if not use, the output dim is `input dim - 2` (remove show and click)
    Returns:
        Variable: A 2-D LodTensor with shape :math:`[N, M]` . if :attr:`use_cvm` = True, M is equal to input dim D. if False, M is equal to `D - 2`. \
        A Tensor with same type as input.
    Examples:
        .. code-block:: python
          import paddle.fluid as fluid
          import paddle
          input = paddle.static.data(name="input", shape=[64, 1], dtype="int64")
          label = paddle.static.data(name="label", shape=[64, 1], dtype="int64")
          w0 = paddle.full(shape=(100, 1), fill_value=2).astype(paddle.float32)
          embed = paddle.nn.functional.embedding(
                            input,
                            w0)
          ones = paddle.full_like(label, 1, dtype="int64")
          show_clk = paddle.cast(paddle.concat([ones, label], axis=1), dtype='float32')
          show_clk.stop_gradient = True
          input_with_cvm = paddle.static.nn.continuous_value_model(embed, show_clk, True)
    """
    helper = LayerHelper('cvm', **locals())
    out = helper.create_variable(dtype=input.dtype)
    check_variable_and_dtype(
        input, 'input', ['float16', 'float32', 'float64'], 'cvm'
    )
    helper.append_op(
        type='cvm',
        inputs={'X': [input], 'CVM': [cvm]},
        outputs={'Y': [out]},
        attrs={"use_cvm": use_cvm},
    )
    return out


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

    **Data Normalization Layer**

    This op can be used as a normalizer function for conv2d and fully_connected operations.
    The required data format for this layer is one of the following:

    1. NHWC `[batch, in_height, in_width, in_channels]`

    2. NCHW `[batch, in_channels, in_height, in_width]`

    :math:`input` is the input features over a mini-batch.

    ..  math::

        \mu_{\beta} &\gets \frac{1}{m} \sum_{i=1}^{m} x_i \qquad &//
        \ mini-batch\ mean \\
        \sigma_{\beta}^{2} &\gets \frac{1}{m} \sum_{i=1}^{m}(x_i -
        \mu_{\beta})^2 \qquad &//\ mini-batch\ variance \\
        \hat{x_i} &\gets \frac{x_i - \mu_\beta} {\sqrt{
        \sigma_{\beta}^{2} + \epsilon}} \qquad &//\ normalize \\
        y_i &\gets \gamma \hat{x_i} + \beta \qquad &//\ scale\ and\ shift

    Args:
        input (Tensor): The input Tensor.
        act (str, optional): Activation type, linear|relu|prelu|... Default: None.
        epsilon(float, optional): Whether to add small values ​in​to the variance during calculations
            to prevent division by zero. Default: 1e-05.
        param_attr (ParamAttr, optional): The parameter attribute for Parameter `scale`. Default: None.
        data_layout (str, optional): Specify the data format of the input, and the data format of the output
            will be consistent with that of the input. An optional string from: `"NCHW"`, `"NHWC"`.
            The default is `"NCHW"`. When it is `"NCHW"`, the data is stored in the order of:
            `[batch_size, input_channels, input_height, input_width]`. Default: `"NCHW"`.
        in_place (bool, optional): Make the input and output of batch norm reuse memory. Default: False.
        name (str, optional): A name for this layer (optional). If set None, the layer
            will be named automatically. Default: None.
        moving_mean_name (str, optional): The name of moving_mean which store the global Mean. Default: None.
        moving_variance_name (str, optional): The name of the moving_variance which store the global Variance. Default: None.
        do_model_average_for_mean_and_var (bool, optional): Whether parameter mean and variance
            should do model average when model average is enabled. Default: True.
        slot_dim (int, optional): The embedding dimension of one slot. Slot is a set of one specific feature. In pslib mode,
            we distinguish feature ids by slot and pull their embeddings from parameter server (pslib). The first
            place of the embedding is the historical show number (occurence time of this feature id with a label 0).
            If the input of this op is concated by slot-wise embeddings, and the show number is zero when this slot
            is new or empty, the normalization result may be impractical. To avoid this, we add slot_dim to locate
            the show number and judge if the show number is zero. If so, we choose to skip normalization on this
            embedding. Default: -1.
        sync_stats (bool, optional): When running with multiple GPU cards, using allreduce to sync the
            summary messages. Default: False.
        summary_decay_rate (float, optional): The decay rate when updating summary. Default: 0.9999999.
        enable_scale_and_shift (bool, optional): do scale&shift after normalization. Default: False.

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
    if len(input_shape) < 2:
        raise ValueError(
            "The shape pf Input < 2 (got {}D input, input shape is: {})".format(
                len(input_shape), input_shape
            )
        )
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


def conv2d(
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
    data_format="NCHW",
):
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
        input (Tensor): The input is 4-D Tensor with shape [N, C, H, W], the data type
            of input is float16 or float32 or float64.
        num_filters(int): The number of filter. It is as same as the output
            image channel.
        filter_size (int|tuple): The filter size. If filter_size
            is a tuple, it must contain two integers, (filter_size_height,
            filter_size_width). Otherwise, filter_size_height = filter_size_width =\
            filter_size.
        stride (int|tuple, optional): The stride size. It means the stride in convolution.
            If stride is a tuple, it must contain two integers, (stride_height, stride_width).
            Otherwise, stride_height = stride_width = stride. Default: stride = 1.
        padding (string|int|list|tuple, optional): The padding size. It means the number of zero-paddings
            on both sides for each dimension.If `padding` is a string, either 'VALID' or
            'SAME' which is the padding algorithm. If padding size is a tuple or list,
            it could be in three forms: `[pad_height, pad_width]` or
            `[pad_height_top, pad_height_bottom, pad_width_left, pad_width_right]`, and when
            `data_format` is `"NCHW"`, `padding` can be in the form `[[0,0], [0,0],
            [pad_height_top, pad_height_bottom], [pad_width_left, pad_width_right]]`.
            when `data_format` is `"NHWC"`, `pool_padding` can be in the form
            `[[0,0], [pad_height_top, pad_height_bottom], [pad_width_left, pad_width_right], [0,0]]`.
            Default: padding = 0.
        dilation (int|tuple, optional): The dilation size. It means the spacing between the kernel
            points. If dilation is a tuple, it must contain two integers, (dilation_height,
            dilation_width). Otherwise, dilation_height = dilation_width = dilation.
            Default: dilation = 1.
        groups (int, optional): The groups number of the Conv2d Layer. According to grouped
            convolution in Alex Krizhevsky's Deep CNN paper: when group=2,
            the first half of the filters is only connected to the first half
            of the input channels, while the second half of the filters is only
            connected to the second half of the input channels. Default: groups=1.
        param_attr (ParamAttr|None, optional): The parameter attribute for learnable parameters/weights
            of conv2d. If it is set to None or one attribute of ParamAttr, conv2d
            will create ParamAttr as param_attr. If the Initializer of the param_attr
            is not set, the parameter is initialized with :math:`Normal(0.0, std)`,
            and the :math:`std` is :math:`(\\frac{2.0 }{filter\_elem\_num})^{0.5}`. Default: None.
        bias_attr (ParamAttr|bool|None, optional): The parameter attribute for the bias of conv2d.
            If it is set to False, no bias will be added to the output units.
            If it is set to None or one attribute of ParamAttr, conv2d
            will create ParamAttr as bias_attr. If the Initializer of the bias_attr
            is not set, the bias is initialized zero. Default: None.
        use_cudnn (bool, optional): Use cudnn kernel or not, it is valid only when the cudnn
            library is installed. Default: True
        act (str, optional): Activation type, if it is set to None, activation is not appended.
            Default: None
        name(str|None, optional): For detailed information, please refer
           to :ref:`api_guide_Name`. Usually name is no need to set and
           None by default.
        data_format (str, optional): Specify the data format of the input, and the data format of the output
            will be consistent with that of the input. An optional string from: `"NCHW"`, `"NHWC"`.
            The default is `"NCHW"`. When it is `"NCHW"`, the data is stored in the order of:
            `[batch_size, input_channels, input_height, input_width]`.

    Returns:
        A Tensor representing the conv2d, whose data type is the
        same with input. If act is None, the tensor storing the convolution
        result, and if act is not None, the tensor storing convolution
        and non-linearity activation result.

    Examples:
        .. code-block:: python

          import paddle
          paddle.enable_static()

          data = paddle.static.data(name='data', shape=[None, 3, 32, 32], dtype='float32')
          conv2d = paddle.static.nn.conv2d(input=data, num_filters=2, filter_size=3, act="relu")
          print(conv2d.shape) # [-1, 2, 30, 30]
    """

    check_variable_and_dtype(
        input, 'input', ['float16', 'float32', 'float64'], 'conv2d'
    )
    if len(input.shape) != 4:
        raise ValueError(
            "Input size should be 4, "
            "but received {}".format(len(input.shape))
        )
    num_channels = input.shape[1]
    if not isinstance(use_cudnn, bool):
        raise ValueError(
            "Attr(use_cudnn) should be True or False. Received "
            "Attr(use_cudnn): %s. " % str(use_cudnn)
        )

    if data_format not in ["NCHW", "NHWC"]:
        raise ValueError(
            "Attr(data_format) should be 'NCHW' or 'NHWC'. Received "
            "Attr(data_format): %s." % str(data_format)
        )

    channel_last = data_format == "NHWC"
    num_channels = input.shape[3] if channel_last else input.shape[1]
    if num_channels < 0:
        raise ValueError(
            "The channel dimmention of the input(%s) should be defined. "
            "Received: %s." % (str(input.shape), str(num_channels))
        )
    assert param_attr is not False, "param_attr should not be False here."

    if groups is None:
        num_filter_channels = num_channels
    elif groups <= 0:
        raise ValueError(
            "the groups of input must be greater than 0, "
            "but received the groups of input is {}".format(groups)
        )
    else:
        if num_channels % groups != 0:
            raise ValueError(
                "the channel of input must be divisible by groups,"
                "received: the channel of input is {}, the shape of input is {}"
                ", the groups is {}".format(num_channels, input.shape, groups)
            )
        num_filter_channels = num_channels // groups

    l_type = 'conv2d'
    if (
        num_channels == groups
        and num_filters % num_channels == 0
        and not use_cudnn
    ):
        l_type = 'depthwise_conv2d'

    if (
        num_channels == groups
        and num_filters % num_channels == 0
        and core.is_compiled_with_rocm()
    ):
        l_type = 'depthwise_conv2d'

    # NPU only supports depthwise_conv2d when  "input_channel = output_channel = groups"
    if core.is_compiled_with_npu():
        if num_channels == groups and num_channels == num_filters:
            l_type = 'depthwise_conv2d'
        else:
            l_type = 'conv2d'

    helper = LayerHelper(l_type, **locals())
    dtype = helper.input_dtype()

    filter_size = utils.convert_to_list(filter_size, 2, 'filter_size')
    stride = utils.convert_to_list(stride, 2, 'stride')
    dilation = utils.convert_to_list(dilation, 2, 'dilation')

    # padding
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
            if utils._is_symmetric_padding(padding, 2):
                padding = [padding[0], padding[2]]

        else:
            padding = utils.convert_to_list(padding, 2, 'padding')

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
            padding = [0, 0]
        elif padding == "SAME":
            padding_algorithm = "SAME"
            padding = [0, 0]

    padding = _update_padding(padding, data_format)

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
        return Normal(0.0, std)

    filter_param = helper.create_parameter(
        attr=helper.param_attr,
        shape=filter_shape,
        dtype=dtype,
        default_initializer=_get_default_param_initializer(),
    )

    pre_bias = helper.create_variable_for_type_inference(dtype)

    if (
        core.is_compiled_with_cuda()
        and paddle.fluid.get_flags("FLAGS_conv2d_disable_cudnn")[
            "FLAGS_conv2d_disable_cudnn"
        ]
    ):
        use_cudnn = False

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
            'fuse_relu_before_depthwise_conv': False,
            "padding_algorithm": padding_algorithm,
            "data_format": data_format,
        },
    )

    if data_format == 'NCHW':
        pre_act = helper.append_bias_op(pre_bias, dim_start=1, dim_end=2)
    else:
        pre_act = helper.append_bias_op(pre_bias, dim_start=3, dim_end=4)

    return helper.append_activation(pre_act)


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

        Out = \sigma (W \ast X + b)

    In the above equation:

    * :math:`X`: Input value, a tensor with NCDHW or NDHWC format.
    * :math:`W`: Filter value, a tensor with MCDHW format.
    * :math:`\ast`: Convolution operation.
    * :math:`b`: Bias value, a 2-D tensor with shape [M, 1].
    * :math:`\sigma`: Activation function.
    * :math:`Out`: Output value, the shape of :math:`Out` and :math:`X` may be different.

    Example:

        - Input:

          Input shape: :math:`(N, C_{in}, D_{in}, H_{in}, W_{in})`

          Filter shape: :math:`(C_{out}, C_{in}, D_f, H_f, W_f)`

        - Output:
          Output shape: :math:`(N, C_{out}, D_{out}, H_{out}, W_{out})`

        Where

        .. math::

            D_{out}&= \frac{(D_{in} + 2 * paddings[0] - (dilations[0] * (D_f - 1) + 1))}{strides[0]} + 1 \\
            H_{out}&= \frac{(H_{in} + 2 * paddings[1] - (dilations[1] * (H_f - 1) + 1))}{strides[1]} + 1 \\
            W_{out}&= \frac{(W_{in} + 2 * paddings[2] - (dilations[2] * (W_f - 1) + 1))}{strides[2]} + 1

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
        A Tensor representing the conv3d, whose data type is
        the same with input. If act is None, the tensor variable storing the
        convolution result, and if act is not None, the tensor variable storing
        convolution and non-linearity activation result.

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
        return Normal(0.0, std)

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

        Out = \sigma (W \ast X + b)

    Where:

    * :math:`X`: Input value, a 4-D Tensor with NCHW or NHWC format.
    * :math:`W`: Filter value, a 4-D Tensor with MCHW format.
    * :math:`\ast`: Convolution operation.
    * :math:`b`: Bias value, a 2-D Tensor with shape [M, 1].
    * :math:`\sigma`: Activation function.
    * :math:`Out`: Output value, a 4-D Tensor with data format 'NCHW' or 'NHWC', the shape of :math:`Out` and :math:`X` may be different.

    Example:

        - Input:

          Input shape: :math:`(N, C_{in}, H_{in}, W_{in})`

          Filter shape: :math:`(C_{in}, C_{out}, H_f, W_f)`

        - Output:

          Output shape: :math:`(N, C_{out}, H_{out}, W_{out})`

        Where

        .. math::

           H^\prime_{out} &= (H_{in} - 1) * strides[0] - 2 * paddings[0] + dilations[0] * (H_f - 1) + 1 \\
           W^\prime_{out} &= (W_{in} - 1) * strides[1] - 2 * paddings[1] + dilations[1] * (W_f - 1) + 1 \\
           H_{out} &\in [ H^\prime_{out}, H^\prime_{out} + strides[0] ] \\
           W_{out} &\in [ W^\prime_{out}, W^\prime_{out} + strides[1] ]

        If `padding` = `"SAME"`:

        .. math::
            H^\prime_{out} &= \frac{(H_{in} + stride[0] - 1)}{stride[0]} \\
            W^\prime_{out} &= \frac{(H_{in} + stride[1] - 1)}{stride[1]}

        If `padding` = `"VALID"`:

        .. math::
            H^\prime_{out} &= (H_{in} - 1) * strides[0] + dilations[0] * (H_f - 1) + 1 \\
            W^\prime_{out} &= (W_{in} − 1) * strides[1] + dilations[1] * (W_f − 1) + 1

        If output_size is None, :math:`H_{out} = H^\prime_{out}, W_{out} = W^\prime_{out}`;
        else, the :math:`H_{out}` of the output size must between :math:`H^\prime_{out}`
        and :math:`H^\prime_{out} + strides[0]`, and the :math:`W_{out}` of the output size must
        between :math:`W^\prime_{out}` and :math:`W^\prime_{out} + strides[1]`,

        Since transposed convolution can be treated as the inverse of convolution, and according to the input-output formula for convolution,
        different sized input feature layers may correspond to the same sized output feature layer,
        the size of the output feature layer for a fixed sized input feature layer is not unique to transposed convolution

        If `output_size` is specified, `conv2d_transpose` can compute the kernel size automatically.

    Args:
        input(Tensor): 4-D Tensor with [N, C, H, W] or [N, H, W, C] format where N is the batch_size,
            C is the input_channels, H is the input_height and W is the input_width.
            Its data type is float32 or float64.
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
        padding(str|int|list|tuple, optional): The padding size. It means the number of zero-paddings
            on both sides for each dimension. If `padding` is a string, either 'VALID' or
            'SAME' which is the padding algorithm. If `padding` is a tuple or list,
            it could be in three forms:
            (1) Contains 4 binary groups: when `data_format` is `"NCHW"`, `padding` can be in the form
            `[[0,0], [0,0], [pad_height_top, pad_height_bottom], [pad_width_left, pad_width_right]]`.
            when `data_format` is `"NHWC"`, `padding` can be in the form
            `[[0,0], [pad_height_top, pad_height_bottom], [pad_width_left, pad_width_right], [0,0]]`.
            (2) Contains 4 integer values：`[pad_height_top, pad_height_bottom, pad_width_left, pad_width_right]`
            (3) Contains 2 integer values：`[pad_height, pad_width]`, in this case, `padding_height_top = padding_height_bottom = padding_height`，
            `padding_width_left = padding_width_right = padding_width`. If an integer, `padding_height = padding_width = padding`. Default: padding = 0.
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
        param_attr (ParamAttr, optional): The parameter attribute for learnable parameters/weights
            of conv2d_transpose. If it is set to None or one attribute of ParamAttr, conv2d_transpose
            will create ParamAttr as param_attr. If the Initializer of the param_attr
            is not set, the parameter is initialized with Xavier. Default: None.
        bias_attr (ParamAttr|bool, optional): Specifies the object for the bias parameter attribute.
            The default value is None, which means that the default bias parameter attribute is used.
            For detailed information, please refer to :ref:`paramattr`.
            The default bias initialisation for the conv2d_transpose operator is 0.0.
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

    if num_filters == 0:
        raise ValueError("num of filters should not be 0.")

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
        raise TypeError("Input of conv2d_transpose must be Tensor")

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
                    "filter_size should not be None when output_size is Tensor or contain Tensor in static graph mode."
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

           D^\prime_{out} &= (D_{in} - 1) * strides[0] - 2 * paddings[0] + dilations[0] * (D_f - 1) + 1 \\
           H^\prime_{out} &= (H_{in} - 1) * strides[1] - 2 * paddings[1] + dilations[1] * (H_f - 1) + 1 \\
           W^\prime_{out} &= (W_{in} - 1) * strides[2] - 2 * paddings[2] + dilations[2] * (W_f - 1) + 1 \\
           D_{out} &\in [ D^\prime_{out}, D^\prime_{out} + strides[0] ] \\
           H_{out} &\in [ H^\prime_{out}, H^\prime_{out} + strides[1] ] \\
           W_{out} &\in [ W^\prime_{out}, W^\prime_{out} + strides[2] ]

    If `padding` = `"SAME"`:

        .. math::
            D^\prime_{out} &= \frac{(D_{in} + stride[0] - 1)}{stride[0]} \\
            H^\prime_{out} &= \frac{(H_{in} + stride[1] - 1)}{stride[1]} \\
            W^\prime_{out} &= \frac{(H_{in} + stride[2] - 1)}{stride[2]}

    If `padding` = `"VALID"`:

    .. math::
        D^\prime_{out} &= (D_{in} - 1) * strides[0] + dilations[0] * (D_f - 1) + 1 \\
        H^\prime_{out} &= (H_{in} - 1) * strides[1] + dilations[1] * (H_f - 1) + 1 \\
        W^\prime_{out} &= (W_{in} − 1) * strides[2] + dilations[2] * (W_f − 1) + 1

    If `output_size` is None, :math:`D_{out} = D^\prime_{out}, :math:`H_{out} = \
    H^\prime_{out}, W_{out} = W^\prime_{out}`; else, the specified `output_size_depth` (the depth of the output feature layer) :math:`D_{out}`
    must between :math:`D^\prime_{out}` and :math:`D^\prime_{out} + strides[0]`(not including :math:`D^\prime_{out} + strides[0]`),
    the specified `output_size_height` (the height of the output feature layer) :math:`H_{out}` must between :math:`H^\prime_{out}`
    and :math:`H^\prime_{out} + strides[1]`(not including :math:`H^\prime_{out} + strides[1]`),
    and the specified `output_size_width` (the width of the output feature layer) :math:`W_{out}` must
    between :math:`W^\prime_{out}` and :math:`W^\prime_{out} + strides[2]`(not including :math:`W^\prime_{out} + strides[2]`).

    Since transposed convolution can be treated as the inverse of convolution,
    and since different sized input feature layers may correspond to the same sized output feature layer according to the input-output formula for convolution,
    the size of the output feature layer for a fixed sized input feature layer is not unique to transposed convolution.

    If `output_size` is specified, `conv3d_transpose` can compute the kernel size automatically.

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
        A Tensor representing the conv3d_transpose, whose data
        type is the same with input and shape is (num_batches, channels, out_d, out_h,
        out_w) or (num_batches, out_d, out_h, out_w, channels). If act is None, the tensor
        variable storing the transposed convolution result, and if act is not None, the tensor
        variable storing transposed convolution and non-linearity activation result.

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
        raise TypeError("Input of conv3d_transpose must be Tensor")
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

            H_{out}&= \frac{(H_{in} + 2 * paddings[0] - (dilations[0] * (H_f - 1) + 1))}{strides[0]} + 1 \\
            W_{out}&= \frac{(W_{in} + 2 * paddings[1] - (dilations[1] * (W_f - 1) + 1))}{strides[1]} + 1

    Args:
        input (Tensor): The input image with [N, C, H, W] format. A Tensor with type
            float32, float64.
        offset (Tensor): The input coordinate offset of deformable convolution layer.
            A Tensor with type float32, float64.
        Mask (Tensor, Optional): The input mask of deformable convolution layer.
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

    if input.ndim != 4:
        raise ValueError(
            f'The input should be of [N, C, H, W] format, but received {input.shape}'
        )

    num_channels = input.shape[1]
    assert param_attr is not False, "param_attr should not be False here."

    helper = LayerHelper('deformable_conv', **locals())
    dtype = helper.input_dtype()

    if not isinstance(input, paddle.static.Variable):
        raise TypeError("Input of deformable_conv must be Tensor")
    if not isinstance(offset, paddle.static.Variable):
        raise TypeError("Input Offset of deformable_conv must be Tensor")

    if groups is None:
        num_filter_channels = num_channels
    else:
        if groups == 0:
            raise ValueError("groups should not be 0.")
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
        return paddle.nn.initializer.normal.Normal(0.0, std)

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

            H_{out}&= \frac{(H_{in} + 2 * paddings[0] - (dilations[0] * (H_f - 1) + 1))}{strides[0]} + 1 \\
            W_{out}&= \frac{(W_{in} + 2 * paddings[1] - (dilations[1] * (W_f - 1) + 1))}{strides[1]} + 1

    Args:
        x (Tensor): The input image with [N, C, H, W] format. A Tensor with type
            float32, float64.
        offset (Tensor): The input coordinate offset of deformable convolution layer.
            A Tensor with type float32, float64.
        mask (Tensor): The input mask of deformable convolution layer.
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
            :math:`std` is :math:`(\frac{2.0 }{filter\_elem\_num})^{0.5}`. Default: None.
        bias_attr (ParamAttr|bool, Optional): The parameter attribute for the bias of
            deformable conv layer. If it is set to False, no bias will be added
            to the output units. If it is set to None or one attribute of ParamAttr, conv2d
            will create ParamAttr as bias_attr. If the Initializer of the bias_attr
            is not set, the bias is initialized zero. Default: None.
        name(str, Optional): For details, please refer to :ref:`api_guide_Name`.
                        Generally, no setting is required. Default: None.

    Returns:
        Tensor: The tensor storing the deformable convolution result. A Tensor with type float32, float64.

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


def bilinear_tensor_product(
    x, y, size, act=None, name=None, param_attr=None, bias_attr=None
):
    r"""
    This layer performs bilinear tensor product on two inputs.

    .. math::

       out_{i} = x * W_{i} * {y^\mathrm{T}}, i=0,1,...,size-1

    In this formula:
      - :math:`x`: the first input contains M elements, shape is [batch_size, M].
      - :math:`y`: the second input contains N elements, shape is [batch_size, N].
      - :math:`W_{i}`: the i-th learned weight, shape is [M, N].
      - :math:`out_{i}`: the i-th element of out, shape is [batch_size, size].
      - :math:`y^\mathrm{T}`: the transpose of :math:`y_{2}`.

    Args:
        x (Tensor): 2-D input tensor with shape [batch_size, M]. Data type
            is float32 or float64.
        y (Tensor): 2-D input tensor with shape [batch_size, N]. Data type
            should be same as **x**.
        size (int): The dimension of this layer.
        act (str|None): Activation to be applied to the output of this layer. Default None.
        name(str|None): For detailed information, please refer to
            :ref:`api_guide_Name` . Usually name is no need to set and None by default.
        param_attr (ParamAttr|None): To specify the weight parameter attribute.
            Default: None, which means the default weight parameter property is
            used. See usage for details in :ref:`api_fluid_ParamAttr` .
        bias_attr (ParamAttr|None): To specify the bias parameter attribute.
            Default: None, which means the default bias parameter property is
            used. See usage for details in :ref:`api_fluid_ParamAttr` .

    Returns:
        Tensor, A 2-D Tensor of shape [batch_size, size]. Data type is the same as input **x**.

    Examples:
        .. code-block:: python

            import paddle
            paddle.enable_static()

            x = paddle.static.data("t1", shape=[-1, 5], dtype="float32")
            y = paddle.static.data("t2", shape=[-1, 4], dtype="float32")
            tensor = paddle.static.nn.bilinear_tensor_product(x, y, size=1000)

    """
    helper = LayerHelper('bilinear_tensor_product', **locals())
    dtype = helper.input_dtype('x')
    if len(x.shape) != 2 or len(y.shape) != 2:
        raise ValueError(
            "Input x and y should be 2D tensor, but received x with the shape of {}, y with the shape of {}".format(
                x.shape, y.shape
            )
        )
    param_shape = [size, x.shape[1], y.shape[1]]

    w = helper.create_parameter(
        attr=helper.param_attr, shape=param_shape, dtype=dtype, is_bias=False
    )
    out = helper.create_variable_for_type_inference(dtype=dtype)

    inputs = {"X": x, "Y": y, "Weight": w}
    if helper.bias_attr:
        bias_size = [1, size]
        bias = helper.create_parameter(
            attr=helper.bias_attr, shape=bias_size, dtype=dtype, is_bias=True
        )
        inputs["Bias"] = bias
    helper.append_op(
        type="bilinear_tensor_product", inputs=inputs, outputs={"Out": out}
    )

    # add activation
    return helper.append_activation(out)


def batch_norm(
    input,
    act=None,
    is_test=False,
    momentum=0.9,
    epsilon=1e-05,
    param_attr=None,
    bias_attr=None,
    data_layout='NCHW',
    in_place=False,
    name=None,
    moving_mean_name=None,
    moving_variance_name=None,
    do_model_average_for_mean_and_var=True,
    use_global_stats=False,
):
    r"""

    **Batch Normalization Layer**

    Can be used as a normalizer function for convolution or fully_connected operations.
    The required data format for this layer is one of the following:

    1. NHWC `[batch, in_height, in_width, in_channels]`

    2. NCHW `[batch, in_channels, in_height, in_width]`

    Refer to `Batch Normalization: Accelerating Deep Network Training by Reducing
    Internal Covariate Shift <https://arxiv.org/pdf/1502.03167.pdf>`_
    for more details.

    :math:input is the input features over a mini-batch.

    ..  math::

        \\mu_{\\beta} &\\gets \\frac{1}{m} \\sum_{i=1}^{m} x_i \\qquad &//\\
        \ mini-batch\ mean \\\\
        \\sigma_{\\beta}^{2} &\\gets \\frac{1}{m} \\sum_{i=1}^{m}(x_i - \\
        \\mu_{\\beta})^2 \\qquad &//\ mini-batch\ variance \\\\
        \\hat{x_i} &\\gets \\frac{x_i - \\mu_\\beta} {\\sqrt{\\
        \\sigma_{\\beta}^{2} + \\epsilon}} \\qquad &//\ normalize \\\\
        y_i &\\gets \\gamma \\hat{x_i} + \\beta \\qquad &//\ scale\ and\ shift

        moving\_mean = moving\_mean * momentum + mini-batch\_mean * (1. - momentum) \\\\
        moving\_var = moving\_var * momentum + mini-batch\_var * (1. - momentum)


    moving_mean is global mean and moving_var is global variance.

    When use_global_stats = True, the :math:`\\mu_{\\beta}`
    and :math:`\\sigma_{\\beta}^{2}` are not the statistics of one mini-batch.
    They are global (or running) statistics. (It usually got from the
    pre-trained model.)
    The training and testing (or inference) have the same behavior:

    ..  math::

        \\hat{x_i} &\\gets \\frac{x_i - \\mu_\\beta} {\\sqrt{\\
        \\sigma_{\\beta}^{2} + \\epsilon}}  \\\\
        y_i &\\gets \\gamma \\hat{x_i} + \\beta

    Note:
        if build_strategy.sync_batch_norm=True, the batch_norm in network will use
        sync_batch_norm automatically.
        `is_test = True` can only be used in test program and inference program, `is_test` CANNOT be set to True in train program, if you want to use global status from pre_train model in train program, please set `use_global_stats = True`.

    Args:
        input(Tensor): The rank of input Tensor can be 2, 3, 4, 5. The data type
            is float16 or float32 or float64.
        act(string, Default None): Activation type, linear|relu|prelu|...
        is_test (bool, Default False): A flag indicating whether it is in
            test phrase or not.
        momentum(float|Tensor, Default 0.9): The value used for the moving_mean and
            moving_var computation. This should be a float number or a Tensor with
            shape [1] and data type as float32. The updated formula is:
            :math:`moving\_mean = moving\_mean * momentum + new\_mean * (1. - momentum)`
            :math:`moving\_var = moving\_var * momentum + new\_var * (1. - momentum)`
            Default is 0.9.
        epsilon(float, Default 1e-05): A value added to the denominator for
            numerical stability. Default is 1e-5.
        param_attr(ParamAttr|None): The parameter attribute for Parameter `scale`
             of batch_norm. If it is set to None or one attribute of ParamAttr, batch_norm
         will create ParamAttr as param_attr, the name of scale can be set in ParamAttr.
         If the Initializer of the param_attr is not set, the parameter is initialized
         with Xavier. Default: None.
        bias_attr(ParamAttr|None): The parameter attribute for the bias of batch_norm.
             If it is set to None or one attribute of ParamAttr, batch_norm
         will create ParamAttr as bias_attr, the name of bias can be set in ParamAttr.
         If the Initializer of the bias_attr is not set, the bias is initialized zero.
         Default: None.
        data_layout (str, optional): Specify the data format of the input, and the data format of the output
             will be consistent with that of the input. An optional string from: `"NCHW"`, `"NHWC"`.
             The default is `"NCHW"`. When it is `"NCHW"`, the data is stored in the order of:
             `[batch_size, input_channels, input_height, input_width]`.
        in_place(bool, Default False): Make the input and output of batch norm reuse memory.
        name(str|None): For detailed information, please refer to :ref:`api_guide_Name`.
            Usually name is no need to set and None by default.
        moving_mean_name(str, Default None): The name of moving_mean which store the global Mean. If it
            is set to None, batch_norm will save global mean with a random name, otherwise, batch_norm
            will save global mean with the string.
        moving_variance_name(str, Default None): The name of the moving_variance which store the global Variance.
            If it is set to None, batch_norm will save global variance with a random name, otherwise, batch_norm
            will save global variance with the string.
        do_model_average_for_mean_and_var(bool, Default True): Whether parameter mean and variance should do model
            average when model average is enabled.
        use_global_stats(bool, Default False): Whether to use global mean and
            variance. In inference or test mode, set use_global_stats to true
            or is_test to true, and the behavior is equivalent.
            In train mode, when setting use_global_stats True, the global mean
            and variance are also used during train period.

    Returns:
        A Tensor which is the result after applying batch normalization on the input,
        has same shape and data type with input.

    Examples:

        .. code-block:: python

            import paddle

            paddle.enable_static()
            x = paddle.static.data(name='x', shape=[3, 7, 3, 7], dtype='float32')
            hidden1 = paddle.static.nn.fc(x=x, size=200)
            print(hidden1.shape)
            # [3, 200]
            hidden2 = paddle.static.nn.batch_norm(input=hidden1)
            print(hidden2.shape)
            # [3, 200]
    """
    assert (
        bias_attr is not False
    ), "bias_attr should not be False in batch_norm."
    helper = LayerHelper('batch_norm', **locals())

    check_variable_and_dtype(
        input, 'input', ['float16', 'float32', 'float64'], 'batch_norm'
    )
    dtype = helper.input_dtype()

    # use fp32 for bn parameter
    if dtype == core.VarDesc.VarType.FP16:
        dtype = core.VarDesc.VarType.FP32

    input_shape = input.shape
    if len(input.shape) < 2 or len(input.shape) > 5:
        raise ValueError(
            'expected 2D or 3D or 4D or 5D input (got {}D input, input shape is: {})'.format(
                len(input.shape), input_shape
            )
        )
    if data_layout == 'NCHW':
        channel_num = input_shape[1]
    else:
        if data_layout == 'NHWC':
            channel_num = input_shape[-1]
        else:
            raise ValueError("unsupported data layout:" + data_layout)

    param_shape = [channel_num]

    # create parameter
    scale = helper.create_parameter(
        attr=helper.param_attr,
        shape=param_shape,
        dtype=dtype,
        default_initializer=paddle.nn.initializer.Constant(1.0),
    )
    bias = helper.create_parameter(
        attr=helper.bias_attr, shape=param_shape, dtype=dtype, is_bias=True
    )

    mean = helper.create_parameter(
        attr=paddle.ParamAttr(
            name=moving_mean_name,
            initializer=paddle.nn.initializer.Constant(0.0),
            trainable=False,
            do_model_average=do_model_average_for_mean_and_var,
        ),
        shape=param_shape,
        dtype=dtype,
    )
    mean.stop_gradient = True

    variance = helper.create_parameter(
        attr=paddle.ParamAttr(
            name=moving_variance_name,
            initializer=paddle.nn.initializer.Constant(1.0),
            trainable=False,
            do_model_average=do_model_average_for_mean_and_var,
        ),
        shape=param_shape,
        dtype=dtype,
    )
    variance.stop_gradient = True

    # create output
    # mean and mean_out share the same memory
    mean_out = mean
    # variance and variance_out share the same memory
    variance_out = variance

    if _non_static_mode():
        inputs_has_MomemtumTensor = False
        attrs_has_momentum = False
        tmp_tensor_type = core.eager.Tensor
        if isinstance(momentum, tmp_tensor_type):
            inputs_has_MomemtumTensor = True
        else:
            attrs_has_momentum = True

        attrs_ = ()
        if attrs_has_momentum:
            attrs_ = (
                'momentum',
                momentum,
                'epsilon',
                epsilon,
                'is_test',
                is_test,
                'data_layout',
                data_layout,
                'use_mkldnn',
                False,
                'fuse_with_relu',
                False,
                'use_global_stats',
                use_global_stats,
            )
        else:
            attrs_ = (
                'epsilon',
                epsilon,
                'is_test',
                is_test,
                'data_layout',
                data_layout,
                'use_mkldnn',
                False,
                'fuse_with_relu',
                False,
                'use_global_stats',
                use_global_stats,
            )
        if inputs_has_MomemtumTensor:
            batch_norm_out, _, _, _, _, _ = paddle._legacy_C_ops.batch_norm(
                input,
                scale,
                bias,
                mean,
                variance,
                momentum,
                mean_out,
                variance_out,
                *attrs_,
            )
        else:
            batch_norm_out, _, _, _, _, _ = paddle._legacy_C_ops.batch_norm(
                input,
                scale,
                bias,
                mean,
                variance,
                None,
                mean_out,
                variance_out,
                *attrs_,
            )

        return paddle.fluid.dygraph_utils._append_activation_in_dygraph(
            batch_norm_out, act=act, use_mkldnn=False
        )

    saved_mean = helper.create_variable_for_type_inference(
        dtype=dtype, stop_gradient=True
    )
    saved_variance = helper.create_variable_for_type_inference(
        dtype=dtype, stop_gradient=True
    )
    reserve_space = None
    if not is_test:
        reserve_space = helper.create_variable_for_type_inference(
            dtype=helper.input_dtype(), stop_gradient=True
        )

    batch_norm_out = (
        input if in_place else helper.create_variable_for_type_inference(dtype)
    )

    inputs = {
        "X": input,
        "Scale": scale,
        "Bias": bias,
        "Mean": mean,
        "Variance": variance,
        "MeanOut": mean_out,
        "VarianceOut": variance_out,
    }
    attrs = {
        "epsilon": epsilon,
        "is_test": is_test,
        "data_layout": data_layout,
        "use_mkldnn": False,
        "fuse_with_relu": False,
        "use_global_stats": use_global_stats,
    }
    if isinstance(momentum, paddle.static.Variable):
        inputs['MomemtumTensor'] = momentum
    else:
        attrs['momentum'] = momentum

    outputs = {
        "Y": batch_norm_out,
        "MeanOut": mean_out,
        "VarianceOut": variance_out,
        "SavedMean": saved_mean,
        "SavedVariance": saved_variance,
    }
    if reserve_space is not None:
        outputs["ReserveSpace"] = reserve_space

    helper.append_op(
        type="batch_norm", inputs=inputs, outputs=outputs, attrs=attrs
    )

    return helper.append_activation(batch_norm_out)


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


class PyFuncRegistry:
    _register_funcs = []

    def __init__(self, func):
        if func is None or not callable(func):
            raise TypeError('func must be a Python function')

        self._func = func
        # find named args using reflection
        args = inspect.getfullargspec(self._func)
        if len(args[0]) == 0 and args[1] is None and args[2] is None:
            # Function with no inputs
            self._named_args = None
        else:
            self._named_args = args[0]
        self._id = core._append_python_callable_object_and_return_id(self)
        '''
        Why record self here?
        1. For debug usage. Users can call
           :code:`py_func.registered_func(idx)` method
           to find the registered function corresponding
           to :code:`idx`.
        2. For increasing reference count of self.
           It seems that to release Python object
           whose reference count is 1 would cause
           segmentation fault error in C++ side.
           May be lack of Python GC in C++ side?
        '''
        PyFuncRegistry._register_funcs.append(self)

    @classmethod
    def registered_func(cls, idx):
        return cls._register_funcs[idx]._func

    @classmethod
    def registered_func_num(cls):
        return len(cls._register_funcs)

    @property
    def id(self):
        return self._id

    def __call__(self, *args):
        if self._named_args is None:
            func_ret = self._func()
        else:
            kwargs = dict()
            idx = 0
            for arg in self._named_args:
                kwargs[arg] = args[idx]
                idx += 1
            func_ret = self._func(*args[idx:], **kwargs)

        if not isinstance(func_ret, (list, tuple)):
            func_ret = (func_ret,)

        ret = []
        for each_ret in func_ret:
            if each_ret is None or isinstance(each_ret, core.LoDTensor):
                ret.append(each_ret)
                continue

            if not isinstance(each_ret, np.ndarray):
                each_ret = np.array(each_ret)

            tensor = core.LoDTensor()
            tensor.set(each_ret, core.CPUPlace())
            ret.append(tensor)

        return tuple(ret)


@static_only
@templatedoc()
def py_func(func, x, out, backward_func=None, skip_vars_in_backward_input=None):
    """
    This is used to register customized Python OP to Paddle. The design
    principe of py_func is that Tensor and numpy array can be converted to each
    other easily. So you can use Python and numpy API to register a python OP.
    The forward function of the registered OP is ``func`` and the backward function
    of that is ``backward_func``. Paddle will call ``func`` at forward runtime and
    call ``backward_func`` at backward runtime(if ``backward_func`` is not  None).
    ``x`` is the input of ``func``, whose type must be Tensor; ``out`` is
    the output of ``func``, whose type can be either Tensor or numpy array.
    The input of the backward function ``backward_func`` is ``x``, ``out`` and
    the gradient of ``out``. If ``out`` have no gradient, the relevant input of
    ``backward_func`` is None. If ``x`` do not have a gradient, the user should
    return None in ``backward_func``.
    The data type and shape of ``out`` should also be set correctly before this
    API is called, and the data type and shape of the gradient of ``out`` and
    ``x`` will be inferred automatically.
    This API can also be used to debug the neural network by setting the ``func``
    as a function that only print variables.

    Args:
        func (callable): The forward function of the registered OP. When the network
            is running, the forward output ``out`` will be calculated according to this
            function and the forward input ``x``. In ``func`` , it's suggested that we
            actively convert Tensor into a numpy array, so that we can use Python and
            numpy API arbitrarily. If not, some operations of numpy may not be compatible.
        x (Tensor|tuple(Tensor)|list[Tensor]): The input of the forward function ``func``.
            It can be Tensor|tuple(Tensor)|list[Tensor]. In addition, Multiple Tensor
            should be passed in the form of tuple(Tensor) or list[Tensor].
        out (T|tuple(T)|list[T]): The output of the forward function ``func``, it can be
            T|tuple(T)|list[T], where T can be either Tensor or numpy array. Since Paddle
            cannot automatically infer the shape and type of ``out``, you must create
            ``out`` in advance.
        backward_func (callable, optional): The backward function of the registered OP.
            Its default value is None, which means there is no reverse calculation. If
            it is not None, ``backward_func`` is called to calculate the gradient of
            ``x`` when the network is at backward runtime.
        skip_vars_in_backward_input (Tensor, optional): It's used to limit the input
            list of ``backward_func``, and it can be Tensor|tuple(Tensor)|list[Tensor].
            It must belong to either ``x`` or ``out``. The default  value is None, which means
            that no tensors need to be removed from ``x`` and ``out``. If it is not None,
            these tensors will not be the input of ``backward_func``. This parameter is only
            useful when ``backward_func`` is not None.

    Returns:
        Tensor|tuple(Tensor)|list[Tensor], The output ``out`` of the forward function ``func``.

    Examples:

        .. code-block:: python
            :name: code-example1

            # example 1:
            import paddle
            import numpy as np
            paddle.enable_static()
            # Creates a forward function, Tensor can be input directly without
            # being converted into numpy array.
            def tanh(x):
                return np.tanh(x)
            # Skip x in backward function and return the gradient of x
            # Tensor must be actively converted to numpy array, otherwise,
            # operations such as +/- can't be used.
            def tanh_grad(y, dy):
                return np.array(dy) * (1 - np.square(np.array(y)))
            # Creates a forward function for debugging running networks(print value)
            def debug_func(x):
                print(x)
            def create_tmp_var(name, dtype, shape):
                return paddle.static.default_main_program().current_block().create_var(
                    name=name, dtype=dtype, shape=shape)
            def simple_net(img, label):
                hidden = img
                for idx in range(4):
                    hidden = paddle.static.nn.fc(hidden, size=200)
                    new_hidden = create_tmp_var(name='hidden_{}'.format(idx),
                        dtype=hidden.dtype, shape=hidden.shape)
                    # User-defined forward and backward
                    hidden = paddle.static.py_func(func=tanh, x=hidden,
                        out=new_hidden, backward_func=tanh_grad,
                        skip_vars_in_backward_input=hidden)
                    # User-defined debug functions that print out the input Tensor
                    paddle.static.py_func(func=debug_func, x=hidden, out=None)
                prediction = paddle.static.nn.fc(hidden, size=10, activation='softmax')
                ce_loss = paddle.nn.loss.CrossEntropyLoss()
                return ce_loss(prediction, label)
            x = paddle.static.data(name='x', shape=[1,4], dtype='float32')
            y = paddle.static.data(name='y', shape=[1], dtype='int64')
            res = simple_net(x, y)
            exe = paddle.static.Executor(paddle.CPUPlace())
            exe.run(paddle.static.default_startup_program())
            input1 = np.random.random(size=[1,4]).astype('float32')
            input2 = np.random.randint(1, 10, size=[1], dtype='int64')
            out = exe.run(paddle.static.default_main_program(),
                          feed={'x':input1, 'y':input2},
                          fetch_list=[res.name])
            print(out)

        .. code-block:: python
            :name: code-example2

            # example 2:
            # This example shows how to turn Tensor into numpy array and
            # use numpy API to register an Python OP
            import paddle
            import numpy as np
            paddle.enable_static()
            def element_wise_add(x, y):
                # Tensor must be actively converted to numpy array, otherwise,
                # numpy.shape can't be used.
                x = np.array(x)
                y = np.array(y)
                if x.shape != y.shape:
                    raise AssertionError("the shape of inputs must be the same!")
                result = np.zeros(x.shape, dtype='int32')
                for i in range(len(x)):
                    for j in range(len(x[0])):
                        result[i][j] = x[i][j] + y[i][j]
                return result
            def create_tmp_var(name, dtype, shape):
                return paddle.static.default_main_program().current_block().create_var(
                            name=name, dtype=dtype, shape=shape)
            def py_func_demo():
                start_program = paddle.static.default_startup_program()
                main_program = paddle.static.default_main_program()
                # Input of the forward function
                x = paddle.static.data(name='x', shape=[2,3], dtype='int32')
                y = paddle.static.data(name='y', shape=[2,3], dtype='int32')
                # Output of the forward function, name/dtype/shape must be specified
                output = create_tmp_var('output','int32', [3,1])
                # Multiple Tensor should be passed in the form of tuple(Tensor) or list[Tensor]
                paddle.static.py_func(func=element_wise_add, x=[x,y], out=output)
                exe=paddle.static.Executor(paddle.CPUPlace())
                exe.run(start_program)
                # Feed numpy array to main_program
                input1 = np.random.randint(1, 10, size=[2,3], dtype='int32')
                input2 = np.random.randint(1, 10, size=[2,3], dtype='int32')
                out = exe.run(main_program,
                            feed={'x':input1, 'y':input2},
                            fetch_list=[output.name])
                print("{0} + {1} = {2}".format(input1, input2, out))
            py_func_demo()
            # Reference output:
            # [[5, 9, 9]   + [[7, 8, 4]  =  [array([[12, 17, 13]
            #  [7, 5, 2]]     [1, 3, 3]]            [8, 8, 5]], dtype=int32)]
    """
    helper = LayerHelper('py_func', **locals())
    check_type(x, 'X', (list, tuple, Variable, type(None)), 'py_func')
    if x is None:
        x = []
    elif isinstance(x, Variable):
        x = [x]
    elif isinstance(x, tuple):
        x = list(x)
    elif not isinstance(x, (list, tuple, Variable)):
        raise TypeError('Input must be Tensor/list(Tensor)/tuple(Tensor)')
    check_type(out, 'Out', (list, tuple, Variable, type(None)), 'py_func')
    if out is None:
        out_list = []
    elif isinstance(out, Variable):
        out_list = [out]
    elif isinstance(out, tuple):
        out_list = list(out)
    elif isinstance(out, list):
        out_list = out
    else:
        raise TypeError('Output must be Tensor/list(Tensor)/tuple(Tensor)')

    fwd_func_id = PyFuncRegistry(func).id
    bwd_func_id = (
        PyFuncRegistry(backward_func).id if backward_func is not None else -1
    )

    for each_out in out_list:
        if len(each_out.shape) == 0:
            raise ValueError(
                'Output shapes of py_func should be provided by users manually'
            )

    backward_skip_vars = set()
    if backward_func is not None and skip_vars_in_backward_input is not None:
        if isinstance(skip_vars_in_backward_input, Variable):
            skip_vars_in_backward_input = [skip_vars_in_backward_input]

        fwd_in_out = [v.name for v in x]
        fwd_in_out.extend([v.name for v in out_list])
        fwd_in_out = set(fwd_in_out)
        backward_skip_vars = set()
        for v in skip_vars_in_backward_input:
            if v.name not in fwd_in_out:
                raise ValueError(
                    'Tensor {} is not found in forward inputs and outputs'.format(
                        v.name
                    )
                )
            backward_skip_vars.add(v.name)

    helper.append_op(
        type='py_func',
        inputs={'X': x},
        outputs={'Out': out_list},
        attrs={
            'forward_callable_id': fwd_func_id,
            'backward_callable_id': bwd_func_id,
            'backward_skip_vars': list(backward_skip_vars),
        },
    )
    return out


@templatedoc()
def row_conv(input, future_context_size, param_attr=None, act=None):
    """
    :api_attr: Static Graph

    ${comment}

    Args:
        input (${x_type}): ${x_comment}.
        future_context_size (int): Future context size. Please note, the shape
            of convolution kernel is [future_context_size + 1, D].
        param_attr (ParamAttr): Attributes of parameters, including
            name, initializer etc.
        act (str): Non-linear activation to be applied to output variable.

    Returns:
        ${out_comment}.

    Examples:

      .. code-block:: python

        # for LodTensor inputs
        import paddle
        paddle.enable_static()
        x = paddle.static.data(name='x', shape=[9, 16],
                               dtype='float32', lod_level=1)
        out_x = paddle.static.nn.row_conv(input=x, future_context_size=2)
        # for Tensor inputs
        y = paddle.static.data(name='y', shape=[9, 4, 16], dtype='float32')
        out_y = paddle.static.nn.row_conv(input=y, future_context_size=2)
    """
    helper = LayerHelper('row_conv', **locals())
    check_variable_and_dtype(input, 'input', ['float32'], 'row_conv')
    dtype = helper.input_dtype()
    filter_shape = [future_context_size + 1, input.shape[-1]]
    filter_param = helper.create_parameter(
        attr=helper.param_attr, shape=filter_shape, dtype=dtype
    )
    out = helper.create_variable_for_type_inference(dtype)
    helper.append_op(
        type='row_conv',
        inputs={'X': [input], 'Filter': [filter_param]},
        outputs={'Out': [out]},
    )
    return helper.append_activation(out)


@templatedoc()
def spectral_norm(weight, dim=0, power_iters=1, eps=1e-12, name=None):
    r"""
    :api_attr: Static Graph

    **Spectral Normalization Layer**

    This operation calculates the spectral normalization value of weight parameters of
    fc, conv1d, conv2d, conv3d layers which should be 2-D, 3-D, 4-D, 5-D
    Parameters. Output tensor will be in same shape with input tensor.
    Calculations are showed as follows.

    Step 1:
    Generate vector U in shape of [H], and V in shape of [W].
    While H is the :attr:`dim` th dimension of the input weights,
    and W is the product result of remaining dimensions.

    Step 2:
    :attr:`power_iters` should be a positive integer, do following
    calculations with U and V for :attr:`power_iters` rounds. Calculations
    as follows:

    .. math::

        \mathbf{v} := \\frac{\mathbf{W}^{T} \mathbf{u}}{\|\mathbf{W}^{T} \mathbf{u}\|_2}

        \mathbf{u} := \\frac{\mathbf{W}^{T} \mathbf{v}}{\|\mathbf{W}^{T} \mathbf{v}\|_2}

    Step 3:
    Calculate :math:`\sigma(\mathbf{W})` and normalize weight values.

    .. math::

        \sigma(\mathbf{W}) = \mathbf{u}^{T} \mathbf{W} \mathbf{v}

        \mathbf{W} = \\frac{\mathbf{W}}{\sigma(\mathbf{W})}


    Refer to `Spectral Normalization <https://arxiv.org/abs/1802.05957>`_ .

    Args:
        weight(Tensor): ${weight_comment}
        dim(int): ${dim_comment}
        power_iters(int): ${power_iters_comment}
        eps(float): ${eps_comment}
        name(str, optional): For detailed information, please refer
                             to :ref:`api_guide_Name`. Usually name is no need to set and
                             None by default.

    Returns:
        Tensor: A tensor of weight parameters after spectral normalization.
                  The data type and shape is same as input tensor.

    Examples:
       .. code-block:: python

            import paddle

            paddle.enable_static()
            weight = paddle.static.data(name='weight', shape=[2, 8, 32, 32], dtype='float32')
            x = paddle.static.nn.spectral_norm(weight=weight, dim=1, power_iters=2)
            print(x.shape) # [2, 8, 32, 32]
    """
    helper = LayerHelper('spectral_norm', **locals())
    check_variable_and_dtype(
        weight, 'weight', ['float32', 'float64'], 'spectral_norm'
    )
    check_type(dim, 'dim', int, 'spectral_norm')
    check_type(power_iters, 'power_iters', int, 'spectral_norm')
    check_type(eps, 'eps', float, 'spectral_norm')
    dtype = weight.dtype

    # create intput and parameters
    input_shape = weight.shape
    assert weight.numel() > 0, "Any dimension of input cannot be equal to 0."

    if dim not in [0, 1]:
        raise ValueError(
            f"The input `dim` must be 0 (if weight in fc) or 1 (if weight in conv), but received dim={dim}"
        )

    h = input_shape[dim]
    w = np.prod(input_shape) // h

    u = helper.create_parameter(
        attr=ParamAttr(),
        shape=[h],
        dtype=dtype,
        default_initializer=Normal(0.0, 1.0),
    )
    u.stop_gradient = True
    v = helper.create_parameter(
        attr=ParamAttr(),
        shape=[w],
        dtype=dtype,
        default_initializer=Normal(0.0, 1.0),
    )
    v.stop_gradient = True

    if paddle.framework.in_dygraph_mode():
        return paddle._C_ops.spectral_norm(weight, u, v, dim, power_iters, eps)

    inputs = {'Weight': weight}
    inputs['U'] = u
    inputs['V'] = v

    # create output
    out = helper.create_variable(dtype=dtype)

    helper.append_op(
        type="spectral_norm",
        inputs=inputs,
        outputs={
            "Out": out,
        },
        attrs={
            "dim": dim,
            "power_iters": power_iters,
            "eps": eps,
        },
    )

    return out


# For debug usage
py_func.registered_func = PyFuncRegistry.registered_func
py_func.registered_func_num = PyFuncRegistry.registered_func_num


@templatedoc()
def layer_norm(
    input,
    scale=True,
    shift=True,
    begin_norm_axis=1,
    epsilon=1e-05,
    param_attr=None,
    bias_attr=None,
    act=None,
    name=None,
):
    r"""

    **Layer Normalization Layer**

    The API implements the function of the Layer Normalization Layer and can be applied to mini-batch input data.
    Refer to `Layer Normalization <https://arxiv.org/pdf/1607.06450v1.pdf>`_

    The formula is as follows:

    ..  math::

        \mu & = \frac{1}{H}\sum_{i=1}^{H} x_i

        \sigma & = \sqrt{\frac{1}{H}\sum_{i=1}^{H}{(x_i - \mu)^2} + \epsilon}

        y & = f(\frac{g}{\sigma}(x - \mu) + b)

    - :math:`x`: the vector representation of the summed inputs to the neurons in that layer.
    - :math:`H`: the number of hidden units in a layers
    - :math:`\\epsilon`: the small value added to the variance to prevent division by zero.
    - :math:`g`: the trainable scale parameter.
    - :math:`b`: the trainable bias parameter.

    Args:
        input(Tensor): A multi-dimension ``Tensor`` , and the data type is float32 or float64.
        scale(bool, optional): Whether to learn the adaptive gain :math:`g` after
            normalization. Default: True.
        shift(bool, optional): Whether to learn the adaptive bias :math:`b` after
            normalization. Default: True.
        begin_norm_axis(int, optional): The normalization will be performed along
            dimensions from :attr:`begin_norm_axis` to :attr:`rank(input)`.
            Default: 1.
        epsilon(float, optional): The small value added to the variance to prevent
            division by zero. Default: 1e-05.
        param_attr(ParamAttr, optional): The parameter attribute for the learnable
            gain :math:`g`. If :attr:`scale` is False, :attr:`param_attr` is
            omitted. If :attr:`scale` is True and :attr:`param_attr` is None,
            a default :code:`ParamAttr` would be added as scale. The
            :attr:`param_attr` is initialized as 1 if it is added. Default: None.
        bias_attr(ParamAttr, optional): The parameter attribute for the learnable
            bias :math:`b`. If :attr:`shift` is False, :attr:`bias_attr` is
            omitted. If :attr:`shift` is True and :attr:`param_attr` is None,
            a default :code:`ParamAttr` would be added as bias. The
            :attr:`bias_attr` is initialized as 0 if it is added. Default: None.
        act(str, optional): Activation to be applied to the output of layer normalization.
                  Default: None.
        name(str, optional): The default value is None.  Normally there is no need for user to set this property.  For more information, please refer to :ref:`api_guide_Name` .

    Returns:
        Tensor: ``Tensor``  indicating the normalized result, the data type is the same as  ``input`` , and the return dimension is the same as  ``input`` .

    Examples:

        .. code-block:: python

            import paddle
            paddle.enable_static()
            x = paddle.static.data(name='x', shape=[8, 32, 32], dtype='float32')
            output = paddle.static.nn.layer_norm(input=x, begin_norm_axis=1)
            print(output.shape)  # [8, 32, 32]
    """
    assert (
        _non_static_mode() is not True
    ), "please use LayerNorm instead of layer_norm in dygraph mode!"
    helper = LayerHelper('layer_norm', **locals())
    check_variable_and_dtype(
        input, 'input', ['float32', 'float64'], 'layer_norm'
    )
    dtype = helper.input_dtype()

    # create intput and parameters
    inputs = {'X': input}
    input_shape = input.shape
    param_shape = [reduce(lambda x, y: x * y, input_shape[begin_norm_axis:])]
    if scale:
        assert (
            param_attr is not False
        ), "param_attr should not be False when using scale."
        scale = helper.create_parameter(
            attr=helper.param_attr,
            shape=param_shape,
            dtype=dtype,
            default_initializer=Constant(1.0),
        )
        inputs['Scale'] = scale
    else:
        if param_attr:
            warnings.warn("param_attr is only available with scale is True.")
    if shift:
        assert (
            bias_attr is not False
        ), "bias_attr should not be False when using shift."
        bias = helper.create_parameter(
            attr=helper.bias_attr, shape=param_shape, dtype=dtype, is_bias=True
        )
        inputs['Bias'] = bias
    else:
        if bias_attr:
            warnings.warn("bias_attr is only available with shift is True.")

    # create output
    mean_out = helper.create_variable_for_type_inference(
        dtype=dtype, stop_gradient=True
    )
    variance_out = helper.create_variable_for_type_inference(
        dtype=dtype, stop_gradient=True
    )
    layer_norm_out = helper.create_variable_for_type_inference(dtype)

    helper.append_op(
        type="layer_norm",
        inputs=inputs,
        outputs={
            "Y": layer_norm_out,
            "Mean": mean_out,
            "Variance": variance_out,
        },
        attrs={"epsilon": epsilon, "begin_norm_axis": begin_norm_axis},
    )

    return helper.append_activation(layer_norm_out)


@static_only
def embedding(
    input,
    size,
    is_sparse=False,
    is_distributed=False,
    padding_idx=None,
    param_attr=None,
    dtype='float32',
):
    r"""
    :api_attr: Static Graph

    The operator is used to lookup embeddings vector of ids provided by :attr:`input` .
    It automatically constructs a 2D embedding matrix based on the
    input :attr:`size` (vocab_size, emb_size) and :attr:`dtype` .

    The shape of output Tensor is generated by appending an emb_size dimension to the
    last dimension of the input Tensor shape.

    **Note:** The id in :attr:`input` must satisfy :math:`0 =< id < size[0]` ,
    otherwise the program will throw an exception and exit.

    .. code-block:: text

        Case 1:

        input is a Tensor. padding_idx = -1
            input.data = [[1, 3], [2, 4], [4, 127]]
            input.shape = [3, 2]
        Given size = [128, 16]
        output is a Tensor:
            out.shape = [3, 2, 16]
            out.data = [[[0.129435295, 0.244512452, ..., 0.436322452],
                        [0.345421456, 0.524563927, ..., 0.144534654]],

                        [[0.345249859, 0.124939536, ..., 0.194353745],
                        [0.945345345, 0.435394634, ..., 0.435345365]],

                        [[0.945345345, 0.435394634, ..., 0.435345365],
                        [0.0,         0.0,         ..., 0.0        ]]]  # padding data
        The input padding_idx is less than 0, it is automatically converted to padding_idx = -1 + 128 = 127
        It will pad all-zero data when ids is 127.

        Case 2:

        input is a LoDTensor with 1-level LoD. padding_idx = 0
            input.lod = [[2, 3]]
            input.data = [[1], [3], [2], [4], [0]]
            input.shape = [5, 1]
        Given size = [128, 16]
        output is a LoDTensor:
            out.lod = [[2, 3]]
            out.shape = [5, 1, 16]
            out.data = [[[0.129435295, 0.244512452, ..., 0.436322452]],
                        [[0.345421456, 0.524563927, ..., 0.144534654]],
                        [[0.345249859, 0.124939536, ..., 0.194353745]],
                        [[0.945345345, 0.435394634, ..., 0.435345365]],
                        [[0.0,         0.0,         ..., 0.0        ]]]  # padding data
        It will pad all-zero data when ids is 0.


    Args:
        input(Tensor): A Tensor or LoDTensor with type int64, which contains the id information.
            The value of the input id should satisfy :math:`0<= id < size[0]` .
        size(tuple|list): The shape of lookup table parameter. It should have two elements which
            indicates the size of the dictionary of embeddings and the size of each embedding vector respectively.
        is_sparse(bool): The flag indicating whether to use sparse update. This parameter only
            affects the performance of the backwards gradient update. It is recommended to set
            True because sparse update is faster. But some optimizer does not support sparse update
            In these case, is_sparse must be False. Default: False.
        is_distributed(bool): Whether to store the embedding matrix in a distributed manner. Only used
            in multi-machine distributed CPU training. Default: False.
        padding_idx(int|long|None): padding_idx needs to be in the interval [-vocab_size, vocab_size).
            If :math:`padding\_idx < 0`, the :math:`padding\_idx` will automatically be converted
            to :math:`vocab\_size + padding\_idx` . It will output all-zero padding data whenever lookup
            encounters :math:`padding\_idx` in id. And the padding data will not be updated while training.
            If set None, it makes no effect to output. Default: None.
        param_attr(ParamAttr): To specify the weight parameter property. Default: None, which means the
            default weight parameter property is used. In addition,
            user-defined or pre-trained word vectors can be loaded with the :attr:`param_attr` parameter.
            The local word vector needs to be transformed into numpy format, and the shape of local word
            vector should be consistent with :attr:`size` .
        dtype(str): It refers to the data type of output Tensor.
            It must be float32 or float64. Default: float32.

    Returns:
        Tensor: Embedding Tensor or LoDTensor mapped by input. The data type is the same as :attr:`dtype` .

    Static Examples:
        .. code-block:: python

            import paddle
            import numpy as np
            paddle.enable_static()

            x = paddle.static.data(name="x", shape = [2, 4], dtype=np.int64)
            output = paddle.static.nn.embedding(x, (10, 3),
                        param_attr=paddle.nn.initializer.Constant(value=1.0))
            m_output=paddle.mean(output)
            place = paddle.CPUPlace()
            exe = paddle.static.Executor(place)
            exe.run(paddle.static.default_startup_program())

            x = np.array([[7, 2, 4, 5],[4, 3, 2, 9]], dtype=np.int64)

            # x is a Numpy.
            # x.data = [[7, 2, 4, 5], [4, 3, 2, 9]]
            # x.shape = [2, 4]

            out, = exe.run(paddle.static.default_main_program(), feed={'x':x}, fetch_list=[output])

            # out is a Numpy.
            # out.data = [[1., 1., 1.],
            #             [1., 1., 1.],
            #             [1., 1., 1.],
            #             [1., 1., 1.]],
            #
            #            [[1., 1., 1.],
            #             [1., 1., 1.],
            #             [1., 1., 1.],
            #             [0., 0., 0.]]]
            # out.shape = [2, 4, 3]
    """

    helper = LayerHelper('embedding', **locals())
    check_variable_and_dtype(input, 'input', ['int64'], 'embedding')
    check_dtype(
        dtype,
        'dtype',
        ['float16', 'float32', 'float64', 'uint16'],
        'embedding',
    )
    remote_prefetch = is_sparse and (not is_distributed)
    if remote_prefetch:
        assert is_sparse is True and is_distributed is False
    w = helper.create_parameter(
        attr=helper.param_attr, shape=size, dtype=dtype, is_bias=False
    )
    tmp = helper.create_variable_for_type_inference(dtype)
    padding_idx = (
        -1
        if padding_idx is None
        else padding_idx
        if padding_idx >= 0
        else (size[0] + padding_idx)
    )
    helper.append_op(
        type='lookup_table_v2',
        inputs={'Ids': input, 'W': w},
        outputs={'Out': tmp},
        attrs={
            'is_sparse': is_sparse,
            'is_distributed': is_distributed,
            'remote_prefetch': remote_prefetch,
            'padding_idx': padding_idx,
        },
    )
    return tmp
