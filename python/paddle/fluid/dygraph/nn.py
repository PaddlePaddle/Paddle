# Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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
from .. import core
from ..layers import utils
from ..layers import nn as F
from .. import dygraph_utils
from . import layers
from ..framework import (
    Variable,
    _non_static_mode,
    OpProtoHolder,
    Parameter,
    _dygraph_tracer,
    _varbase_creator,
    default_main_program,
    _global_flags,
    in_dygraph_mode,
    _in_legacy_dygraph,
)

from ..data_feeder import (
    convert_dtype,
    check_variable_and_dtype,
    check_type,
    check_dtype,
)
from ..param_attr import ParamAttr
from ..initializer import Normal, Constant, NumpyArrayInitializer
from .. import unique_name
from .layer_object_helper import LayerObjectHelper
from ..data_feeder import check_variable_and_dtype, check_type
import numpy as np
import numbers
import logging
import os
import paddle.utils.deprecated as deprecated
from paddle import _C_ops, _legacy_C_ops

__all__ = [
    'Conv3D',
    'Linear',
    'BatchNorm',
    'Conv3DTranspose',
]


class Conv3D(layers.Layer):
    r"""
    **Convlution3D Layer**

    The convolution3D layer calculates the output based on the input, filter
    and strides, paddings, dilations, groups parameters. Input(Input) and
    Output(Output) are multidimensional tensors with a shape of
    :math:`[N, C, D, H, W]` . Where N is batch size, C is the number of
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

    Parameters:
        num_channels(int): The number of channels in the input image.
        num_filters(int): The number of filter. It is as same as the output image channel.
        filter_size (int|tuple, optional): The filter size. If filter_size is a tuple,
            it must contain three integers, (filter_size_D, filter_size_H, filter_size_W).
            Otherwise, the filter will be a square, filter_size_depth = filter_size_height
            = filter_size_width = filter_size.
        stride (int|tuple, optional): The stride size. If stride is a tuple, it must
            contain three integers, (stride_D, stride_H, stride_W). Otherwise, the
            stride_D = stride_H = stride_W = stride. The default value is 1.
        padding (int|tuple, optional): The padding size. If padding is a tuple, it must
            contain three integers, (padding_D, padding_H, padding_W). Otherwise, the
            padding_D = padding_H = padding_W = padding. The default value is 0.
        dilation (int|tuple, optional): The dilation size. If dilation is a tuple, it must
            contain three integers, (dilation_D, dilation_H, dilation_W). Otherwise, the
            dilation_D = dilation_H = dilation_W = dilation. The default value is 1.
        groups (int, optional): The groups number of the Conv3D Layer. According to grouped
            convolution in Alex Krizhevsky's Deep CNN paper: when group=2,
            the first half of the filters is only connected to the first half
            of the input channels, while the second half of the filters is only
            connected to the second half of the input channels. The default value is 1.
        param_attr (ParamAttr, optional): The parameter attribute for learnable parameters/weights
            of conv3d. If it is set to None or one attribute of ParamAttr, conv3d
            will create ParamAttr as param_attr. If it is set to None, the parameter
            is initialized with :math:`Normal(0.0, std)`, and the :math:`std` is
            :math:`(\\frac{2.0 }{filter\_elem\_num})^{0.5}`. The default value is None.
        bias_attr (ParamAttr|bool, optional): The parameter attribute for the bias of conv3d.
            If it is set to False, no bias will be added to the output units.
            If it is set to None or one attribute of ParamAttr, conv3d
            will create ParamAttr as bias_attr. If the Initializer of the bias_attr
            is not set, the bias is initialized zero. The default value is None.
        use_cudnn (bool, optional): Use cudnn kernel or not, it is valid only when the cudnn
            library is installed. The default value is True.
        act (str, optional): Activation type, if it is set to None, activation is not appended.
            The default value is None.
        dtype (str, optional): Data type, it can be "float32" or "float64". Default: "float32".

    Attribute:
        **weight** (Parameter): the learnable weights of filters of this layer.

        **bias** (Parameter): the learnable bias of this layer.

    Returns:
        None.

    Raises:
        ValueError: If the shapes of input, filter_size, stride, padding and
                    groups mismatch.

    Examples:
        .. code-block:: python

          import paddle.fluid as fluid
          import numpy

          with fluid.dygraph.guard():
              data = numpy.random.random((5, 3, 12, 32, 32)).astype('float32')
              conv3d = fluid.dygraph.nn.Conv3D(
                    num_channels=3, num_filters=2, filter_size=3, act="relu")
              ret = conv3d(fluid.dygraph.base.to_variable(data))

    """

    def __init__(
        self,
        num_channels,
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
        dtype='float32',
    ):
        assert param_attr is not False, "param_attr should not be False here."
        super().__init__()
        self._num_channels = num_channels
        self._groups = groups
        self._stride = utils.convert_to_list(stride, 3, 'stride')
        self._padding = utils.convert_to_list(padding, 3, 'padding')
        self._dilation = utils.convert_to_list(dilation, 3, 'dilation')
        self._act = act
        self._use_cudnn = use_cudnn
        self._filter_size = filter_size
        self._num_filters = num_filters
        self._param_attr = param_attr
        self._bias_attr = bias_attr
        self._dtype = dtype

        if self._groups is None:
            num_filter_channels = self._num_channels
        else:
            if self._num_channels % self._groups != 0:
                raise ValueError("num_channels must be divisible by groups.")
            num_filter_channels = self._num_channels // self._groups

        filter_size = utils.convert_to_list(self._filter_size, 3, 'filter_size')
        filter_shape = [self._num_filters, num_filter_channels] + filter_size

        def _get_default_param_initializer():
            filter_elem_num = (
                filter_size[0]
                * filter_size[1]
                * filter_size[2]
                * self._num_channels
            )
            std = (2.0 / filter_elem_num) ** 0.5
            return Normal(0.0, std, 0)

        self.weight = self.create_parameter(
            attr=self._param_attr,
            shape=filter_shape,
            dtype=self._dtype,
            default_initializer=_get_default_param_initializer(),
        )

        self.bias = self.create_parameter(
            attr=self._bias_attr,
            shape=[self._num_filters],
            dtype=self._dtype,
            is_bias=True,
        )

    def forward(self, input):
        pre_bias = self._helper.create_variable_for_type_inference(
            dtype=self._dtype
        )

        self._helper.append_op(
            type='conv3d',
            inputs={
                'Input': input,
                'Filter': self.weight,
            },
            outputs={"Output": pre_bias},
            attrs={
                'strides': self._stride,
                'paddings': self._padding,
                'dilations': self._dilation,
                'groups': self._groups if self._groups else 1,
                'use_cudnn': self._use_cudnn,
                'use_mkldnn': False,
            },
        )

        if self.bias is not None:
            pre_act = self._helper.create_variable_for_type_inference(
                dtype=self._dtype
            )
            self._helper.append_op(
                type='elementwise_add',
                inputs={'X': [pre_bias], 'Y': [self.bias]},
                outputs={'Out': [pre_act]},
                attrs={'axis': 1},
            )
        else:
            pre_act = pre_bias

        return self._helper.append_activation(pre_act, act=self._act)


class Conv3DTranspose(layers.Layer):
    r"""
    **Convlution3D transpose layer**

    The convolution3D transpose layer calculates the output based on the input,
    filter, and dilations, strides, paddings. Input(Input) and output(Output)
    are in NCDHW format. Where N is batch size, C is the number of channels,
    D is the depth of the feature, H is the height of the feature, and W
    is the width of the feature. Parameters(dilations, strides, paddings) are
    two elements. These two elements represent height and width, respectively.
    The details of convolution transpose layer, please refer to the following
    explanation and references `therein <http://www.matthewzeiler.com/wp-content/uploads/2017/07/cvpr2010.pdf>`_.
    If bias attribution and activation type are provided, bias is added to
    the output of the convolution, and the corresponding activation function
    is applied to the final result.

    For each input :math:`X`, the equation is:

    .. math::

        Out = \sigma (W \\ast X + b)

    In the above equation:

    * :math:`X`: Input value, a tensor with NCDHW format.
    * :math:`W`: Filter value, a tensor with MCDHW format.
    * :math:`\\ast`: Convolution operation.
    * :math:`b`: Bias value, a 2-D tensor with shape [M, 1].
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

    **Note**:

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


    Parameters:
        num_channels(int): The number of channels in the input image.
        num_filters(int): The number of the filter. It is as same as the output
            image channel.
        filter_size(int|tuple): The filter size. If filter_size is a tuple,
            it must contain three integers, (filter_size_D, filter_size_H, filter_size_W).
            Otherwise, the filter will be a square.
        padding(int|tuple, optional): The padding size. The padding argument effectively
             adds `dilation * (kernel - 1)` amount of zero-padding on both sides of input. If `padding` is a string,
             either 'VALID' or 'SAME' supported, which is the padding algorithm. If `padding`
             is a tuple or list, it could be in three forms: `[pad_depth, pad_height, pad_width]` or
            `[pad_depth_front, pad_depth_back, pad_height_top, pad_height_bottom, pad_width_left, pad_width_right]`,
            and when `data_format` is `'NCDHW'`, `padding` can be in the form
            `[[0,0], [0,0], [pad_depth_front, pad_depth_back], [pad_height_top, pad_height_bottom], [pad_width_left, pad_width_right]]`.
            when `data_format` is `'NDHWC'`, `padding` can be in the form
            `[[0,0], [pad_depth_front, pad_depth_back], [pad_height_top, pad_height_bottom], [pad_width_left, pad_width_right], [0,0]]`.
            The default value is 0.
        stride(int|tuple, optional): The stride size. It means the stride in transposed convolution.
            If stride is a tuple, it must contain three integers, (stride_depth, stride_height,
            stride_width). Otherwise, stride_depth = stride_height = stride_width = stride.
            The default value is 1.
        dilation(int|tuple, optional): The dilation size. If dilation is a tuple, it must
            contain three integers, (dilation_D, dilation_H, dilation_W). Otherwise, the
            dilation_D = dilation_H = dilation_W = dilation. The default value is 1.
        groups(int, optional): The groups number of the Conv3D transpose layer. Inspired by
            grouped convolution in Alex Krizhevsky's Deep CNN paper, in which
            when group=2, the first half of the filters is only connected to the
            first half of the input channels, while the second half of the
            filters is only connected to the second half of the input channels.
            The default value is 1.
        param_attr (ParamAttr, optional): The parameter attribute for learnable parameters/weights
            of conv3d_transpose. If it is set to None or one attribute of ParamAttr, conv3d_transpose
            will create ParamAttr as param_attr. If the Initializer of the param_attr
            is not set, the parameter is initialized with Xavier. The default value is None.
        bias_attr (ParamAttr|bool, optional): The parameter attribute for the bias of conv3d_transpose.
            If it is set to False, no bias will be added to the output units.
            If it is set to None or one attribute of ParamAttr, conv3d_transpose
            will create ParamAttr as bias_attr. If the Initializer of the bias_attr
            is not set, the bias is initialized zero. The default value is None.
        use_cudnn(bool, optional): Use cudnn kernel or not, it is valid only when the cudnn
            library is installed. The default value is True.
        act (str, optional): Activation type, if it is set to None, activation is not appended.
            The default value is None.
        name(str, optional): The default value is None. Normally there is no need for user
            to set this property. For more information, please refer to :ref:`api_guide_Name`.

    Attribute:
        **weight** (Parameter): the learnable weights of filters of this layer.

        **bias** (Parameter): the learnable bias of this layer.

    Returns:
        None.

    Raises:
        ValueError: If the shapes of input, filter_size, stride, padding and
                    groups mismatch.

    Examples:
       .. code-block:: python

         import paddle.fluid as fluid
         import numpy

         with fluid.dygraph.guard():
             data = numpy.random.random((5, 3, 12, 32, 32)).astype('float32')
             conv3dTranspose = fluid.dygraph.nn.Conv3DTranspose(
                    num_channels=3,
                    num_filters=12,
                    filter_size=12,
                    use_cudnn=False)
             ret = conv3dTranspose(fluid.dygraph.base.to_variable(data))

    """

    def __init__(
        self,
        num_channels,
        num_filters,
        filter_size,
        padding=0,
        stride=1,
        dilation=1,
        groups=None,
        param_attr=None,
        bias_attr=None,
        use_cudnn=True,
        act=None,
        dtype='float32',
    ):
        super().__init__()
        if not isinstance(use_cudnn, bool):
            raise ValueError("use_cudnn should be True or False")
        assert (
            param_attr is not False
        ), "param_attr should not be False in conv3d_transpose."
        self._padding = utils.convert_to_list(padding, 3, 'padding')
        self._stride = utils.convert_to_list(stride, 3, 'stride')
        self._dilation = utils.convert_to_list(dilation, 3, 'dilation')
        self._param_attr = param_attr
        self._num_channels = num_channels
        self._filter_size = filter_size
        self._groups = 1 if groups is None else groups
        self._num_filters = num_filters
        self._use_cudnn = use_cudnn
        self._bias_attr = bias_attr
        self._act = act
        self._dtype = dtype

        self._filter_size = utils.convert_to_list(
            self._filter_size, 3, 'conv3d_transpose.filter_size'
        )

        filter_shape = [
            self._num_channels,
            self._num_filters // self._groups,
        ] + self._filter_size
        self.weight = self.create_parameter(
            dtype=self._dtype, shape=filter_shape, attr=self._param_attr
        )
        self.bias = self.create_parameter(
            attr=self._bias_attr,
            shape=[self._num_filters],
            dtype=self._dtype,
            is_bias=True,
        )

    def forward(self, input):
        pre_bias = self._helper.create_variable_for_type_inference(
            dtype=self._dtype
        )
        self._helper.append_op(
            type="conv3d_transpose",
            inputs={'Input': [input], 'Filter': [self.weight]},
            outputs={'Output': pre_bias},
            attrs={
                'strides': self._stride,
                'paddings': self._padding,
                'dilations': self._dilation,
                'groups': self._groups if self._groups else 1,
                'use_cudnn': self._use_cudnn,
            },
        )

        if self._bias_attr:
            pre_act = self._helper.create_variable_for_type_inference(
                dtype=self._dtype
            )
            self._helper.append_op(
                type='elementwise_add',
                inputs={'X': [pre_bias], 'Y': [self.bias]},
                outputs={'Out': [pre_act]},
                attrs={'axis': 1},
            )
        else:
            pre_act = pre_bias

        # Currently, we don't support inplace in imperative mode
        return self._helper.append_activation(pre_act, act=self._act)


class Linear(layers.Layer):
    """

    Fully-connected linear transformation layer:

    .. math::

        Out = Act({XW + b})

    where :math:`X` is the input Tensor, :math:`W` and :math:`b` are weight and bias respectively.

    Linear layer takes only one ``Tensor`` input.
    The Linear layer multiplies input tensor with weight matrix and
    produces an output Tensor of shape [N, *, `output_dim`],
    where N is batch size and `*` means any number of additional dimensions.
    If ``bias_attr`` is not None, a bias variable will be created and added to the output.
    Finally, if ``act`` is not None, it will be applied to the output as well.

    Parameters:
        input_dim(int): The number of input units in this layer.
        output_dim(int): The number of output units in this layer.
        param_attr(ParamAttr or list of ParamAttr, optional): The parameter attribute for learnable
            weights(Parameter) of this layer. Default: None.
        bias_attr(ParamAttr or list of ParamAttr, optional): The attribute for the bias
            of this layer. If it is set to False, no bias will be added to the output units.
            If it is set to None, the bias is initialized zero. Default: None.
        act(str, optional): Activation to be applied to the output of this layer. Default: None.
        dtype(str, optional): Dtype used for weight, it can be "float32" or "float64". Default: "float32".

    Attributes:
        **weight** (Parameter): the learnable weights of this layer.

        **bias** (Parameter or None): the learnable bias of this layer.

    Returns:
        None

    Examples:
        .. code-block:: python

          from paddle.fluid.dygraph.base import to_variable
          import paddle.fluid as fluid
          from paddle.fluid.dygraph import Linear
          import numpy as np

          data = np.random.uniform(-1, 1, [30, 10, 32]).astype('float32')
          with fluid.dygraph.guard():
              linear = Linear(32, 64)
              data = to_variable(data)
              res = linear(data)  # [30, 10, 64]
    """

    def __init__(
        self,
        input_dim,
        output_dim,
        param_attr=None,
        bias_attr=None,
        act=None,
        dtype="float32",
    ):
        super().__init__()
        self._act = act
        self._dtype = dtype
        self.weight = self.create_parameter(
            shape=[input_dim, output_dim],
            attr=param_attr,
            dtype=dtype,
            is_bias=False,
        )
        self.bias = self.create_parameter(
            shape=[output_dim], attr=bias_attr, dtype=dtype, is_bias=True
        )

        self._use_mkldnn = _global_flags()["FLAGS_use_mkldnn"]

    def forward(self, input):
        if _non_static_mode():
            pre_bias = _varbase_creator(dtype=input.dtype)
            _legacy_C_ops.matmul(
                input,
                self.weight,
                pre_bias,
                'transpose_X',
                False,
                'transpose_Y',
                False,
                "alpha",
                1,
                "use_mkldnn",
                self._use_mkldnn,
            )
            pre_act = dygraph_utils._append_bias_in_dygraph(
                pre_bias,
                self.bias,
                axis=len(input.shape) - 1,
                use_mkldnn=self._use_mkldnn,
            )

            return dygraph_utils._append_activation_in_dygraph(
                pre_act, self._act, use_mkldnn=self._use_mkldnn
            )

        check_variable_and_dtype(
            input, 'input', ['float16', 'float32', 'float64'], "Linear"
        )

        attrs = {
            "transpose_X": False,
            "transpose_Y": False,
            "alpha": 1,
            "use_mkldnn": self._use_mkldnn,
        }
        inputs = {"X": [input], "Y": [self.weight]}

        tmp = self._helper.create_variable_for_type_inference(self._dtype)
        self._helper.append_op(
            type="matmul", inputs=inputs, outputs={"Out": tmp}, attrs=attrs
        )
        if self.bias is not None:
            pre_activation = self._helper.create_variable_for_type_inference(
                dtype=self._dtype
            )
            self._helper.append_op(
                type='elementwise_add',
                inputs={'X': [tmp], 'Y': [self.bias]},
                outputs={'Out': [pre_activation]},
                attrs={
                    'axis': len(input.shape) - 1,
                    'use_mkldnn': self._use_mkldnn,
                },
            )
        else:
            pre_activation = tmp
        return self._helper.append_activation(pre_activation, act=self._act)


class BatchNorm(layers.Layer):
    r"""

    This interface is used to construct a callable object of the ``BatchNorm`` class.
    For more details, refer to code examples.
    It implements the function of the Batch Normalization Layer and can be used
    as a normalizer function for conv2d and fully connected operations.
    The data is normalized by the mean and variance of the channel based on the current batch data.
    Refer to `Batch Normalization: Accelerating Deep Network Training by Reducing
    Internal Covariate Shift <https://arxiv.org/pdf/1502.03167.pdf>`_
    for more details.

    When use_global_stats = False, the :math:`\mu_{\beta}`
    and :math:`\sigma_{\beta}^{2}` are the statistics of one mini-batch.
    Calculated as follows:

    ..  math::

        \mu_{\beta} &\gets \frac{1}{m} \sum_{i=1}^{m} x_i \qquad &
        //\ mini-batch\ mean \\
        \sigma_{\beta}^{2} &\gets \frac{1}{m} \sum_{i=1}^{m}(x_i - \mu_{\beta})^2 \qquad &
        //\ mini-batch\ variance \\

    - :math:`x` : mini-batch data
    - :math:`m` : the size of the mini-batch data

    When use_global_stats = True, the :math:`\\mu_{\\beta}`
    and :math:`\\sigma_{\\beta}^{2}` are not the statistics of one mini-batch.
    They are global or running statistics (moving_mean and moving_variance). It usually got from the
    pre-trained model. Calculated as follows:

    .. math::
        moving\_mean = moving\_mean * momentum + \mu_{\beta} * (1. - momentum) \quad &// global mean \\
        moving\_variance = moving\_variance * momentum + \sigma_{\beta}^{2} * (1. - momentum) \quad &// global variance \\

    The normalization function formula is as follows:

    ..  math::

        \hat{x_i} &\gets \frac{x_i - \mu_\beta} {\sqrt{\
        \sigma_{\beta}^{2} + \epsilon}} \qquad &//\ normalize \\
        y_i &\gets \gamma \hat{x_i} + \beta \qquad &//\ scale\ and\ shift


    - :math:`\epsilon` : add a smaller value to the variance to prevent division by zero
    - :math:`\gamma` : trainable proportional parameter
    - :math:`\beta` : trainable deviation parameter

    Parameters:
        num_channels(int): Indicate the number of channels of the input ``Tensor``.
        act(str, optional): Activation to be applied to the output of batch normalization. Default: None.
        is_test (bool, optional): A flag indicating whether it is in test phrase or not.
             This flag only has effect on static graph mode. For dygraph mode, please use ``eval()``.
             Default: False.
        momentum(float, optional): The value used for the moving_mean and moving_var computation. Default: 0.9.
        epsilon(float, optional): The small value added to the variance to prevent division by zero. Default: 1e-5.
        param_attr(ParamAttr, optional): The parameter attribute for Parameter `scale`
             of batch_norm. If it is set to None or one attribute of ParamAttr, batch_norm
             will create ParamAttr as param_attr. If the Initializer of the param_attr
             is not set, the parameter is initialized with Xavier. Default: None.
        bias_attr(ParamAttr, optional): The parameter attribute for the bias of batch_norm.
             If it is set to None or one attribute of ParamAttr, batch_norm
             will create ParamAttr as bias_attr. If the Initializer of the bias_attr
             is not set, the bias is initialized zero. Default: None.
        dtype(str, optional): Indicate the data type of the input ``Tensor``,
             which can be float32 or float64. Default: float32.
        data_layout(str, optional): Specify the input data format, the data format can be "NCHW" or "NHWC". Default: NCHW.
        in_place(bool, optional): Make the input and output of batch norm reuse memory. Default: False.
        moving_mean_name(str, optional): The name of moving_mean which store the global Mean. Default: None.
        moving_variance_name(str, optional): The name of the moving_variance which store the global Variance. Default: None.
        do_model_average_for_mean_and_var(bool, optional): Whether parameter mean and variance should do model
            average when model average is enabled. Default: True.
        use_global_stats(bool, optional): Whether to use global mean and
            variance. In inference or test mode, set use_global_stats to true
            or is_test to true, and the behavior is equivalent.
            In train mode, when setting use_global_stats True, the global mean
            and variance are also used during train period. Default: False.
        trainable_statistics(bool, optional): Whether to calculate mean and var in eval mode. In eval mode, when
            setting trainable_statistics True, mean and variance will be calculated by current batch statistics.
            Default: False.

    Returns:
        None

    Examples:
        .. code-block:: python

          import paddle.fluid as fluid
          from paddle.fluid.dygraph.base import to_variable
          import numpy as np

          x = np.random.random(size=(3, 10, 3, 7)).astype('float32')
          with fluid.dygraph.guard():
              x = to_variable(x)
              batch_norm = fluid.BatchNorm(10)
              hidden1 = batch_norm(x)
    """

    def __init__(
        self,
        num_channels,
        act=None,
        is_test=False,
        momentum=0.9,
        epsilon=1e-05,
        param_attr=None,
        bias_attr=None,
        dtype='float32',
        data_layout='NCHW',
        in_place=False,
        moving_mean_name=None,
        moving_variance_name=None,
        do_model_average_for_mean_and_var=True,
        use_global_stats=False,
        trainable_statistics=False,
    ):
        super().__init__()
        self._param_attr = param_attr
        self._bias_attr = bias_attr
        self._act = act
        self._use_mkldnn = _global_flags()["FLAGS_use_mkldnn"]

        assert (
            bias_attr is not False
        ), "bias_attr should not be False in batch_norm."

        if dtype == "float16":
            self._dtype = "float32"
        else:
            self._dtype = dtype

        param_shape = [num_channels]

        # create parameter
        self.weight = self.create_parameter(
            attr=self._param_attr,
            shape=param_shape,
            dtype=self._dtype,
            default_initializer=Constant(1.0),
        )
        self.weight.stop_gradient = (
            use_global_stats and self._param_attr.learning_rate == 0.0
        )

        self.bias = self.create_parameter(
            attr=self._bias_attr,
            shape=param_shape,
            dtype=self._dtype,
            is_bias=True,
        )
        self.bias.stop_gradient = (
            use_global_stats and self._param_attr.learning_rate == 0.0
        )

        self._mean = self.create_parameter(
            attr=ParamAttr(
                name=moving_mean_name,
                initializer=Constant(0.0),
                trainable=False,
                do_model_average=do_model_average_for_mean_and_var,
            ),
            shape=param_shape,
            dtype=self._dtype,
        )
        self._mean.stop_gradient = True

        self._variance = self.create_parameter(
            attr=ParamAttr(
                name=moving_variance_name,
                initializer=Constant(1.0),
                trainable=False,
                do_model_average=do_model_average_for_mean_and_var,
            ),
            shape=param_shape,
            dtype=self._dtype,
        )
        self._variance.stop_gradient = True

        self._in_place = in_place
        self._data_layout = data_layout
        self._momentum = momentum
        self._epsilon = epsilon
        self._is_test = is_test
        self._fuse_with_relu = False
        self._use_global_stats = use_global_stats
        self._trainable_statistics = trainable_statistics

    def forward(self, input):
        # create output
        # mean and mean_out share the same memory
        mean_out = self._mean
        # variance and variance out share the same memory
        variance_out = self._variance

        if _non_static_mode():
            if in_dygraph_mode():
                batch_norm_out, t1, t2, t3, t4, _ = _C_ops.batch_norm(
                    input,
                    self._mean,
                    self._variance,
                    self.weight,
                    self.bias,
                    not self.training,
                    self._momentum,
                    self._epsilon,
                    self._data_layout,
                    self._use_global_stats,
                    self._trainable_statistics,
                )
                return dygraph_utils._append_activation_in_dygraph(
                    batch_norm_out, act=self._act, use_mkldnn=self._use_mkldnn
                )

            elif _in_legacy_dygraph():
                attrs = (
                    "momentum",
                    self._momentum,
                    "epsilon",
                    self._epsilon,
                    "is_test",
                    not self.training,
                    "data_layout",
                    self._data_layout,
                    "use_mkldnn",
                    self._use_mkldnn,
                    "fuse_with_relu",
                    self._fuse_with_relu,
                    "use_global_stats",
                    self._use_global_stats,
                    'trainable_statistics',
                    self._trainable_statistics,
                )
                batch_norm_out, _, _, _, _, _ = _legacy_C_ops.batch_norm(
                    input,
                    self.weight,
                    self.bias,
                    self._mean,
                    self._variance,
                    None,
                    mean_out,
                    variance_out,
                    *attrs
                )

            return dygraph_utils._append_activation_in_dygraph(
                batch_norm_out, act=self._act, use_mkldnn=self._use_mkldnn
            )

        check_variable_and_dtype(
            input, 'input', ['float16', 'float32', 'float64'], 'BatchNorm'
        )

        attrs = {
            "momentum": self._momentum,
            "epsilon": self._epsilon,
            "is_test": self._is_test,
            "data_layout": self._data_layout,
            "use_mkldnn": False,
            "fuse_with_relu": self._fuse_with_relu,
            "use_global_stats": self._use_global_stats,
            "trainable_statistics": self._trainable_statistics,
        }

        inputs = {
            "X": [input],
            "Scale": [self.weight],
            "Bias": [self.bias],
            "Mean": [self._mean],
            "Variance": [self._variance],
        }

        saved_mean = self._helper.create_variable_for_type_inference(
            dtype=self._dtype, stop_gradient=True
        )
        saved_variance = self._helper.create_variable_for_type_inference(
            dtype=self._dtype, stop_gradient=True
        )
        reserve_space = self._helper.create_variable_for_type_inference(
            dtype=self._helper.input_dtype(input), stop_gradient=True
        )

        batch_norm_out = (
            input
            if self._in_place
            else self._helper.create_variable_for_type_inference(self._dtype)
        )

        outputs = {
            "Y": [batch_norm_out],
            "MeanOut": [mean_out],
            "VarianceOut": [variance_out],
            "SavedMean": [saved_mean],
            "SavedVariance": [saved_variance],
        }
        if reserve_space is not None:
            outputs["ReserveSpace"] = [reserve_space]

        self._helper.append_op(
            type="batch_norm", inputs=inputs, outputs=outputs, attrs=attrs
        )

        # Currently, we don't support inplace in dygraph mode
        return self._helper.append_activation(batch_norm_out, self._act)


class RowConv(layers.Layer):
    """
    ***Row-convolution operator***

    The row convolution is called lookahead convolution.  This operator was introduced in the following paper for DeepSpeech2:
    http://www.cs.cmu.edu/~dyogatam/papers/wang+etal.iclrworkshop2016.pdf

    The main motivation is that a bidirectional RNN, useful in DeepSpeech like speech models, learns representation for a sequence by performing a
    forward and a backward pass through the entire sequence. However, unlike
    unidirectional RNNs, bidirectional RNNs are challenging to deploy in an online
    and low-latency setting. The lookahead convolution incorporates information
    from future subsequences in a computationally efficient manner to improve
    unidirectional recurrent neural networks. The row convolution operator is
    different from the 1D sequence convolution, and is computed as follows:

    Given an input sequence X of length t and input dimension D, and a filter (W) of size context * D.

    More details about row_conv please refer to the design document https://github.com/PaddlePaddle/Paddle/issues/2228#issuecomment-303903645 .

    Parameters:
        name_scope(str): The name of this class.
        future_context_size (int): Future context size. Please note, the shape
            of convolution kernel is [future_context_size + 1, D].
        param_attr (ParamAttr): Attributes of parameters, including
            name, initializer etc. Default: None.
        act (str): Non-linear activation to be applied to output variable. Default: None.

    Attributes:
        weight (Parameter): the learnable weights of this layer.

    Returns:
        the output(Out) is a LodTensor, which supports variable time-length input sequences.
        The underlying tensor in this LodTensor is a matrix with shape T x N, i.e., the same shape as X.

    Examples:
        .. code-block:: python

          import paddle.fluid as fluid
          import numpy

          with fluid.dygraph.guard():
              x = numpy.random.random((16)).astype('float32')
              rowConv = fluid.dygraph.nn.RowConv(
                    'RowConv', future_context_size=2)
              ret = rowConv(fluid.dygraph.base.to_variable(x))

    """

    def __init__(
        self, name_scope, future_context_size, param_attr=None, act=None
    ):
        assert (
            not _non_static_mode()
        ), "RowConv is not supported by dynamic graph mode yet!"
        super().__init__(name_scope)
        self._act = act
        self._param_attr = param_attr
        self._future_context_size = future_context_size

    def _build_once(self, input):
        self._dtype = self._helper.input_dtype(input)
        filter_shape = [self._future_context_size + 1, input.shape[1]]
        self.weight = self.create_parameter(
            attr=self._param_attr,
            shape=filter_shape,
            dtype=self._dtype,
            is_bias=False,
        )

    def forward(self, input):
        out = self._helper.create_variable_for_type_inference(self._dtype)
        self._helper.append_op(
            type='row_conv',
            inputs={'X': [input], 'Filter': [self.weight]},
            outputs={'Out': [out]},
        )
        return self._helper.append_activation(out, act=self._act)
