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

from __future__ import print_function

from six.moves import reduce

from .. import core
from ..layers import utils
from . import layers
from ..framework import Variable, in_dygraph_mode, OpProtoHolder, Parameter
from ..param_attr import ParamAttr
from ..initializer import Normal, Constant, NumpyArrayInitializer
import numpy as np
import logging

__all__ = [
    'Conv2D', 'Conv3D', 'Pool2D', 'FC', 'BatchNorm', 'Embedding', 'GRUUnit',
    'LayerNorm', 'NCE', 'PRelu', 'BilinearTensorProduct', 'Conv2DTranspose',
    'Conv3DTranspose', 'GroupNorm', 'SpectralNorm', 'TreeConv'
]


class Conv2D(layers.Layer):
    """
    This interface is used to construct a callable object of the ``Conv2D`` class.
    For more details, refer to code examples.
    The convolution2D layer calculates the output based on the input, filter
    and strides, paddings, dilations, groups parameters. Input and
    Output are in NCHW format, where N is batch size, C is the number of
    the feature map, H is the height of the feature map, and W is the width of the feature map.
    Filter's shape is [MCHW] , where M is the number of output feature map,
    C is the number of input feature map, H is the height of the filter,
    and W is the width of the filter. If the groups is greater than 1,
    C will equal the number of input feature map divided by the groups.
    Please refer to UFLDL's `convolution
    <http://ufldl.stanford.edu/tutorial/supervised/FeatureExtractionUsingConvolution/>`_
    for more detials.
    If bias attribution and activation type are provided, bias is added to the
    output of the convolution, and the corresponding activation function is
    applied to the final result.

    For each input :math:`X`, the equation is:

    .. math::

        Out = \\sigma (W \\ast X + b)

    Where:

    * :math:`X`: Input value, a ``Tensor`` with NCHW format.
    * :math:`W`: Filter value, a ``Tensor`` with shape [MCHW] .
    * :math:`\\ast`: Convolution operation.
    * :math:`b`: Bias value, a 2-D ``Tensor`` with shape [M, 1].
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

    Parameters:
        name_scope(str): The name for this class.
        num_filters(int): The number of filter. It is as same as the output
            feature map.
        filter_size (int or tuple): The filter size. If filter_size is a tuple,
            it must contain two integers, (filter_size_H, filter_size_W).
            Otherwise, the filter will be a square.
        stride (int or tuple, optional): The stride size. If stride is a tuple, it must
            contain two integers, (stride_H, stride_W). Otherwise, the
            stride_H = stride_W = stride. Default: 1.
        padding (int or tuple, optional): The padding size. If padding is a tuple, it must
            contain two integers, (padding_H, padding_W). Otherwise, the
            padding_H = padding_W = padding. Default: 0.
        dilation (int or tuple, optional): The dilation size. If dilation is a tuple, it must
            contain two integers, (dilation_H, dilation_W). Otherwise, the
            dilation_H = dilation_W = dilation. Default: 1.
        groups (int, optional): The groups number of the Conv2d Layer. According to grouped
            convolution in Alex Krizhevsky's Deep CNN paper: when group=2,
            the first half of the filters is only connected to the first half
            of the input channels, while the second half of the filters is only
            connected to the second half of the input channels. Default: 1.
        param_attr (ParamAttr, optional): The parameter attribute for learnable weights(Parameter)
            of conv2d. If it is set to None or one attribute of ParamAttr, conv2d
            will create ParamAttr as param_attr. If the Initializer of the param_attr
            is not set, the parameter is initialized with :math:`Normal(0.0, std)`,
            and the :math:`std` is :math:`(\\frac{2.0 }{filter\_elem\_num})^{0.5}`. Default: None.
        bias_attr (ParamAttr or bool, optional): The attribute for the bias of conv2d.
            If it is set to False, no bias will be added to the output units.
            If it is set to None or one attribute of ParamAttr, conv2d
            will create ParamAttr as bias_attr. If the Initializer of the bias_attr
            is not set, the bias is initialized zero. Default: None.
        use_cudnn (bool, optional): Use cudnn kernel or not, it is valid only when the cudnn
            library is installed. Default: True.
        act (str, optional): Activation type, if it is set to None, activation is not appended.
            Default: None.
        dtype (str, optional): Data type, it can be "float32" or "float64". Default: "float32".

    Attribute:
        **weight** (Parameter): the learnable weights of filter of this layer.

        **bias** (Parameter or None): the learnable bias of this layer.

    Returns:
        None
    
    Raises:
        ValueError: if ``use_cudnn`` is not a bool value.

    Examples:
        .. code-block:: python

          from paddle.fluid.dygraph.base import to_variable
          import paddle.fluid as fluid
          from paddle.fluid.dygraph import Conv2D
          import numpy as np

          data = np.random.uniform(-1, 1, [10, 3, 32, 32]).astype('float32')
          with fluid.dygraph.guard():
              conv2d = Conv2D("conv2d", 2, 3)
              data = to_variable(data)
              conv = conv2d(data)

    """

    def __init__(self,
                 name_scope,
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
                 dtype='float32'):
        assert param_attr is not False, "param_attr should not be False here."
        super(Conv2D, self).__init__(name_scope, dtype)
        self._groups = groups
        self._stride = utils.convert_to_list(stride, 2, 'stride')
        self._padding = utils.convert_to_list(padding, 2, 'padding')
        self._dilation = utils.convert_to_list(dilation, 2, 'dilation')
        self._act = act
        if not isinstance(use_cudnn, bool):
            raise ValueError("use_cudnn should be True or False")
        self._use_cudnn = use_cudnn
        self._filter_size = filter_size
        self._num_filters = num_filters
        self._param_attr = param_attr
        self._bias_attr = bias_attr
        self._dtype = dtype
        # if (self._num_channels == self._groups and
        #         num_filters % self._num_channels == 0 and not self._use_cudnn):
        #     self._l_type = 'depthwise_conv2d'
        # else:
        # TODO(jiabin): recover the usage of depthwise_conv2d when it's
        #  kernel fixed https://github.com/PaddlePaddle/Paddle/issues/17275
        self._l_type = 'conv2d'

    def _build_once(self, input):
        self._num_channels = input.shape[1]
        if self._groups is None:
            num_filter_channels = self._num_channels
        else:
            if self._num_channels % self._groups != 0:
                raise ValueError("num_channels must be divisible by groups.")
            num_filter_channels = self._num_channels // self._groups
        filter_size = utils.convert_to_list(self._filter_size, 2, 'filter_size')
        filter_shape = [self._num_filters, int(num_filter_channels)
                        ] + filter_size

        def _get_default_param_initializer():
            filter_elem_num = filter_size[0] * filter_size[
                1] * self._num_channels
            std = (2.0 / filter_elem_num)**0.5
            return Normal(0.0, std, 0)

        self._filter_param = self.create_parameter(
            attr=self._param_attr,
            shape=filter_shape,
            dtype=self._dtype,
            default_initializer=_get_default_param_initializer())

        self._bias_param = self.create_parameter(
            attr=self._bias_attr,
            shape=[self._num_filters],
            dtype=self._dtype,
            is_bias=True)

    @property
    def weight(self):
        return self._filter_param

    @weight.setter
    def weight(self, value):
        self._filter_param = value

    @property
    def bias(self):
        return self._bias_param

    @bias.setter
    def bias(self, value):
        self._bias_param = value

    def forward(self, input):
        pre_bias = self._helper.create_variable_for_type_inference(
            dtype=self._dtype)

        self._helper.append_op(
            type=self._l_type,
            inputs={
                'Input': input,
                'Filter': self._filter_param,
            },
            outputs={"Output": pre_bias},
            attrs={
                'strides': self._stride,
                'paddings': self._padding,
                'dilations': self._dilation,
                'groups': self._groups if self._groups else 1,
                'use_cudnn': self._use_cudnn,
                'use_mkldnn': False,
            })

        if self._bias_param is not None:
            pre_act = self._helper.create_variable_for_type_inference(
                dtype=self._dtype)
            self._helper.append_op(
                type='elementwise_add',
                inputs={'X': [pre_bias],
                        'Y': [self._bias_param]},
                outputs={'Out': [pre_act]},
                attrs={'axis': 1})
        else:
            pre_act = pre_bias

        # Currently, we don't support inplace in dygraph mode
        return self._helper.append_activation(pre_act, act=self._act)


class Conv3D(layers.Layer):
    """
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
        name_scope(str) : The name for this class.
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
        groups (int, optional): The groups number of the Conv3d Layer. According to grouped
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
                    'Conv3D', num_filters=2, filter_size=3, act="relu")
              ret = conv3d(fluid.dygraph.base.to_variable(data))

    """

    def __init__(self,
                 name_scope,
                 num_filters,
                 filter_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=None,
                 param_attr=None,
                 bias_attr=None,
                 use_cudnn=True,
                 act=None):
        assert param_attr is not False, "param_attr should not be False here."
        super(Conv3D, self).__init__(name_scope)
        self._groups = groups
        self._stride = utils.convert_to_list(stride, 3, 'stride')
        self._padding = utils.convert_to_list(padding, 3, 'padding')
        self._dilation = utils.convert_to_list(dilation, 3, 'dilation')
        self._act = act
        if not isinstance(use_cudnn, bool):
            raise ValueError("use_cudnn should be True or False")
        self._use_cudnn = use_cudnn
        self._filter_size = filter_size
        self._num_filters = num_filters
        self._param_attr = param_attr
        self._bias_attr = bias_attr

    def _build_once(self, input):
        num_channels = input.shape[1]
        self._dtype = self._helper.input_dtype(input)

        if self._groups is None:
            num_filter_channels = num_channels
        else:
            if num_channels % self._groups != 0:
                raise ValueError("num_channels must be divisible by groups.")
            num_filter_channels = num_channels // self._groups

        filter_size = utils.convert_to_list(self._filter_size, 3, 'filter_size')

        filter_shape = [self._num_filters, num_filter_channels] + filter_size

        def _get_default_param_initializer():
            filter_elem_num = filter_size[0] * filter_size[1] * filter_size[
                2] * num_channels
            std = (2.0 / filter_elem_num)**0.5
            return Normal(0.0, std, 0)

        self._filter_param = self.create_parameter(
            attr=self._param_attr,
            shape=filter_shape,
            dtype=self._dtype,
            default_initializer=_get_default_param_initializer())

        self._bias_param = self.create_parameter(
            attr=self._bias_attr,
            shape=[self._num_filters],
            dtype=self._dtype,
            is_bias=True)

    @property
    def weight(self):
        return self._filter_param

    @weight.setter
    def weight(self, value):
        self._filter_param = value

    @property
    def bias(self):
        return self._bias_param

    @bias.setter
    def bias(self, value):
        self._bias_param = value

    def forward(self, input):
        pre_bias = self._helper.create_variable_for_type_inference(
            dtype=self._dtype)

        self._helper.append_op(
            type='conv3d',
            inputs={
                'Input': input,
                'Filter': self._filter_param,
            },
            outputs={"Output": pre_bias},
            attrs={
                'strides': self._stride,
                'paddings': self._padding,
                'dilations': self._dilation,
                'groups': self._groups if self._groups else 1,
                'use_cudnn': self._use_cudnn,
                'use_mkldnn': False
            })

        if self._bias_param is not None:
            pre_act = self._helper.create_variable_for_type_inference(
                dtype=self._dtype)
            self._helper.append_op(
                type='elementwise_add',
                inputs={'X': [pre_bias],
                        'Y': [self._bias_param]},
                outputs={'Out': [pre_act]},
                attrs={'axis': 1})
        else:
            pre_act = pre_bias

        return self._helper.append_activation(pre_act, act=self._act)


class Conv3DTranspose(layers.Layer):
    """
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
        name_scope(str) : The name for this class.
        num_filters(int): The number of the filter. It is as same as the output
            image channel.
        output_size(int|tuple, optional): The output image size. If output size is a
            tuple, it must contain three integers, (image_depth, image_height, image_width). This
            parameter only works when filter_size is None. If output_size and filter_size are 
            specified at the same time, They should follow the formula above. The default value is None.
            Output_size and filter_size should not be None at the same time.
        filter_size(int|tuple, optional): The filter size. If filter_size is a tuple,
            it must contain three integers, (filter_size_D, filter_size_H, filter_size_W).
            Otherwise, the filter will be a square. None if use output size to
            calculate filter_size. The default value is None.
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
        groups(int, optional): The groups number of the Conv3d transpose layer. Inspired by
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
                    'Conv3DTranspose',
                    num_filters=12,
                    filter_size=12,
                    use_cudnn=False)
             ret = conv3dTranspose(fluid.dygraph.base.to_variable(data))

    """

    def __init__(self,
                 name_scope,
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
                 name=None):
        super(Conv3DTranspose, self).__init__(name_scope)
        if not isinstance(use_cudnn, bool):
            raise ValueError("use_cudnn should be True or False")
        assert param_attr is not False, "param_attr should not be False in conv3d_transpose."
        self._padding = utils.convert_to_list(padding, 3, 'padding')
        self._stride = utils.convert_to_list(stride, 3, 'stride')
        self._dilation = utils.convert_to_list(dilation, 3, 'dilation')
        self._param_attr = param_attr
        self._filter_size = filter_size
        self._output_size = output_size
        self._groups = 1 if groups is None else groups
        self._num_filters = num_filters
        self._use_cudnn = use_cudnn
        self._bias_attr = bias_attr
        self._act = act

    def _build_once(self, input):
        self._dtype = self._helper.input_dtype(input)
        self._input_channel = input.shape[1]

        if self._filter_size is None:
            if self._output_size is None:
                raise ValueError(
                    "output_size must be set when filter_size is None")
            if isinstance(self._output_size, int):
                self._output_size = [self._output_size, self._output_size]

            d_in = input.shape[2]
            h_in = input.shape[3]
            w_in = input.shape[4]

            filter_size_d = (self._output_size[0] -
                             (d_in - 1) * self._stride[0] + 2 * self._padding[0]
                             - 1) // self._dilation[0] + 1
            filter_size_h = (self._output_size[1] -
                             (h_in - 1) * self._stride[1] + 2 * self._padding[1]
                             - 1) // self._dilation[1] + 1
            filter_size_w = (self._output_size[2] -
                             (w_in - 1) * self._stride[2] + 2 * self._padding[2]
                             - 1) // self._dilation[2] + 1
            self._filter_size = [filter_size_d, filter_size_h, filter_size_w]
        else:
            self._filter_size = utils.convert_to_list(
                self._filter_size, 3, 'conv3d_transpose.filter_size')

        filter_shape = [
            self._input_channel, self._num_filters // self._groups
        ] + self._filter_size
        self._img_filter = self.create_parameter(
            dtype=self._dtype, shape=filter_shape, attr=self._param_attr)
        if self._bias_attr:
            self._bias_param = self.create_parameter(
                attr=self._bias_attr,
                shape=[self._num_filters],
                dtype=self._dtype,
                is_bias=True)

    @property
    def weight(self):
        return self._img_filter

    @weight.setter
    def weight(self, value):
        self._img_filter = value

    @property
    def bias(self):
        return self._bias_param

    @bias.setter
    def bias(self, value):
        self._bias_param = value

    def forward(self, input):
        pre_bias = self._helper.create_variable_for_type_inference(
            dtype=self._dtype)
        self._helper.append_op(
            type="conv3d_transpose",
            inputs={'Input': [input],
                    'Filter': [self._img_filter]},
            outputs={'Output': pre_bias},
            attrs={
                'strides': self._stride,
                'paddings': self._padding,
                'dilations': self._dilation,
                'groups': self._groups if self._groups else 1,
                'use_cudnn': self._use_cudnn
            })

        if self._bias_attr:
            pre_act = self._helper.create_variable_for_type_inference(
                dtype=self._dtype)
            self._helper.append_op(
                type='elementwise_add',
                inputs={'X': [pre_bias],
                        'Y': [self._bias_param]},
                outputs={'Out': [pre_act]},
                attrs={'axis': 1})
        else:
            pre_act = pre_bias

        # Currently, we don't support inplace in imperative mode
        return self._helper.append_activation(pre_act, act=self._act)


class Pool2D(layers.Layer):
    """
    This interface is used to construct a callable object of the ``Pool2D`` class.
    For more details, refer to code examples.
    The pooling2d operation calculates the output based on the input, pool_type and pool_size, pool_stride,
    pool_padding parameters.Input and output are in NCHW format, where N is batch size, C is the number of feature map,
    H is the height of the feature map, and W is the width of the feature map.
    Parameters(ksize, strides, paddings) are two elements. These two elements represent height and width, respectively.
    The input(X) size and output(Out) size may be different.

    Example:

        - Input:

          Input shape: :math:`(N, C, H_{in}, W_{in})`

        - Output:

          Output shape: :math:`(N, C, H_{out}, W_{out})`

        If ``ceil_mode`` = False:

        .. math::

            H_{out} = \\frac{(H_{in} - ksize[0] + 2 * paddings[0])}{strides[0]} + 1 \\\\
            W_{out} = \\frac{(W_{in} - ksize[1] + 2 * paddings[1])}{strides[1]} + 1

        If ``ceil_mode`` = True:

        .. math::

            H_{out} = \\frac{(H_{in} - ksize[0] + 2 * paddings[0] + strides[0] - 1)}{strides[0]} + 1 \\\\
            W_{out} = \\frac{(W_{in} - ksize[1] + 2 * paddings[1] + strides[1] - 1)}{strides[1]} + 1

        If ``exclusive`` = False:

        .. math::

            hstart &= i * strides[0] - paddings[0] \\\\
            hend   &= hstart + ksize[0] \\\\
            wstart &= j * strides[1] - paddings[1] \\\\
            wend   &= wstart + ksize[1] \\\\
            Output(i ,j) &= \\frac{sum(Input[hstart:hend, wstart:wend])}{ksize[0] * ksize[1]}

        If ``exclusive`` = True:

        .. math::

            hstart &= max(0, i * strides[0] - paddings[0])\\\\
            hend &= min(H, hstart + ksize[0]) \\\\
            wstart &= max(0, j * strides[1] - paddings[1]) \\\\
            wend & = min(W, wstart + ksize[1]) \\\\
            Output(i ,j) & = \\frac{sum(Input[hstart:hend, wstart:wend])}{(hend - hstart) * (wend - wstart)}

    Parameters:
        name_scope(str) : The name of this class.
        pool_size (int or list or tuple, optional): The pool kernel size. If pool kernel size is a tuple or list,
            it must contain two integers, (pool_size_Height, pool_size_Width).
            Otherwise, the pool kernel size will be a square of an int. Default: -1.
        pool_type(str, optional) : The pooling type, can be "max" for max-pooling and "avg" for average-pooling. 
            Default: max.
        pool_stride (int or list or tuple, optional): The pool stride size. If pool stride size is a tuple or list,
            it must contain two integers, (pool_stride_Height, pool_stride_Width). Otherwise,
            the pool stride size will be a square of an int. Default: 1.
        pool_padding (int or list or tuple, optional): The padding size for pooling operation. 
            If ``pool_padding`` is a tuple,
            it must contain two integers, (pool_padding_on_Height, pool_padding_on_Width).
            Otherwise, the padding size for pooling operation will be a square of an int. Default: 0.
        global_pooling (bool, optional): Whether to use the global pooling. If global_pooling = true,
            kernel size and paddings will be ignored. Default: False.
        use_cudnn (bool, optional): Only used in cudnn kernel, need install cudnn. Default: True.
        ceil_mode (bool, optional): Whether to use the ceil function to calculate output height and width.
            False is the default. If it is set to False, the floor function will be used. Default: False.
        exclusive (bool, optional): Whether to exclude padding points in average pooling mode. Default: True.
        dtype (str, optional): Data type, it can be "float32" or "float64". Default: "float32". 

    Returns:
        None

    Raises:
        ValueError: If 'pool_type' is not "max" nor "avg"
        ValueError: If 'global_pooling' is False and 'pool_size' is -1
        ValueError: If 'use_cudnn' is not a bool value.

    Examples:

        .. code-block:: python

          import paddle.fluid as fluid
          from paddle.fluid.dygraph.base import to_variable
          import numpy as np

          with fluid.dygraph.guard():
             data = numpy.random.random((3, 32, 32, 5)).astype('float32')
             pool2d = fluid.dygraph.Pool2D("pool2d",pool_size=2,
                            pool_type='max',
                            pool_stride=1,
                            global_pooling=False)
             pool2d_res = pool2d(to_variable(data))

    """

    def __init__(self,
                 name_scope,
                 pool_size=-1,
                 pool_type="max",
                 pool_stride=1,
                 pool_padding=0,
                 global_pooling=False,
                 use_cudnn=True,
                 ceil_mode=False,
                 exclusive=True,
                 dtype=core.VarDesc.VarType.FP32):
        if pool_type not in ["max", "avg"]:
            raise ValueError(
                "Unknown pool_type: '%s'. It can only be 'max' or 'avg'.",
                str(pool_type))

        if global_pooling is False and pool_size == -1:
            raise ValueError(
                "When the global_pooling is False, pool_size must be passed "
                "and be a valid value. Received pool_size: " + str(pool_size))

        if not isinstance(use_cudnn, bool):
            raise ValueError("use_cudnn should be True or False")

        super(Pool2D, self).__init__(name_scope, dtype=dtype)

        self._pool_type = pool_type
        self._pool_size = utils.convert_to_list(pool_size, 2, 'pool_size')
        self._pool_padding = utils.convert_to_list(pool_padding, 2,
                                                   'pool_padding')
        self._pool_stride = utils.convert_to_list(pool_stride, 2, 'pool_stride')
        self._global_pooling = global_pooling
        self._use_cudnn = use_cudnn
        self._ceil_mode = ceil_mode
        self._exclusive = exclusive
        self._l_type = 'pool2d'

    def forward(self, input):
        pool_out = self._helper.create_variable_for_type_inference(self._dtype)

        self._helper.append_op(
            type=self._l_type,
            inputs={"X": input},
            outputs={"Out": pool_out},
            attrs={
                "pooling_type": self._pool_type,
                "ksize": self._pool_size,
                "global_pooling": self._global_pooling,
                "strides": self._pool_stride,
                "paddings": self._pool_padding,
                "use_cudnn": self._use_cudnn,
                "ceil_mode": self._ceil_mode,
                "use_mkldnn": False,
                "exclusive": self._exclusive,
            })
        return pool_out


class FC(layers.Layer):
    """
    This interface is used to construct a callable object of the ``FC`` class.
    For more details, refer to code examples.
    It creates a fully connected layer in the network. It can take
    one or multiple ``Tensor`` as its inputs. It creates a Variable called weights for each input tensor,
    which represents a fully connected weight matrix from each input unit to
    each output unit. The fully connected layer multiplies each input tensor
    with its corresponding weight to produce an output Tensor with shape [N, `size`],
    where N is batch size. If multiple input tensors are given, the results of
    multiple output tensors with shape [N, `size`] will be summed up. If ``bias_attr``
    is not None, a bias variable will be created and added to the output.
    Finally, if ``act`` is not None, it will be applied to the output as well.

    When the input is single ``Tensor`` :

    .. math::

        Out = Act({XW + b})

    When the input are multiple ``Tensor`` :

    .. math::

        Out = Act({\sum_{i=0}^{N-1}X_iW_i + b})

    In the above equation:

    * :math:`N`: Number of the input. N equals to len(input) if input is list of ``Tensor`` .
    * :math:`X_i`: The i-th input ``Tensor`` .
    * :math:`W_i`: The i-th weights matrix corresponding i-th input tensor.
    * :math:`b`: The bias parameter created by this layer (if needed).
    * :math:`Act`: The activation function.
    * :math:`Out`: The output ``Tensor`` .

    See below for an example.

    .. code-block:: text

        Given:
            data_1.data = [[[0.1, 0.2]]]
            data_1.shape = (1, 1, 2) # 1 is batch_size

            data_2.data = [[[0.1, 0.2, 0.3]]]
            data_2.shape = (1, 1, 3) # 1 is batch_size

            fc = FC("fc", 2, num_flatten_dims=2)
            out = fc(input=[data_1, data_2])

        Then:
            out.data = [[[0.182996 -0.474117]]]
            out.shape = (1, 1, 2)

    Parameters:
        name_scope(str): The name of this class.
        size(int): The number of output units in this layer.
        num_flatten_dims (int, optional): The fc layer can accept an input tensor with more than
            two dimensions. If this happens, the multi-dimension tensor will first be flattened
            into a 2-dimensional matrix. The parameter `num_flatten_dims` determines how the input
            tensor is flattened: the first `num_flatten_dims` (inclusive, index starts from 1)
            dimensions will be flatten to form the first dimension of the final matrix (height of
            the matrix), and the rest `rank(X) - num_flatten_dims` dimensions are flattened to
            form the second dimension of the final matrix (width of the matrix). For example, suppose
            `X` is a 5-dimensional tensor with a shape [2, 3, 4, 5, 6], and `num_flatten_dims` = 3.
            Then, the flattened matrix will have a shape [2 x 3 x 4, 5 x 6] = [24, 30]. Default: 1
        param_attr (ParamAttr or list of ParamAttr, optional): The parameter attribute for learnable
            weights(Parameter) of this layer. Default: None.
        bias_attr (ParamAttr or list of ParamAttr, optional): The attribute for the bias
            of this layer. If it is set to False, no bias will be added to the output units.
            If it is set to None, the bias is initialized zero. Default: None.
        act (str, optional): Activation to be applied to the output of this layer. Default: None.
        is_test(bool, optional): A flag indicating whether execution is in test phase. Default: False.
        dtype(str, optional): Dtype used for weight, it can be "float32" or "float64". Default: "float32".

    Attribute:
        **weight** (list of Parameter): the learnable weights of this layer.

        **bias** (Parameter or None): the learnable bias of this layer.

    Returns:
        None
    
    Examples:
        .. code-block:: python

          from paddle.fluid.dygraph.base import to_variable
          import paddle.fluid as fluid
          from paddle.fluid.dygraph import FC
          import numpy as np

          data = np.random.uniform(-1, 1, [30, 10, 32]).astype('float32')
          with fluid.dygraph.guard():
              fc = FC("fc", 64, num_flatten_dims=2)
              data = to_variable(data)
              conv = fc(data)

    """

    def __init__(self,
                 name_scope,
                 size,
                 num_flatten_dims=1,
                 param_attr=None,
                 bias_attr=None,
                 act=None,
                 is_test=False,
                 dtype="float32"):
        super(FC, self).__init__(name_scope, dtype)

        self._size = size
        self._num_flatten_dims = num_flatten_dims
        self._dtype = dtype
        self._param_attr = param_attr
        self._bias_attr = bias_attr
        self._act = act
        self.__w = list()

    def _build_once(self, input):
        i = 0
        for inp, param in self._helper.iter_inputs_and_params(input,
                                                              self._param_attr):
            input_shape = inp.shape

            param_shape = [
                reduce(lambda a, b: a * b, input_shape[self._num_flatten_dims:],
                       1)
            ] + [self._size]
            self.__w.append(
                self.add_parameter(
                    '_w%d' % i,
                    self.create_parameter(
                        attr=param,
                        shape=param_shape,
                        dtype=self._dtype,
                        is_bias=False)))
            i += 1

        size = list([self._size])
        self._b = self.create_parameter(
            attr=self._bias_attr, shape=size, dtype=self._dtype, is_bias=True)

    # TODO(songyouwei): We should remove _w property
    @property
    def _w(self, i=0):
        return self.__w[i]

    @_w.setter
    def _w(self, value, i=0):
        assert isinstance(self.__w[i], Variable)
        self.__w[i].set_value(value)

    @property
    def weight(self):
        if len(self.__w) > 1:
            return self.__w
        else:
            return self.__w[0]

    @weight.setter
    def weight(self, value):
        if len(self.__w) == 1:
            self.__w[0] = value

    @property
    def bias(self):
        return self._b

    @bias.setter
    def bias(self, value):
        self._b = value

    def forward(self, input):
        mul_results = list()
        i = 0
        for inp, param in self._helper.iter_inputs_and_params(input,
                                                              self._param_attr):
            tmp = self._helper.create_variable_for_type_inference(self._dtype)
            self._helper.append_op(
                type="mul",
                inputs={"X": inp,
                        "Y": self.__w[i]},
                outputs={"Out": tmp},
                attrs={
                    "x_num_col_dims": self._num_flatten_dims,
                    "y_num_col_dims": 1
                })
            i += 1
            mul_results.append(tmp)

        if len(mul_results) == 1:
            pre_bias = mul_results[0]
        else:
            pre_bias = self._helper.create_variable_for_type_inference(
                self._dtype)
            self._helper.append_op(
                type="sum",
                inputs={"X": mul_results},
                outputs={"Out": pre_bias},
                attrs={"use_mkldnn": False})

        if self._b:
            pre_activation = self._helper.create_variable_for_type_inference(
                dtype=self._dtype)
            self._helper.append_op(
                type='elementwise_add',
                inputs={'X': [pre_bias],
                        'Y': [self._b]},
                outputs={'Out': [pre_activation]},
                attrs={'axis': self._num_flatten_dims})
        else:
            pre_activation = pre_bias
        # Currently, we don't support inplace in dygraph mode
        return self._helper.append_activation(pre_activation, act=self._act)


class BatchNorm(layers.Layer):
    """
    This interface is used to construct a callable object of the ``BatchNorm`` class.
    For more details, refer to code examples.
    It implements the function of the Batch Normalization Layer and can be used 
    as a normalizer function for conv2d and fully connected operations.
    The data is normalized by the mean and variance of the channel based on the current batch data.
    Refer to `Batch Normalization: Accelerating Deep Network Training by Reducing
    Internal Covariate Shift <https://arxiv.org/pdf/1502.03167.pdf>`_
    for more details.

    When use_global_stats = False, the :math:`\\mu_{\\beta}` 
    and :math:`\\sigma_{\\beta}^{2}` are the statistics of one mini-batch.
    Calculated as follows:

    ..  math::

        \\mu_{\\beta} &\\gets \\frac{1}{m} \\sum_{i=1}^{m} x_i \\qquad &//\\
        \ mini-batch\ mean \\\\
        \\sigma_{\\beta}^{2} &\\gets \\frac{1}{m} \\sum_{i=1}^{m}(x_i - \\
        \\mu_{\\beta})^2 \\qquad &//\ mini-batch\ variance \\\\

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

        \\hat{x_i} &\\gets \\frac{x_i - \\mu_\\beta} {\\sqrt{\\
        \\sigma_{\\beta}^{2} + \\epsilon}} \\qquad &//\ normalize \\\\
        y_i &\\gets \\gamma \\hat{x_i} + \\beta \\qquad &//\ scale\ and\ shift

    - :math:`\\epsilon` : add a smaller value to the variance to prevent division by zero
    - :math:`\\gamma` : trainable proportional parameter
    - :math:`\\beta` : trainable deviation parameter

    Parameters:
        name_scope(str): The name of this class.
        num_channels(int): Indicate the number of channels of the input ``Tensor``.
        act(str, optional): Activation to be applied to the output of batch normalizaiton. Default: None.
        is_test (bool, optional): A flag indicating whether it is in test phrase or not. Default: False.
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
        fuse_with_relu (bool, optional): When setting fuse_with_relu True, this OP performs relu after batch norm. 
            Default: False.
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
              batch_norm = fluid.BatchNorm("batch_norm", 10)
              hidden1 = batch_norm(x)
    """

    def __init__(self,
                 name_scope,
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
                 fuse_with_relu=False,
                 use_global_stats=False,
                 trainable_statistics=False):
        super(BatchNorm, self).__init__(name_scope, dtype)
        self._param_attr = param_attr
        self._bias_attr = bias_attr
        self._act = act

        assert bias_attr is not False, "bias_attr should not be False in batch_norm."

        if dtype == "float16":
            self._dtype = "float32"
        else:
            self._dtype = dtype

        param_shape = [num_channels]

        # create parameter
        self._scale = self.create_parameter(
            attr=self._param_attr,
            shape=param_shape,
            dtype=self._dtype,
            default_initializer=Constant(1.0))
        if use_global_stats and self._param_attr.learning_rate == 0.:
            self._scale.stop_gradient = True

        self._bias = self.create_parameter(
            attr=self._bias_attr,
            shape=param_shape,
            dtype=self._dtype,
            is_bias=True)
        if use_global_stats and self._param_attr.learning_rate == 0.:
            self._bias.stop_gradient = True

        self._mean = self.create_parameter(
            attr=ParamAttr(
                name=moving_mean_name,
                initializer=Constant(0.0),
                trainable=False,
                do_model_average=do_model_average_for_mean_and_var),
            shape=param_shape,
            dtype=self._dtype)
        self._mean.stop_gradient = True

        self._variance = self.create_parameter(
            attr=ParamAttr(
                name=moving_variance_name,
                initializer=Constant(1.0),
                trainable=False,
                do_model_average=do_model_average_for_mean_and_var),
            shape=param_shape,
            dtype=self._dtype)
        self._variance.stop_gradient = True

        self._in_place = in_place
        self._data_layout = data_layout
        self._momentum = momentum
        self._epsilon = epsilon
        self._is_test = is_test
        self._fuse_with_relu = fuse_with_relu
        self._use_global_stats = use_global_stats
        self._trainable_statistics = trainable_statistics

    def _build_once(self, input):
        pass

    def forward(self, input):
        # create output
        # mean and mean_out share the same memory
        mean_out = self._mean
        # variance and variance out share the same memory
        variance_out = self._variance

        saved_mean = self._helper.create_variable_for_type_inference(
            dtype=self._dtype, stop_gradient=True)
        saved_variance = self._helper.create_variable_for_type_inference(
            dtype=self._dtype, stop_gradient=True)
        batch_norm_out = input if self._in_place else self._helper.create_variable_for_type_inference(
            self._dtype)

        self._helper.append_op(
            type="batch_norm",
            inputs={
                "X": input,
                "Scale": self._scale,
                "Bias": self._bias,
                "Mean": self._mean,
                "Variance": self._variance
            },
            outputs={
                "Y": batch_norm_out,
                "MeanOut": mean_out,
                "VarianceOut": variance_out,
                "SavedMean": saved_mean,
                "SavedVariance": saved_variance
            },
            attrs={
                "momentum": self._momentum,
                "epsilon": self._epsilon,
                "is_test": self._is_test,
                "data_layout": self._data_layout,
                "use_mkldnn": False,
                "fuse_with_relu": self._fuse_with_relu,
                "use_global_stats": self._use_global_stats,
                "trainable_statistics": self._trainable_statistics
            })

        # Currently, we don't support inplace in dygraph mode
        return self._helper.append_activation(batch_norm_out, self._act)


class Embedding(layers.Layer):
    """
    **Embedding Layer**

    This interface is used to construct a callable object of the ``Embedding`` class.
    For specific usage, refer to code examples. It implements the function of the Embedding Layer.
    This layer is used to lookup embeddings vector of ids provided by :attr:`input` .
    It automatically constructs a 2D embedding matrix based on the
    input :attr:`size` (vocab_size, emb_size) and :attr:`dtype` .

    This layer requires the last dimension of Tensor shape must be equal to 1. The shape
    of output Tensor is generated by replacing the last dimension of the input Tensor shape
    with emb_size.

    The id in :attr:`input` must satisfy :math:`0 =< id < size[0]` , 
    otherwise the program will throw an exception and exit.

    .. code-block:: text

        Case 1:

        input is a Tensor. padding_idx = -1
            input.data = [[[1], [3]], [[2], [4]], [[4], [127]]]
            input.shape = [3, 2, 1]
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

    Parameters:
        name_scope(str): The name of this class.
        size(tuple|list): The shape of the look up table parameter. It should have two elements which indicate the size
            of the dictionary of embeddings and the size of each embedding vector respectively.
        is_sparse(bool): The flag indicating whether to use sparse update. This parameter only
            affects the performance of the backwards gradient update. It is recommended to set 
            True because sparse update is faster. But some optimizer does not support sparse update,
            such as :ref:`api_fluid_optimizer_AdadeltaOptimizer` , :ref:`api_fluid_optimizer_AdamaxOptimizer` , 
            :ref:`api_fluid_optimizer_DecayedAdagradOptimizer` , :ref:`api_fluid_optimizer_FtrlOptimizer` ,
            :ref:`api_fluid_optimizer_LambOptimizer` and :ref:`api_fluid_optimizer_LarsMomentumOptimizer` .
            In these case, is_sparse must be False. Default: False.
        is_distributed(bool): Whether to store the embedding matrix in a distributed manner. Only used
            in multi-machine distributed CPU training. Default: False.
        padding_idx(int|long|None): padding_idx needs to be in the interval [-vocab_size, vocab_size). 
            If :math:`padding\_idx < 0`, the :math:`padding\_idx` will automatically be converted
            to :math:`vocab\_size + padding\_idx` . It will output all-zero padding data whenever lookup
            encounters :math:`padding\_idx` in id. And the padding data will not be updated while training.
            If set None, it makes no effect to output. Default: None.
        param_attr(ParamAttr): To specify the weight parameter property. Default: None, which means the
            default weight parameter property is used. See usage for details in :ref:`api_fluid_ParamAttr` . In addition,
            user-defined or pre-trained word vectors can be loaded with the :attr:`param_attr` parameter. 
            The local word vector needs to be transformed into numpy format, and the shape of local word
            vector shoud be consistent with :attr:`size` . Then :ref:`api_fluid_initializer_NumpyArrayInitializer`
            is used to load custom or pre-trained word vectors. See code example 2 for details.
        dtype(np.dtype|core.VarDesc.VarType|str): It refers to the data type of output Tensor.
            It must be "float32" or "float64". Default: "float32".

    Attribute:
        **weight** (Parameter): the learnable weights of this layer.

    Returns:
        Variable: Embedding Tensor or LoDTensor mapped by input. The data type is the same as :attr:`dtype` .

    Examples:

        .. code-block:: python

          import paddle.fluid as fluid
          import paddle.fluid.dygraph.base as base
          import numpy as np

          # example 1
          inp_word = np.array([[[1]]]).astype('int64')
          dict_size = 20
          with fluid.dygraph.guard():
              emb = fluid.dygraph.Embedding(
                  name_scope='embedding',
                  size=[dict_size, 32],
                  param_attr='emb.w',
                  is_sparse=False)
              static_rlt3 = emb(base.to_variable(inp_word))

          # example 2: load custom or pre-trained word vectors
          weight_data = np.random.random(size=(128, 100))  # word vectors with numpy format
          w_param_attrs = fluid.ParamAttr(
              name="emb_weight",
              learning_rate=0.5,
              initializer=fluid.initializer.NumpyArrayInitializer(weight_data),
              trainable=True)
          with fluid.dygraph.guard():
              emb = fluid.dygraph.Embedding(
                  name_scope='embedding',
                  size=[128, 100],
                  param_attr= w_param_attrs,
                  is_sparse=False)
              static_rlt3 = emb(base.to_variable(inp_word))          
    """

    def __init__(self,
                 name_scope,
                 size,
                 is_sparse=False,
                 is_distributed=False,
                 padding_idx=None,
                 param_attr=None,
                 dtype='float32'):
        super(Embedding, self).__init__(name_scope, dtype)
        self._size = size
        self._is_sparse = is_sparse
        self._is_distributed = is_distributed
        self._padding_idx = -1 if padding_idx is None else padding_idx if padding_idx >= 0 else (
            size[0] + padding_idx)

        self._param_attr = param_attr
        self._dtype = dtype
        self._remote_prefetch = self._is_sparse and (not self._is_distributed)
        if self._remote_prefetch:
            assert self._is_sparse is True and self._is_distributed is False

        self._w = self.create_parameter(
            attr=self._param_attr,
            shape=self._size,
            dtype=self._dtype,
            is_bias=False)

    @property
    def weight(self):
        return self._w

    @weight.setter
    def weight(self, value):
        self._w = value

    def forward(self, input):
        out = self._helper.create_variable_for_type_inference(self._dtype)
        self._helper.append_op(
            type='lookup_table',
            inputs={'Ids': input,
                    'W': self._w},
            outputs={'Out': out},
            attrs={
                'is_sparse': self._is_sparse,
                'is_distributed': self._is_distributed,
                'remote_prefetch': self._remote_prefetch,
                'padding_idx': self._padding_idx
            })

        return out


class LayerNorm(layers.Layer):
    """
    This interface is used to construct a callable object of the ``LayerNorm`` class.
    For more details, refer to code examples.
    It implements the function of the Layer Normalization Layer and can be applied to mini-batch input data.
    Refer to `Layer Normalization <https://arxiv.org/pdf/1607.06450v1.pdf>`_

    The formula is as follows:

    ..  math::

        \\mu & = \\frac{1}{H}\\sum_{i=1}^{H} x_i

        \\sigma & = \\sqrt{\\frac{1}{H}\sum_{i=1}^{H}{(x_i - \\mu)^2} + \\epsilon}

        y & = f(\\frac{g}{\\sigma}(x - \\mu) + b)

    - :math:`x`: the vector representation of the summed inputs to the neurons in that layer.
    - :math:`H`: the number of hidden units in a layers
    - :math:`\\epsilon`: the small value added to the variance to prevent division by zero.
    - :math:`g`: the trainable scale parameter.
    - :math:`b`: the trainable bias parameter.

    Parameters:
        name_scope(str): The name of this class.
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
        act(str, optional): Activation to be applied to the output of layer normalizaiton.
                  Default: None.
    Returns:
        None

    Examples:

        .. code-block:: python

          import paddle.fluid as fluid
          from paddle.fluid.dygraph.base import to_variable
          import numpy

          x = numpy.random.random((3, 32, 32)).astype('float32')
          with fluid.dygraph.guard():
              x = to_variable(x)
              layerNorm = fluid.LayerNorm('LayerNorm', begin_norm_axis=1)
              ret = layerNorm(x)

    """

    def __init__(self,
                 name_scope,
                 scale=True,
                 shift=True,
                 begin_norm_axis=1,
                 epsilon=1e-05,
                 param_attr=None,
                 bias_attr=None,
                 act=None):
        super(LayerNorm, self).__init__(name_scope)
        self._scale = scale
        self._shift = shift
        self._begin_norm_axis = begin_norm_axis
        self._epsilon = epsilon
        self._param_attr = param_attr
        self._bias_attr = bias_attr
        self._act = act

    def _build_once(self, input):
        self._dtype = self._helper.input_dtype(input)
        input_shape = input.shape
        param_shape = [
            reduce(lambda x, y: x * y, input_shape[self._begin_norm_axis:])
        ]
        if self._scale:
            self._scale_w = self.create_parameter(
                attr=self._param_attr,
                shape=param_shape,
                dtype=self._dtype,
                default_initializer=Constant(1.0))
        else:
            if self._param_attr:
                logging.warn("param_attr are only avaliable with scale is True")

        if self._shift:
            assert self._bias_attr is not False
            self._bias_w = self.create_parameter(
                attr=self._bias_attr,
                shape=param_shape,
                dtype=self._dtype,
                is_bias=True)
        else:
            if self._bias_attr:
                logging.warn("bias_attr are only avaliable with shift is True")

    def forward(self, input):
        inputs = dict()
        inputs['X'] = input
        if self._scale:
            inputs['Scale'] = self._scale_w
        if self._shift:
            inputs['Bias'] = self._bias_w
        # create output
        mean_out = self._helper.create_variable_for_type_inference(
            dtype=self._dtype, stop_gradient=True)
        variance_out = self._helper.create_variable_for_type_inference(
            dtype=self._dtype, stop_gradient=True)
        layer_norm_out = self._helper.create_variable_for_type_inference(
            self._dtype)

        self._helper.append_op(
            type="layer_norm",
            inputs=inputs,
            outputs={
                "Y": layer_norm_out,
                "Mean": mean_out,
                "Variance": variance_out,
            },
            attrs={
                "epsilon": self._epsilon,
                "begin_norm_axis": self._begin_norm_axis
            })

        return self._helper.append_activation(layer_norm_out, act=self._act)


class GRUUnit(layers.Layer):
    """
    **GRU unit layer**
    
    It creates a callable object from GRUUnit class.
    If origin_mode is True, then the equation of a gru step is from paper
    `Learning Phrase Representations using RNN Encoder-Decoder for Statistical 
    Machine Translation <https://arxiv.org/pdf/1406.1078.pdf>`_

        .. math::
            u_t & = actGate(xu_{t} + W_u h_{t-1} + b_u)

            r_t & = actGate(xr_{t} + W_r h_{t-1} + b_r)

            m_t & = actNode(xm_t + W_c dot(r_t, h_{t-1}) + b_m)

            h_t & = dot(u_t, h_{t-1}) + dot((1-u_t), m_t)

    If origin_mode is False, then the equation of a gru step is from paper
    `Empirical Evaluation of Gated Recurrent Neural Networks on Sequence
    Modeling <https://arxiv.org/pdf/1412.3555.pdf>`_

        .. math::
            u_t & = actGate(xu_{t} + W_u h_{t-1} + b_u)

            r_t & = actGate(xr_{t} + W_r h_{t-1} + b_r)

            m_t & = actNode(xm_t + W_c dot(r_t, h_{t-1}) + b_m)

            h_t & = dot((1-u_t), h_{t-1}) + dot(u_t, m_t)


    The inputs of gru unit includes :math:`z_t`, :math:`h_{t-1}`. In terms
    of the equation above, the :math:`z_t` is split into 3 parts -
    :math:`xu_t`, :math:`xr_t` and :math:`xm_t`. This means that in order to
    implement a full GRU unit operator for an input, a fully
    connected layer has to be applied, such that :math:`z_t = W_{fc}x_t`.

    The terms :math:`u_t` and :math:`r_t` represent the update and reset gates
    of the GRU cell. Unlike LSTM, GRU has one lesser gate. However, there is
    an intermediate candidate hidden output, which is denoted by :math:`m_t`.
    This layer has three outputs :math:`h_t`, :math:`dot(r_t, h_{t-1})`
    and concatenation of :math:`u_t`, :math:`r_t` and :math:`m_t`.

    Parameters:
        name_scope(str): The name of this class.
        size (int): The input dimension value.
        param_attr(ParamAttr, optional): The parameter attribute for the learnable
            hidden-hidden weight matrix. 
            
            **Note**:
    
                1. The shape of the weight matrix is :math:`[T, 3*D]`, where D is the hidden size.
                2. All elements in the weight matrix can be divided into two parts. The first 
                   part are weights of the update gate and reset gate with shape :math:`[D, 2*D]`, 
                   and the second part are weights for candidate hidden state with shape :math:`[D, D]`.


            If it is set to None or one attribute of ParamAttr, gru_unit will
            create ParamAttr as param_attr. If the Initializer of the param_attr
            is not set, the parameter is initialized with Xavier. The default 
            value is None.
        bias_attr (ParamAttr|bool, optional): The parameter attribute for the bias
            of GRU.Note that the bias with :math:`[1, 3*D]` concatenates
            the bias in the update gate, reset gate and candidate calculations.
            If it is set to False, no bias will be applied to the update gate,
            reset gate and candidate calculations. If it is set to None or one
            attribute of ParamAttr, gru_unit will create ParamAttr as
            bias_attr. If the Initializer of the bias_attr is not set, the bias
            is initialized zero. The default value is None.
        activation (str): The activation type for cell (actNode).
                             The default value is 'tanh'.
        gate_activation (str): The activation type for gates (actGate).
                                  The default value is 'sigmoid'.
        dtype(str): The dtype of the layers. The data type can be set as
            'float32', 'float64'. The default value is 'float32'.

    Attribute:
        **weight** (Parameter): the learnable weights of this layer.

        **bias** (Parameter): the learnable bias of this layer.

    Returns:
        tuple: The hidden value, reset-hidden value and gate values. The hidden value
        is a 2-D tensor with shape  :math:`[T, D]` . The reset-hidden value is a
        2-D tensor with shape  :math:`[T, D]` . The gate value is a 2-D tensor with 
        shape  :math:`[T, 3*D]`.

    Examples:

        .. code-block:: python

          import paddle.fluid as fluid
          import paddle.fluid.dygraph.base as base
          import numpy

          lod = [[2, 4, 3]]
          D = 5
          T = sum(lod[0])

          input = numpy.random.rand(T, 3 * D).astype('float32')
          hidden_input = numpy.random.rand(T, D).astype('float32')
          with fluid.dygraph.guard():
              x = numpy.random.random((3, 32, 32)).astype('float32')
              gru = fluid.dygraph.GRUUnit('gru', size=D * 3)
              dy_ret = gru(
                base.to_variable(input), base.to_variable(hidden_input))

    """

    def __init__(self,
                 name_scope,
                 size,
                 param_attr=None,
                 bias_attr=None,
                 activation='tanh',
                 gate_activation='sigmoid',
                 origin_mode=False,
                 dtype='float32'):
        super(GRUUnit, self).__init__(name_scope, dtype)
        self._bias_attr = bias_attr

        activation_dict = dict(
            identity=0,
            sigmoid=1,
            tanh=2,
            relu=3, )
        self.activation = activation_dict[activation]
        self.gate_activation = activation_dict[gate_activation]

        self._dtype = dtype
        size = size // 3
        # create weight
        self._weight = self.create_parameter(
            attr=param_attr, shape=[size, 3 * size], dtype=dtype)

        # create bias
        bias_size = [1, 3 * size]
        self._bias_size = bias_size
        self._bias = self.create_parameter(
            attr=bias_attr, shape=bias_size, dtype=dtype, is_bias=True)

    @property
    def weight(self):
        return self._weight

    @weight.setter
    def weight(self, value):
        self._weight = value

    @property
    def bias(self):
        return self._bias

    @bias.setter
    def bias(self, value):
        self._bias = value

    def forward(self, input, hidden):
        inputs = {'Input': input, 'HiddenPrev': hidden, 'Weight': self._weight}
        if self._bias:
            inputs['Bias'] = self._bias

        gate = self._helper.create_variable_for_type_inference(self._dtype)
        reset_hidden_pre = self._helper.create_variable_for_type_inference(
            self._dtype)
        updated_hidden = self._helper.create_variable_for_type_inference(
            self._dtype)
        self._helper.append_op(
            type='gru_unit',
            inputs=inputs,
            outputs={
                'Gate': gate,
                'ResetHiddenPrev': reset_hidden_pre,
                'Hidden': updated_hidden,
            },
            attrs={
                'activation': self.activation,
                'gate_activation': self.gate_activation,
            })

        return updated_hidden, reset_hidden_pre, gate


class NCE(layers.Layer):
    """
    This interface is used to construct a callable object of the ``NCE`` class.
    For more details, refer to code examples.
    It implements the function of the ``NCE`` loss function.
    By default this function uses a uniform distribution for sampling, and it
    compute and return the noise-contrastive estimation training loss. See
    `Noise-contrastive estimation: A new estimation principle for unnormalized statistical models <http://www.jmlr.org/proceedings/papers/v9/gutmann10a/gutmann10a.pdf>`_ .

    Parameters:
        name_scope(str): The name of this class.
        num_total_classes (int): Total number of classes in all samples
        param_attr (ParamAttr, optional): The parameter attribute for learnable weights(Parameter)
             of nce. If it is set to None or one attribute of ParamAttr, nce
             will create ParamAttr as param_attr. If the Initializer of the param_attr
             is not set, the parameter is initialized with Xavier. Default: None.
        bias_attr (ParamAttr or bool, optional): The attribute for the bias of nce.
             If it is set to False, no bias will be added to the output units.
             If it is set to None or one attribute of ParamAttr, nce
             will create ParamAttr as bias_attr. If the Initializer of the bias_attr
             is not set, the bias is initialized zero. Default: None.
        num_neg_samples (int, optional): The number of negative classes. The default value is 10.
        sampler (str, optional): The sampler used to sample class from negtive classes.
                       It can be 'uniform', 'log_uniform' or 'custom_dist'.
                       default: 'uniform'.
        custom_dist (float[], optional): A float[] with size=num_total_classes.
                       It is used when sampler is set to 'custom_dist'.
                       custom_dist[i] is the probability of i-th class to be sampled.
                       Default: None.
        seed (int, optional): The seed used in sampler. Default: 0.
        is_sparse(bool, optional): The flag indicating whether to use sparse update. If is_sparse is True, the weight@GRAD and bias@GRAD will be changed to SelectedRows. Default: False.

    Attribute:
        **weight** (Parameter): the learnable weights of this layer.

        **bias** (Parameter or None): the learnable bias of this layer.
    
    Returns:
        None

    Examples:
        .. code-block:: python

            import numpy as np
            import paddle.fluid as fluid

            window_size = 5
            dict_size = 20
            label_word = int(window_size // 2) + 1
            inp_word = np.array([[[1]], [[2]], [[3]], [[4]], [[5]]]).astype('int64')
            nid_freq_arr = np.random.dirichlet(np.ones(20) * 1000).astype('float32')

            with fluid.dygraph.guard():
                words = []
                for i in range(window_size):
                    words.append(fluid.dygraph.base.to_variable(inp_word[i]))

                emb = fluid.Embedding(
                    'embedding',
                    size=[dict_size, 32],
                    param_attr='emb.w',
                    is_sparse=False)

                embs3 = []
                for i in range(window_size):
                    if i == label_word:
                        continue

                    emb_rlt = emb(words[i])
                    embs3.append(emb_rlt)

                embs3 = fluid.layers.concat(input=embs3, axis=1)
                nce = fluid.NCE('nce',
                             num_total_classes=dict_size,
                             num_neg_samples=2,
                             sampler="custom_dist",
                             custom_dist=nid_freq_arr.tolist(),
                             seed=1,
                             param_attr='nce.w',
                             bias_attr='nce.b')

                nce_loss3 = nce(embs3, words[label_word])

    """

    def __init__(self,
                 name_scope,
                 num_total_classes,
                 sample_weight=None,
                 param_attr=None,
                 bias_attr=None,
                 num_neg_samples=None,
                 sampler="uniform",
                 custom_dist=None,
                 seed=0,
                 is_sparse=False):
        super(NCE, self).__init__(name_scope)
        self._param_attr = param_attr
        self._bias_attr = bias_attr
        self._num_total_classes = num_total_classes

        self._inputs = dict()
        self._inputs['SampleWeight'] = sample_weight if sample_weight is not None else []
        if sampler == "uniform":
            sampler = 0
        elif sampler == "log_uniform":
            sampler = 1
        elif sampler == "custom_dist":
            assert custom_dist is not None
            # assert isinstance(custom_dist, Variable)

            custom_dist_len = len(custom_dist)
            alias_probs_ = [0] * custom_dist_len
            alias_ = [0] * custom_dist_len
            bigs = []
            littles = []
            for i in range(custom_dist_len):
                normal_prob = custom_dist[i] * custom_dist_len
                if normal_prob - 1.0 > 0:
                    bigs.append((i, normal_prob))
                elif 1.0 - normal_prob > 0:
                    littles.append((i, normal_prob))
                else:
                    alias_probs_[i] = normal_prob
                    alias_[i] = -1

            while len(bigs) and len(littles):
                big = bigs.pop(0)
                little = littles.pop(0)

                big_idx = big[0]
                big_prob = big[1]

                alias_probs_[little[0]] = little[1]
                alias_[little[0]] = big_idx
                big_left = big[1] + little[1] - 1
                if big_left - 1.0 > 0:
                    bigs.append((big_idx, big_left))
                elif 1.0 - big_left > 0:
                    littles.append((big_idx, big_left))
                else:
                    alias_probs_[big_idx] = big_left
                    alias_[big_idx] = -1

            if len(bigs):
                big = bigs.pop(0)
                alias_probs_[big[0]] = 1.0
                alias_[big[0]] = -1
            if len(littles):
                little = littles.pop(0)
                alias_probs_[little[0]] = 1.0
                alias_[little[0]] = -1

            def _init_by_numpy_array(numpy_array):
                ret = self.create_parameter(
                    attr=ParamAttr(),
                    shape=numpy_array.shape,
                    dtype=numpy_array.dtype,
                    default_initializer=NumpyArrayInitializer(numpy_array))
                ret.stop_gradient = True
                return ret

            self._inputs['CustomDistProbs'] = _init_by_numpy_array(
                np.array(custom_dist).astype('float32'))
            self._inputs['CustomDistAlias'] = _init_by_numpy_array(
                np.array(alias_).astype('int32'))
            self._inputs['CustomDistAliasProbs'] = _init_by_numpy_array(
                np.array(alias_probs_).astype('float32'))
            sampler = 2
        else:
            raise Exception("Unsupported sampler type.")

        if num_neg_samples is None:
            num_neg_samples = 10
        else:
            num_neg_samples = int(num_neg_samples)
        self._num_neg_samples = num_neg_samples
        remote_prefetch = is_sparse
        print(
            "With sparse mode, if your models has only small parameter prefetch may cause speed down"
        )
        self._attrs = {
            'num_total_classes': int(num_total_classes),
            'num_neg_samples': num_neg_samples,
            'seed': seed,
            'sampler': sampler,
            'is_sparse': is_sparse,
            'remote_prefetch': remote_prefetch
        }

    def _build_once(self, input, label, sample_weight=None):
        assert isinstance(input, Variable)
        assert isinstance(label, Variable)

        dim = input.shape[1]
        num_true_class = label.shape[1]
        self._w = self.create_parameter(
            attr=self._param_attr,
            shape=[self._num_total_classes, dim],
            is_bias=False,
            dtype=input.dtype)
        if self._bias_attr:
            self._b = self.create_parameter(
                attr=self._bias_attr,
                shape=[self._num_total_classes, 1],
                is_bias=True,
                dtype=input.dtype)
            self._inputs['Bias'] = self._b
        self._inputs['Weight'] = self._w

    @property
    def weight(self):
        return self._w

    @weight.setter
    def weight(self, value):
        self._w = value

    @property
    def bias(self):
        return self._b

    @bias.setter
    def bias(self, value):
        self._b = value

    def forward(self, input, label, sample_weight=None):
        assert isinstance(input, Variable)
        assert isinstance(label, Variable)

        self._inputs['Input'] = input
        self._inputs['Label'] = label
        self._inputs['SampleWeight'] = sample_weight if sample_weight is not None else []

        cost = self._helper.create_variable_for_type_inference(
            dtype=input.dtype)
        sample_logits = self._helper.create_variable_for_type_inference(
            dtype=input.dtype)
        sample_labels = self._helper.create_variable_for_type_inference(
            dtype=label.dtype)

        self._helper.append_op(
            type='nce',
            inputs=self._inputs,
            outputs={
                'Cost': cost,
                'SampleLogits': sample_logits,
                'SampleLabels': sample_labels
            },
            attrs=self._attrs)
        return cost / (self._num_neg_samples + 1)


class PRelu(layers.Layer):
    """
    This interface is used to construct a callable object of the ``PRelu`` class.
    For more details, refer to code examples.
    It implements three activation methods of the ``PRelu`` activation function.

    Equation:

    .. math::
        y = \max(0, x) + \\alpha * \min(0, x)

    Parameters:
        name_scope(str): The name of this class.
        mode (str): The mode for weight sharing. It supports all, channel
          and element. all: all elements share same weight
          channel:elements in a channel share same weight
          element:each element has a weight
        param_attr(ParamAttr, optional): The parameter attribute for the learnable
          weight (alpha). Default: None.

    Attribute:
        **weight** (Parameter): the learnable weights of this layer.
    
    Returns:
        None

    Examples:

        .. code-block:: python

          import paddle.fluid as fluid
          from paddle.fluid.dygraph.base import to_variable
          import numpy as np

          inp_np = np.ones([5, 200, 100, 100]).astype('float32')
          with fluid.dygraph.guard():
              inp_np = to_variable(inp_np)
              mode = 'channel'
              prelu = fluid.PRelu(
                 'prelu',
                 mode=mode,
                 param_attr=fluid.ParamAttr(initializer=fluid.initializer.Constant(1.0)))
              dy_rlt = prelu(inp_np)

    """

    def __init__(self, name_scope, mode, param_attr=None):

        super(PRelu, self).__init__(name_scope)
        self._mode = mode
        self._param_attr = param_attr
        if self._mode not in ['all', 'channel', 'element']:
            raise ValueError('mode should be one of all, channel, element.')
        self._alpha_shape = [1]

    def _build_once(self, input):
        if self._mode == 'channel':
            self._alpha_shape = [1, input.shape[1], 1, 1]
        elif self._mode == 'element':
            self._alpha_shape = input.shape
        self._dtype = self._helper.input_dtype(input)
        self._alpha = self.create_parameter(
            attr=self._param_attr,
            shape=self._alpha_shape,
            dtype='float32',
            is_bias=False,
            default_initializer=Constant(1.0))

    @property
    def weight(self):
        return self._alpha

    @weight.setter
    def weight(self, value):
        self._alpha = value

    def forward(self, input):

        out = self._helper.create_variable_for_type_inference(self._dtype)
        self._helper.append_op(
            type="prelu",
            inputs={"X": input,
                    'Alpha': self._alpha},
            attrs={"mode": self._mode},
            outputs={"Out": out})
        return out


class BilinearTensorProduct(layers.Layer):
    """
    **Add Bilinear Tensor Product Layer**

    This layer performs bilinear tensor product on two inputs.
    For example:

    .. math::
      out_{i} = x * W_{i} * {y^\mathrm{T}}, i=0,1,...,size-1

    In this formula:
     - :math:`x`: the first input contains M elements, shape is [batch_size, M].
     - :math:`y`: the second input contains N elements, shape is [batch_size, N].
     - :math:`W_{i}`: the i-th learned weight, shape is [M, N]
     - :math:`out_{i}`: the i-th element of out, shape is [batch_size, size].
     - :math:`y^\mathrm{T}`: the transpose of :math:`y`.

    Parameters:
       name_scope(str): The name of this class.
       size (int): The dimension of this layer.
       name (str): The default value is None. Normally there is no need for user 
           to set this property. For more information, please refer to :ref:`api_guide_Name`.
       act (str, optional): Activation to be applied to the output of this layer. The default value is None.
       param_attr (ParamAttr, optional): The parameter attribute for the learnable w, parameters/weights of 
           this layer. The default value is None.
       bias_attr (ParamAttr, optional): The parameter attribute for the bias
           of this layer. If it is set to False, no bias will be added to the output units.
           If it is set to None, the bias is initialized zero. The default value is None.

    Attribute:
        **weight** (Parameter): the learnable weights of this layer.

        **bias** (Parameter): the learnable bias of this layer.

    Returns:
       Variable: A 2-D Tensor of shape [batch_size, size].

    Examples:
       .. code-block:: python

         import paddle.fluid as fluid
         import numpy

         with fluid.dygraph.guard():
             layer1 = numpy.random.random((5, 5)).astype('float32')
             layer2 = numpy.random.random((5, 4)).astype('float32')
             bilinearTensorProduct = fluid.dygraph.nn.BilinearTensorProduct(
                    'BilinearTensorProduct', size=1000)
             ret = bilinearTensorProduct(fluid.dygraph.base.to_variable(layer1),
                                fluid.dygraph.base.to_variable(layer2))
    """

    def __init__(self,
                 name_scope,
                 size,
                 name=None,
                 act=None,
                 param_attr=None,
                 bias_attr=None):
        super(BilinearTensorProduct, self).__init__(name_scope)
        self._param_attr = param_attr
        self._bias_attr = bias_attr
        self._act = act
        self._size = size
        self._name = name
        self._inputs = dict()

    def _build_once(self, x, y):
        self._dtype = self._helper.input_dtype(x)

        param_shape = [self._size, x.shape[1], y.shape[1]]

        self._w = self.create_parameter(
            attr=self._param_attr,
            shape=param_shape,
            dtype=self._dtype,
            is_bias=False)

        bias_size = [1, self._size]
        self._bias_param = self.create_parameter(
            attr=self._bias_attr,
            shape=bias_size,
            dtype=self._dtype,
            is_bias=True)

    @property
    def weight(self):
        return self._w

    @weight.setter
    def weight(self, value):
        self._w = value

    @property
    def bias(self):
        return self._bias_param

    @bias.setter
    def bias(self, value):
        self._bias_param = value

    def forward(self, x, y):
        self._inputs = {"X": x, "Y": y, "Weight": self._w}
        if self._bias_param:
            self._inputs["Bias"] = self._bias_param
        if self._name is not None:
            out = self._helper.create_variable(
                name=".".join([self.full_name(), self._name]),
                dtype=self._dtype,
                persistable=False)
        else:
            out = self._helper.create_variable(
                dtype=self._dtype, persistable=False)
        self._helper.append_op(
            type="bilinear_tensor_product",
            inputs=self._inputs,
            outputs={"Out": out})

        # add activation
        return self._helper.append_activation(out, act=self._act)


class Conv2DTranspose(layers.Layer):
    """
    This interface is used to construct a callable object of the ``Conv2DTranspose`` class.
    For more details, refer to code examples.
    The convolution2D transpose layer calculates the output based on the input,
    filter, and dilations, strides, paddings. Input and output
    are in NCHW format. Where N is batch size, C is the number of feature map,
    H is the height of the feature map, and W is the width of the feature map.
    Filter's shape is [MCHW] , where M is the number of output feature map,
    C is the number of input feature map, H is the height of the filter,
    and W is the width of the filter. If the groups is greater than 1,
    C will equal the number of input feature map divided by the groups.
    If bias attribution and activation type are provided, bias is added to
    the output of the convolution, and the corresponding activation function
    is applied to the final result.
    The details of convolution transpose layer, please refer to the following explanation and references
    `conv2dtranspose <http://www.matthewzeiler.com/wp-content/uploads/2017/07/cvpr2010.pdf>`_ .

    For each input :math:`X`, the equation is:

    .. math::

        Out = \sigma (W \\ast X + b)

    Where:

    * :math:`X`: Input value, a ``Tensor`` with NCHW format.
    * :math:`W`: Filter value, a ``Tensor`` with shape [MCHW] .
    * :math:`\\ast`: Convolution operation.
    * :math:`b`: Bias value, a 2-D ``Tensor`` with shape [M, 1].
    * :math:`\\sigma`: Activation function.
    * :math:`Out`: Output value, the shape of :math:`Out` and :math:`X` may be different.

    Example:

        - Input:

          Input shape: :math:`(N, C_{in}, H_{in}, W_{in})`

          Filter shape: :math:`(C_{in}, C_{out}, H_f, W_f)`

        - Output:

          Output shape: :math:`(N, C_{out}, H_{out}, W_{out})`

        Where

        .. math::

           H^\prime_{out} &= (H_{in} - 1) * strides[0] - 2 * paddings[0] + dilations[0] * (H_f - 1) + 1 \\\\
           W^\prime_{out} &= (W_{in} - 1) * strides[1] - 2 * paddings[1] + dilations[1] * (W_f - 1) + 1 \\\\
           H_{out} &\in [ H^\prime_{out}, H^\prime_{out} + strides[0] ) \\\\
           W_{out} &\in [ W^\prime_{out}, W^\prime_{out} + strides[1] )

    Parameters:
        name_scope(str): The name of this class.
        num_filters(int): The number of the filter. It is as same as the output
            feature map.
        output_size(int or tuple, optional): The output image size. If output size is a
            tuple, it must contain two integers, (image_H, image_W). None if use
            filter_size, padding, and stride to calculate output_size.
            if output_size and filter_size are specified at the same time, They
            should follow the formula above. Default: None.
        filter_size(int or tuple, optional): The filter size. If filter_size is a tuple,
            it must contain two integers, (filter_size_H, filter_size_W).
            Otherwise, the filter will be a square. None if use output size to
            calculate filter_size. Default: None.
        padding(int or tuple, optional): The padding size. If padding is a tuple, it must
            contain two integers, (padding_H, padding_W). Otherwise, the
            padding_H = padding_W = padding. Default: 0.
        stride(int or tuple, optional): The stride size. If stride is a tuple, it must
            contain two integers, (stride_H, stride_W). Otherwise, the
            stride_H = stride_W = stride. Default: 1.
        dilation(int or tuple, optional): The dilation size. If dilation is a tuple, it must
            contain two integers, (dilation_H, dilation_W). Otherwise, the
            dilation_H = dilation_W = dilation. Default: 1.
        groups(int, optional): The groups number of the Conv2d transpose layer. Inspired by
            grouped convolution in Alex Krizhevsky's Deep CNN paper, in which
            when group=2, the first half of the filters is only connected to the
            first half of the input channels, while the second half of the
            filters is only connected to the second half of the input channels.
            Default: 1.
        param_attr (ParamAttr, optional): The parameter attribute for learnable weights(Parameter)
            of conv2d_transpose. If it is set to None or one attribute of ParamAttr, conv2d_transpose
            will create ParamAttr as param_attr. If the Initializer of the param_attr
            is not set, the parameter is initialized with Xavier. Default: None.
        bias_attr (ParamAttr or bool, optional): The attribute for the bias of conv2d_transpose.
            If it is set to False, no bias will be added to the output units.
            If it is set to None or one attribute of ParamAttr, conv2d_transpose
            will create ParamAttr as bias_attr. If the Initializer of the bias_attr
            is not set, the bias is initialized zero. Default: None.
        use_cudnn(bool, optional): Use cudnn kernel or not, it is valid only when the cudnn
            library is installed. Default: True.
        act (str, optional): Activation type, if it is set to None, activation is not appended.
            Default: None.

    Attribute:
        **weight** (Parameter): the learnable weights of filters of this layer.

        **bias** (Parameter or None): the learnable bias of this layer.

    Returns:
        None

    Examples:
       .. code-block:: python

          import paddle.fluid as fluid
          import numpy as np

          with fluid.dygraph.guard():
              data = np.random.random((3, 32, 32, 5)).astype('float32')
              conv2DTranspose = fluid.dygraph.nn.Conv2DTranspose(
                    'Conv2DTranspose', num_filters=2, filter_size=3)
              ret = conv2DTranspose(fluid.dygraph.base.to_variable(data))

    """

    def __init__(self,
                 name_scope,
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
                 act=None):
        super(Conv2DTranspose, self).__init__(name_scope)
        assert param_attr is not False, "param_attr should not be False in conv2d_transpose."
        self._param_attr = param_attr
        self._bias_attr = bias_attr
        self._act = act
        self._groups = groups
        self._num_filters = num_filters
        self._use_cudnn = use_cudnn
        self._padding = padding
        self._stride = stride
        self._dilation = dilation
        self._filter_size = filter_size
        self._output_size = output_size
        self._op_type = 'conv2d_transpose'

    def _build_once(self, input):
        input_channel = input.shape[1]
        if (input_channel == self._groups and
                self._num_filters == input_channel and not self._use_cudnn):
            self._op_type = 'depthwise_conv2d_transpose'

        if not isinstance(input, Variable):
            raise TypeError("Input of conv2d_transpose must be Variable")

        self._padding = utils.convert_to_list(self._padding, 2, 'padding')
        self._stride = utils.convert_to_list(self._stride, 2, 'stride')
        self._dilation = utils.convert_to_list(self._dilation, 2, 'dilation')

        if not isinstance(self._use_cudnn, bool):
            raise ValueError("use_cudnn should be True or False")

        if self._filter_size is None:
            if self._output_size is None:
                raise ValueError(
                    "output_size must be set when filter_size is None")
            if isinstance(self._output_size, int):
                self._output_size = [self._output_size, self._output_size]

            h_in = input.shape[2]
            w_in = input.shape[3]

            filter_size_h = (self._output_size[0] -
                             (h_in - 1) * self._stride[0] + 2 * self._padding[0]
                             - 1) // self._dilation[0] + 1
            filter_size_w = (self._output_size[1] -
                             (w_in - 1) * self._stride[1] + 2 * self._padding[1]
                             - 1) // self._dilation[1] + 1
            self._filter_size = [filter_size_h, filter_size_w]
        else:
            self._filter_size = utils.convert_to_list(
                self._filter_size, 2, 'conv2d_transpose.filter_size')

        if self._output_size is None:
            self._output_size = []
        elif isinstance(self._output_size, list) or isinstance(
                self._output_size, int):
            self._output_size = utils.convert_to_list(self._output_size, 2,
                                                      'output_size')
        else:
            raise ValueError("output_size should be list or int")
        self._padding = utils.convert_to_list(self._padding, 2, 'padding')
        self._groups = 1 if self._groups is None else self._groups
        filter_shape = [input_channel, self._num_filters // self._groups
                        ] + self._filter_size

        self._img_filter = self.create_parameter(
            dtype=input.dtype, shape=filter_shape, attr=self._param_attr)

        self._bias_param = self.create_parameter(
            attr=self._bias_attr,
            shape=[self._num_filters],
            dtype=self._dtype,
            is_bias=True)

    @property
    def weight(self):
        return self._img_filter

    @weight.setter
    def weight(self, value):
        self._img_filter = value

    @property
    def bias(self):
        return self._bias_param

    @bias.setter
    def bias(self, value):
        self._bias_param = value

    def forward(self, input):
        pre_bias = self._helper.create_variable_for_type_inference(
            dtype=input.dtype)
        self._helper.append_op(
            type=self._op_type,
            inputs={'Input': [input],
                    'Filter': [self._img_filter]},
            outputs={'Output': pre_bias},
            attrs={
                'output_size': self._output_size,
                'strides': self._stride,
                'paddings': self._padding,
                'dilations': self._dilation,
                'groups': self._groups,
                'use_cudnn': self._use_cudnn
            })

        if self._bias_param is not None:
            pre_act = self._helper.create_variable_for_type_inference(
                dtype=self._dtype)
            self._helper.append_op(
                type='elementwise_add',
                inputs={'X': [pre_bias],
                        'Y': [self._bias_param]},
                outputs={'Out': [pre_act]},
                attrs={'axis': 1})
        else:
            pre_act = pre_bias

        out = self._helper.append_activation(pre_act, act=self._act)
        return out


class SequenceConv(layers.Layer):
    """
    This function creates the op for sequence_conv, using the inputs and
    other convolutional configurations for the filters and stride as given
    in the input parameters to the function.

    Parameters:
        name_scope(str): The name of this class.
        num_filters (int): number of filters.
        filter_size (int): the filter size (H and W). Default: 3.
        filter_stride (int): stride of the filter. Default: 1.
        padding (bool|None): if True, add paddings. Default: None
        bias_attr (ParamAttr|bool|None): The parameter attribute for the bias of sequence_conv.
            If it is set to False, no bias will be added to the output units.
            If it is set to None or one attribute of ParamAttr, sequence_conv
            will create ParamAttr as bias_attr. If the Initializer of the bias_attr
            is not set, the bias is initialized zero. Default: None.
        param_attr (ParamAttr|None): The parameter attribute for learnable parameters/weights
            of sequence_conv. If it is set to None or one attribute of ParamAttr, sequence_conv
            will create ParamAttr as param_attr. If the Initializer of the param_attr
            is not set, the parameter is initialized with Xavier. Default: None.
        act (str): Activation type, if it is set to None, activation is not appended.
            Default: None.

    Attributes:
        weight (Parameter): the learnable weights of filters of this layer.
        bias (Parameter|None): the learnable bias of this layer.

    Returns:
        Variable: output of sequence_conv
    """

    def __init__(self,
                 name_scope,
                 num_filters,
                 filter_size=3,
                 filter_stride=1,
                 padding=None,
                 bias_attr=None,
                 param_attr=None,
                 act=None):
        assert not in_dygraph_mode(
        ), "SequenceConv is not supported by dynamic graph mode yet!"
        super(SequenceConv, self).__init__(name_scope)
        self._num_filters = num_filters
        self._filter_size = filter_size
        self._filter_stride = filter_stride
        self._padding = padding
        self._bias_attr = bias_attr
        self._param_attr = param_attr
        self._act = act

    def _build_once(self, input):
        self._dtype = self._helper.input_dtype(input)
        filter_shape = [self._filter_size * input.shape[1], self._num_filters]
        self._filter_param = self.create_parameter(
            attr=self._param_attr, shape=filter_shape, dtype=self._dtype)

        self._bias_param = self.create_parameter(
            attr=self._bias_attr,
            shape=[self._num_filters],
            dtype=self._dtype,
            is_bias=True)

    def forward(self, input):
        pre_bias = self._helper.create_variable_for_type_inference(self._dtype)
        self._helper.append_op(
            type='sequence_conv',
            inputs={
                'X': [input],
                'Filter': [self._filter_param],
            },
            outputs={"Out": pre_bias},
            attrs={
                'contextStride': self._filter_stride,
                'contextStart': -int(self._filter_size // 2),
                'contextLength': self._filter_size
            })

        if self._bias_param is not None:
            pre_act = self._helper.create_variable_for_type_inference(
                dtype=self._dtype)
            self._helper.append_op(
                type='elementwise_add',
                inputs={'X': [pre_bias],
                        'Y': [self._bias_param]},
                outputs={'Out': [pre_act]},
                attrs={'axis': 1})
        else:
            pre_act = pre_bias

        return self._helper.append_activation(pre_act, act=self._act)


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

    def __init__(self,
                 name_scope,
                 future_context_size,
                 param_attr=None,
                 act=None):
        assert not in_dygraph_mode(
        ), "RowConv is not supported by dynamic graph mode yet!"
        super(RowConv, self).__init__(name_scope)
        self._act = act
        self._param_attr = param_attr
        self._future_context_size = future_context_size

    def _build_once(self, input):
        self._dtype = self._helper.input_dtype(input)
        filter_shape = [self._future_context_size + 1, input.shape[1]]
        self._filter_param = self.create_parameter(
            attr=self._param_attr,
            shape=filter_shape,
            dtype=self._dtype,
            is_bias=False)

    def forward(self, input):
        out = self._helper.create_variable_for_type_inference(self._dtype)
        self._helper.append_op(
            type='row_conv',
            inputs={'X': [input],
                    'Filter': [self._filter_param]},
            outputs={'Out': [out]})
        return self._helper.append_activation(out, act=self._act)


class GroupNorm(layers.Layer):
    """
    This interface is used to construct a callable object of the ``GroupNorm`` class.
    For more details, refer to code examples.
    It implements the function of the Group Normalization Layer.
    Refer to `Group Normalization <https://arxiv.org/abs/1803.08494>`_ .

    Parameters:
        name_scope(str): The name of this class.
        groups(int): The number of groups that divided from channels.
        epsilon(float, optional): The small value added to the variance to prevent
                                  division by zero. Default: 1e-05.
        param_attr(ParamAttr, optional): The parameter attribute for the learnable
                                         scale :math:`g`. If it is set to False, no scale will be added to the output units.
                                         If it is set to None, the bias is initialized one. Default: None.
        bias_attr(ParamAttr, optional): The parameter attribute for the learnable
                                        bias :math:`b`. If it is set to False, no bias will be added to the output units.
                                        If it is set to None, the bias is initialized zero. Default: None.
        act(str, optional): Activation to be applied to the output of group normalizaiton. Default: None.
        data_layout(str, optional): Specify the input data format. Only NCHW is supported. Default: NCHW.

    Returns:
        None

    Examples:
        .. code-block:: python

          import paddle.fluid as fluid
          import numpy as np

          with fluid.dygraph.guard():
              x = np.random.random((8, 32, 32)).astype('float32')
              groupNorm = fluid.dygraph.nn.GroupNorm('GroupNorm', groups=4)
              ret = groupNorm(fluid.dygraph.base.to_variable(x))

    """

    def __init__(self,
                 name_scope,
                 groups,
                 epsilon=1e-05,
                 param_attr=None,
                 bias_attr=None,
                 act=None,
                 data_layout='NCHW'):
        super(GroupNorm, self).__init__(name_scope)
        self._param_attr = param_attr
        self._bias_attr = bias_attr
        self._epsilon = epsilon
        self._groups = groups
        self._act = act
        if data_layout != 'NCHW':
            raise ValueError("unsupported data layout:" + data_layout)

    def _build_once(self, input):
        self._dtype = self._helper.input_dtype(input)
        param_shape = [input.shape[1]]
        if self._bias_attr:
            self._bias = self.create_parameter(
                attr=self._bias_attr,
                shape=param_shape,
                dtype=self._dtype,
                is_bias=True)

        if self._param_attr:
            self._scale = self.create_parameter(
                attr=self._param_attr,
                shape=param_shape,
                dtype=self._dtype,
                default_initializer=Constant(1.0))

    def forward(self, input):
        inputs = {'X': input}
        if self._bias_attr:
            inputs['Bias'] = self._bias
        if self._param_attr:
            inputs['Scale'] = self._scale

        # create output
        mean_out = self._helper.create_variable_for_type_inference(
            dtype=self._dtype, stop_gradient=True)
        variance_out = self._helper.create_variable_for_type_inference(
            dtype=self._dtype, stop_gradient=True)
        group_norm_out = self._helper.create_variable_for_type_inference(
            dtype=self._dtype)

        self._helper.append_op(
            type="group_norm",
            inputs=inputs,
            outputs={
                "Y": group_norm_out,
                "Mean": mean_out,
                "Variance": variance_out,
            },
            attrs={"epsilon": self._epsilon,
                   "groups": self._groups})

        return self._helper.append_activation(group_norm_out, self._act)


class SpectralNorm(layers.Layer):
    """
    This interface is used to construct a callable object of the ``SpectralNorm`` class.
    For more details, refer to code examples. It implements the function of the Spectral Normalization Layer.
    This layer calculates the spectral normalization value of weight parameters of
    fc, conv1d, conv2d, conv3d layers which should be 2-D, 3-D, 4-D, 5-D
    Parameters. Calculations are showed as follows.

    Step 1:
    Generate vector U in shape of [H], and V in shape of [W].
    While H is the :attr:`dim` th dimension of the input weights,
    and W is the product result of remaining dimensions.

    Step 2:
    :attr:`power_iters` shoule be a positive interger, do following
    calculations with U and V for :attr:`power_iters` rounds.

    .. math::

        \mathbf{v} := \\frac{\mathbf{W}^{T} \mathbf{u}}{\|\mathbf{W}^{T} \mathbf{u}\|_2}

        \mathbf{u} := \\frac{\mathbf{W}^{T} \mathbf{v}}{\|\mathbf{W}^{T} \mathbf{v}\|_2}

    Step 3:
    Calculate :math:`\sigma(\mathbf{W})` and normalize weight values.

    .. math::

        \sigma(\mathbf{W}) = \mathbf{u}^{T} \mathbf{W} \mathbf{v}

        \mathbf{W} = \\frac{\mathbf{W}}{\sigma(\mathbf{W})}


    Refer to `Spectral Normalization <https://arxiv.org/abs/1802.05957>`_ .

    Parameters:
        name_scope(str): The name of this class.
        dim(int, optional): The index of dimension which should be permuted to the first before reshaping Input(Weight) to matrix, it should be set as 0 if Input(Weight) is the weight of fc layer, and should be set as 1 if Input(Weight) is the weight of conv layer. Default: 0.
        power_iters(int, optional): The number of power iterations to calculate spectral norm. Default: 1.
        eps(float, optional): The epsilon for numerical stability in calculating norms. Default: 1e-12.
        name (str, optional): The default value is None.  Normally there is no need for user to set this property.  For more information, please refer to :ref:`api_guide_Name` .

    Returns:
        None

    Examples:
       .. code-block:: python

            import paddle.fluid as fluid
            import numpy as np

            with fluid.dygraph.guard():
                x = np.random.random((2, 8, 32, 32)).astype('float32')
                spectralNorm = fluid.dygraph.nn.SpectralNorm('SpectralNorm', dim=1, power_iters=2)
                ret = spectralNorm(fluid.dygraph.base.to_variable(x))

    """

    def __init__(self, name_scope, dim=0, power_iters=1, eps=1e-12, name=None):
        super(SpectralNorm, self).__init__(name_scope)
        self._power_iters = power_iters
        self._eps = eps
        self._dim = dim

    def _build_once(self, weight):
        self._dtype = self._helper.input_dtype(weight)
        input_shape = weight.shape
        h = input_shape[self._dim]
        w = np.prod(input_shape) // h

        self.u = self.create_parameter(
            attr=ParamAttr(),
            shape=[h],
            dtype=self._dtype,
            default_initializer=Normal(0., 1.))
        self.u.stop_gradient = True

        self.v = self.create_parameter(
            attr=ParamAttr(),
            shape=[w],
            dtype=self._dtype,
            default_initializer=Normal(0., 1.))
        self.v.stop_gradient = True

    def forward(self, weight):
        inputs = {'Weight': weight, 'U': self.u, 'V': self.v}
        out = self._helper.create_variable_for_type_inference(self._dtype)
        self._helper.append_op(
            type="spectral_norm",
            inputs=inputs,
            outputs={"Out": out, },
            attrs={
                "dim": self._dim,
                "power_iters": self._power_iters,
                "eps": self._eps,
            })

        return out


class TreeConv(layers.Layer):
    """
    This interface is used to construct a callable object of the ``TreeConv`` class.
    For more details, refer to code examples.
    Tree-Based Convolution is a kind of convolution based on tree structure.
    Tree-Based Convolution is a part of Tree-Based Convolution Neural Network(TBCNN),
    which is used to classify tree structures, such as Abstract Syntax Tree.
    Tree-Based Convolution proposed a kind of data structure called continuous binary tree,
    which regards multiway tree as binary tree.
    The paper of Tree-Based Convolution Operator is here: `tree-based convolution <https://arxiv.org/abs/1409.5718v1/>`_ .
    
    Parameters:
        name_scope(str): The name of this class.
        output_size(int): output feature width.
        num_filters(int, optional): number of filters, Default: 1.
        max_depth(int, optional): max depth of filters, Default: 2.
        act(str, optional): activation function, Default: tanh.
        param_attr(ParamAttr, optional): the parameter attribute for the filters, Default: None.
        bias_attr(ParamAttr, optional): the parameter attribute for the bias of this layer, Default: None.
        name(str, optional): The default value is None. Normally there is no need for user to set this property. For more information, please refer to :ref:`api_guide_Name` .

    Attribute:
        **weight** (Parameter): the learnable weights of filters of this layer.

        **bias** (Parameter or None): the learnable bias of this layer.

    Returns:
        None

    Examples:

        .. code-block:: python

          import paddle.fluid as fluid
          import numpy

          with fluid.dygraph.guard():
              nodes_vector = numpy.random.random((1, 10, 5)).astype('float32')
              edge_set = numpy.random.random((1, 9, 2)).astype('int32')
              treeConv = fluid.dygraph.nn.TreeConv(
                'TreeConv', output_size=6, num_filters=1, max_depth=2)
              ret = treeConv(fluid.dygraph.base.to_variable(nodes_vector), fluid.dygraph.base.to_variable(edge_set))
    """

    def __init__(self,
                 name_scope,
                 output_size,
                 num_filters=1,
                 max_depth=2,
                 act='tanh',
                 param_attr=None,
                 bias_attr=None,
                 name=None):
        super(TreeConv, self).__init__(name_scope)
        self._name = name
        self._output_size = output_size
        self._act = act
        self._max_depth = max_depth
        self._num_filters = num_filters
        self._bias_attr = bias_attr
        self._param_attr = param_attr

    def _build_once(self, nodes_vector, edge_set):
        assert isinstance(nodes_vector, Variable)
        assert isinstance(edge_set, Variable)
        self._dtype = self._helper.input_dtype(nodes_vector)

        feature_size = nodes_vector.shape[2]
        w_shape = [feature_size, 3, self._output_size, self._num_filters]
        if self._bias_attr:
            self._bias_param = self.create_parameter(
                attr=self._bias_attr,
                shape=[self._num_filters],
                dtype=self._dtype,
                is_bias=True)
        self.W = self.create_parameter(
            attr=self._param_attr,
            shape=w_shape,
            dtype=self._dtype,
            is_bias=False)

    @property
    def weight(self):
        return self.W

    @weight.setter
    def weight(self, value):
        self.W = value

    @property
    def bias(self):
        return self._bias_param

    @bias.setter
    def bias(self, value):
        self._bias_param = value

    def forward(self, nodes_vector, edge_set):

        if self._name:
            out = self.create_variable(
                name=self._name, dtype=self._dtype, persistable=False)
        else:

            out = self._helper.create_variable_for_type_inference(
                dtype=self._dtype)

        self._helper.append_op(
            type='tree_conv',
            inputs={
                'NodesVector': nodes_vector,
                'EdgeSet': edge_set,
                'Filter': self.W
            },
            outputs={'Out': out, },
            attrs={'max_depth': self._max_depth})
        if self._bias_attr:
            pre_activation = self._helper.create_variable_for_type_inference(
                dtype=self._dtype)
            self._helper.append_op(
                type='elementwise_add',
                inputs={'X': [out],
                        'Y': [self._bias_param]},
                outputs={'Out': [pre_activation]},
                attrs={'axis': 1})
        else:
            pre_activation = out
        return self._helper.append_activation(pre_activation, act=self._act)
