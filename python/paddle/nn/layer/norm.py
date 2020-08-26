# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

# TODO: define normalization api  

from ...fluid.dygraph.nn import InstanceNorm

from ...fluid.dygraph import BatchNorm  #DEFINE_ALIAS
#from ...fluid.dygraph import GroupNorm  #DEFINE_ALIAS

#from ...fluid.dygraph import LayerNorm  #DEFINE_ALIAS
from ...fluid.dygraph import SpectralNorm  #DEFINE_ALIAS

from ...fluid.dygraph import layers

from ...framework import get_default_dtype, set_default_dtype
from ...fluid.framework import in_dygraph_mode

from ...fluid.initializer import Constant
from ...fluid.param_attr import ParamAttr
from ...fluid.data_feeder import check_variable_and_dtype, check_type
from ...fluid import core, dygraph_utils

from ..functional import batch_norm, layer_norm, instance_norm

import numpy as np
import numbers
import warnings

__all__ = [
    'BatchNorm', 'GroupNorm', 'LayerNorm', 'SpectralNorm', 'InstanceNorm',
    'BatchNorm1d', 'BatchNorm2d', 'BatchNorm3d', 'InstanceNorm1d',
    'InstanceNorm2d', 'InstanceNorm3d', 'SyncBatchNorm'
]


class _InstanceNormBase(layers.Layer):
    """
    This class is based class for InstanceNorm1d, 2d, 3d. 

    See InstaceNorm1d, InstanceNorm2d or InstanceNorm3d for more details.
    """

    def __init__(self,
                 num_features,
                 epsilon=1e-5,
                 momentum=0.9,
                 weight_attr=None,
                 bias_attr=None,
                 track_running_stats=False,
                 data_format="NCHW",
                 name=None):
        super(_InstanceNormBase, self).__init__()

        if weight_attr == False or bias_attr == False:
            assert weight_attr == param_attr, "weight_attr and bias_attr must be set to Fasle at the same time in InstanceNorm"
        self._epsilon = epsilon
        self._weight_attr = weight_attr
        self._bias_attr = bias_attr

        if weight_attr != False and bias_attr != False:
            self.scale = self.create_parameter(
                attr=self._weight_attr,
                shape=[num_features],
                default_initializer=Constant(1.0),
                is_bias=False)
            self.bias = self.create_parameter(
                attr=self._bias_attr,
                shape=[num_features],
                default_initializer=Constant(0.0),
                is_bias=True)
        else:
            self.scale = None
            self.bias = None

    def _check_input_dim(self, input):
        raise NotImplementedError("InstanceNorm Base error")

    def forward(self, input):
        self._check_input_dim(input)

        return instance_norm(
            input, weight=self.scale, bias=self.bias, eps=self._epsilon)


class InstanceNorm1d(_InstanceNormBase):
    """
    Applies Instance Normalization over a 3D input (a mini-batch of 1D inputs with additional channel dimension) as described in the paper Instance Normalization: The Missing Ingredient for Fast Stylization .

    DataLayout: NCL `[batch, in_channels, length]`

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

    Parameters:
        num_features(int): Indicate the number of channels of the input ``Tensor``.
        epsilon(float, optional): A value added to the denominator for
            numerical stability. Default is 1e-5.
        momentum(float, optional): The value used for the moving_mean and moving_var computation. Default: 0.9.
        track_running_stats(bool, optional): Whether to use global mean and
            variance. In train mode, when setting track_running_stats True, the global mean
            and variance are also used during train period. Default: False.
        weight_attr(ParamAttr|bool, optional): The parameter attribute for Parameter `scale`
             of instance_norm. If it is set to None or one attribute of ParamAttr, instance_norm
	     will create ParamAttr as weight_attr, the name of scale can be set in ParamAttr.
	     If the Initializer of the weight_attr is not set, the parameter is initialized 
	     one. If it is set to False, will not create weight_attr. Default: None.
        bias_attr(ParamAttr|bool, optional): The parameter attribute for the bias of instance_norm.
             If it is set to None or one attribute of ParamAttr, instance_norm
	     will create ParamAttr as bias_attr, the name of bias can be set in ParamAttr. 
	     If the Initializer of the bias_attr is not set, the bias is initialized zero. 
             If it is set to False, will not create bias_attr. Default: None.
        data_format(str, optional): Specify the input data format, may be "NC", "NCL". Defalut "NCL".
        name(str, optional): Name for the InstanceNorm, default is None. For more information, please refer to :ref:`api_guide_Name`..


    Shape:
        - x: 2-D or 3-D tensor with shape: (batch, num_features) or (batch, num_features, length).
        - output: 3-D tensor with same shape as input x.

    Returns:
        None.

    **Note**:
        Momentum and track_running_stats is not effective. The next version will fix the problem .


    Examples:

        .. code-block:: python

          import paddle
          import numpy as np

          paddle.disable_static()
          np.random.seed(123)
          x_data = np.random.random(size=(2, 2, 3)).astype('float32')
          x = paddle.to_tensor(x_data) 
          instance_norm = paddle.nn.InstanceNorm1d(2)
          instance_norm_out = instance_norm(x)

          print(instance_norm_out.numpy)

    """

    def _check_input_dim(self, input):
        if len(input.shape) != 2 and len(input.shape) != 3:
            raise ValueError('expected 2D or 3D input (got {}D input)'.format(
                len(input.shape)))


class InstanceNorm2d(_InstanceNormBase):
    """
    Applies Instance Normalization over a 4D input (a mini-batch of 2D inputs with additional channel dimension) as described in the paper Instance Normalization: The Missing Ingredient for Fast Stylization .

    DataLayout: NCHW `[batch, in_channels, in_height, in_width]`


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

    Parameters:
        num_features(int): Indicate the number of channels of the input ``Tensor``.
        epsilon(float, optional): A value added to the denominator for
            numerical stability. Default is 1e-5.
        momentum(float, optional): The value used for the moving_mean and moving_var computation. Default: 0.9.
        track_running_stats(bool, optional): Whether to use global mean and
            variance. In train mode, when setting track_running_stats True, the global mean
            and variance are also used during train period. Default: False.
        weight_attr(ParamAttr|bool, optional): The parameter attribute for Parameter `scale`
             of instance_norm. If it is set to None or one attribute of ParamAttr, instance_norm
	     will create ParamAttr as weight_attr, the name of scale can be set in ParamAttr.
	     If the Initializer of the weight_attr is not set, the parameter is initialized 
	     one. If it is set to False, will not create weight_attr. Default: None.
        bias_attr(ParamAttr|bool, optional): The parameter attribute for the bias of instance_norm.
             If it is set to None or one attribute of ParamAttr, instance_norm
	     will create ParamAttr as bias_attr, the name of bias can be set in ParamAttr. 
	     If the Initializer of the bias_attr is not set, the bias is initialized zero. 
             If it is set to False, will not create bias_attr. Default: None.
        data_format(str, optional): Specify the input data format, could be "NCHW". Default: NCHW.
        name(str, optional): Name for the InstanceNorm, default is None. For more information, please refer to :ref:`api_guide_Name`..

    Shape:
        - x: 4-D tensor with shape: (batch, num_features, height, weight).
        - output: 4-D tensor with same shape as input x.

    Returns:
        None.

    **Note**:
        Momentum and track_running_stats is not effective. The next version will fix the problem .

    Examples:

        .. code-block:: python

          import paddle
          import numpy as np

          paddle.disable_static()
          np.random.seed(123)
          x_data = np.random.random(size=(2, 2, 2, 3)).astype('float32')
          x = paddle.to_tensor(x_data) 
          instance_norm = paddle.nn.InstanceNorm2d(2)
          instance_norm_out = instance_norm(x)

          print(instance_norm_out.numpy)
    """

    def _check_input_dim(self, input):
        if len(input.shape) != 4:
            raise ValueError('expected 4D input (got {}D input)'.format(
                len(input.shape)))


class InstanceNorm3d(_InstanceNormBase):
    """
    Applies Instance Normalization over a 5D input (a mini-batch of 3D inputs with additional channel dimension) as described in the paper Instance Normalization: The Missing Ingredient for Fast Stylization .

    DataLayout: NCHW `[batch, in_channels, D, in_height, in_width]`


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

    Parameters:
        num_features(int): Indicate the number of channels of the input ``Tensor``.
        epsilon(float, optional): A value added to the denominator for
            numerical stability. Default is 1e-5.
        momentum(float, optional): The value used for the moving_mean and moving_var computation. Default: 0.9.
        track_running_stats(bool, optional): Whether to use global mean and
            variance. In train mode, when setting track_running_stats True, the global mean
            and variance are also used during train period. Default: False.
        weight_attr(ParamAttr|bool, optional): The parameter attribute for Parameter `scale`
             of instance_norm. If it is set to None or one attribute of ParamAttr, instance_norm
	     will create ParamAttr as weight_attr, the name of scale can be set in ParamAttr.
	     If the Initializer of the weight_attr is not set, the parameter is initialized 
	     one. If it is set to False, will not create weight_attr. Default: None.
        bias_attr(ParamAttr|bool, optional): The parameter attribute for the bias of instance_norm.
             If it is set to None or one attribute of ParamAttr, instance_norm
	     will create ParamAttr as bias_attr, the name of bias can be set in ParamAttr. 
	     If the Initializer of the bias_attr is not set, the bias is initialized zero. 
             If it is set to False, will not create bias_attr. Default: None.
        data_format(str, optional): Specify the input data format, could be "NCDHW". Default: NCDHW.
        name(str, optional): Name for the InstanceNorm, default is None. For more information, please refer to :ref:`api_guide_Name`..

    Shape:
        - x: 5-D tensor with shape: (batch, num_features, dims, height, weight).
        - output: 5-D tensor with same shape as input x.

    Returns:
        None.

    **Note**:
        Momentum and track_running_stats is not effective. The next version will fix the problem .

    Examples:

        .. code-block:: python

          import paddle
          import numpy as np

          paddle.disable_static()
          np.random.seed(123)
          x_data = np.random.random(size=(2, 2, 2, 2, 3)).astype('float32')
          x = paddle.to_tensor(x_data) 
          instance_norm = paddle.nn.InstanceNorm3d(2)
          instance_norm_out = instance_norm(x)

          print(instance_norm_out.numpy)
    """

    def _check_input_dim(self, input):
        if len(input.shape) != 5:
            raise ValueError('expected 5D input (got {}D input)'.format(
                len(input.shape)))


class GroupNorm(layers.Layer):
    """
    This interface is used to construct a callable object of the ``GroupNorm`` class.
    For more details, refer to code examples.
    It implements the function of the Group Normalization Layer.
    Refer to `Group Normalization <https://arxiv.org/abs/1803.08494>`_ .

    Parameters:
        num_channels(int): The number of channels of input.
        num_groups(int): The number of groups that divided from channels.
        epsilon(float, optional): The small value added to the variance to prevent
                                  division by zero. Default: 1e-05.
        weight_attr(ParamAttr|bool, optional): The parameter attribute for the learnable
                                         scale :math:`g`. If it is set to False, no scale will be added to the output units.
                                         If it is set to None, the bias is initialized one. Default: None.
        bias_attr(ParamAttr|bool, optional): The parameter attribute for the learnable
                                        bias :math:`b`. If it is set to False, no bias will be added to the output units.
                                        If it is set to None, the bias is initialized zero. Default: None.
        data_format(str, optional): Specify the input data format. Only NCHW is supported. Default: NCHW.
        name(str, optional): Name for the GroupNorm, default is None. For more information, please refer to :ref:`api_guide_Name`..

    Shape:
        - x: 4-D tensor with shape: (batch, num_features, height, weight).
        - output: 4-D tensor with same shape as input x.

    Returns:
        None

    Examples:
        .. code-block:: python
          import paddle
          import numpy as np

          paddle.disable_static()
          np.random.seed(123)
          x_data = np.random.random(size=(2, 6, 2, 2)).astype('float32')
          x = paddle.to_tensor(x_data) 
          group_norm = paddle.nn.GroupNorm(num_channels=3, num_groups=6)
          group_norm_out = group_norm(x)

          print(group_norm_out.numpy)
    """

    def __init__(self,
                 num_channels,
                 num_groups,
                 epsilon=1e-05,
                 weight_attr=None,
                 bias_attr=None,
                 data_layout='NCHW',
                 name=None):
        super(GroupNorm, self).__init__()
        self._weight_attr = weight_attr
        self._bias_attr = bias_attr
        self._epsilon = epsilon
        self._num_channels = num_channels
        self._num_groups = num_groups
        if data_layout != 'NCHW':
            raise ValueError("unsupported data layout:" + data_layout)

        param_shape = [self._num_channels]

        self.weight = self.create_parameter(
            attr=self._weight_attr or False,
            shape=param_shape,
            default_initializer=Constant(1.0))

        self.bias = self.create_parameter(
            attr=self._weight_attr or False, shape=param_shape, is_bias=True)

    def forward(self, input):
        inputs = {'X': input}
        if self.bias is not None:
            inputs['Bias'] = self.bias
        if self.weight is not None:
            inputs['Scale'] = self.weight

        # create output
        mean_out = self._helper.create_variable_for_type_inference(
            dtype=input.dtype, stop_gradient=True)
        variance_out = self._helper.create_variable_for_type_inference(
            dtype=input.dtype, stop_gradient=True)
        group_norm_out = self._helper.create_variable_for_type_inference(
            dtype=input.dtype)

        self._helper.append_op(
            type="group_norm",
            inputs=inputs,
            outputs={
                "Y": group_norm_out,
                "Mean": mean_out,
                "Variance": variance_out,
            },
            attrs={"epsilon": self._epsilon,
                   "groups": self._num_groups})

        return self._helper.append_activation(group_norm_out, None)


class LayerNorm(layers.Layer):
    """
    :alias_main: paddle.nn.LayerNorm
	:alias: paddle.nn.LayerNorm,paddle.nn.layer.LayerNorm,paddle.nn.layer.norm.LayerNorm
	:old_api: paddle.fluid.dygraph.LayerNorm

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
        normalized_shape(int|list|tuple): Input shape from an expected input of
            size :math:`[*, normalized_shape[0], normalized_shape[1], ..., normalized_shape[-1]]`.
            If it is a single integer, this module will normalize over the last dimension
            which is expected to be of that specific size.
        epsilon(float, optional): The small value added to the variance to prevent
            division by zero. Default: 1e-05.
        weight_attr(ParamAttr|bool, optional): The parameter attribute for the learnable
            gain :math:`g`. If False, weight is None. If is None, a default :code:`ParamAttr` would be added as scale. The
            :attr:`param_attr` is initialized as 1 if it is added. Default: None.
        bias_attr(ParamAttr|bool, optional): The parameter attribute for the learnable
            bias :math:`b`. If is False, bias is None. If is None, a default :code:`ParamAttr` would be added as bias. The
            :attr:`bias_attr` is initialized as 0 if it is added. Default: None.
        name(str, optional): Name for the LayerNorm, default is None. For more information, please refer to :ref:`api_guide_Name`..

    Shape:
        - x: 2-D, 3-D, 4-D or 5-D tensor.
        - output: same shape as input x.

    Returns:
        None

    Examples:

        .. code-block:: python

          import paddle
          import numpy as np

          paddle.disable_static()
          np.random.seed(123)
          x_data = np.random.random(size=(2, 2, 2, 3)).astype('float32')
          x = paddle.to_tensor(x_data) 
          layer_norm = paddle.nn.LayerNorm(x_data.shape[1:])
          layer_norm_out = layer_norm(x)

          print(layer_norm_out.numpy)
    """

    def __init__(self,
                 normalized_shape,
                 epsilon=1e-05,
                 weight_attr=None,
                 bias_attr=None,
                 name=None):
        super(LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = [normalized_shape]

        self._normalized_shape = list(normalized_shape)
        self._epsilon = epsilon
        self._weight_attr = weight_attr
        self._bias_attr = bias_attr
        param_shape = [np.prod(self._normalized_shape)]

        if weight_attr is False:
            self.weight = None
        else:
            self.weight = self.create_parameter(
                attr=self._weight_attr,
                shape=param_shape,
                default_initializer=Constant(1.0))

        if bias_attr is False:
            self.bias = None
        else:
            self.bias = self.create_parameter(
                attr=self._bias_attr, shape=param_shape, is_bias=True)

    def forward(self, input):
        return layer_norm(
            input,
            normalized_shape=self._normalized_shape,
            weight=self.weight,
            bias=self.bias,
            epsilon=self._epsilon)


class _BatchNormBase(layers.Layer):
    """
    BatchNorm base .
    """

    def __init__(self,
                 num_features,
                 momentum=0.9,
                 epsilon=1e-05,
                 weight_attr=None,
                 bias_attr=None,
                 data_format='NCHW',
                 track_running_stats=True,
                 name=None):
        super(_BatchNormBase, self).__init__()
        self._num_features = num_features
        self._weight_attr = weight_attr
        self._bias_attr = bias_attr

        if get_default_dtype() == 'float16':
            set_default_dtype('float32')

        param_shape = [num_features]

        # create parameter
        self.weight = self.create_parameter(
            attr=self._weight_attr,
            shape=param_shape,
            default_initializer=Constant(1.0))
        self.weight.stop_gradient = (self._weight_attr is False) or (
            self._weight_attr and self._weight_attr.learning_rate == 0.)

        self.bias = self.create_parameter(
            attr=self._bias_attr, shape=param_shape, is_bias=True)
        self.bias.stop_gradient = (self._bias_attr is False) or (
            self._bias_attr and self._bias_attr.learning_rate == 0.)

        moving_mean_name = None
        moving_variance_name = None

        if name is not None:
            moving_mean_name = name + "_mean"
            moving_variance_name = name + "_variance"

        self._mean = self.create_parameter(
            attr=ParamAttr(
                name=moving_mean_name,
                initializer=Constant(0.0),
                trainable=False,
                do_model_average=True),
            shape=param_shape,
            dtype=self._dtype)
        self._mean.stop_gradient = True

        self._variance = self.create_parameter(
            attr=ParamAttr(
                name=moving_variance_name,
                initializer=Constant(1.0),
                trainable=False,
                do_model_average=True),
            shape=param_shape,
            dtype=self._dtype)
        self._variance.stop_gradient = True

        self._data_format = data_format
        self._in_place = False
        self._momentum = momentum
        self._epsilon = epsilon
        self._fuse_with_relu = False
        self._track_running_stats = track_running_stats

    def _check_input_dim(self, input):
        raise NotImplementedError("BatchNorm Base error")

    def forward(self, input):

        self._check_input_dim(input)

        if not self.training and not self._track_running_stats:
            raise ValueError(
                'When inference, expected track_running_stats is True.')

        if self.training and not self._track_running_stats:
            warnings.warn(
                "When training, we now always track global mean and variance.")

        return batch_norm(
            input,
            self._mean,
            self._variance,
            weight=self.weight,
            bias=self.bias,
            training=self.training,
            momentum=self._momentum,
            epsilon=self._epsilon,
            data_format=self._data_format)


class BatchNorm1d(_BatchNormBase):
    """
    Applies Batch Normalization over a 2D or 3D input (a mini-batch of 1D inputswith additional channel dimension) as described in the paper Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift .

    When track_running_stats = False, the :math:`\\mu_{\\beta}`
    and :math:`\\sigma_{\\beta}^{2}` are the statistics of one mini-batch.
    Calculated as follows:

    ..  math::

        \\mu_{\\beta} &\\gets \\frac{1}{m} \\sum_{i=1}^{m} x_i \\qquad &//\\
        \ mini-batch\ mean \\\\
        \\sigma_{\\beta}^{2} &\\gets \\frac{1}{m} \\sum_{i=1}^{m}(x_i - \\
        \\mu_{\\beta})^2 \\qquad &//\ mini-batch\ variance \\\\

    When track_running_stats = True, the :math:`\\mu_{\\beta}`
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
        num_features(int): Indicate the number of channels of the input ``Tensor``.
        epsilon(float, optional): The small value added to the variance to prevent division by zero. Default: 1e-5.
        momentum(float, optional): The value used for the moving_mean and moving_var computation. Default: 0.9.
        weight_attr(ParamAttr|bool, optional): The parameter attribute for Parameter `scale`
            of batch_norm. If it is set to None or one attribute of ParamAttr, batch_norm
            will create ParamAttr as weight_attr. If it is set to Fasle, the weight is not learnable.
            If the Initializer of the weight_attr is not set, the parameter is initialized with Xavier. Default: None.
        bias_attr(ParamAttr|bool, optional): The parameter attribute for the bias of batch_norm.
            If it is set to None or one attribute of ParamAttr, batch_norm
            will create ParamAttr as bias_attr. If it is set to Fasle, the weight is not learnable.
            If the Initializer of the bias_attr is not set, the bias is initialized zero. Default: None.
        data_format(str, optional): Specify the input data format, may be "NC", "NCL". Defalut "NCL".
        track_running_stats(bool, optional): Whether to use global mean and variance. In train period, 
            True will track global mean and variance used for inference. When inference, track_running_stats must be 
            True. Default: True.
        name(str, optional): Name for the BatchNorm, default is None. For more information, please refer to :ref:`api_guide_Name`..

    Shape:
        - x: 2-D or 3-D tensor with shape: (batch, num_features) or (batch, num_features, length).
        - output: 3-D tensor with same shape as input x.

    Returns:
        None.

    **Note**:
        Now track_running_stats is actucal always true. The next version will fix the problem .
    

    Examples:
        .. code-block:: python

          import paddle
          import numpy as np

          paddle.disable_static()
          np.random.seed(123)
          x_data = np.random.random(size=(2, 1, 3)).astype('float32')
          x = paddle.to_tensor(x_data) 
          batch_norm = paddle.nn.BatchNorm1d(1)
          batch_norm_out = batch_norm(x)

          print(batch_norm_out.numpy)
    """

    def _check_input_dim(self, input):
        if len(input.shape) != 2 and len(input.shape) != 3:
            raise ValueError('expected 2D or 3D input (got {}D input)'.format(
                len(input.shape)))


class BatchNorm2d(_BatchNormBase):
    """
    Applies Batch Normalization over a 4D input (a mini-batch of 2D inputswith additional channel dimension) as described in the paper Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift .

    When track_running_stats = False, the :math:`\\mu_{\\beta}`
    and :math:`\\sigma_{\\beta}^{2}` are the statistics of one mini-batch.
    Calculated as follows:

    ..  math::

        \\mu_{\\beta} &\\gets \\frac{1}{m} \\sum_{i=1}^{m} x_i \\qquad &//\\
        \ mini-batch\ mean \\\\
        \\sigma_{\\beta}^{2} &\\gets \\frac{1}{m} \\sum_{i=1}^{m}(x_i - \\
        \\mu_{\\beta})^2 \\qquad &//\ mini-batch\ variance \\\\

    When track_running_stats = True, the :math:`\\mu_{\\beta}`
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
        num_features(int): Indicate the number of channels of the input ``Tensor``.
        epsilon(float, optional): The small value added to the variance to prevent division by zero. Default: 1e-5.
        momentum(float, optional): The value used for the moving_mean and moving_var computation. Default: 0.9.
        weight_attr(ParamAttr|bool, optional): The parameter attribute for Parameter `scale`
            of batch_norm. If it is set to None or one attribute of ParamAttr, batch_norm
            will create ParamAttr as weight_attr. If it is set to Fasle, the weight is not learnable.
            If the Initializer of the weight_attr is not set, the parameter is initialized with Xavier. Default: None.
        bias_attr(ParamAttr|bool, optional): The parameter attribute for the bias of batch_norm.
            If it is set to None or one attribute of ParamAttr, batch_norm
            will create ParamAttr as bias_attr. If it is set to Fasle, the weight is not learnable.
            If the Initializer of the bias_attr is not set, the bias is initialized zero. Default: None.
        data_format(str, optional): Specify the input data format, the data format can be "NCHW" or "NHWC". Default: NCHW.
        track_running_stats(bool, optional): Whether to use global mean and variance. In train period, 
            True will track global mean and variance used for inference. When inference, track_running_stats must be 
            True. Default: True.
        name(str, optional): Name for the BatchNorm, default is None. For more information, please refer to :ref:`api_guide_Name`..

    Shape:
        - x: 4-D tensor with shape: (batch, num_features, height, weight).
        - output: 4-D tensor with same shape as input x.

    Returns:
        None

    **Note**:
        Now track_running_stats is actucal always true. The next version will fix the problem .

    Examples:
        .. code-block:: python

          import paddle
          import numpy as np

          paddle.disable_static()
          np.random.seed(123)
          x_data = np.random.random(size=(2, 1, 2, 3)).astype('float32')
          x = paddle.to_tensor(x_data) 
          batch_norm = paddle.nn.BatchNorm2d(1)
          batch_norm_out = batch_norm(x)

          print(batch_norm_out.numpy)
    """

    def _check_input_dim(self, input):
        if len(input.shape) != 4:
            raise ValueError('expected 4D input (got {}D input)'.format(
                len(input.shape)))


class BatchNorm3d(_BatchNormBase):
    """
    Applies Batch Normalization over a 5D input (a mini-batch of 3D inputswith additional channel dimension) as described in the paper Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift .

    When track_running_stats = False, the :math:`\\mu_{\\beta}`
    and :math:`\\sigma_{\\beta}^{2}` are the statistics of one mini-batch.
    Calculated as follows:

    ..  math::

        \\mu_{\\beta} &\\gets \\frac{1}{m} \\sum_{i=1}^{m} x_i \\qquad &//\\
        \ mini-batch\ mean \\\\
        \\sigma_{\\beta}^{2} &\\gets \\frac{1}{m} \\sum_{i=1}^{m}(x_i - \\
        \\mu_{\\beta})^2 \\qquad &//\ mini-batch\ variance \\\\

    When track_running_stats = True, the :math:`\\mu_{\\beta}`
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
        num_features(int): Indicate the number of channels of the input ``Tensor``.
        epsilon(float, optional): The small value added to the variance to prevent division by zero. Default: 1e-5.
        momentum(float, optional): The value used for the moving_mean and moving_var computation. Default: 0.9.
        weight_attr(ParamAttr|bool, optional): The parameter attribute for Parameter `scale`
            of batch_norm. If it is set to None or one attribute of ParamAttr, batch_norm
            will create ParamAttr as weight_attr. If it is set to Fasle, the weight is not learnable.
            If the Initializer of the weight_attr is not set, the parameter is initialized with Xavier. Default: None.
        bias_attr(ParamAttr|bool, optional): The parameter attribute for the bias of batch_norm.
            If it is set to None or one attribute of ParamAttr, batch_norm
            will create ParamAttr as bias_attr. If it is set to Fasle, the weight is not learnable.
            If the Initializer of the bias_attr is not set, the bias is initialized zero. Default: None.
        data_format(str, optional): Specify the input data format, the data format can be "NCDHW". Default: NCDHW.
        track_running_stats(bool, optional): Whether to use global mean and variance. In train period, 
            True will track global mean and variance used for inference. When inference, track_running_stats must be 
            True. Default: True.
        name(str, optional): Name for the BatchNorm, default is None. For more information, please refer to :ref:`api_guide_Name`..

    Shape:
        - x: 5-D tensor with shape: (batch, num_features, dims, height, weight).
        - output: 5-D tensor with same shape as input x.

    Returns:
        None

    **Note**:
        Now track_running_stats is actucal always true. The next version will fix the problem .

    Examples:
        .. code-block:: python

          import paddle
          import numpy as np

          paddle.disable_static()
          np.random.seed(123)
          x_data = np.random.random(size=(2, 1, 2, 2, 3)).astype('float32')
          x = paddle.to_tensor(x_data) 
          batch_norm = paddle.nn.BatchNorm3d(1)
          batch_norm_out = batch_norm(x)

          print(batch_norm_out.numpy)
    """

    def _check_input_dim(self, input):
        if len(input.shape) != 5:
            raise ValueError('expected 5D input (got {}D input)'.format(
                len(input.shape)))


class SyncBatchNorm(layers.Layer):
    """
    This interface is used to construct a callable object of the ``SyncBatchNorm`` class.
    It implements the function of the Cross-GPU Synchronized Batch Normalization Layer, and can 
    be used as a normalizer function for other operations, such as conv2d and fully connected 
    operations.
    The data is normalized by the mean and variance of the channel based on whole mini-batch
    , which including data in all gpus.
    Refer to `Batch Normalization: Accelerating Deep Network Training by Reducing
    Internal Covariate Shift <https://arxiv.org/pdf/1502.03167.pdf>`_
    for more details.

    When model in training mode, the :math:`\\mu_{\\beta}` 
    and :math:`\\sigma_{\\beta}^{2}` are the statistics of whole mini-batch data in all gpus.
    Calculated as follows:

    ..  math::

        \\mu_{\\beta} &\\gets \\frac{1}{m} \\sum_{i=1}^{m} x_i \\qquad &//\\
        \ mini-batch\ mean \\\\
        \\sigma_{\\beta}^{2} &\\gets \\frac{1}{m} \\sum_{i=1}^{m}(x_i - \\
        \\mu_{\\beta})^2 \\qquad &//\ mini-batch\ variance \\\\

    - :math:`x` : whole mini-batch data in all gpus
    - :math:`m` : the size of the whole mini-batch data

    When model in evaluation mode, the :math:`\\mu_{\\beta}`
    and :math:`\\sigma_{\\beta}^{2}` are global statistics (moving_mean and moving_variance, 
    which usually got from the pre-trained model). Global statistics calculated as follows:

    .. math::
        moving\_mean = moving\_mean * momentum + \mu_{\beta} * (1. - momentum) \quad &// global mean \\
        moving\_variance = moving\_variance * momentum + \sigma_{\beta}^{2} * (1. - momentum) \quad &// global variance \\

    The formula of normalization is as follows:
 
    ..  math::

        \\hat{x_i} &\\gets \\frac{x_i - \\mu_\\beta} {\\sqrt{\\
        \\sigma_{\\beta}^{2} + \\eps}} \\qquad &//\ normalize \\\\
        y_i &\\gets \\gamma \\hat{x_i} + \\beta \\qquad &//\ scale\ and\ shift

    - :math:`\\eps` : add a smaller value to the variance to prevent division by zero
    - :math:`\\gamma` : trainable scale parameter vector
    - :math:`\\beta` : trainable shift parameter vector 

    Parameters:
        num_features(int): Indicate the number of channels of the input ``Tensor``.
        epsilon(float, optional): The small value added to the variance to prevent division by zero. Default: 1e-5.
        momentum(float, optional): The value used for the moving_mean and moving_var computation. Default: 0.9.
        weight_attr(ParamAttr|bool, optional): The parameter attribute for Parameter `scale`
             of this layer. If it is set to None or one attribute of ParamAttr, this layerr
             will create ParamAttr as param_attr. If the Initializer of the param_attr
             is not set, the parameter is initialized with Xavier. If it is set to False, 
             this layer will not have trainable scale parameter. Default: None.
        bias_attr(ParamAttr|bool, optional): The parameter attribute for the bias of this layer.
             If it is set to None or one attribute of ParamAttr, this layer
             will create ParamAttr as bias_attr. If the Initializer of the bias_attr
             is not set, the bias is initialized zero. If it is set to False, this layer will not 
             have trainable bias parameter. Default: None.
        track_running_stats(bool, optional): Whether to compute global stats, which including running mean and 
             running variance. Default: True.

    Shapes:
        input: Tensor that the dimension from 2 to 5.
        output: Tensor with the same shape as input.

    Examples:
        .. code-block:: python

          import paddle
          import paddle.nn as nn
          import numpy as np

          x = np.array([[[[0.3, 0.4], [0.3, 0.07]], [[0.83, 0.37], [0.18, 0.93]]]]).astype('float32')
          paddle.disable_static()
          x = paddle.to_tensor(x)
          if paddle.fluid.is_compiled_with_cuda():
              sync_batch_norm = nn.SyncBatchNorm(2)
              hidden1 = sync_batch_norm(x)
              print(hidden1.numpy())
              # [[[[0.26824948, 1.0936325],[0.26824948, -1.6301316]],[[ 0.8095662, -0.665287],[-1.2744656, 1.1301866 ]]]]
    """

    def __init__(self,
                 num_features,
                 epsilon=1e-05,
                 momentum=0.9,
                 track_running_stats=True,
                 weight_attr=None,
                 bias_attr=None,
                 data_format='NCHW',
                 name=None):
        super(SyncBatchNorm, self).__init__()
        self._weight_attr = weight_attr
        self._bias_attr = bias_attr
        self._num_features = num_features
        self._data_layout = data_format
        self._momentum = momentum
        self._epsilon = epsilon
        self._track_running_stats = track_running_stats

        if self._track_running_stats == False:
            warnings.warn(
                "moving mean and moving variance will be calculated whether `track_running_stats` is set to `True` or `False`, we will fix it in the next version."
            )

        param_shape = [self._num_features]

        # create parameter
        if weight_attr == False:
            self.weight = self.create_parameter(
                attr=None, shape=param_shape, default_initializer=Constant(1.0))
            self.weight.stop_gradient = True
        else:
            self.weight = self.create_parameter(
                attr=self._weight_attr,
                shape=param_shape,
                default_initializer=Constant(1.0))
            self.weight.stop_gradient = self._weight_attr != None and self._weight_attr.learning_rate == 0.

        if bias_attr == False:
            self.bias = self.create_parameter(
                attr=None,
                shape=param_shape,
                default_initializer=Constant(0.0),
                is_bias=True)
            self.bias.stop_gradient = True
        else:
            self.bias = self.create_parameter(
                attr=self._bias_attr, shape=param_shape, is_bias=True)
            self.bias.stop_gradient = self._weight_attr != None and self._weight_attr.learning_rate == 0.

        self._mean = self.create_parameter(
            attr=ParamAttr(
                name=None,
                initializer=Constant(0.0),
                trainable=False,
                do_model_average=True),
            shape=param_shape,
            dtype=self._dtype)
        self._mean.stop_gradient = True

        self._variance = self.create_parameter(
            attr=ParamAttr(
                name=None,
                initializer=Constant(1.0),
                trainable=False,
                do_model_average=True),
            shape=param_shape,
            dtype=self._dtype)
        self._variance.stop_gradient = True

    def forward(self, x):
        # create output
        # mean and mean_out share the same memory
        mean_out = self._mean
        # variance and variance out share the same memory
        variance_out = self._variance

        ### train mode: use mini-batch stats, eval mode: use global stats
        ### use_global_stats only support False in sync_batch_norm
        if in_dygraph_mode():
            attrs = ("momentum", self._momentum, "epsilon", self._epsilon,
                     "is_test", not self.training, "data_layout",
                     self._data_layout, "use_mkldnn", False, "fuse_with_relu",
                     False, "use_global_stats", False, 'trainable_statistics',
                     False)
            sync_batch_norm_out, _, _, _, _, _ = core.ops.sync_batch_norm(
                x, self.weight, self.bias, self._mean, self._variance, mean_out,
                variance_out, *attrs)

            return sync_batch_norm_out

        check_variable_and_dtype(x, 'input', ['float16', 'float32', 'float64'],
                                 'BatchNorm')

        attrs = {
            "momentum": self._momentum,
            "epsilon": self._epsilon,
            "is_test": not self.training,
            "data_layout": self._data_layout,
            "use_mkldnn": False,
            "fuse_with_relu": False,
            "use_global_stats": False,
            "trainable_statistics": False,
        }

        inputs = {
            "X": [x],
            "Scale": [self.weight],
            "Bias": [self.bias],
            "Mean": [self._mean],
            "Variance": [self._variance]
        }

        saved_mean = self._helper.create_variable_for_type_inference(
            dtype=self._dtype, stop_gradient=True)
        saved_variance = self._helper.create_variable_for_type_inference(
            dtype=self._dtype, stop_gradient=True)
        sync_batch_norm_out = self._helper.create_variable_for_type_inference(
            self._dtype)

        outputs = {
            "Y": [sync_batch_norm_out],
            "MeanOut": [mean_out],
            "VarianceOut": [variance_out],
            "SavedMean": [saved_mean],
            "SavedVariance": [saved_variance]
        }

        self._helper.append_op(
            type="sync_batch_norm", inputs=inputs, outputs=outputs, attrs=attrs)
        return sync_batch_norm_out
