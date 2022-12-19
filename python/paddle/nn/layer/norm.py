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

import numbers
import warnings

import numpy as np

from paddle import _C_ops, _legacy_C_ops, in_dynamic_mode
from paddle.device import get_all_custom_device_type
from paddle.fluid.framework import _in_legacy_dygraph, in_dygraph_mode

from ...fluid import dygraph_utils
from ...fluid.data_feeder import check_variable_and_dtype
from ...framework import (
    ParamAttr,
    _global_flags,
    _non_static_mode,
    get_default_dtype,
    no_grad,
)
from .. import Layer
from .. import functional as F
from ..functional import batch_norm, instance_norm, layer_norm
from ..initializer import Constant, Normal

__all__ = []


class _InstanceNormBase(Layer):
    """
    This class is based class for InstanceNorm1D, 2d, 3d.

    See InstaceNorm1D, InstanceNorm2D or InstanceNorm3D for more details.
    """

    def __init__(
        self,
        num_features,
        epsilon=1e-5,
        momentum=0.9,
        weight_attr=None,
        bias_attr=None,
        data_format="NCHW",
        name=None,
    ):
        super().__init__()

        if weight_attr is False or bias_attr is False:
            assert (
                weight_attr == bias_attr
            ), "weight_attr and bias_attr must be set to False at the same time in InstanceNorm"
        self._epsilon = epsilon
        self._weight_attr = weight_attr
        self._bias_attr = bias_attr
        self._num_features = num_features

        if weight_attr is not False and bias_attr is not False:
            self.scale = self.create_parameter(
                attr=self._weight_attr,
                shape=[num_features],
                default_initializer=Constant(1.0),
                is_bias=False,
            )
            self.bias = self.create_parameter(
                attr=self._bias_attr,
                shape=[num_features],
                default_initializer=Constant(0.0),
                is_bias=True,
            )
        else:
            self.scale = None
            self.bias = None

    def _check_input_dim(self, input):
        raise NotImplementedError("InstanceNorm Base error")

    def forward(self, input):
        self._check_input_dim(input)

        return instance_norm(
            input, weight=self.scale, bias=self.bias, eps=self._epsilon
        )

    def extra_repr(self):
        return 'num_features={}, epsilon={}'.format(
            self._num_features, self._epsilon
        )


class InstanceNorm1D(_InstanceNormBase):
    r"""
    Create a callable object of `InstanceNorm1D`. Applies Instance Normalization over a 3D input (a mini-batch of 1D inputs with additional channel dimension) as described in the paper Instance Normalization: The Missing Ingredient for Fast Stylization .

    DataLayout: NCL `[batch, in_channels, length]`

    :math:`input` is the input features over a mini-batch.

    ..  math::

        \mu_{\beta} &\gets \frac{1}{HW} \sum_{i=1}^{HW} x_i \qquad &//\
        \ mean\ of\ one\  feature\ map\ in\ mini-batch \\
        \sigma_{\beta}^{2} &\gets \frac{1}{HW} \sum_{i=1}^{HW}(x_i - \
        \mu_{\beta})^2 \qquad &//\ variance\ of\ one\ feature\ map\ in\ mini-batch \\
        \hat{x_i} &\gets \frac{x_i - \mu_\beta} {\sqrt{\
        \sigma_{\beta}^{2} + \epsilon}} \qquad &//\ normalize \\
        y_i &\gets \gamma \hat{x_i} + \beta \qquad &//\ scale\ and\ shift

    Where `H` means height of feature map, `W` means width of feature map.

    Parameters:
        num_features(int): Indicate the number of channels of the input ``Tensor``.
        epsilon(float, optional): A value added to the denominator for
            numerical stability. Default is 1e-5.
        momentum(float, optional): The value used for the moving_mean and moving_var computation. Default: 0.9.
        weight_attr(ParamAttr|bool, optional): The parameter attribute for Parameter `scale` of instance_norm.
            If it is set to None or one attribute of ParamAttr, instance_norm
            will create ParamAttr as weight_attr, the name of scale can be set in ParamAttr.
            If the Initializer of the weight_attr is not set, the parameter is initialized
            one. If it is set to False, will not create weight_attr. Default: None. For more information, please refer to :ref:`api_paddle_ParamAttr` .
        bias_attr(ParamAttr|bool, optional): The parameter attribute for the bias of instance_norm.
            If it is set to None or one attribute of ParamAttr, instance_norm
            will create ParamAttr as bias_attr, the name of bias can be set in ParamAttr.
            If the Initializer of the bias_attr is not set, the bias is initialized zero.
            If it is set to False, will not create bias_attr. Default: None. For more information, please refer to :ref:`api_paddle_ParamAttr` .
        data_format(str, optional): Specify the input data format, may be "NC", "NCL". Default "NCL".
        name(str, optional): Name for the InstanceNorm, default is None. For more information, please refer to :ref:`api_guide_Name` .


    Shape:
        - x: 2-D or 3-D tensor with shape: (batch, num_features) or (batch, num_features, length).
        - output: 3-D tensor with same shape as input x.

    Returns:
        None.


    Examples:

        .. code-block:: python

          import paddle

          x = paddle.rand((2, 2, 3))
          instance_norm = paddle.nn.InstanceNorm1D(2)
          instance_norm_out = instance_norm(x)

          print(instance_norm_out)

    """

    def __init__(
        self,
        num_features,
        epsilon=0.00001,
        momentum=0.9,
        weight_attr=None,
        bias_attr=None,
        data_format="NCL",
        name=None,
    ):
        super().__init__(
            num_features,
            epsilon,
            momentum,
            weight_attr,
            bias_attr,
            data_format,
            name,
        )

    def _check_input_dim(self, input):
        if len(input.shape) != 2 and len(input.shape) != 3:
            raise ValueError(
                'expected 2D or 3D input (got {}D input)'.format(
                    len(input.shape)
                )
            )


class InstanceNorm2D(_InstanceNormBase):
    r"""
    Create a callable object of `InstanceNorm2D`. Applies Instance Normalization over a 4D input (a mini-batch of 2D inputs with additional channel dimension) as described in the paper Instance Normalization: The Missing Ingredient for Fast Stylization .

    DataLayout: NCHW `[batch, in_channels, in_height, in_width]`


    :math:`input` is the input features over a mini-batch.

    ..  math::

        \mu_{\beta} &\gets \frac{1}{HW} \sum_{i=1}^{HW} x_i \qquad &//\
        \ mean\ of\ one\  feature\ map\ in\ mini-batch \\
        \sigma_{\beta}^{2} &\gets \frac{1}{HW} \sum_{i=1}^{HW}(x_i - \
        \mu_{\beta})^2 \qquad &//\ variance\ of\ one\ feature\ map\ in\ mini-batch \\
        \hat{x_i} &\gets \frac{x_i - \mu_\beta} {\sqrt{\
        \sigma_{\beta}^{2} + \epsilon}} \qquad &//\ normalize \\
        y_i &\gets \gamma \hat{x_i} + \beta \qquad &//\ scale\ and\ shift

    Where `H` means height of feature map, `W` means width of feature map.

    Parameters:
        num_features(int): Indicate the number of channels of the input ``Tensor``.
        epsilon(float, optional): A value added to the denominator for
            numerical stability. Default is 1e-5.
        momentum(float, optional): The value used for the moving_mean and moving_var computation. Default: 0.9.
        weight_attr(ParamAttr|bool, optional): The parameter attribute for Parameter `scale`
            of instance_norm. If it is set to None or one attribute of ParamAttr, instance_norm
            will create ParamAttr as weight_attr, the name of scale can be set in ParamAttr.
            If the Initializer of the weight_attr is not set, the parameter is initialized
            one. If it is set to False, will not create weight_attr. Default: None. For more information, please refer to :ref:`api_paddle_ParamAttr` .
        bias_attr(ParamAttr|bool, optional): The parameter attribute for the bias of instance_norm.
            If it is set to None or one attribute of ParamAttr, instance_norm
            will create ParamAttr as bias_attr, the name of bias can be set in ParamAttr.
            If the Initializer of the bias_attr is not set, the bias is initialized zero.
            If it is set to False, will not create bias_attr. Default: None. For more information, please refer to :ref:`api_paddle_ParamAttr` .
        data_format(str, optional): Specify the input data format, could be "NCHW". Default: NCHW.
        name(str, optional): Name for the InstanceNorm, default is None. For more information, please refer to :ref:`api_guide_Name` .

    Shape:
        - x: 4-D tensor with shape: (batch, num_features, height, weight).
        - output: 4-D tensor with same shape as input x.

    Returns:
        None.


    Examples:

        .. code-block:: python

            import paddle

            x = paddle.rand((2, 2, 2, 3))
            instance_norm = paddle.nn.InstanceNorm2D(2)
            instance_norm_out = instance_norm(x)

            print(instance_norm_out)
    """

    def __init__(
        self,
        num_features,
        epsilon=0.00001,
        momentum=0.9,
        weight_attr=None,
        bias_attr=None,
        data_format="NCHW",
        name=None,
    ):
        super().__init__(
            num_features,
            epsilon,
            momentum,
            weight_attr,
            bias_attr,
            data_format,
            name,
        )

    def _check_input_dim(self, input):
        if len(input.shape) != 4:
            raise ValueError(
                'expected 4D input (got {}D input)'.format(len(input.shape))
            )


class InstanceNorm3D(_InstanceNormBase):
    r"""
    Create a callable object of `InstanceNorm3D`. Applies Instance Normalization over a 5D input (a mini-batch of 3D inputs with additional channel dimension) as described in the paper Instance Normalization: The Missing Ingredient for Fast Stylization .

    DataLayout: NCDHW `[batch, in_channels, D, in_height, in_width]`


    :math:`input` is the input features over a mini-batch.

    ..  math::

        \mu_{\beta} &\gets \frac{1}{HW} \sum_{i=1}^{HW} x_i \qquad &//\
        \ mean\ of\ one\  feature\ map\ in\ mini-batch \\
        \sigma_{\beta}^{2} &\gets \frac{1}{HW} \sum_{i=1}^{HW}(x_i - \
        \mu_{\beta})^2 \qquad &//\ variance\ of\ one\ feature\ map\ in\ mini-batch \\
        \hat{x_i} &\gets \frac{x_i - \mu_\beta} {\sqrt{\
        \sigma_{\beta}^{2} + \epsilon}} \qquad &//\ normalize \\
        y_i &\gets \gamma \hat{x_i} + \beta \qquad &//\ scale\ and\ shift

    Where `H` means height of feature map, `W` means width of feature map.

    Parameters:
        num_features(int): Indicate the number of channels of the input ``Tensor``.
        epsilon(float, optional): A value added to the denominator for
            numerical stability. Default is 1e-5.
        momentum(float, optional): The value used for the moving_mean and moving_var computation. Default: 0.9.
        weight_attr(ParamAttr|bool, optional): The parameter attribute for Parameter `scale`
            of instance_norm. If it is set to None or one attribute of ParamAttr, instance_norm
            will create ParamAttr as weight_attr, the name of scale can be set in ParamAttr.
            If the Initializer of the weight_attr is not set, the parameter is initialized
            one. If it is set to False, will not create weight_attr. Default: None. For more information, please refer to :ref:`api_paddle_ParamAttr` .
        bias_attr(ParamAttr|bool, optional): The parameter attribute for the bias of instance_norm.
            If it is set to None or one attribute of ParamAttr, instance_norm
            will create ParamAttr as bias_attr, the name of bias can be set in ParamAttr.
            If the Initializer of the bias_attr is not set, the bias is initialized zero.
            If it is set to False, will not create bias_attr. Default: None. For more information, please refer to :ref:`api_paddle_ParamAttr` .
        data_format(str, optional): Specify the input data format, could be "NCDHW". Default: NCDHW.
        name(str, optional): Name for the InstanceNorm, default is None. For more information, please refer to :ref:`api_guide_Name` .

    Shape:
        - x: 5-D tensor with shape: (batch, num_features, dims, height, weight).
        - output: 5-D tensor with same shape as input x.

    Returns:
        None.


    Examples:

        .. code-block:: python

            import paddle

            x = paddle.rand((2, 2, 2, 2, 3))
            instance_norm = paddle.nn.InstanceNorm3D(2)
            instance_norm_out = instance_norm(x)

            print(instance_norm_out.numpy)
    """

    def __init__(
        self,
        num_features,
        epsilon=0.00001,
        momentum=0.9,
        weight_attr=None,
        bias_attr=None,
        data_format="NCDHW",
        name=None,
    ):
        super().__init__(
            num_features,
            epsilon,
            momentum,
            weight_attr,
            bias_attr,
            data_format,
            name,
        )

    def _check_input_dim(self, input):
        if len(input.shape) != 5:
            raise ValueError(
                'expected 5D input (got {}D input)'.format(len(input.shape))
            )


class GroupNorm(Layer):
    """

    This interface is used to construct a callable object of the ``GroupNorm`` class.
    For more details, refer to code examples.
    It implements the function of the Group Normalization Layer.
    Refer to `Group Normalization <https://arxiv.org/abs/1803.08494>`_ .

    Parameters:
        num_groups(int): The number of groups that divided from channels.
        num_channels(int): The number of channels of input.
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
        - x: Tensor with shape: attr:`(batch, num_features, *)`.
        - output: The same shape as input x.

    Returns:
        None

    Examples:
        .. code-block:: python

            import paddle

            x = paddle.arange(48, dtype="float32").reshape((2, 6, 2, 2))
            group_norm = paddle.nn.GroupNorm(num_channels=6, num_groups=6)
            group_norm_out = group_norm(x)

            print(group_norm_out)
    """

    def __init__(
        self,
        num_groups,
        num_channels,
        epsilon=1e-05,
        weight_attr=None,
        bias_attr=None,
        data_format='NCHW',
        name=None,
    ):
        super().__init__()
        self._weight_attr = weight_attr
        self._bias_attr = bias_attr
        self._epsilon = epsilon
        self._num_channels = num_channels
        self._num_groups = num_groups
        if data_format not in ['NCHW', 'NHWC']:
            raise ValueError("unsupported data layout:" + data_format)
        self._data_format = data_format

        param_shape = [self._num_channels]

        if weight_attr is False:
            self.weight = self.create_parameter(
                attr=None, shape=param_shape, default_initializer=Constant(1.0)
            )
            self.weight.stop_gradient = True
        else:
            self.weight = self.create_parameter(
                attr=self._weight_attr,
                shape=param_shape,
                default_initializer=Constant(1.0),
            )
            self.weight.stop_gradient = self._weight_attr is not None and (
                hasattr(self._weight_attr, "learning_rate")
                and self._weight_attr.learning_rate == 0.0
            )

        if bias_attr is False:
            self.bias = self.create_parameter(
                attr=None,
                shape=param_shape,
                default_initializer=Constant(0.0),
                is_bias=True,
            )
            self.bias.stop_gradient = True
        else:
            self.bias = self.create_parameter(
                attr=self._bias_attr, shape=param_shape, is_bias=True
            )
            self.bias.stop_gradient = self._bias_attr is not None and (
                hasattr(self._bias_attr, "learning_rate")
                and self._bias_attr.learning_rate == 0.0
            )

    def forward(self, input):
        if in_dygraph_mode():
            return _C_ops.group_norm(
                input,
                self.weight,
                self.bias,
                self._epsilon,
                self._num_groups,
                self._data_format,
            )

        mean_out = self._helper.create_variable_for_type_inference(
            dtype=input.dtype, stop_gradient=True
        )
        variance_out = self._helper.create_variable_for_type_inference(
            dtype=input.dtype, stop_gradient=True
        )

        if _in_legacy_dygraph():
            pre_act, _, _ = _legacy_C_ops.group_norm(
                input,
                self.weight,
                self.bias,
                mean_out,
                variance_out,
                'epsilon',
                self._epsilon,
                'groups',
                self._num_groups,
            )
            return pre_act

        inputs = {'X': input}
        if self.bias is not None:
            inputs['Bias'] = self.bias
        if self.weight is not None:
            inputs['Scale'] = self.weight

        # create output
        group_norm_out = self._helper.create_variable_for_type_inference(
            dtype=input.dtype
        )

        self._helper.append_op(
            type="group_norm",
            inputs=inputs,
            outputs={
                "Y": group_norm_out,
                "Mean": mean_out,
                "Variance": variance_out,
            },
            attrs={"epsilon": self._epsilon, "groups": self._num_groups},
        )

        return self._helper.append_activation(group_norm_out, None)

    def extra_repr(self):
        return 'num_groups={}, num_channels={}, epsilon={}'.format(
            self._num_groups, self._num_channels, self._epsilon
        )


class LayerNorm(Layer):
    r"""
    Construct a callable object of the ``LayerNorm`` class.
    For more details, refer to code examples.
    It implements the function of the Layer Normalization Layer and can be applied to mini-batch input data.
    Refer to `Layer Normalization <https://arxiv.org/pdf/1607.06450v1.pdf>`_

    The formula is as follows:

    ..  math::

        \mu & = \frac{1}{H}\sum_{i=1}^{H} x_i

        \sigma & = \sqrt{\frac{1}{H}\sum_{i=1}^{H}{(x_i - \mu)^2} + \epsilon}

        y & = f(\frac{g}{\sigma}(x - \mu) + b)

    - :math:`x`: the vector representation of the summed inputs to the neurons in that layer.
    - :math:`H`: the number of hidden units in a layers
    - :math:`\epsilon`: the small value added to the variance to prevent division by zero.
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
            :attr:`param_attr` is initialized as 1 if it is added. Default: None. For more information, please refer to :ref:`api_paddle_ParamAttr` .
        bias_attr(ParamAttr|bool, optional): The parameter attribute for the learnable
            bias :math:`b`. If is False, bias is None. If is None, a default :code:`ParamAttr` would be added as bias. The
            :attr:`bias_attr` is initialized as 0 if it is added. Default: None. For more information, please refer to :ref:`api_paddle_ParamAttr` .
        name(str, optional): Name for the LayerNorm, default is None. For more information, please refer to :ref:`api_guide_Name` .

    Shape:
        - x: 2-D, 3-D, 4-D or 5-D tensor.
        - output: same shape as input x.

    Returns:
        None

    Examples:

        .. code-block:: python

          import paddle

          x = paddle.rand((2, 2, 2, 3))
          layer_norm = paddle.nn.LayerNorm(x.shape[1:])
          layer_norm_out = layer_norm(x)

          print(layer_norm_out)
    """

    def __init__(
        self,
        normalized_shape,
        epsilon=1e-05,
        weight_attr=None,
        bias_attr=None,
        name=None,
    ):
        super().__init__()
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
                default_initializer=Constant(1.0),
            )

        if bias_attr is False:
            self.bias = None
        else:
            self.bias = self.create_parameter(
                attr=self._bias_attr, shape=param_shape, is_bias=True
            )

    def forward(self, input):
        return layer_norm(
            input,
            normalized_shape=self._normalized_shape,
            weight=self.weight,
            bias=self.bias,
            epsilon=self._epsilon,
        )

    def extra_repr(self):
        return 'normalized_shape={}, epsilon={}'.format(
            self._normalized_shape, self._epsilon
        )


class _BatchNormBase(Layer):
    """
    BatchNorm base .
    """

    def __init__(
        self,
        num_features,
        momentum=0.9,
        epsilon=1e-05,
        weight_attr=None,
        bias_attr=None,
        data_format='NCHW',
        use_global_stats=None,
        name=None,
    ):
        super().__init__()
        self._num_features = num_features
        self._weight_attr = weight_attr
        self._bias_attr = bias_attr
        self._use_global_stats = use_global_stats

        if get_default_dtype() == 'float16':
            self._dtype = 'float32'
        else:
            self._dtype = get_default_dtype()

        param_shape = [num_features]

        # create parameter
        if weight_attr is False:
            self.weight = self.create_parameter(
                attr=None,
                shape=param_shape,
                dtype=self._dtype,
                default_initializer=Constant(1.0),
            )
            self.weight.stop_gradient = True
        else:
            self.weight = self.create_parameter(
                attr=self._weight_attr,
                shape=param_shape,
                dtype=self._dtype,
                default_initializer=Constant(1.0),
            )
            self.weight.stop_gradient = (
                self._weight_attr is not None
                and self._weight_attr.learning_rate == 0.0
            )

        if bias_attr is False:
            self.bias = self.create_parameter(
                attr=None,
                shape=param_shape,
                dtype=self._dtype,
                default_initializer=Constant(0.0),
                is_bias=True,
            )
            self.bias.stop_gradient = True
        else:
            self.bias = self.create_parameter(
                attr=self._bias_attr,
                shape=param_shape,
                dtype=self._dtype,
                is_bias=True,
            )
            self.bias.stop_gradient = (
                self._bias_attr is not None
                and self._bias_attr.learning_rate == 0.0
            )

        moving_mean_name = None
        moving_variance_name = None

        if name is not None:
            moving_mean_name = name + "_mean"
            moving_variance_name = name + "_variance"

        self._mean = self.create_parameter(
            dtype=self._dtype,
            attr=ParamAttr(
                name=moving_mean_name,
                initializer=Constant(0.0),
                trainable=False,
                do_model_average=True,
            ),
            shape=param_shape,
        )
        self._mean.stop_gradient = True

        self._variance = self.create_parameter(
            dtype=self._dtype,
            attr=ParamAttr(
                name=moving_variance_name,
                initializer=Constant(1.0),
                trainable=False,
                do_model_average=True,
            ),
            shape=param_shape,
        )
        self._variance.stop_gradient = True

        # TODO(qili93): temporary for ascned npu performance to be removed along with npu_identity op
        if (
            _global_flags()['FLAGS_npu_storage_format']
            and 'npu' in get_all_custom_device_type()
        ):
            with no_grad():
                weight_trans = _C_ops.npu_identity(
                    self.weight, 3
                )  # ACL_FORMAT_NC1HWC0 = 3
                bias_trans = _C_ops.npu_identity(
                    self.bias, 3
                )  # ACL_FORMAT_NC1HWC0 = 3
                mean_trans = _C_ops.npu_identity(
                    self._mean, 3
                )  # ACL_FORMAT_NC1HWC0 = 3
                var_trans = _C_ops.npu_identity(
                    self._variance, 3
                )  # ACL_FORMAT_NC1HWC0 = 3
                weight_trans._share_underline_tensor_to(self.weight)
                bias_trans._share_underline_tensor_to(self.bias)
                mean_trans._share_underline_tensor_to(self._mean)
                var_trans._share_underline_tensor_to(self._variance)

        self._data_format = data_format
        self._in_place = False
        self._momentum = momentum
        self._epsilon = epsilon
        self._fuse_with_relu = False
        self._name = name

    def _check_input_dim(self, input):
        raise NotImplementedError("BatchNorm Base error")

    def _check_data_format(self, input):
        raise NotImplementedError("BatchNorm Base data format error")

    def forward(self, input):

        self._check_data_format(self._data_format)

        self._check_input_dim(input)

        if self.training:
            warnings.warn(
                "When training, we now always track global mean and variance."
            )

        return batch_norm(
            input,
            self._mean,
            self._variance,
            weight=self.weight,
            bias=self.bias,
            training=self.training,
            momentum=self._momentum,
            epsilon=self._epsilon,
            data_format=self._data_format,
            use_global_stats=self._use_global_stats,
        )

    def extra_repr(self):
        main_str = 'num_features={}, momentum={}, epsilon={}'.format(
            self._num_features, self._momentum, self._epsilon
        )
        if self._data_format != 'NCHW':
            main_str += ', data_format={}'.format(self._data_format)
        if self._name is not None:
            main_str += ', name={}'.format(self._name)
        return main_str


class BatchNorm(Layer):
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
        data_layout(str, optional): Specify the input data format, the data format can be "NCHW" or "NHWC", where `N` is batch size, `C` is the number of the feature map, `H` is the height of the feature map, `W` is the width of the feature map. Default: NCHW.
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
          import paddle.nn as nn
          from paddle.fluid.dygraph.base import to_variable
          import numpy as np


          x = np.random.random(size=(3, 10, 3, 7)).astype('float32')
          with fluid.dygraph.guard():
              x = to_variable(x)
              batch_norm = nn.layer.norm.BatchNorm(10)
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


class BatchNorm1D(_BatchNormBase):
    r"""
    Applies Batch Normalization over a 2D or 3D input (a mini-batch of 1D inputswith additional channel dimension) as described in the paper Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift .

    When use_global_stats = False, the :math:`\mu_{\beta}`
    and :math:`\sigma_{\beta}^{2}` are the statistics of one mini-batch.
    Calculated as follows:

    ..  math::

        \mu_{\beta} &\gets \frac{1}{m} \sum_{i=1}^{m} x_i \qquad &//\
        \ mini-batch\ mean \\
        \sigma_{\beta}^{2} &\gets \frac{1}{m} \sum_{i=1}^{m}(x_i - \
        \mu_{\beta})^2 \qquad &//\ mini-batch\ variance \\

    When use_global_stats = True, the :math:`\mu_{\beta}`
    and :math:`\sigma_{\beta}^{2}` are not the statistics of one mini-batch.
    They are global or running statistics (moving_mean and moving_variance). It usually got from the
    pre-trained model. Calculated as follows:

    .. math::
        moving\_mean = moving\_mean * momentum + \mu_{\beta} * (1. - momentum) \quad &// global \ mean \\
        moving\_variance = moving\_variance * momentum + \sigma_{\beta}^{2} * (1. - momentum) \quad &// global \ variance \\

    The normalization function formula is as follows:

    ..  math::

        \hat{x_i} &\gets \frac{x_i - \mu_\beta} {\sqrt{\sigma_{\beta}^{2} + \epsilon}} \qquad &//\ normalize \\
        y_i &\gets \gamma \hat{x_i} + \beta \qquad &//\ scale\ and\ shift

    - :math:`\epsilon` : add a smaller value to the variance to prevent division by zero
    - :math:`\gamma` : trainable proportional parameter
    - :math:`\beta` : trainable deviation parameter

    Parameters:
        num_features(int): Indicate the number of channels of the input ``Tensor``.
        epsilon(float, optional): The small value added to the variance to prevent division by zero. Default: 1e-5.
        momentum(float, optional): The value used for the moving_mean and moving_var computation. Default: 0.9.
        weight_attr(ParamAttr|bool, optional): The parameter attribute for Parameter `scale`
            of batch_norm. If it is set to None or one attribute of ParamAttr, batch_norm
            will create ParamAttr as weight_attr. If it is set to False, the weight is not learnable.
            If the Initializer of the weight_attr is not set, the parameter is initialized with ones. Default: None.
        bias_attr(ParamAttr|bool, optional): The parameter attribute for the bias of batch_norm.
            If it is set to None or one attribute of ParamAttr, batch_norm
            will create ParamAttr as bias_attr. If it is set to False, the weight is not learnable.
            If the Initializer of the bias_attr is not set, the bias is initialized zero. Default: None.
        data_format(str, optional): Specify the input data format, may be "NC", "NCL" or "NLC", where `N` is batch size, `C` is the number of the feature map, `L` is the length of the feature map. Default "NCL".
        use_global_stats(bool|None, optional): Whether to use global mean and variance. If set to False, use the statistics of one mini-batch, if set to True, use the global statistics, if set to None, use global statistics in the test phase and use the statistics of one mini-batch in the training phase. Default: None.
        name(str, optional): Name for the BatchNorm, default is None. For more information, please refer to :ref:`api_guide_Name`..

    Shape:
        - x: 2-D or 3-D tensor with shape: (batch, num_features) or (batch, num_features, length) when data_format is "NC" or "NCL",
            (batch, length, num_features) when data_format is "NLC".
        - output: 3-D tensor with same shape as input x.

    Returns:
        None.


    Examples:
        .. code-block:: python

          import paddle

          x = paddle.rand((2, 1, 3))
          batch_norm = paddle.nn.BatchNorm1D(1)
          batch_norm_out = batch_norm(x)

          print(batch_norm_out)
    """

    def __init__(
        self,
        num_features,
        momentum=0.9,
        epsilon=1e-05,
        weight_attr=None,
        bias_attr=None,
        data_format='NCL',
        use_global_stats=None,
        name=None,
    ):
        super().__init__(
            num_features,
            momentum,
            epsilon,
            weight_attr,
            bias_attr,
            data_format,
            use_global_stats,
            name,
        )

    def _check_data_format(self, input):
        if input == 'NCHW' or input == 'NC' or input == 'NCL':
            self._data_format = 'NCHW'
        elif input == "NHWC" or input == 'NLC':
            self._data_format = "NHWC"
        else:
            raise ValueError(
                'expected NC , NCL, NLC or None for data_format input'
            )

    def _check_input_dim(self, input):
        if len(input.shape) != 2 and len(input.shape) != 3:
            raise ValueError(
                'expected 2D or 3D input (got {}D input)'.format(
                    len(input.shape)
                )
            )


class BatchNorm2D(_BatchNormBase):
    r"""
    Applies Batch Normalization over a 4D input (a mini-batch of 2D inputswith additional channel dimension) as described in the paper Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift .

    When use_global_stats = False, the :math:`\mu_{\beta}`
    and :math:`\sigma_{\beta}^{2}` are the statistics of one mini-batch.
    Calculated as follows:

    ..  math::

        \mu_{\beta} &\gets \frac{1}{m} \sum_{i=1}^{m} x_i \qquad &//
        \ mini-batch\ mean \\
        \sigma_{\beta}^{2} &\gets \frac{1}{m} \sum_{i=1}^{m}(x_i -
        \mu_{\beta})^2 \qquad &//\ mini-batch\ variance \\

    When use_global_stats = True, the :math:`\mu_{\beta}`
    and :math:`\sigma_{\beta}^{2}` are not the statistics of one mini-batch.
    They are global or running statistics (moving_mean and moving_variance). It usually got from the
    pre-trained model. Calculated as follows:

    .. math::
        moving\_mean = moving\_mean * momentum + \mu_{\beta} * (1. - momentum) \quad &// global \ mean \\
        moving\_variance = moving\_variance * momentum + \sigma_{\beta}^{2} * (1. - momentum) \quad &// global \ variance \\

    The normalization function formula is as follows:

    ..  math::

        \hat{x_i} &\gets \frac{x_i - \mu_\beta} {\sqrt{\sigma_{\beta}^{2} + \epsilon}} \qquad &//\ normalize \\
        y_i &\gets \gamma \hat{x_i} + \beta \qquad &//\ scale\ and\ shift

    - :math:`\epsilon` : add a smaller value to the variance to prevent division by zero
    - :math:`\gamma` : trainable proportional parameter
    - :math:`\beta` : trainable deviation parameter

    Parameters:
        num_features(int): Indicate the number of channels of the input ``Tensor``.
        epsilon(float, optional): The small value added to the variance to prevent division by zero. Default: 1e-5.
        momentum(float, optional): The value used for the moving_mean and moving_var computation. Default: 0.9.
        weight_attr(ParamAttr|bool, optional): The parameter attribute for Parameter `scale`
            of batch_norm. If it is set to None or one attribute of ParamAttr, batch_norm
            will create ParamAttr as weight_attr. If it is set to False, the weight is not learnable.
            If the Initializer of the weight_attr is not set, the parameter is initialized with ones. Default: None.
        bias_attr(ParamAttr|bool, optional): The parameter attribute for the bias of batch_norm.
            If it is set to None or one attribute of ParamAttr, batch_norm
            will create ParamAttr as bias_attr. If it is set to False, the weight is not learnable.
            If the Initializer of the bias_attr is not set, the bias is initialized zero. Default: None.
        data_format(str, optional): Specify the input data format, the data format can be "NCHW" or "NHWC", where `N` is batch size, `C` is the number of the feature map, `H` is the height of the feature map, `W` is the width of the feature map. Default: NCHW.
        use_global_stats(bool|None, optional): Whether to use global mean and variance. If set to False, use the statistics of one mini-batch, if set to True, use the global statistics, if set to None, use global statistics in the test phase and use the statistics of one mini-batch in the training phase. Default: None.
        name(str, optional): Name for the BatchNorm, default is None. For more information, please refer to :ref:`api_guide_Name`..

    Shape:
        - x: 4-D tensor with shape: (batch, num_features, height, weight) when data_format is "NCHW",
            or (batch, height, weight, num_features) when data_format is "NHWC".
        - output: 4-D tensor with same shape as input x.

    Returns:
        None

    Examples:
        .. code-block:: python

          import paddle

          x = paddle.rand((2, 1, 2, 3))
          batch_norm = paddle.nn.BatchNorm2D(1)
          batch_norm_out = batch_norm(x)

          print(batch_norm_out)
    """

    def _check_data_format(self, input):
        if input == 'NCHW':
            self._data_format = input
        elif input == "NHWC":
            self._data_format = input
        else:
            raise ValueError('expected NCHW or NHWC for data_format input')

    def _check_input_dim(self, input):
        if len(input.shape) != 4:
            raise ValueError(
                'expected 4D input (got {}D input)'.format(len(input.shape))
            )


class BatchNorm3D(_BatchNormBase):
    r"""
    Applies Batch Normalization over a 5D input (a mini-batch of 3D inputswith additional channel dimension) as described in the paper Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift .

    When use_global_stats = False, the :math:`\mu_{\beta}`
    and :math:`\sigma_{\beta}^{2}` are the statistics of one mini-batch.
    Calculated as follows:

    ..  math::

        \mu_{\beta} &\gets \frac{1}{m} \sum_{i=1}^{m} x_i \qquad &//\
        \ mini-batch\ mean \\
        \sigma_{\beta}^{2} &\gets \frac{1}{m} \sum_{i=1}^{m}(x_i - \
        \mu_{\beta})^2 \qquad &//\ mini-batch\ variance \\

    When use_global_stats = True, the :math:`\\mu_{\\beta}`
    and :math:`\\sigma_{\\beta}^{2}` are not the statistics of one mini-batch.
    They are global or running statistics (moving_mean and moving_variance). It usually got from the
    pre-trained model. Calculated as follows:

    .. math::
        moving\_mean = moving\_mean * momentum + \mu_{\beta} * (1. - momentum) \quad &// global \ mean \\
        moving\_variance = moving\_variance * momentum + \sigma_{\beta}^{2} * (1. - momentum) \quad &// global \ variance \\

    The normalization function formula is as follows:

    ..  math::

        \hat{x_i} &\gets \frac{x_i - \mu_\beta} {\sqrt{\sigma_{\beta}^{2} + \epsilon}} \qquad &//\ normalize \\
        y_i &\gets \gamma \hat{x_i} + \beta \qquad &//\ scale\ and\ shift

    - :math:`\epsilon` : add a smaller value to the variance to prevent division by zero
    - :math:`\gamma` : trainable proportional parameter
    - :math:`\beta` : trainable deviation parameter

    Parameters:
        num_features(int): Indicate the number of channels of the input ``Tensor``.
        epsilon(float, optional): The small value added to the variance to prevent division by zero. Default: 1e-5.
        momentum(float, optional): The value used for the moving_mean and moving_var computation. Default: 0.9.
        weight_attr(ParamAttr|bool, optional): The parameter attribute for Parameter `scale`
            of batch_norm. If it is set to None or one attribute of ParamAttr, batch_norm
            will create ParamAttr as weight_attr. If it is set to False, the weight is not learnable.
            If the Initializer of the weight_attr is not set, the parameter is initialized with ones. Default: None.
        bias_attr(ParamAttr|bool, optional): The parameter attribute for the bias of batch_norm.
            If it is set to None or one attribute of ParamAttr, batch_norm
            will create ParamAttr as bias_attr. If it is set to False, the weight is not learnable.
            If the Initializer of the bias_attr is not set, the bias is initialized zero. Default: None.
        data_format(str, optional): Specify the input data format, the data format can be "NCDHW" or "NDHWC", where `N` is batch size, `C` is the number of the feature map, `D` is the depth of the feature, `H` is the height of the feature map, `W` is the width of the feature map. Default: NCDHW.
        use_global_stats(bool|None, optional): Whether to use global mean and variance. If set to False, use the statistics of one mini-batch, if set to True, use the global statistics, if set to None, use global statistics in the test phase and use the statistics of one mini-batch in the training phase. Default: None.
        name(str, optional): Name for the BatchNorm, default is None. For more information, please refer to :ref:`api_guide_Name`..

    Shape:
        - x: 5-D tensor with shape: (batch, num_features, dims, height, weight) when data_format is "NCDHW",
            or (batch, dims, height, weight, num_features) when data_format is "NDHWC".
        - output: 5-D tensor with same shape as input x.

    Returns:
        None

    Examples:
        .. code-block:: python

          import paddle

          x = paddle.rand((2, 1, 2, 2, 3))
          batch_norm = paddle.nn.BatchNorm3D(1)
          batch_norm_out = batch_norm(x)

          print(batch_norm_out)
    """

    def __init__(
        self,
        num_features,
        momentum=0.9,
        epsilon=1e-05,
        weight_attr=None,
        bias_attr=None,
        data_format='NCDHW',
        use_global_stats=None,
        name=None,
    ):
        super().__init__(
            num_features,
            momentum,
            epsilon,
            weight_attr,
            bias_attr,
            data_format,
            use_global_stats,
            name,
        )

    def _check_data_format(self, input):
        if input == 'NCHW' or input == 'NCDHW':
            self._data_format = 'NCHW'
        elif input == "NHWC" or input == "NDHWC":
            self._data_format = 'NHWC'
        else:
            raise ValueError(
                'expected NCDHW, NDHWC or None for data_format input'
            )

    def _check_input_dim(self, input):
        if len(input.shape) != 5:
            raise ValueError(
                'expected 5D input (got {}D input)'.format(len(input.shape))
            )


class SyncBatchNorm(_BatchNormBase):
    r"""

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

        \mu_{\beta} &\gets \frac{1}{m} \sum_{i=1}^{m} x_i \qquad &//\
        \ mini-batch\ mean \\
        \sigma_{\beta}^{2} &\gets \frac{1}{m} \sum_{i=1}^{m}(x_i - \
        \mu_{\beta})^2 \qquad &//\ mini-batch\ variance \\

    - :math:`x` : whole mini-batch data in all gpus
    - :math:`m` : the size of the whole mini-batch data

    When model in evaluation mode, the :math:`\\mu_{\\beta}`
    and :math:`\sigma_{\beta}^{2}` are global statistics (moving_mean and moving_variance,
    which usually got from the pre-trained model). Global statistics calculated as follows:

    .. math::
        moving\_mean = moving\_mean * momentum + \mu_{\beta} * (1. - momentum) \quad &// global \ mean \\
        moving\_variance = moving\_variance * momentum + \sigma_{\beta}^{2} * (1. - momentum) \quad &// global \ variance \\

    The formula of normalization is as follows:

    ..  math::

        \hat{x_i} &\gets \frac{x_i - \mu_\beta} {\sqrt{\
        \sigma_{\beta}^{2} + \epsilon}} \qquad &//\ normalize \\
        y_i &\gets \gamma \hat{x_i} + \beta \qquad &//\ scale\ and\ shift

    - :math:`\epsilon` : add a smaller value to the variance to prevent division by zero
    - :math:`\gamma` : trainable scale parameter vector
    - :math:`\beta` : trainable shift parameter vector

    Note:
        If you want to use container to pack your model and has :ref:`api_paddle_nn_SyncBatchNorm` in the
        evaluation phase, please use :ref:`api_paddle_nn_LayerList` or :ref:`api_paddle_nn_Sequential` instead of
        :ref:`api_paddle_hub_list` to pack the model.

    Parameters:
        num_features(int): Indicate the number of channels of the input ``Tensor``.
        epsilon(float, optional): The small value added to the variance to prevent division by zero. Default: 1e-5.
        momentum(float, optional): The value used for the moving_mean and moving_var computation. Default: 0.9.
        weight_attr(ParamAttr|bool, optional): The parameter attribute for Parameter `scale`
             of this layer. If it is set to None or one attribute of ParamAttr, this layerr
             will create ParamAttr as param_attr. If the Initializer of the param_attr
             is not set, the parameter is initialized with ones. If it is set to False,
             this layer will not have trainable scale parameter. Default: None.
        bias_attr(ParamAttr|bool, optional): The parameter attribute for the bias of this layer.
             If it is set to None or one attribute of ParamAttr, this layer
             will create ParamAttr as bias_attr. If the Initializer of the bias_attr
             is not set, the bias is initialized zero. If it is set to False, this layer will not
             have trainable bias parameter. Default: None.

    Shapes:
        - input: Tensor that the dimension from 2 to 5.
        - output: Tensor with the same shape as input.

    Examples:
        .. code-block:: python

            # required: gpu

            import paddle
            import paddle.nn as nn

            x = paddle.to_tensor([[[[0.3, 0.4], [0.3, 0.07]], [[0.83, 0.37], [0.18, 0.93]]]]).astype('float32')

            if paddle.is_compiled_with_cuda():
                sync_batch_norm = nn.SyncBatchNorm(2)
                hidden1 = sync_batch_norm(x)
                print(hidden1)
                # Tensor(shape=[1, 2, 2, 2], dtype=float32, place=Place(gpu:0), stop_gradient=False,
                #        [[[[ 0.26824948,  1.09363246],
                #           [ 0.26824948, -1.63013160]],

                #          [[ 0.80956620, -0.66528702],
                #           [-1.27446556,  1.13018656]]]])

    """

    def __init__(
        self,
        num_features,
        momentum=0.9,
        epsilon=1e-05,
        weight_attr=None,
        bias_attr=None,
        data_format='NCHW',
        name=None,
    ):
        super().__init__(
            num_features,
            momentum,
            epsilon,
            weight_attr,
            bias_attr,
            data_format,
            None,
            name,
        )

    def _check_data_format(self):
        if self._data_format in ['NCHW', 'NCDHW', 'NC', 'NCL']:
            self._data_format = 'NCHW'
        elif self._data_format in ["NHWC", "NDHWC", 'NLC']:
            self._data_format = 'NHWC'
        else:
            raise ValueError(
                'expected \'NCDHW\', \'NDHWC\', \'NCL\', \'NLC\', \'NC\', \'NCHW\', \'NHWC\' for data_format'
            )

    def forward(self, x):
        self._check_data_format()
        # create output
        # mean and mean_out share the same memory
        mean_out = self._mean
        # variance and variance out share the same memory
        variance_out = self._variance

        # train mode: use mini-batch stats, eval mode: use global stats
        # use_global_stats only support False in sync_batch_norm
        if in_dygraph_mode():
            sync_batch_norm_out, _, _, _, _, _ = _C_ops.sync_batch_norm_(
                x,
                self._mean,
                self._variance,
                self.weight,
                self.bias,
                not self.training,
                self._momentum,
                self._epsilon,
                self._data_format,
                False,
                False,
            )
            return sync_batch_norm_out

        elif in_dynamic_mode():
            attrs = (
                "momentum",
                self._momentum,
                "epsilon",
                self._epsilon,
                "is_test",
                not self.training,
                "data_layout",
                self._data_format,
                "use_mkldnn",
                False,
                "fuse_with_relu",
                False,
                "use_global_stats",
                False,
                'trainable_statistics',
                False,
            )
            sync_batch_norm_out, _, _, _, _, _ = _legacy_C_ops.sync_batch_norm(
                x,
                self.weight,
                self.bias,
                self._mean,
                self._variance,
                mean_out,
                variance_out,
                *attrs
            )
            return sync_batch_norm_out

        check_variable_and_dtype(
            x, 'input', ['float16', 'float32', 'float64'], 'SyncBatchNorm'
        )

        attrs = {
            "momentum": self._momentum,
            "epsilon": self._epsilon,
            "is_test": not self.training,
            "data_layout": self._data_format,
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
            "Variance": [self._variance],
        }

        saved_mean = self._helper.create_variable_for_type_inference(
            dtype=self._dtype, stop_gradient=True
        )
        saved_variance = self._helper.create_variable_for_type_inference(
            dtype=self._dtype, stop_gradient=True
        )
        sync_batch_norm_out = self._helper.create_variable_for_type_inference(
            self._dtype
        )

        outputs = {
            "Y": [sync_batch_norm_out],
            "MeanOut": [mean_out],
            "VarianceOut": [variance_out],
            "SavedMean": [saved_mean],
            "SavedVariance": [saved_variance],
        }

        self._helper.append_op(
            type="sync_batch_norm", inputs=inputs, outputs=outputs, attrs=attrs
        )
        return sync_batch_norm_out

    @classmethod
    def convert_sync_batchnorm(cls, layer):
        """
        Helper function to convert :class: `paddle.nn.BatchNorm*d` layers in the model to :class: `paddle.nn.SyncBatchNorm` layers.

        Parameters:
            layer(paddle.nn.Layer): model containing one or more `BatchNorm*d` layers.

        Returns:
            The original model with converted SyncBatchNorm layers. If BatchNorm*d layer in the model, use SyncBatchNorm layer instead.

        Examples:
            .. code-block:: python

                import paddle
                import paddle.nn as nn

                model = nn.Sequential(nn.Conv2D(3, 5, 3), nn.BatchNorm2D(5))
                sync_model = nn.SyncBatchNorm.convert_sync_batchnorm(model)

        """
        layer_output = layer
        if isinstance(layer, _BatchNormBase):
            if (
                layer._weight_attr is not None
                and not isinstance(layer._weight_attr, bool)
                and layer._weight_attr.name is not None
            ):
                layer._weight_attr.name = layer._weight_attr.name + '_sync'
            if (
                layer._bias_attr is not None
                and not isinstance(layer._bias_attr, bool)
                and layer._bias_attr.name is not None
            ):
                layer._bias_attr.name = layer._bias_attr.name + '_sync'

            layer_output = SyncBatchNorm(
                layer._num_features,
                layer._momentum,
                layer._epsilon,
                layer._weight_attr,
                layer._bias_attr,
                layer._data_format,
                layer._name,
            )

            if (
                layer._weight_attr is not False
                and layer._bias_attr is not False
            ):
                with no_grad():
                    layer_output.weight = layer.weight
                    layer_output.bias = layer.bias
            layer_output._mean = layer._mean
            layer_output._variance = layer._variance

        for name, sublayer in layer.named_children():
            layer_output.add_sublayer(
                name, cls.convert_sync_batchnorm(sublayer)
            )
        del layer
        return layer_output


class LocalResponseNorm(Layer):
    """
    Local Response Normalization performs a type of "lateral inhibition" by normalizing over local input regions.
    For more information, please refer to `ImageNet Classification with Deep Convolutional Neural Networks <https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf>`_

    See more details in :ref:`api_paddle_nn_functional_local_response_norm` .

    Parameters:
        size (int): The number of channels to sum over.
        alpha (float, optional): The scaling parameter, positive. Default:1e-4
        beta (float, optional): The exponent, positive. Default:0.75
        k (float, optional): An offset, positive. Default: 1.0
        data_format (str, optional): Specify the data format of the input, and the data format of the output
            will be consistent with that of the input. An optional string from:
            If input is 3-D Tensor, the string could be `"NCL"` or `"NLC"` . When it is `"NCL"`,
            the data is stored in the order of: `[batch_size, input_channels, feature_length]`.
            If input is 4-D Tensor, the string could be  `"NCHW"`, `"NHWC"`. When it is `"NCHW"`,
            the data is stored in the order of: `[batch_size, input_channels, input_height, input_width]`.
            If input is 5-D Tensor, the string could be  `"NCDHW"`, `"NDHWC"` . When it is `"NCDHW"`,
            the data is stored in the order of: `[batch_size, input_channels, input_depth, input_height, input_width]`.
        name (str, optional): Name for the operation (optional, default is None). For more information,
            please refer to :ref:`api_guide_Name`.

    Shape:
        - input: 3-D/4-D/5-D tensor.
        - output: 3-D/4-D/5-D tensor, the same shape as input.

    Examples:

    .. code-block:: python

        import paddle

        x = paddle.rand(shape=(3, 3, 112, 112), dtype="float32")
        m = paddle.nn.LocalResponseNorm(size=5)
        y = m(x)
        print(y.shape)  # [3, 3, 112, 112]
    """

    def __init__(
        self,
        size,
        alpha=0.0001,
        beta=0.75,
        k=1.0,
        data_format="NCHW",
        name=None,
    ):
        super().__init__()
        self.size = size
        self.alpha = alpha
        self.beta = beta
        self.k = k
        self.data_format = data_format
        self.name = name

    def forward(self, input):
        out = F.local_response_norm(
            input,
            self.size,
            self.alpha,
            self.beta,
            self.k,
            self.data_format,
            self.name,
        )
        return out

    def extra_repr(self):
        main_str = 'size={}, alpha={}, beta={}, k={}'.format(
            self.size, self.alpha, self.beta, self.k
        )
        if self.data_format != 'NCHW':
            main_str += ', data_format={}'.format(self.data_format)
        if self.name is not None:
            main_str += ', name={}'.format(self.name)
        return main_str


class SpectralNorm(Layer):
    r"""
    This interface is used to construct a callable object of the ``SpectralNorm`` class.
    For more details, refer to code examples. It implements the function of the Spectral Normalization Layer.
    This layer calculates the spectral normalization value of weight parameters of
    fc, conv1d, conv2d, conv3d layers which should be 2-D, 3-D, 4-D, 5-D
    Parameters. Calculations are showed as follows.

    Step 1:
    Generate vector U in shape of [H], and V in shape of [W].
    While H is the :attr:`axis` th dimension of the input weights,
    and W is the product result of remaining dimensions.

    Step 2:
    :attr:`power_iters` should be a positive integer, do following
    calculations with U and V for :attr:`power_iters` rounds.

    .. math::

        \mathbf{v} := \frac{\mathbf{W}^{T} \mathbf{u}}{\|\mathbf{W}^{T} \mathbf{u}\|_2}

        \mathbf{u} := \frac{\mathbf{W}^{T} \mathbf{v}}{\|\mathbf{W}^{T} \mathbf{v}\|_2}

    Step 3:
    Calculate :math:`\sigma(\mathbf{W})` and normalize weight values.

    .. math::

        \sigma(\mathbf{W}) = \mathbf{u}^{T} \mathbf{W} \mathbf{v}

        \mathbf{W} = \frac{\mathbf{W}}{\sigma(\mathbf{W})}


    Refer to `Spectral Normalization <https://arxiv.org/abs/1802.05957>`_ .

    Parameters:
        weight_shape(list or tuple): The shape of weight parameter.
        axis(int, optional): The index of dimension which should be permuted to the first before reshaping Input(Weight) to matrix, it should be set as 0 if Input(Weight) is the weight of fc layer, and should be set as 1 if Input(Weight) is the weight of conv layer. Default: 0.
        power_iters(int, optional): The number of power iterations to calculate spectral norm. Default: 1.
        epsilon(float, optional): The epsilon for numerical stability in calculating norms. Default: 1e-12.
        name (str, optional): The default value is None.  Normally there is no need for user to set this property.  For more information, please refer to :ref:`api_guide_Name` .
        dtype (str, optional): Data type, it can be "float32" or "float64". Default: "float32".

    Returns:
        None

    Examples:
       .. code-block:: python

            import paddle
            x = paddle.rand((2,8,32,32))

            spectral_norm = paddle.nn.SpectralNorm(x.shape, axis=1, power_iters=2)
            spectral_norm_out = spectral_norm(x)

            print(spectral_norm_out.shape) # [2, 8, 32, 32]

    """

    def __init__(
        self,
        weight_shape,
        axis=0,
        power_iters=1,
        epsilon=1e-12,
        dtype='float32',
    ):
        super().__init__()
        self._power_iters = power_iters
        self._epsilon = epsilon
        self._dim = axis
        self._dtype = dtype

        self._weight_shape = list(weight_shape)
        assert (
            np.prod(self._weight_shape) > 0
        ), "Any dimension of `weight_shape` cannot be equal to 0."
        assert axis < len(self._weight_shape), (
            "The input `axis` should be less than the "
            "length of `weight_shape`, but received axis="
            "{}".format(axis)
        )
        h = self._weight_shape[self._dim]
        w = np.prod(self._weight_shape) // h

        self.weight_u = self.create_parameter(
            attr=ParamAttr(),
            shape=[h],
            dtype=self._dtype,
            default_initializer=Normal(0.0, 1.0),
        )
        self.weight_u.stop_gradient = True

        self.weight_v = self.create_parameter(
            attr=ParamAttr(),
            shape=[w],
            dtype=self._dtype,
            default_initializer=Normal(0.0, 1.0),
        )
        self.weight_v.stop_gradient = True

    def forward(self, x):
        weight = x
        if in_dygraph_mode():
            return _C_ops.spectral_norm(
                weight,
                self.weight_u,
                self.weight_v,
                self._dim,
                self._power_iters,
                self._epsilon,
            )

        check_variable_and_dtype(
            weight, "weight", ['float32', 'float64'], 'SpectralNorm'
        )
        inputs = {'Weight': weight, 'U': self.weight_u, 'V': self.weight_v}
        out = self._helper.create_variable_for_type_inference(self._dtype)
        self._helper.append_op(
            type="spectral_norm",
            inputs=inputs,
            outputs={
                "Out": out,
            },
            attrs={
                "dim": self._dim,
                "power_iters": self._power_iters,
                "eps": self._epsilon,
            },
        )

        return out
