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

# TODO: define normalization api  

from ...fluid.dygraph.nn import InstanceNorm

from ...fluid.dygraph import BatchNorm  #DEFINE_ALIAS
from ...fluid.dygraph import GroupNorm  #DEFINE_ALIAS
from ...fluid.dygraph import LayerNorm  #DEFINE_ALIAS
from ...fluid.dygraph import SpectralNorm  #DEFINE_ALIAS

__all__ = [
    'BatchNorm', 'GroupNorm', 'LayerNorm', 'SpectralNorm', 'InstanceNorm',
    'BatchNorm', 'BatchNorm2d'
]


class BatchNorm2d(layers.Layer):
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
        weight_attr(ParamAttr, optional): The parameter attribute for Parameter `scale`
             of batch_norm. If it is set to None or one attribute of ParamAttr, batch_norm
             will create ParamAttr as weight_attr. If the Initializer of the weight_attr
             is not set, the parameter is initialized with Xavier. Default: None.
        bias_attr(ParamAttr, optional): The parameter attribute for the bias of batch_norm.
             If it is set to None or one attribute of ParamAttr, batch_norm
             will create ParamAttr as bias_attr. If the Initializer of the bias_attr
             is not set, the bias is initialized zero. Default: None.
        data_format(str, optional): Specify the input data format, the data format can be "NCHW" or "NHWC". Default: NCHW.
        track_running_stats(bool, optional): Whether to use global mean and
            variance. In train mode, when setting track_running_stats True, the global mean
            and variance are also used during train period. Default: True.
        name(str, optional): Default: None.

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
              batch_norm = fluid.BatchNorm2d(10)
              hidden1 = batch_norm(x)
    """

    def __init__(self,
                 num_features,
                 momentum=0.9,
                 epsilon=1e-05,
                 weight_attr=None,
                 bias_attr=None,
                 data_format='NCHW',
                 track_running_stats=False,
                 name=None):
        super(BatchNorm2d, self).__init__()
        self._weight_attr = weight_attr
        self._bias_attr = bias_attr

        assert bias_attr is not False, "bias_attr should not be False in batch_norm."

        self._dtype = 'float32'

        param_shape = [num_features]

        # create parameter
        self.weight = self.create_parameter(
            attr=self._weight_attr,
            shape=param_shape,
            dtype=self._dtype,
            default_initializer=Constant(1.0))
        self.weight.stop_gradient = self._weight_attr.learning_rate == 0.

        self.bias = self.create_parameter(
            attr=self._bias_attr,
            shape=param_shape,
            dtype=self._dtype,
            is_bias=True)
        self.bias.stop_gradient = self._param_attr.learning_rate == 0.

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
        self._in_place = True
        self._momentum = momentum
        self._epsilon = epsilon
        self._fuse_with_relu = False
        self._track_running_stats = track_running_stats

    def forward(self, input):
        # create output
        # mean and mean_out share the same memory
        mean_out = self._mean
        # variance and variance out share the same memory
        variance_out = self._variance

        if in_dygraph_mode():
            attrs = ("momentum", self._momentum, "epsilon", self._epsilon,
                     "is_test", not self.training, "data_layout",
                     self._data_format, "use_mkldnn", False, "fuse_with_relu",
                     self._fuse_with_relu, "use_global_stats",
                     self._track_running_stats, 'trainable_statistics',
                     self._track_running_stats)
            batch_norm_out, _, _, _, _, _ = core.ops.batch_norm(
                input, self.weight, self.bias, self._mean, self._variance,
                mean_out, variance_out, *attrs)

            return dygraph_utils._append_activation_in_dygraph(
                batch_norm_out, act=None)

        check_variable_and_dtype(
            input, 'input', ['float16', 'float32', 'float64'], 'BatchNorm2d')

        attrs = {
            "momentum": self._momentum,
            "epsilon": self._epsilon,
            "is_test": not self.training,
            "data_layout": self._data_format,
            "use_mkldnn": False,
            "fuse_with_relu": self._fuse_with_relu,
            "use_global_stats": self._track_running_stats,
            "trainable_statistics": self._track_running_stats,
        }

        inputs = {
            "X": [input],
            "Scale": [self.weight],
            "Bias": [self.bias],
            "Mean": [self._mean],
            "Variance": [self._variance]
        }

        saved_mean = self._helper.create_variable_for_type_inference(
            dtype=self._dtype, stop_gradient=True)
        saved_variance = self._helper.create_variable_for_type_inference(
            dtype=self._dtype, stop_gradient=True)
        batch_norm_out = input if self._in_place else self._helper.create_variable_for_type_inference(
            self._dtype)

        outputs = {
            "Y": [batch_norm_out],
            "MeanOut": [mean_out],
            "VarianceOut": [variance_out],
            "SavedMean": [saved_mean],
            "SavedVariance": [saved_variance]
        }

        self._helper.append_op(
            type="batch_norm", inputs=inputs, outputs=outputs, attrs=attrs)

        # Currently, we don't support inplace in dygraph mode
        return self._helper.append_activation(batch_norm_out, None)
