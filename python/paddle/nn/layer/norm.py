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

import warnings
from ...fluid.dygraph.nn import InstanceNorm

from ...fluid.dygraph import BatchNorm  #DEFINE_ALIAS
from ...fluid.dygraph import GroupNorm  #DEFINE_ALIAS
from ...fluid.dygraph import LayerNorm  #DEFINE_ALIAS
from ...fluid.dygraph import SpectralNorm  #DEFINE_ALIAS

from ...fluid.dygraph import layers
from ...fluid.framework import in_dygraph_mode

from ...fluid.initializer import Constant
from ...fluid.param_attr import ParamAttr
from ...fluid.data_feeder import check_variable_and_dtype, check_type
from ...fluid import core

__all__ = [
    'BatchNorm', 'GroupNorm', 'LayerNorm', 'SpectralNorm', 'InstanceNorm',
    'SyncBatchNorm'
]


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
