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
# __all__ = ['BatchNorm',
#            'GroupNorm',
#            'LayerNorm',
#            'SpectralNorm']
__all__ = ['InstanceNorm']

from ...fluid.dygraph.layers import Layer
from ...fluid import core
from ...fluid.framework import in_dygraph_mode


class InstanceNorm(Layer):
    """
    This interface is used to construct a callable object of the ``InstanceNorm`` class.
    For more details, refer to code examples.

    Can be used as a normalizer function for convolution or fully_connected operations.
    The required data format for this layer is one of the following:

    DataLayout: NCHW `[batch, in_channels, in_height, in_width]`

    Refer to `Instance Normalization: The Missing Ingredient for Fast Stylization <https://arxiv.org/pdf/1607.08022.pdf>`_
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

    Parameters:
        num_channels(int): Indicate the number of channels of the input ``Tensor``.
        epsilon(float, optional): A value added to the denominator for
            numerical stability. Default is 1e-5.
        param_attr(ParamAttr, optional): The parameter attribute for Parameter `scale`
             of instance_norm. If it is set to None or one attribute of ParamAttr, instance_norm
	     will create ParamAttr as param_attr, the name of scale can be set in ParamAttr.
	     If the Initializer of the param_attr is not set, the parameter is initialized 
	     one. Default: None.
        bias_attr(ParamAttr, optional): The parameter attribute for the bias of instance_norm.
             If it is set to None or one attribute of ParamAttr, instance_norm
	     will create ParamAttr as bias_attr, the name of bias can be set in ParamAttr. 
	     If the Initializer of the bias_attr is not set, the bias is initialized zero. 
	     Default: None.
        dtype(str, optional): Indicate the data type of the input ``Tensor``,
             which can be float32 or float64. Default: float32.

    Returns:
        None.

    Examples:

        .. code-block:: python

          import paddle.fluid as fluid
          from paddle.fluid.dygraph.base import to_variable
          import numpy as np
          import paddle

          x = np.random.random((10, 3, 32, 32)).astype('float32')
          with fluid.dygraph.guard():
              x = to_variable(x)
              instanceNorm = paddle.nn.InstanceNorm(3)
              ret = instanceNorm(x)
              print(ret)

    """

    def __init__(self,
                 num_channels,
                 epsilon=1e-5,
                 param_attr=None,
                 bias_attr=None,
                 dtype='float32'):
        super(InstanceNorm, self).__init__()
        assert bias_attr is not False, "bias_attr should not be False in InstanceNorm."

        self._epsilon = epsilon
        self._param_attr = param_attr
        self._bias_attr = bias_attr
        self._dtype = dtype

        self.scale = self.create_parameter(
            attr=self._param_attr,
            shape=[num_channels],
            dtype=self._dtype,
            default_initializer=Constant(1.0),
            is_bias=False)
        self.bias = self.create_parameter(
            attr=self._bias_attr,
            shape=[num_channels],
            dtype=self._dtype,
            default_initializer=Constant(0.0),
            is_bias=True)

    def forward(self, input):
        if in_dygraph_mode():
            out, _, _ = core.ops.instance_norm(input, self.scale, self.bias,
                                               'epsilon', self.epsilon)
            return out

        attrs = {"epsilon": self._epsilon}

        inputs = {"X": [input], "Scale": [self.scale], "Bias": [self.bias]}

        saved_mean = self._helper.create_variable_for_type_inference(
            dtype=self._dtype, stop_gradient=True)
        saved_variance = self._helper.create_variable_for_type_inference(
            dtype=self._dtype, stop_gradient=True)
        instance_norm_out = self._helper.create_variable_for_type_inference(
            self._dtype)

        outputs = {
            "Y": [instance_norm_out],
            "SavedMean": [saved_mean],
            "SavedVariance": [saved_variance]
        }

        self._helper.append_op(
            type="instance_norm", inputs=inputs, outputs=outputs, attrs=attrs)
        return instance_norm_out
