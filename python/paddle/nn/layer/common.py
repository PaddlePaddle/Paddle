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

# TODO: define the common classes to build a neural network  
# __all__ = ['BilinearTensorProduct',
#            'Pool2D',
#            'Embedding',
#            'Linear',
#            'UpSample']

__all__ = ['Linear']

from ...fluid.dygraph.layers import Layer
from ...fluid import core
from ...fluid import dygraph_utils
from ...fluid.framework import in_dygraph_mode


class Linear(Layer):
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
          from paddle.nn import Linear
          import numpy as np

          data = np.random.uniform(-1, 1, [30, 10, 32]).astype('float32')
          with fluid.dygraph.guard():
              linear = Linear(32, 64)
              data = to_variable(data)
              res = linear(data)  # [30, 10, 64]
    """

    def __init__(self,
                 input_dim,
                 output_dim,
                 param_attr=None,
                 bias_attr=None,
                 act=None,
                 dtype="float32"):
        super(Linear, self).__init__()
        self._act = act
        self._dtype = dtype
        self.weight = self.create_parameter(
            shape=[input_dim, output_dim],
            attr=param_attr,
            dtype=dtype,
            is_bias=False)
        self.bias = self.create_parameter(
            shape=[output_dim], attr=bias_attr, dtype=dtype, is_bias=True)

    def forward(self, input):
        attrs = {
            "x_num_col_dims": len(input.shape) - 1,
            "y_num_col_dims": 1,
        }
        inputs = {"X": [input], "Y": [self.weight]}

        if in_dygraph_mode():
            outs = core.ops.mul(inputs, attrs)
            pre_bias = outs['Out'][0]

            pre_act = dygraph_utils._append_bias_in_dygraph(
                pre_bias, self.bias, axis=len(input.shape) - 1)

            return dygraph_utils._append_activation_in_dygraph(pre_act,
                                                               self._act)

        tmp = self._helper.create_variable_for_type_inference(self._dtype)
        self._helper.append_op(
            type="mul", inputs=inputs, outputs={"Out": tmp}, attrs=attrs)
        if self.bias:
            pre_activation = self._helper.create_variable_for_type_inference(
                dtype=self._dtype)
            self._helper.append_op(
                type='elementwise_add',
                inputs={'X': [tmp],
                        'Y': [self.bias]},
                outputs={'Out': [pre_activation]},
                attrs={'axis': len(input.shape) - 1})
        else:
            pre_activation = tmp
        return self._helper.append_activation(pre_activation, act=self._act)
