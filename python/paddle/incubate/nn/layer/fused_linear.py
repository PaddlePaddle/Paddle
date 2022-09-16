# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

from paddle.nn import Layer
from paddle.incubate.nn import functional as F


class FusedLinear(Layer):
    """
    Linear layer takes only one multi-dimensional tensor as input with the
    shape :math:`[batch\_size, *, in\_features]` , where :math:`*` means any
    number of additional dimensions. It multiplies input tensor with the weight
    (a 2-D tensor of shape :math:`[in\_features, out\_features]` ) and produces
    an output tensor of shape :math:`[batch\_size, *, out\_features]` .
    If :math:`bias\_attr` is not False, the bias (a 1-D tensor of
    shape :math:`[out\_features]` ) will be created and added to the output.

    Parameters:
        in_features (int): The number of input units.
        out_features (int): The number of output units.
        weight_attr (ParamAttr, optional): The attribute for the learnable
            weight of this layer. The default value is None and the weight will be
            initialized to zero. For detailed information, please refer to
            paddle.ParamAttr.
        transpose_weight (bool): Whether to transpose the `weight` Tensor before
            multiplication.
        bias_attr (ParamAttr|bool, optional): The attribute for the learnable bias
            of this layer. If it is set to False, no bias will be added to the output.
            If it is set to None or one kind of ParamAttr, a bias parameter will
            be created according to ParamAttr. For detailed information, please refer
            to paddle.ParamAttr. The default value is None and the bias will be
            initialized to zero.
        name (str, optional): Normally there is no need for user to set this parameter.
            For detailed information, please refer to :ref:`api_guide_Name` .

    Attribute:
        **weight** (Parameter): the learnable weight of this layer.

        **bias** (Parameter): the learnable bias of this layer.

    Shape:
        - input: Multi-dimentional tensor with shape :math:`[batch\_size, *, in\_features]` .
        - output: Multi-dimentional tensor with shape :math:`[batch\_size, *, out\_features]` .

    Examples:
        .. code-block:: python

            # required: gpu
            import paddle
            from paddle.incubate.nn import FusedLinear

            x = paddle.randn([3, 4])
            linear = FusedLinear(4, 5)
            y = linear(x)
            print(y.shape) # [3, 5]
    """

    def __init__(self,
                 in_features,
                 out_features,
                 weight_attr=None,
                 bias_attr=None,
                 transpose_weight=False,
                 name=None):
        super(FusedLinear, self).__init__()
        if transpose_weight:
            weight_shape = [out_features, in_features]
        else:
            weight_shape = [in_features, out_features]
        dtype = self._helper.get_default_dtype()
        self.weight = self.create_parameter(shape=weight_shape,
                                            attr=weight_attr,
                                            dtype=dtype,
                                            is_bias=False)
        self.bias = self.create_parameter(shape=[out_features],
                                          attr=bias_attr,
                                          dtype=dtype,
                                          is_bias=True)
        self.transpose_weight = transpose_weight
        self.name = name

    def forward(self, input):
        return F.fused_linear(input, self.weight, self.bias,
                              self.transpose_weight, self.name)
