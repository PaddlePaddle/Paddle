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

from paddle.incubate.nn import functional as F
from paddle.nn import Layer


class FusedEcMoe(Layer):
    r"""A FusedEcMoe Layer.

    Parameters:
        hidden_size (int): The dim size of input units.
        inter_size (int): The dim size of feed forward network.
        num_expert (int): The number of experts.
        act_type (string): The activation type. Currently only support `gelu`, `relu`.
        weight_attr (ParamAttr, optional): The attribute for the learnable
            weight of this layer. The default value is None and the weight will be
            initialized to zero. For detailed information, please refer to
            paddle.ParamAttr.
        bias_attr (ParamAttr|bool, optional): The attribute for the learnable bias
            of this layer. If it is set to False, no bias will be added to the output.
            If it is set to None or one kind of ParamAttr, a bias parameter will
            be created according to ParamAttr. For detailed information, please refer
            to paddle.ParamAttr. The default value is None and the bias will be
            initialized to zero.

    Attribute:
        **weight** (Parameter): the learnable weight of this layer.
        **bias** (Parameter): the learnable bias of this layer.

    Shape:
        - input: Multi-dimentional tensor with shape :math:`[batch\_size, seq\_len, d\_model]` .
        - output: Multi-dimentional tensor with shape :math:`[batch\_size, seq\_len, d\_model]` .

    Examples:
        .. code-block:: python

            # required: gpu
            import paddle
            from paddle.incubate.nn.layer.fused_ec_moe import FusedEcMoe

            x = paddle.randn([10, 128, 1024]) # [bsz, seq_len, d_model]
            gate = paddle.randn([10, 128, 8]) # [bsz, seq_len, num_experts]
            moe = FusedEcMoe(1024, 4096, 8, act_type="gelu")
            y = moe(x, gate)
            print(y.shape) # [10, 128, 1024]
    """

    def __init__(
        self,
        hidden_size,
        inter_size,
        num_experts,
        act_type,
        weight_attr=None,
        bias_attr=None,
    ):
        super().__init__()
        weight0_shape = [num_experts, hidden_size, inter_size]
        bias0_shape = [num_experts, 1, inter_size]
        weight1_shape = [num_experts, inter_size, hidden_size]
        bias1_shape = [num_experts, 1, hidden_size]

        dtype = self._helper.get_default_dtype()
        self.bmm_weight0 = self.create_parameter(
            shape=weight0_shape, attr=weight_attr, dtype=dtype, is_bias=False
        )
        self.bmm_bias0 = self.create_parameter(
            shape=bias0_shape, attr=bias_attr, dtype=dtype, is_bias=True
        )
        self.bmm_weight1 = self.create_parameter(
            shape=weight1_shape, attr=weight_attr, dtype=dtype, is_bias=False
        )
        self.bmm_bias1 = self.create_parameter(
            shape=bias1_shape, attr=bias_attr, dtype=dtype, is_bias=True
        )
        self.act_type = act_type
        if self.act_type not in ["gelu", "relu"]:
            raise NotImplementedError("Currently only support `gelu`, `relu`. ")

    def forward(self, x, gate):
        return F.fused_ec_moe(
            x,
            gate,
            self.bmm_weight0,
            self.bmm_bias0,
            self.bmm_weight1,
            self.bmm_bias1,
            self.act_type,
        )
