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


class FusedMoe(Layer):
    r"""A FusedMoe Layer.

    Parameters:

    Attribute:

    Shape:

    Examples:

    """

    def __init__(
        self,
        hidden_size,
        inter_size,
        num_experts,
        act_type=None,
        weight_attr=None,
        bias_attr=None,
        moe_topk=2,
    ):
        super().__init__()
        gate_shape = [hidden_size, num_experts]
        weight0_shape = [num_experts, hidden_size, inter_size]
        bias0_shape = [num_experts, 1, inter_size]
        weight1_shape = [num_experts, inter_size, hidden_size]
        bias1_shape = [num_experts, 1, hidden_size]

        dtype = self._helper.get_default_dtype()
        self.gate = self.create_parameter(
            shape=gate_shape, attr=weight_attr, dtype=dtype, is_bias=False
        )
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

    def forward(self, x):
        return F.fused_moe(
            x,
            self.gate,
            self.bmm_weight0,
            self.bmm_bias0,
            self.bmm_weight1,
            self.bmm_bias1,
            None,
            2,
        )
