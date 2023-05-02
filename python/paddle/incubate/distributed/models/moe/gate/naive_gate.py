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
#
# The file has been adapted from the file:
#     https://github.com/laekov/fastmoe/blob/master/fmoe/gates/naive_gate.py
#     Git commit hash: 295a615aacce7e54a37e7935274ba15e901c78e4
# We retain the following license from the original files:
#     Copyright 2021, Jiaao He. All rights reserved.
#   Licensed under the Apache License, Version 2.0 (the "License").

import paddle
from paddle import nn

from .base_gate import BaseGate


class NaiveGate(BaseGate):
    def __init__(self, d_model, num_expert, world_size, topk=2):
        super().__init__(num_expert, world_size)
        self.gate = nn.Linear(d_model, self.tot_expert)
        self.gate.weight.name = "gate_" + self.gate.weight.name
        self.gate.bias.name = "gate_" + self.gate.bias.name
        self.top_k = topk

    def forward(self, inp, return_all_scores=False):
        gate = self.gate(inp)
        gate_top_k_val, gate_top_k_idx = paddle.topk(
            gate, k=self.top_k, axis=-1, largest=True, sorted=False
        )

        if return_all_scores:
            return gate_top_k_val, gate_top_k_idx, gate
        return gate_top_k_val, gate_top_k_idx
