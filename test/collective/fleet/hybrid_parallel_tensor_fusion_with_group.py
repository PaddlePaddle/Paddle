# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

import unittest

import paddle
from paddle.distributed import fleet
from paddle.distributed.fleet.utils.tensor_fusion_helper import (
    HOOK_ACTION,
    fused_parameters,
)


class SimpleDPNet(paddle.nn.Layer):
    def __init__(self, vocab_size, hidden_size, inner_size, output_size):
        super().__init__()
        self.linear1 = paddle.nn.Linear(
            hidden_size,
            inner_size,
        )

        self.linear2 = paddle.nn.Linear(
            inner_size,
            hidden_size,
        )

        self.linear3 = paddle.nn.Linear(
            hidden_size,
            output_size,
        )

        self.embedding = paddle.nn.Embedding(
            vocab_size,
            hidden_size,
        )

    def forward(self, x):
        x = self.embedding(x)
        x = self.linear1(x)
        x = self.linear2(x)
        x = self.linear3(x)
        x = paddle.matmul(x, self.embedding.weight, transpose_y=True)
        return x


class TestDistSharding(unittest.TestCase):
    def setUp(self):
        self.strategy = fleet.DistributedStrategy()
        self.strategy.hybrid_configs = {
            "sharding_degree": 1,
            "dp_degree": 2,
            "mp_degree": 1,
            "pp_degree": 1,
        }
        fleet.init(is_collective=True, strategy=self.strategy)

    def test_fusion(self):
        model = SimpleDPNet(20, 10, 8, 10)
        parameters = model.parameters()
        parameters[0].optimize_attr = {'lr': 1}
        param_group = [{'params': parameters}, {'params': parameters}]
        fused_parameters(
            param_group,
            act=HOOK_ACTION.ALL_REDUCE,
            comm_overlap=True,
            group_params=True,
        )


if __name__ == "__main__":
    unittest.main()
