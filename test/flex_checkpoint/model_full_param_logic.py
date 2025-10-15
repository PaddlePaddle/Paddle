# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

import os

import paddle
from paddle import nn
from paddle.distributed import fleet
from paddle.distributed.fleet.layers.mpu import (
    ColumnParallelLinear,
    RowParallelLinear,
    VocabParallelEmbedding,
)


class SimpleMLP(nn.Layer):
    def __init__(self, hidden_size=100, has_bias=False):
        super().__init__()
        self.embedding = VocabParallelEmbedding(24, hidden_size)
        self.linear1 = ColumnParallelLinear(
            hidden_size, hidden_size, gather_output=False, has_bias=has_bias
        )
        self.linear2 = RowParallelLinear(
            hidden_size, hidden_size, input_is_parallel=True, has_bias=has_bias
        )
        self.llm_head = self.embedding

    def forward(self, x):
        x = self.embedding(x)
        x = self.linear1(x)
        x = self.linear2(x)
        x = paddle.matmul(x, self.llm_head.weight, transpose_y=True)
        return x


class TestFullParamLogic:
    def __init__(self):
        self.tp_degree = int(os.getenv("tp", "1"))
        self.dp_degree = int(os.getenv("dp", "1"))
        self.sharding_degree = int(os.getenv("sharding_degree", "1"))
        self.world_size = int(os.getenv("world_size"))
        self.has_bias = os.getenv("has_bias", "True").lower() == "true"
        self.batch_size = 2
        self.hidden_size = 32
        self.vocab_size = 24
        self.seq_len = 2
        self.hcg = None

    def run_test(self):
        strategy = fleet.DistributedStrategy()
        strategy.hybrid_configs = {
            "dp_degree": self.dp_degree,
            "mp_degree": self.tp_degree,
            "sharding_degree": self.sharding_degree,
            "pp_degree": 1,
        }
        fleet.init(is_collective=True, strategy=strategy)
        self.run_full_param_test()
        self.run_full_param_with_aoa_test()

    def run_full_param_test(self):
        model = SimpleMLP(hidden_size=self.hidden_size, has_bias=self.has_bias)
        model = fleet.distributed_model(model)
        model.train()
        model_state_dict = model.state_dict()

        for k, v in model_state_dict.items():
            ones = paddle.ones_like(v)
            paddle.assign(ones, v)

        full_param_iter = model.full()
        full_param = dict(full_param_iter)

        param_shape = {
            "_layers.embedding.weight": [24, 32],
            "_layers.linear1.weight": [32, 32],
            "_layers.linear1.bias": [32],
            "_layers.linear2.weight": [32, 32],
            "_layers.linear2.bias": [32],
            "_layers.llm_head.weight": [24, 32],
        }
        for name, shape in param_shape.items():
            if not self.has_bias:
                if ".bias" in name:
                    continue
            assert name in full_param.keys()
            tensor = full_param[name]
            answer = paddle.ones_like(tensor)
            assert tensor._md5sum() == answer._md5sum()

    def run_full_param_with_aoa_test(self):
        model = SimpleMLP(hidden_size=self.hidden_size, has_bias=self.has_bias)
        model = paddle.amp.decorate(
            models=model, optimizers=None, level="O2", dtype="float16"
        )
        model = fleet.distributed_model(model)
        model.train()
        model_state_dict = model.state_dict()

        for k, v in model_state_dict.items():
            ones = paddle.ones_like(v)
            paddle.assign(ones, v)
            if k == "_layers.linear1.weight":
                zeros = paddle.zeros_like(v)
                paddle.assign(zeros, v)

        aoa_config = {
            "aoa_statements": [
                "_layers.linear1.weight, _layers.linear2.weight -> _layers.fused_weight, axis=1"
            ]
        }

        full_param_iter = model.full(aoa_config, None)
        full_param = dict(full_param_iter)

        param_shape = {
            # "_layers.linear1.weight" : [32,32],
            # "_layers.linear2.weight" : [32, 32],
            "_layers.embedding.weight": [24, 32],
            "_layers.linear1.bias": [32],
            "_layers.linear2.bias": [32],
            "_layers.llm_head.weight": [24, 32],
            "_layers.fused_weight": [32, 64],
        }

        for name, shape in param_shape.items():
            if name == "_layers.fused_weight":
                continue
            if not self.has_bias:
                if ".bias" in name:
                    continue
            assert name in full_param.keys()
            tensor = full_param[name]
            answer = paddle.ones_like(tensor)
            assert tensor._md5sum() == answer._md5sum()

        assert "_layers.fused_weight" in full_param.keys()
        ones = paddle.ones([32, 32], 'float16')
        zeros = paddle.zeros([32, 32], 'float16')
        answer = paddle.concat([zeros, ones], axis=1)
        assert full_param["_layers.fused_weight"]._md5sum() == answer._md5sum()


if __name__ == '__main__':
    TestFullParamLogic().run_test()
