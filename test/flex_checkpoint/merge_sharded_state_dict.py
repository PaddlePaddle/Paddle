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

import numpy as np

import paddle.distributed as dist
from paddle.distributed import fleet
from paddle.distributed.fleet.layers.mpu import (
    ColumnParallelLinear,
)
from paddle.nn import Layer


class SimpleMLP(Layer):
    def __init__(self, hidden_size=1024):
        super().__init__()
        self.linear = ColumnParallelLinear(
            hidden_size, hidden_size * 2, has_bias=True
        )
        self.linear1 = ColumnParallelLinear(
            hidden_size, hidden_size * 2, has_bias=True
        )

    def forward(self, x):
        x = self.linear(x)
        x = self.linear1(x)
        return x


class TestDistCheckpoint:
    def __init__(self):
        np.random.seed(42)
        self.temp_dir = "./state_dict_merge"
        self.test_type = os.getenv("test_type")
        self.layer_type = os.getenv("layer_type")
        self.tp_degree = int(os.getenv("tp"))
        self.dp_degree = int(os.getenv("dp"))
        self.world_size = int(os.getenv("world_size"))
        self.has_bias = os.getenv("has_bias", "True").lower() == "true"

        self.hidden_size = 32
        self.vocab_size = 1024

    def run_layer_test(self):
        strategy = fleet.DistributedStrategy()
        strategy.hybrid_configs = {
            "dp_degree": self.dp_degree,
            "mp_degree": self.tp_degree,
            "pp_degree": 1,
        }
        fleet.init(is_collective=True, strategy=strategy)
        hcg = fleet.get_hybrid_communicate_group()
        tp_group = hcg.get_model_parallel_group()

        model_path = os.path.join(self.temp_dir, 'model')
        single_path = os.path.join(self.temp_dir, 'single_model')
        model = SimpleMLP()
        sharded_state_dict = model.sharded_state_dict()
        state_dict = model.state_dict()

        dist.save_state_dict(sharded_state_dict, model_path, safetensors=False)

        dist.flex_checkpoint.dcp.load_state_dict.merge_sharded_state_dict(
            model_path,
            single_path,
            offload=True,
            safetensors=False,
        )
        import safetensors

        load_result = {}
        for i in range(1, 3):
            load_result.update(
                safetensors.paddle.load_file(
                    f"{single_path}/model-0000{i}-of-00002.safetensors"
                )
            )
        assert len(load_result) == 4


if __name__ == '__main__':
    TestDistCheckpoint().run_layer_test()
