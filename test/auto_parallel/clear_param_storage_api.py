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

import unittest

import paddle
from paddle.distributed import fleet
from paddle.distributed.fleet.meta_optimizers.dygraph_optimizer.dygraph_sharding_optimizer import (
    DygraphShardingOptimizerV2,
)


class TestClearParamStorage(unittest.TestCase):
    def test_clear_param_storage(self):
        class TestLayer(paddle.nn.Layer):
            def __init__(self, dtype):
                super().__init__()
                self._w = self.create_parameter([2, 3], dtype=dtype)
                self._b = self.create_parameter([2, 3], dtype=dtype)
                self._w.color = {"color": "_w"}
                self._b.color = {"color": "_b"}

            @paddle.amp.debugging.check_layer_numerics
            def forward(self, x):
                return x * self._w + self._b

        strategy = fleet.DistributedStrategy()
        strategy.hybrid_configs = {
            "dp_degree": 1,
            "mp_degree": 1,
            "pp_degree": 1,
            "sharding_degree": 2,
        }
        fleet.init(is_collective=True, strategy=strategy)
        hcg = fleet.get_hybrid_communicate_group()
        dtype = 'float32'
        model = TestLayer(dtype)

        optimizer = paddle.optimizer.AdamW(parameters=model.parameters())
        optimizer = DygraphShardingOptimizerV2(optimizer, hcg)
        optimizer.clear_param_storage("_w")
        optimizer.clear_param_storage("_b")
        optimizer.clear_param_storage(None)
        optimizer.reset_param_storage()


if __name__ == '__main__':
    unittest.main()
