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

import numpy as np
from hybrid_parallel_mp_model import SimpleMPNet

import paddle
from paddle.distributed import fleet

vocab_size = 20
hidden_size = 10
inner_size = 8
output_size = 10


class TestMPClipCatchError(unittest.TestCase):
    def setUp(self):
        strategy = fleet.DistributedStrategy()
        strategy.hybrid_configs = {
            "dp_degree": 1,
            "mp_degree": 2,
            "pp_degree": 1,
        }
        fleet.init(is_collective=True, strategy=strategy)

    def build_model_optimizer(self):
        hcg = fleet.get_hybrid_communicate_group()
        mp_id = hcg.get_model_parallel_rank()

        np_fc1 = np.random.random_sample((hidden_size, inner_size))
        np_fc2 = np.random.random_sample((inner_size, hidden_size))

        model = SimpleMPNet(
            vocab_size,
            hidden_size,
            inner_size,
            output_size,
            np_fc1,
            np_fc2,
            mp_id,
        )

        grad_clip = paddle.nn.ClipGradByGlobalNorm(0.0)
        optimizer = paddle.optimizer.AdamW(
            learning_rate=0.1,
            parameters=model.parameters(),
            grad_clip=grad_clip,
        )

        return model, optimizer

    def test_clip_catch_error(self):
        _, optimizer = self.build_model_optimizer()

        with self.assertRaises(ValueError):
            optimizer = fleet.distributed_optimizer(optimizer)


if __name__ == '__main__':
    unittest.main()
