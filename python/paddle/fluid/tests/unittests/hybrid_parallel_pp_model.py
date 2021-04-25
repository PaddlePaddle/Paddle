# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import division
from __future__ import print_function

import paddle
import numpy as np
import random
import paddle.distributed as dist
import paddle.fluid as fluid
import paddle.distributed.fleet as fleet
from paddle.io import DataLoader, Dataset
import unittest


def set_random_seed(seed, dp_id, rank_id):
    """Set random seed for reproducability."""
    random.seed(seed)
    np.random.seed(seed + dp_id)
    paddle.seed(seed + rank_id)


HIDDEN_DIM = 32
LAYERS = 8


def sequential_model():
    model = paddle.nn.Sequential(
        paddle.nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
        paddle.nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
        paddle.nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
        paddle.nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
        paddle.nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
        paddle.nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
        paddle.nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
        paddle.nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
        paddle.nn.Linear(HIDDEN_DIM, 1), )
    return model


class TestDistPPTraning(unittest.TestCase):
    def setUp(self):
        strategy = fleet.DistributedStrategy()
        self.model_parallel_size = 1
        self.data_parallel_size = 1
        self.pipeline_parallel_size = 2
        strategy.hybrid_configs = {
            "dp_degree": self.data_parallel_size,
            "mp_degree": self.model_parallel_size,
            "pp_degree": self.pipeline_parallel_size,
        }
        strategy.pipeline_configs = {"accumulate_steps": 2}
        paddle.distributed.init_parallel_env()
        fleet.init(is_collective=True, strategy=strategy)

    def test_mp_model(self):
        batch_input = paddle.randn(shape=(1, HIDDEN_DIM), dtype="float32")
        pipe_model = sequential_model()
        sgd = paddle.optimizer.SGD(learning_rate=0.0003, parameters=[])
        pipe_model = paddle.distributed.fleet.distributed_model(pipe_model)

        if pipe_model.stage_id == 0 or pipe_model.stage_id == 1:
            pipe_input = batch_input.clone().detach()
            pipe_input = paddle.cast(pipe_input, 'float32')

            def data_gen():
                gen = True
                while gen:
                    yield [pipe_input, 0]
                    gen = False

            loader = paddle.io.DataLoader.from_generator(capacity=5)
            loader.set_batch_generator(data_gen)
            data_iter = iter(loader)
        else:
            data_iter = None
        return True


if __name__ == "__main__":
    unittest.main()
