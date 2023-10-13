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

import unittest

import numpy as np

import paddle
from paddle.distributed import fleet
from paddle.distributed.fleet.utils import mix_precision_utils

vocab_size = 20
hidden_size = 10
inner_size = 8
output_size = 10
seq_length = 2
batch_size = 4


class SimpleMPNet(paddle.nn.Layer):
    def __init__(self, vocab_size, hidden_size, inner_size, output_size):
        super().__init__()

        self.linear1 = fleet.meta_parallel.ColumnParallelLinear(
            hidden_size,
            inner_size,
            gather_output=False,
            has_bias=True,
        )

        self.linear2 = fleet.meta_parallel.RowParallelLinear(
            inner_size,
            hidden_size,
            input_is_parallel=True,
            has_bias=True,
        )

        self.linear3 = paddle.nn.Linear(
            hidden_size,
            output_size,
        )

        self.embedding = fleet.meta_parallel.VocabParallelEmbedding(
            vocab_size,
            hidden_size,
        )

    def forward(self, x):
        x = self.embedding(x)
        x = self.linear1(x)
        x = self.linear2(x)
        x = self.linear3(x)
        return x


def train_batch(batch, model, optimizer):
    output = model(batch)
    loss = output.mean()
    loss.backward()  # do backward
    optimizer.step()  # update parameters
    optimizer.clear_grad()
    return loss


def build_model_optimizer(main_grad):
    model = SimpleMPNet(vocab_size, hidden_size, inner_size, output_size)
    optimizer = paddle.optimizer.SGD(
        learning_rate=0.001, parameters=model.parameters()
    )
    if main_grad:
        model = mix_precision_utils.MixPrecisionLayer(model, dtype="bfloat16")
        optimizer = mix_precision_utils.MixPrecisionOptimizer(optimizer)
    model = fleet.distributed_model(model)
    optimizer = fleet.distributed_optimizer(optimizer)

    return model, optimizer


class TestDistMPTraining(unittest.TestCase):
    def setUp(self):
        strategy = fleet.DistributedStrategy()
        strategy.hybrid_configs = {
            "dp_degree": 1,
            "mp_degree": 2,
            "pp_degree": 1,
        }
        strategy.hybrid_configs["mp_configs"].mp_async_allreduce = True
        fleet.init(is_collective=True, strategy=strategy)

    def test_mp_model(self):
        model, optimizer = build_model_optimizer(False)

        np_data = np.random.randint(
            0,
            vocab_size,
            (
                batch_size,
                seq_length,
            ),
        )
        batch = paddle.to_tensor(np_data)
        train_batch(batch, model, optimizer)


class TestDistMPTrainingWithFusedLinear(unittest.TestCase):
    def setUp(self):
        strategy = fleet.DistributedStrategy()
        strategy.hybrid_configs = {
            "dp_degree": 1,
            "mp_degree": 2,
            "pp_degree": 1,
        }
        strategy.hybrid_configs["mp_configs"].mp_async_allreduce = True
        strategy.hybrid_configs["mp_configs"].mp_skip_c_identity = True
        strategy.hybrid_configs[
            "mp_configs"
        ].mp_fused_linear_param_grad_add = True
        fleet.init(is_collective=True, strategy=strategy)

    def test_mp_model(self):
        model, optimizer = build_model_optimizer(True)

        np_data = np.random.randint(
            0,
            vocab_size,
            (
                batch_size,
                seq_length,
            ),
        )
        batch = paddle.to_tensor(np_data)
        train_batch(batch, model, optimizer)


if __name__ == "__main__":
    unittest.main()
