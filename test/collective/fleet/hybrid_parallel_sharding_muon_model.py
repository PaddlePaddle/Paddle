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
#
# Validates Muon optimizer with sharding v2 (split_param), no tensor parallelism.
# Topology: sharding_degree=2, mp_degree=1 (2 ranks total)

import os
import random
import unittest

import numpy as np

import paddle
from paddle.distributed import fleet
from paddle.distributed.fleet.utils.mix_precision_utils import (
    MixPrecisionLayer,
    MixPrecisionOptimizer,
)

g_shard_split_param = int(os.environ.get("FLAGS_shard_split_param", 0))
g_sharding_v3 = os.environ.get("FLAGS_sharding_v3", "0") == "1"

vocab_size = 20
hidden_size = 100
inner_size = 100
output_size = 100
seq_length = 2
batch_size = 4
STEPS = 3

sharding_degree = 2
mp_degree = 1


class SimpleNet(paddle.nn.Layer):
    """Model used by both distributed (sharding) and reference."""

    def __init__(self, np_fc1, np_fc2, np_fc3, np_emb):
        super().__init__()

        self.embedding = paddle.nn.Embedding(
            vocab_size,
            hidden_size,
            weight_attr=paddle.framework.ParamAttr(
                initializer=paddle.nn.initializer.Assign(np_emb)
            ),
        )

        self.linear1 = paddle.nn.Linear(
            hidden_size,
            inner_size,
            weight_attr=paddle.framework.ParamAttr(
                initializer=paddle.nn.initializer.Assign(np_fc1)
            ),
        )

        self.linear2 = paddle.nn.Linear(
            inner_size,
            hidden_size,
            weight_attr=paddle.framework.ParamAttr(
                initializer=paddle.nn.initializer.Assign(np_fc2)
            ),
        )

        self.linear3 = paddle.nn.Linear(
            hidden_size,
            output_size,
            weight_attr=paddle.framework.ParamAttr(
                initializer=paddle.nn.initializer.Assign(np_fc3)
            ),
            bias_attr=paddle.framework.ParamAttr(
                initializer=paddle.nn.initializer.Constant(0.5)
            ),
        )

    def forward(self, x):
        x = self.embedding(x)
        x = self.linear1(x)
        x = self.linear2(x)
        x = self.linear3(x)
        x = paddle.matmul(x, self.embedding.weight, transpose_y=True)
        return x


class TestDistShardingMuonTraining(unittest.TestCase):
    def setUp(self):
        random.seed(2021)
        np.random.seed(2021)
        paddle.seed(2021)

        self.strategy = fleet.DistributedStrategy()
        self.strategy.hybrid_configs = {
            "sharding_degree": sharding_degree,
            "dp_degree": 1,
            "mp_degree": mp_degree,
            "pp_degree": 1,
        }
        self.strategy.hybrid_configs[
            "sharding_configs"
        ].split_param = g_shard_split_param

        fleet.init(is_collective=True, strategy=self.strategy)
        self.data = [
            np.random.randint(0, vocab_size, (batch_size, seq_length))
            for _ in range(STEPS)
        ]

    def train_batch(self, batch, model, optimizer):
        output = model(batch)
        loss = output.mean()
        loss.backward()
        optimizer.step()
        optimizer.clear_grad()
        return loss

    def build_optimizer(self, model):
        clip = paddle.nn.ClipGradByGlobalNorm(0.5)
        return paddle.optimizer.Muon(
            parameters=model.parameters(),
            learning_rate=0.001,
            weight_decay=0.00001,
            grad_clip=clip,
        )

    def sharding_model(self, amp_level=None):
        np_fc1 = np.random.random_sample((hidden_size, inner_size)).astype(
            "float32"
        )
        np_fc2 = np.random.random_sample((inner_size, hidden_size)).astype(
            "float32"
        )
        np_fc3 = np.random.random_sample((hidden_size, output_size)).astype(
            "float32"
        )
        np_emb = np.random.random_sample((vocab_size, hidden_size)).astype(
            "float32"
        )

        model_a = SimpleNet(np_fc1, np_fc2, np_fc3, np_emb)
        optimizer_a = self.build_optimizer(model_a)

        model_b = SimpleNet(np_fc1, np_fc2, np_fc3, np_emb)
        optimizer_b = self.build_optimizer(model_b)

        if amp_level == "O2":
            model_a = MixPrecisionLayer(model_a)
            optimizer_a = MixPrecisionOptimizer(optimizer_a)
            model_b = MixPrecisionLayer(model_b)
            optimizer_b = MixPrecisionOptimizer(optimizer_b)

        model_a = fleet.distributed_model(model_a)
        optimizer_a = fleet.distributed_optimizer(optimizer_a)

        hcg = fleet.get_hybrid_communicate_group()
        sharding_rank = hcg.get_sharding_parallel_rank()
        local_batch_size = batch_size // sharding_degree

        for idx in range(STEPS):
            start = sharding_rank * local_batch_size
            batch_a = paddle.to_tensor(
                self.data[idx][start : start + local_batch_size]
            )
            batch_b = paddle.to_tensor(self.data[idx])

            loss_a = self.train_batch(batch_a, model_a, optimizer_a)
            loss_b = self.train_batch(batch_b, model_b, optimizer_b)

            for param_a, param_b in zip(
                model_a.parameters(), model_b.parameters()
            ):
                # V3 uses reduce+broadcast (vs V2 allgather), so allow
                # slightly larger numerical tolerance.
                tol = 5e-4 if g_sharding_v3 else 1e-4
                np.testing.assert_allclose(
                    param_a.numpy(),
                    param_b.numpy(),
                    rtol=tol,
                    atol=tol,
                    err_msg=f"Param {param_a.name} mismatch at step {idx}!",
                )

    @unittest.skipIf(
        not paddle.is_compiled_with_cuda()
        or paddle.device.cuda.get_device_capability()[0] < 8,
        "BF16 matmul requires GPU compute capability >= 80 (Ampere+)",
    )
    def test_sharding_muon(self):
        if g_shard_split_param:
            self.sharding_model(amp_level="O2")


if __name__ == "__main__":
    unittest.main()
