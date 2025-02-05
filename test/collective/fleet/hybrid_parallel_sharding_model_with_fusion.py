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

import random
import unittest

import numpy as np

import paddle
from paddle.distributed import fleet
from paddle.distributed.fleet.meta_optimizers.dygraph_optimizer.dygraph_sharding_optimizer import (
    DygraphShardingOptimizer,
)

vocab_size = 20
hidden_size = 10
inner_size = 8
output_size = 10
seq_length = 2
batch_size = 4
STEPS = 10


class SimpleDPNet(paddle.nn.Layer):
    def __init__(
        self, vocab_size, hidden_size, inner_size, output_size, np_fc1, np_fc2
    ):
        super().__init__()
        self.linear1 = paddle.nn.Linear(
            hidden_size,
            inner_size,
            weight_attr=paddle.framework.ParamAttr(
                initializer=paddle.nn.initializer.Assign(np_fc1)
            ),
            bias_attr=paddle.framework.ParamAttr(
                initializer=paddle.nn.initializer.Constant(0.0)
            ),
        )

        self.linear2 = paddle.nn.Linear(
            inner_size,
            hidden_size,
            weight_attr=paddle.framework.ParamAttr(
                initializer=paddle.nn.initializer.Assign(np_fc2)
            ),
            bias_attr=paddle.framework.ParamAttr(
                initializer=paddle.nn.initializer.Constant(0.0)
            ),
        )

        self.linear3 = paddle.nn.Linear(
            hidden_size,
            output_size,
            weight_attr=paddle.framework.ParamAttr(
                initializer=paddle.nn.initializer.Constant(0.0)
            ),
            bias_attr=paddle.framework.ParamAttr(
                initializer=paddle.nn.initializer.Constant(0.0)
            ),
        )

        self.embedding = paddle.nn.Embedding(
            vocab_size,
            hidden_size,
            weight_attr=paddle.nn.initializer.Constant(value=0.5),
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
        random.seed(2021)
        np.random.seed(2021)
        paddle.seed(2021)

        self.strategy = fleet.DistributedStrategy()
        self.strategy.hybrid_configs = {
            "sharding_degree": 2,
            "dp_degree": 1,
            "mp_degree": 1,
            "pp_degree": 1,
        }
        self.strategy.hybrid_configs["sharding_configs"].tensor_fusion = True
        self.strategy.hybrid_configs["sharding_configs"].comm_overlap = True
        self.strategy.hybrid_configs["sharding_configs"].accumulate_steps = 1
        self.strategy.hybrid_configs["sharding_configs"].fuse_optimizer = False
        fleet.init(is_collective=True, strategy=self.strategy)
        self.data = np.random.randint(
            0,
            vocab_size,
            (
                batch_size,
                seq_length,
            ),
        )

        if paddle.distributed.get_rank() == 0:
            self.batch_sharding = paddle.to_tensor(self.data[:2])
        else:
            self.batch_sharding = paddle.to_tensor(self.data[2:])

        self.batch_single = paddle.to_tensor(self.data)

    def train_batch(self, batch, model, optimizer):
        output = model(batch)
        loss = output.mean()
        loss.backward()
        optimizer.step()
        optimizer.clear_grad()
        return loss

    def build_optimizer(self, model):
        clip = paddle.nn.ClipGradByGlobalNorm(0.5)
        optimizer = paddle.optimizer.AdamW(
            parameters=model.parameters(),
            learning_rate=0.001,
            weight_decay=0.001,
            grad_clip=clip,
        )
        return optimizer

    def build_model_optimizer(self):
        np_fc1 = np.random.random_sample((hidden_size, inner_size))
        np_fc2 = np.random.random_sample((inner_size, hidden_size))

        model_a = SimpleDPNet(
            vocab_size, hidden_size, inner_size, output_size, np_fc1, np_fc2
        )
        optimizer_a = self.build_optimizer(model_a)

        model_b = SimpleDPNet(
            vocab_size, hidden_size, inner_size, output_size, np_fc1, np_fc2
        )
        optimizer_b = self.build_optimizer(model_b)

        model_a = fleet.distributed_model(model_a)
        optimizer_a = fleet.distributed_optimizer(optimizer_a)

        return model_a, optimizer_a, model_b, optimizer_b

    def sharding_model(self):
        (
            model_a,
            optimizer_a,
            model_b,
            optimizer_b,
        ) = self.build_model_optimizer()

        self.assertTrue(
            isinstance(optimizer_a._inner_opt, DygraphShardingOptimizer)
        )

        for idx in range(STEPS):
            loss_a = self.train_batch(self.batch_sharding, model_a, optimizer_a)
            loss_b = self.train_batch(self.batch_single, model_b, optimizer_b)
            np.testing.assert_allclose(loss_a, loss_b, rtol=1e-6, atol=1e-6)

            for j in range(len(model_a.parameters())):
                np.testing.assert_allclose(
                    model_a.parameters()[j].numpy(),
                    model_b.parameters()[j].numpy(),
                    rtol=1e-6,
                    atol=1e-7,
                )

    def test_sharding_adam(self):
        self.sharding_model()


if __name__ == "__main__":
    unittest.main()
