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
from hybrid_parallel_sharding_model import SimpleDPNet

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


class TestDistMPTraning(unittest.TestCase):
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
        fleet.init(is_collective=True, strategy=self.strategy)
        self.data = [
            np.random.randint(
                0,
                vocab_size,
                (
                    batch_size,
                    seq_length,
                ),
            )
            for _ in range(STEPS)
        ]

    def train_batch(self, batch, model, optimizer):
        output = model(batch)
        loss = output.mean()
        loss.backward()  # do backward
        optimizer.step()  # update parameters
        optimizer.clear_grad()
        return loss

    def build_optimizer(self, model, strategy=None, Optimizer="adam"):
        clip = paddle.nn.ClipGradByNorm(0.7)
        param_groups = [
            {
                "params": model.linear1.parameters(),
                "weight_decay": 0.0001,
                "learning_rate": 0.1,
            },
            {
                "params": model.linear2.parameters(),
                "weight_decay": 0.020,
                "learning_rate": 0.01,
            },
            {
                "params": model.linear3.parameters(),
                "weight_decay": 0.1,
                "learning_rate": 0.1,
            },
        ]

        if Optimizer == "adam":
            optimizer = paddle.optimizer.AdamW(
                parameters=param_groups,
                learning_rate=0.001,
                weight_decay=0.00001,
                grad_clip=clip,
            )
        else:
            optimizer = paddle.optimizer.Momentum(
                learning_rate=0.001,
                parameters=model.parameters(),
                grad_clip=clip,
            )
        return optimizer

    def build_model_optimizer(self, Optimizer="adam"):
        np_fc1 = np.random.random_sample((hidden_size, inner_size))
        np_fc2 = np.random.random_sample((inner_size, hidden_size))

        model_a = SimpleDPNet(
            vocab_size, hidden_size, inner_size, output_size, np_fc1, np_fc2
        )
        optimizer_a = self.build_optimizer(
            model_a,
            strategy=self.strategy,
            Optimizer=Optimizer,
        )
        model_a = fleet.distributed_model(model_a)
        optimizer_a = fleet.distributed_optimizer(optimizer_a)

        model_b = SimpleDPNet(
            vocab_size, hidden_size, inner_size, output_size, np_fc1, np_fc2
        )
        optimizer_b = self.build_optimizer(
            model_b,
            strategy=self.strategy,
            Optimizer=Optimizer,
        )

        return model_a, optimizer_a, model_b, optimizer_b

    def sharding_model(self, Optimizer, sharded_accumulators):
        model_a, optimizer_a, model_b, optimizer_b = self.build_model_optimizer(
            Optimizer=Optimizer
        )

        self.assertTrue(
            isinstance(optimizer_a._inner_opt, DygraphShardingOptimizer)
        )

        for idx in range(STEPS):
            if idx > 1:
                self.assertTrue(
                    set(optimizer_a._inner_opt._inner_opt.state_dict().keys())
                    == sharded_accumulators[paddle.distributed.get_rank()]
                )

            if paddle.distributed.get_rank() == 0:
                batch_sharding = paddle.to_tensor(self.data[idx][:2])
            else:
                batch_sharding = paddle.to_tensor(self.data[idx][2:])

            batch_single = paddle.to_tensor(self.data[idx])
            loss_a = self.train_batch(batch_sharding, model_a, optimizer_a)
            loss_b = self.train_batch(batch_single, model_b, optimizer_b)

            np.testing.assert_allclose(loss_a.numpy(), loss_b.numpy())
            for j in range(len(model_a.parameters())):
                np.testing.assert_allclose(
                    model_a.parameters()[j].numpy(),
                    model_b.parameters()[j].numpy(),
                    rtol=1e-6,
                )

    def test_sharding_adam(self):
        sharded_accumulators = [
            {
                'linear_0.b_0_moment1_0',
                'linear_1.b_0_moment1_0',
                'linear_2.w_0_moment1_0',
                'linear_2.b_0_moment1_0',
                'linear_0.b_0_moment2_0',
                'linear_1.b_0_moment2_0',
                'linear_2.w_0_moment2_0',
                'linear_2.b_0_moment2_0',
                'linear_0.b_0_beta1_pow_acc_0',
                'linear_1.b_0_beta1_pow_acc_0',
                'linear_2.w_0_beta1_pow_acc_0',
                'linear_2.b_0_beta1_pow_acc_0',
                'linear_0.b_0_beta2_pow_acc_0',
                'linear_1.b_0_beta2_pow_acc_0',
                'linear_2.w_0_beta2_pow_acc_0',
                'linear_2.b_0_beta2_pow_acc_0',
            },
            {
                'linear_0.w_0_moment1_0',
                'linear_1.w_0_moment1_0',
                'linear_0.w_0_moment2_0',
                'linear_1.w_0_moment2_0',
                'linear_0.w_0_beta1_pow_acc_0',
                'linear_1.w_0_beta1_pow_acc_0',
                'linear_0.w_0_beta2_pow_acc_0',
                'linear_1.w_0_beta2_pow_acc_0',
            },
        ]

        self.sharding_model(
            Optimizer="adam",
            sharded_accumulators=sharded_accumulators,
        )


if __name__ == "__main__":
    unittest.main()
