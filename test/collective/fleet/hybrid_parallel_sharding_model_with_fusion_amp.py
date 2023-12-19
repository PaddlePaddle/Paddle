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

import paddle
from paddle.distributed import fleet

vocab_size = 20
hidden_size = 10
inner_size = 8
output_size = 10
seq_length = 2
batch_size = 4
STEPS = 10


class SimpleDPNet(paddle.nn.Layer):
    def __init__(self, vocab_size, hidden_size, inner_size, output_size):
        super().__init__()
        self.linear1 = paddle.nn.Linear(hidden_size, inner_size)

        self.linear2 = paddle.nn.Linear(inner_size, hidden_size)

        self.linear3 = paddle.nn.Linear(hidden_size, output_size)

        self.embedding = paddle.nn.Embedding(vocab_size, hidden_size)

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
            "sharding_degree": 2,
            "dp_degree": 1,
            "mp_degree": 1,
            "pp_degree": 1,
        }
        self.strategy.hybrid_configs["sharding_configs"].tensor_fusion = True
        self.strategy.hybrid_configs["sharding_configs"].comm_overlap = True
        self.strategy.hybrid_configs["sharding_configs"].accumulate_steps = 1
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

    def build_optimizer(self, model):
        clip = paddle.nn.ClipGradByGlobalNorm(0.5)
        optimizer = paddle.optimizer.AdamW(
            parameters=model.parameters(),
            learning_rate=0.001,
            weight_decay=0.001,
            grad_clip=clip,
        )
        return optimizer

    def build_model_optimizer(self, diff_lr=False):
        model = SimpleDPNet(vocab_size, hidden_size, inner_size, output_size)
        optimizer = self.build_optimizer(model)
        model, optimizer = paddle.amp.decorate(
            model, optimizers=optimizer, level="O2", dtype="float16"
        )
        if diff_lr:
            for param in model.parameters():
                if 'w' in param.name:
                    param.optimize_attr = {"learning_rate": 1.0}
                else:
                    param.optimize_attr = {"learning_rate": 2.0}
        scaler = paddle.amp.GradScaler(init_loss_scaling=1024)
        scaler = fleet.distributed_scaler(scaler)
        model = fleet.distributed_model(model)
        optimizer = fleet.distributed_optimizer(optimizer)
        return model, optimizer, scaler

    def sharding_model(self):
        model, optimizer, scaler = self.build_model_optimizer()

        for idx in range(STEPS):
            with paddle.amp.auto_cast(enable=True, level='O2'):
                output = model(self.batch_sharding)
            loss = output.mean()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.clear_grad()

    def sharding_different_lr(self):
        model, optimizer, scaler = self.build_model_optimizer(diff_lr=True)
        assert optimizer._inner_opt.fuse_optimizer is False

    def test_sharding_adam(self):
        self.sharding_model()
        self.sharding_different_lr()


if __name__ == "__main__":
    unittest.main()
