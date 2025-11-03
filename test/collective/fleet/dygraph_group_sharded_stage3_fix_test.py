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

import numpy as np
from dist_amp_base import create_optimizer

import paddle
import paddle.distributed as dist
import paddle.nn.functional as F
from paddle import nn
from paddle.distributed import fleet
from paddle.distributed.fleet.meta_parallel.sharding.group_sharded_stage3 import (
    GroupShardedStage3,
)


class RandomDataset(paddle.io.Dataset):
    def __init__(self, num_samples=2000, shape=(4, 8, 16)):
        self.num_samples = num_samples
        self.shape = shape

    def __getitem__(self, idx):
        img = np.random.rand(*self.shape).astype('float32')
        label = np.ones(1).astype('int64')
        return img, label

    def __len__(self):
        return self.num_samples


def train_step(model, use_pure_bf16=False, use_main_grad=False):
    optimizer = create_optimizer(
        model=model, use_pure_bf16=use_pure_bf16, use_main_grad=use_main_grad
    )
    hcg = fleet.get_hybrid_communicate_group()
    group = hcg.get_sharding_parallel_group()
    model = GroupShardedStage3(model, optimizer, group=group)
    local_rank = paddle.distributed.get_rank()
    epoch = 1
    batch_size = 500
    paddle.seed(2025)
    np.random.seed(2025)
    train_loader = paddle.io.DataLoader(
        RandomDataset(),
        batch_size=batch_size,
        shuffle=False,
        drop_last=True,
        num_workers=0,
    )
    for eop in range(epoch):
        model.train()
        for batch_id, data in enumerate(train_loader()):
            print("<<<<<<<<<<<< forward >>>>>>>>>>>")
            print(
                f"-- [rank={local_rank}] epoch {eop}, batch {batch_id}, {data[0].shape=}"
            )
            score, out = model(data[0])
            print(f"after forward, {score=}, {out.shape=}")

            loss = out.mean()

            print(
                f"-- [rank={local_rank}] epoch {eop}, batch {batch_id}, loss: {loss.astype(paddle.float32).numpy()}"
            )
            print("<<<<<<<<<<<< backward >>>>>>>>>>>")
            loss.backward()
            print("<<<<<<<<<<<< optimizer >>>>>>>>>>>")
            optimizer.step()


class MulLinear(nn.Layer):
    def __init__(self, input_dim, output_dim, scale=1.0):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, output_dim)
        self.linear2 = nn.Linear(input_dim, output_dim)
        self.scale1 = self.create_parameter(
            shape=[1], default_initializer=nn.initializer.Constant(scale)
        )
        self.scale2 = self.create_parameter(
            shape=[1], default_initializer=nn.initializer.Constant(1.0 - scale)
        )

    def forward(self, x):
        out1 = self.linear1(x)
        out2 = self.linear2(x)
        output1 = self.scale1 * out1
        output2 = self.scale2 * out2
        score1 = output1.mean()
        score2 = output2.mean()
        combined = paddle.stack([output1, output2], axis=0)
        combined.stop_gradient = True
        return score1.item(), score2.item(), output1, output2, combined


class MyModel(nn.Layer):
    def __init__(self, input_dim, hidden_dim, output_dim, scale):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.mullinear = MulLinear(hidden_dim, hidden_dim, scale)
        self.linear2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, input):
        hidden_states = self.linear1(input)
        hidden_states = F.relu(hidden_states)
        (
            score1,
            score2,
            hidden_states1,
            hidden_states2,
            combined_hidden_states,
        ) = self.mullinear(hidden_states)
        final_score = score1 + score2
        w1 = score1 / final_score
        w2 = score2 / final_score
        hidden_states = w1 * hidden_states1 + w2 * hidden_states2
        hidden_states = F.relu(hidden_states)
        output = self.linear2(hidden_states)
        return final_score, output


class TestStage3Bugfix(unittest.TestCase):
    def setUp(self):
        strategy = fleet.DistributedStrategy()
        self.model_parallel_size = 1
        self.data_parallel_size = 1
        self.pipeline_parallel_size = 1
        self.sharding_parallel_size = 2
        strategy.hybrid_configs = {
            "dp_degree": self.data_parallel_size,
            "mp_degree": self.model_parallel_size,
            "pp_degree": self.pipeline_parallel_size,
            "sharding_degree": self.sharding_parallel_size,
        }
        fleet.init(is_collective=True, strategy=strategy)

    def test_stage3(self):
        b, s, h = 4, 8, 16
        model = MyModel(input_dim=h, hidden_dim=32, output_dim=h, scale=0.4)
        dist.init_parallel_env()
        train_step(model)


if __name__ == "__main__":
    unittest.main()
