# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

import paddle
import paddle.distributed as dist


class TestSemiAutoParallelShardingStage3:
    def __init__(self):
        self._backend = os.getenv("backend")
        self._seed = eval(os.getenv("seed"))
        self._mesh = dist.ProcessMesh([0, 1], dim_names=["x"])

    def check_tensor_eq(self, a, b, rtol=1e-05, atol=0, verbose=True):
        np.testing.assert_allclose(a, b, rtol=rtol, atol=atol, verbose=verbose)

    def get_single_card_rst(self):
        paddle.seed(self._seed)
        linear = paddle.nn.Linear(10, 10)
        batch = paddle.rand(shape=[10, 10])
        opt = paddle.optimizer.AdamW(parameters=linear.parameters())
        for _ in range(5):
            loss = linear(batch)
            loss.backward()
            opt.step()
            opt.clear_grad()
        self.weight = linear.weight.numpy()
        self.bias = linear.bias.numpy()

    def test_pure_sharding_stage_3(self):
        paddle.seed(self._seed)
        linear = paddle.nn.Linear(10, 10)
        batch = paddle.rand(shape=[10, 10])
        # shard the input by sharding degree
        batch = dist.shard_tensor(batch, self._mesh, [dist.Shard(0)])
        # shard optimizer with stage 1 fn
        opt = paddle.optimizer.AdamW(parameters=linear.parameters())
        opt = dist.shard_optimizer(opt, dist.ShardingStage3(self._mesh))
        for _ in range(5):
            loss = linear(batch)
            loss.backward()
            opt.step()
            opt.clear_grad()
        self.check_tensor_eq(self.weight, linear.weight.numpy())
        self.check_tensor_eq(self.bias, linear.bias.numpy())

    def run_test_case(self):
        if self._backend == "cpu":
            paddle.set_device("cpu")
        elif self._backend == "gpu":
            paddle.set_device("gpu:" + str(dist.get_rank()))
        else:
            raise ValueError("Only support cpu or gpu backend.")

        self.get_single_card_rst()
        self.test_pure_sharding_stage_3()


if __name__ == '__main__':
    TestSemiAutoParallelShardingStage3().run_test_case()
