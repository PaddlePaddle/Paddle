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

import os

import paddle
import paddle.distributed as dist


class TestSemiAutoParallelShardingStage123:
    def __init__(self):
        self._backend = os.getenv("backend")
        self._seed = eval(os.getenv("seed"))
        self._mesh = dist.ProcessMesh([0, 1], dim_names=["x"])

    def test_pure_sharding_stage_1(self):
        paddle.seed(self._seed)
        linear = paddle.nn.Linear(10, 10)
        batch = paddle.rand(shape=[10, 10])
        # shard the input by sharding degree
        batch = dist.shard_tensor(batch, self._mesh, [dist.Shard(0)])
        # shard optimizer with stage 1 fn
        opt = paddle.optimizer.AdamW(parameters=linear.parameters())
        linear, opt = paddle.amp.decorate(
            linear, optimizers=opt, level='O2', master_grad=True
        )
        opt = dist.shard_optimizer(opt, dist.ShardingStage1("x", self._mesh))
        stage_losses = []
        for _ in range(5):
            with paddle.amp.auto_cast(level='O2'):
                loss = linear(batch)
                loss.backward()
                opt.step()
                opt.clear_grad()
                stage_losses.append(loss._md5sum())
        return stage_losses

    def test_pure_sharding_stage_2(self):
        paddle.seed(self._seed)
        linear = paddle.nn.Linear(10, 10)
        batch = paddle.rand(shape=[10, 10])
        # shard the input by sharding degree
        batch = dist.shard_tensor(batch, self._mesh, [dist.Shard(0)])
        # shard optimizer with stage 2 fn
        opt = paddle.optimizer.AdamW(parameters=linear.parameters())
        linear, opt = paddle.amp.decorate(
            linear, optimizers=opt, level='O2', master_grad=True
        )
        opt = dist.shard_optimizer(opt, dist.ShardingStage2("x", self._mesh))
        stage_losses = []
        for _ in range(5):
            with paddle.amp.auto_cast(level='O2'):
                loss = linear(batch)
                loss.backward()
                opt.step()
                opt.clear_grad()
                stage_losses.append(loss._md5sum())
        return stage_losses

    def test_pure_sharding_stage_3(self):
        paddle.seed(self._seed)
        linear = paddle.nn.Linear(10, 10)
        batch = paddle.rand(shape=[10, 10])
        # shard the input by sharding degree
        batch = dist.shard_tensor(batch, self._mesh, [dist.Shard(0)])
        # shard optimizer with stage 3 fn
        opt = paddle.optimizer.AdamW(parameters=linear.parameters())
        linear, opt = paddle.amp.decorate(
            linear, optimizers=opt, level='O2', master_grad=True
        )
        opt = dist.shard_optimizer(opt, dist.ShardingStage3("x", self._mesh))
        stage_losses = []
        for _ in range(5):
            with paddle.amp.auto_cast(level='O2'):
                loss = linear(batch)
                loss.backward()
                opt.step()
                opt.clear_grad()
                stage_losses.append(loss._md5sum())
        os.environ["skip_sharding3_output_reshard"] = "0"
        return stage_losses

    def run_test_case(self):
        if self._backend == "cpu":
            paddle.set_device("cpu")
        elif self._backend == "gpu":
            paddle.set_device("gpu:" + str(dist.get_rank()))
        else:
            raise ValueError("Only support cpu or gpu backend.")

        stage_losses1 = self.test_pure_sharding_stage_1()
        stage_losses2 = self.test_pure_sharding_stage_2()
        stage_losses3 = self.test_pure_sharding_stage_3()
        assert stage_losses3 == stage_losses2
        assert stage_losses2 == stage_losses1


if __name__ == '__main__':
    TestSemiAutoParallelShardingStage123().run_test_case()
