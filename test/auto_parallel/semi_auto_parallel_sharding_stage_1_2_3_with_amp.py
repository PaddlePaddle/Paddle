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
        self.gradient_accumulation_steps = 2

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
        tr_loss_add = float(0)
        for step in range(6):
            with paddle.amp.auto_cast(level='O2'):
                tr_loss = linear(batch)
                tr_loss.backward()
                tr_loss_add += tr_loss
                if (step + 1) % self.gradient_accumulation_steps == 0:
                    tr_loss_add /= self.gradient_accumulation_steps
                    tr_loss = tr_loss_add
                    opt.step()
                    opt.clear_grad()
                stage_losses.append(tr_loss._md5sum())
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
        tr_loss_add = float(0)
        for step in range(6):
            with paddle.amp.auto_cast(level='O2'):
                tr_loss = linear(batch)
                tr_loss.backward()
                tr_loss_add += tr_loss
                if (step + 1) % self.gradient_accumulation_steps == 0:
                    tr_loss_add /= self.gradient_accumulation_steps
                    tr_loss = tr_loss_add
                    opt.step()
                    opt.clear_grad()
                stage_losses.append(tr_loss._md5sum())
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
        tr_loss_add = float(0)
        for step in range(6):
            with paddle.amp.auto_cast(level='O2'):
                tr_loss = linear(batch)
                tr_loss.backward()
                tr_loss_add += tr_loss
                if (step + 1) % self.gradient_accumulation_steps == 0:
                    tr_loss_add /= self.gradient_accumulation_steps
                    tr_loss = tr_loss_add
                    opt.step()
                    opt.clear_grad()
                stage_losses.append(tr_loss._md5sum())
        os.environ["skip_sharding3_output_reshard"] = "0"
        return stage_losses

    def run_test_case(self):
        if self._backend == "cpu":
            paddle.set_device("cpu")
        elif self._backend == "gpu":
            paddle.set_device("gpu:" + str(dist.get_rank()))
        else:
            raise ValueError("Only support cpu or gpu backend.")

        losses_stage_1 = self.test_pure_sharding_stage_1()
        losses_stage_2 = self.test_pure_sharding_stage_2()
        losses_stage_3 = self.test_pure_sharding_stage_3()
        assert losses_stage_1 == losses_stage_2
        assert losses_stage_2 == losses_stage_3


if __name__ == '__main__':
    TestSemiAutoParallelShardingStage123().run_test_case()
