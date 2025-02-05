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

import os

import numpy as np

import paddle
import paddle.distributed as dist


class TestSemiAutoParallelShardOptimizer:
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

    def test_adamw_dp(self):
        paddle.seed(self._seed)
        linear = paddle.nn.Linear(10, 10)
        batch = paddle.rand(shape=[10, 10])
        batch = dist.shard_tensor(batch, self._mesh, [dist.Shard(0)])
        opt = paddle.optimizer.AdamW(parameters=linear.parameters())
        for _ in range(5):
            loss = linear(batch)
            loss.backward()
            opt.step()
            opt.clear_grad()
        assert linear.bias.is_dist()
        assert linear.weight.is_dist()
        self.check_tensor_eq(self.weight, linear.weight.numpy())
        self.check_tensor_eq(self.bias, linear.bias.numpy())

    def shard_fn(self, layer_name, layer, process_mesh):
        layer.weight = dist.shard_tensor(
            layer.weight, process_mesh, [dist.Shard(1)]
        )
        layer.bias = dist.shard_tensor(
            layer.bias, process_mesh, [dist.Shard(0)]
        )

    def test_adamw_mp(self):
        paddle.seed(self._seed)
        linear = paddle.nn.Linear(10, 10)
        dist.shard_layer(linear, self._mesh, self.shard_fn)
        batch = paddle.rand(shape=[10, 10])
        opt = paddle.optimizer.AdamW(parameters=linear.parameters())
        for _ in range(5):
            loss = linear(batch)
            loss.backward()
            opt.step()
            opt.clear_grad()
        for key in opt._accumulators.keys():
            for k, v in opt._accumulators[key].items():
                if 'moment' in key:
                    assert opt._accumulators[key][k].is_dist()
                    assert (
                        opt._accumulators[key][k].shape[-1]
                        == opt._accumulators[key][k]._local_shape[-1] * 2
                    )
        self.check_tensor_eq(self.weight, linear.weight.numpy())
        self.check_tensor_eq(self.bias, linear.bias.numpy())

    def test_adamw_shard_optimizer(self, stage1=False):
        paddle.seed(self._seed)
        linear = paddle.nn.Linear(10, 10)
        batch = paddle.rand(shape=[10, 10])
        if stage1:
            batch = dist.shard_tensor(batch, self._mesh, [dist.Shard(0)])
        opt = paddle.optimizer.AdamW(parameters=linear.parameters())
        opt.helper = paddle.base.layer_helper.LayerHelper(
            opt.__class__.__name__
        )
        opt._create_accumulators(
            paddle.base.framework.default_main_program().global_block(),
            [linear.weight, linear.bias],
        )
        for key in opt._accumulators.keys():
            for k, v in opt._accumulators[key].items():
                if 'beta' in key:
                    opt._accumulators[key][k] = dist.shard_tensor(
                        v, self._mesh, [dist.Replicate()]
                    )
                else:
                    opt._accumulators[key][k] = dist.shard_tensor(
                        v, self._mesh, [dist.Shard(0)]
                    )
        for _ in range(5):
            loss = linear(batch)
            loss.backward()
            opt.step()
            opt.clear_grad()
        assert linear.bias.is_dist()
        assert linear.weight.is_dist()
        assert linear.bias.shape == [10]
        assert linear.weight.shape == [10, 10]
        assert linear.bias._local_shape == [5]
        assert linear.weight._local_shape == [5, 10]
        for k, v in opt._master_weights.items():
            assert v.is_dist()
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
        self.test_adamw_dp()
        if self._backend == "gpu":
            self.test_adamw_mp()
            self.test_adamw_shard_optimizer(stage1=True)
            self.test_adamw_shard_optimizer(stage1=False)


if __name__ == '__main__':
    TestSemiAutoParallelShardOptimizer().run_test_case()
