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


class TestSemiAutoParallelShardOptimizerAPI:
    def __init__(self):
        self._backend = os.getenv("backend")
        self._seed = eval(os.getenv("seed"))
        self._mesh = dist.ProcessMesh([0, 1], dim_names=["x"])
        self._ckpt_path = os.getenv("ckpt_path")

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

    def shard_layer_fn(self, layer_name, layer, process_mesh):
        layer.weight = dist.shard_tensor(
            layer.weight, process_mesh, [dist.Shard(1)]
        )
        layer.bias = dist.shard_tensor(
            layer.bias, process_mesh, [dist.Shard(0)]
        )

    def test_opt(self, opt):
        for key in opt._accumulators.keys():
            for k, v in opt._accumulators[key].items():
                assert opt._accumulators[key][k].is_dist()
                if 'moment' in key:
                    assert (
                        opt._accumulators[key][k].shape[-1]
                        == opt._accumulators[key][k]._local_shape[-1] * 2
                    )
                else:
                    assert opt._accumulators[key][k].shape == [1]
                    assert opt._accumulators[key][k]._local_shape == [1]

    def test_shard_optimizer_mp(self):
        paddle.seed(self._seed)
        linear = paddle.nn.Linear(10, 10)
        dist.shard_layer(linear, self._mesh, self.shard_layer_fn)
        batch = paddle.rand(shape=[10, 10])
        opt = paddle.optimizer.AdamW(parameters=linear.parameters())
        opt = dist.shard_optimizer(opt)
        for _ in range(5):
            loss = linear(batch)
            loss.backward()
            opt.step()
            opt.clear_grad()
        self.test_opt(opt)
        self.check_tensor_eq(self.weight, linear.weight.numpy())
        self.check_tensor_eq(self.bias, linear.bias.numpy())
        # save load
        ckpt_state_dict = opt.state_dict()
        ckpt_state_dict_keys = list(ckpt_state_dict.keys())
        dist.save_state_dict(ckpt_state_dict, self._ckpt_path)
        linear = paddle.nn.Linear(10, 10)
        dist.shard_layer(linear, self._mesh, self.shard_layer_fn)
        new_opt = paddle.optimizer.AdamW(parameters=linear.parameters())
        new_opt = dist.shard_optimizer(new_opt)
        new_state_dict = new_opt.state_dict()
        new_state_dict = {
            ckpt_state_dict_keys[i]: v
            for i, (k, v) in enumerate(new_state_dict.items())
        }
        dist.load_state_dict(new_state_dict, self._ckpt_path)
        assert len(new_state_dict) > 0, "load_state_dict fail"
        for k, v in new_state_dict.items():
            assert k in ckpt_state_dict
            if k in ["master_weights", "LR_Scheduler"]:
                continue
            self.check_tensor_eq(v, ckpt_state_dict[k])

    def test_shard_optimizer_from_non_shard_layer(self):
        paddle.seed(self._seed)
        linear = paddle.nn.Linear(10, 10)
        batch = paddle.rand(shape=[10, 10])
        opt = paddle.optimizer.AdamW(parameters=linear.parameters())
        opt = dist.shard_optimizer(opt)
        for _ in range(5):
            loss = linear(batch)
            loss.backward()
            opt.step()
            opt.clear_grad()
        self.check_tensor_eq(self.weight, linear.weight.numpy())
        self.check_tensor_eq(self.bias, linear.bias.numpy())
        # save load
        ckpt_state_dict = opt.state_dict()
        ckpt_state_dict_keys = list(ckpt_state_dict.keys())
        ckpt_path = os.path.join(
            self._ckpt_path, "test_shard_optimizer_from_non_shard_layer"
        )
        dist.save_state_dict(ckpt_state_dict, ckpt_path)
        linear = paddle.nn.Linear(10, 10)
        new_opt = paddle.optimizer.AdamW(parameters=linear.parameters())
        new_opt = dist.shard_optimizer(new_opt)
        new_state_dict = new_opt.state_dict()
        new_state_dict = {
            ckpt_state_dict_keys[i]: v
            for i, (k, v) in enumerate(new_state_dict.items())
        }
        dist.load_state_dict(new_state_dict, ckpt_path)
        assert len(new_state_dict) > 0, "load_state_dict fail"
        for k, v in new_state_dict.items():
            assert k in ckpt_state_dict
            if k in ["master_weights", "LR_Scheduler"]:
                continue
            self.check_tensor_eq(v, ckpt_state_dict[k])

    def shard_opt_fn(self, accumulator_name, param, accumulator):
        if param.is_dist():
            if 'beta' not in accumulator_name:
                placements = param.placements
            else:
                placements = [
                    dist.Replicate()
                    for _ in range(len(param.process_mesh.shape))
                ]
            return dist.shard_tensor(
                accumulator, param.process_mesh, placements
            )
        return accumulator

    def test_shard_optimizer_shard_fn(self):
        paddle.seed(self._seed)
        linear = paddle.nn.Linear(10, 10)
        dist.shard_layer(linear, self._mesh, self.shard_layer_fn)
        batch = paddle.rand(shape=[10, 10])
        opt = paddle.optimizer.AdamW(parameters=linear.parameters())
        opt = dist.shard_optimizer(opt, self.shard_opt_fn)
        loss = linear(batch)
        loss.backward()
        opt.step()
        opt.clear_grad()
        self.test_opt(opt)

    def test_shard_optimizer_master_params(self):
        paddle.seed(self._seed)
        linear = paddle.nn.Linear(10, 10)
        batch = paddle.rand(shape=[10, 10], dtype="float16")
        linear = paddle.amp.decorate(linear, level="O2", dtype="float16")
        dist.shard_layer(linear, self._mesh, self.shard_layer_fn)
        opt = paddle.optimizer.AdamW(
            parameters=linear.parameters(), multi_precision=True
        )
        opt = dist.shard_optimizer(opt)
        loss = linear(batch)
        loss.backward()
        opt.step()
        self.test_opt(opt)
        for k, v in opt._master_weights.items():
            assert v.dtype == paddle.float32
            assert v.is_dist()
            assert v.shape[-1] == v._local_shape[-1] * 2

        # save load
        ckpt_state_dict = opt.state_dict()
        ckpt_path = os.path.join(
            self._ckpt_path, "test_shard_optimizer_master_params"
        )
        dist.save_state_dict(ckpt_state_dict, ckpt_path)
        paddle.distributed.barrier()
        expected_local_state_dict = {}
        expected_local_state_dict.setdefault("master_weights", {})
        need_load_state_dict = {}
        need_load_state_dict.setdefault("master_weights", {})
        for k, v in ckpt_state_dict.items():
            if k == "LR_Scheduler":
                continue
            elif k == "master_weights":
                assert isinstance(v, dict), v
                for mk, mv in v.items():
                    expected_local_state_dict[k][mk] = mv._local_value().clone()
                    need_load_state_dict[k][mk] = paddle.zeros_like(mv)
            else:
                expected_local_state_dict[k] = v._local_value().clone()
                need_load_state_dict[k] = paddle.zeros_like(v)
        opt.set_state_dict(need_load_state_dict)
        after_set_state_dict = opt.state_dict()
        for k, v in after_set_state_dict.items():
            if k == "master_weights":
                assert isinstance(v, dict), v
                for mk, mv in v.items():
                    assert (
                        mv.numpy().sum() == 0.0
                    ), f"state_dict {k} in master_weights is not zero"
                    assert (
                        need_load_state_dict[k][mk].numpy().sum() == 0.0
                    ), f"state_dict {k} in master_weights is not zero"
            else:
                assert v.numpy().sum() == 0.0, f"state_dict {k} is not zero"
                assert k in need_load_state_dict, f"state_dict {k} is not found"
                assert (
                    need_load_state_dict[k].numpy().sum() == 0.0
                ), f"state_dict {k} is not zero"
        dist.load_state_dict(need_load_state_dict, ckpt_path)
        opt.set_state_dict(need_load_state_dict)
        new_state_dict = opt.state_dict()
        assert "master_weights" in new_state_dict, new_state_dict
        for k, v in new_state_dict.items():
            assert k in expected_local_state_dict
            if k == "master_weights":
                for mk, mv in v.items():
                    np.testing.assert_equal(
                        mv._local_value().numpy(),
                        expected_local_state_dict[k][mk].numpy(),
                    )
            else:
                np.testing.assert_equal(
                    v._local_value().numpy(),
                    expected_local_state_dict[k].numpy(),
                )

    def test_shard_optimizer_params_group(self):
        paddle.seed(self._seed)
        linear = paddle.nn.Linear(10, 10)
        dist.shard_layer(linear, self._mesh, self.shard_layer_fn)
        batch = paddle.rand(shape=[10, 10])
        linear.weight.optimize_attr = {'lr': 1}
        linear.bias.optimize_attr = {'lr': 1}
        params_group = [{'params': linear.weight}, {'params': linear.bias}]
        opt = paddle.optimizer.AdamW(parameters=params_group)
        opt = dist.shard_optimizer(opt)
        loss = linear(batch)
        loss.backward()
        opt.step()
        opt.clear_grad()
        self.test_opt(opt)

    def run_test_case(self):
        if self._backend == "cpu":
            paddle.set_device("cpu")
        elif self._backend == "gpu":
            paddle.set_device("gpu:" + str(dist.get_rank()))
        else:
            raise ValueError("Only support cpu or gpu backend.")

        self.get_single_card_rst()
        self.test_shard_optimizer_params_group()
        self.test_shard_optimizer_shard_fn()
        if self._backend == "gpu":
            self.test_shard_optimizer_master_params()
            self.test_shard_optimizer_mp()
            self.test_shard_optimizer_from_non_shard_layer()


if __name__ == '__main__':
    TestSemiAutoParallelShardOptimizerAPI().run_test_case()
