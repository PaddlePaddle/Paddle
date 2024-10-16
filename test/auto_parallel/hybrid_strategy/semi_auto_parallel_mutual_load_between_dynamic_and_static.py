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
import random

import numpy as np
from auto_parallel.semi_auto_parallel_dist_to_static_mlp import RandomDataset
from auto_parallel.semi_auto_parallel_simple_net import (
    BATCH_SIZE,
    CLASS_NUM,
    IMAGE_SIZE,
    DemoNet,
    TestSimpleNetForSemiAutoParallel,
)

import paddle
import paddle.distributed as dist
from paddle import nn
from paddle.io import DataLoader


class TestSemiAutoParallelMutualLoadBetweenDynamicAndStatic(
    TestSimpleNetForSemiAutoParallel
):
    def __init__(self):
        self._ckpt_path = os.environ.get("ckpt_path")
        self._seed = os.environ.get("seed", 123)
        self.mesh = dist.ProcessMesh([0, 1], dim_names=["x"])

    def create_data_loader(self):
        images = np.random.rand(BATCH_SIZE, IMAGE_SIZE).astype('float32')
        labels = np.random.rand(BATCH_SIZE, CLASS_NUM).astype('float32')
        dataset = RandomDataset(images, labels, BATCH_SIZE)
        loader = DataLoader(dataset, batch_size=BATCH_SIZE)
        return loader

    def run_dynamic(self, layer, opt, data_loader, is_recompute=False):
        loss_fn = nn.MSELoss()

        loss_list = []
        for _ in range(5):
            for batch_id, (image, label) in enumerate(data_loader()):
                if is_recompute:
                    image.stop_gradient = False
                out = layer(image)
                loss = loss_fn(out, label)
                loss_list.append(loss.numpy())
                loss.backward()

                opt.step()
                opt.clear_grad()
        return np.array(loss_list)

    def run_dy2static(self, layer, opt, data_loader):
        # create loss
        loss_fn = nn.MSELoss()
        dist_loader = dist.shard_dataloader(
            dataloader=data_loader,
            meshes=[self.mesh],
            shard_dims=None,
        )
        # static training
        dist_model = dist.to_static(layer, dist_loader, loss_fn, opt)
        loss_list = []
        dist_model.train()
        for epoch in range(5):
            for batch_id, (image, label) in enumerate(dist_loader()):
                loss = dist_model(image, label)
                loss_list.append(loss)

        return np.array(loss_list), dist_model

    def set_random_seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        paddle.seed(seed)

    def test_dygraph_save_static_load(self):
        paddle.disable_static()
        # set seed to promise the same input for different tp rank
        self.set_random_seed(self._seed)
        data_loader = self.create_data_loader()
        dist_loader = dist.shard_dataloader(
            dataloader=data_loader,
            meshes=[self.mesh],
            shard_dims=None,
        )
        dy_layer = dist.shard_layer(
            DemoNet("dp_mp_hybrid_strategy"), self.mesh, self.shard_fn
        )
        dy_opt = paddle.optimizer.SGD(
            learning_rate=0.1, parameters=dy_layer.parameters()
        )
        dy_losses = self.run_dynamic(dy_layer, dy_opt, dist_loader)
        saved_dy_layer_state_dict = dy_layer.state_dict()
        ckpt_path = os.path.join(
            self._ckpt_path, "test_dygraph_save_static_load"
        )
        dist.save_state_dict(saved_dy_layer_state_dict, ckpt_path)
        dist.barrier()

        dy2static_opt = paddle.optimizer.SGD(
            learning_rate=0.1, parameters=dy_layer.parameters()
        )

        loss_fn = nn.MSELoss()
        dist_model = dist.to_static(
            dy_layer, dist_loader, loss_fn, dy2static_opt
        )
        need_load_state_dict = {}
        expected_state_dict = {}
        with paddle.base.dygraph.guard():
            for k, v in saved_dy_layer_state_dict.items():
                expected_state_dict[k] = v._local_value().clone()
                need_load_state_dict[k] = paddle.zeros_like(v)
        dist_model.train()
        dist_model.set_state_dict(need_load_state_dict)
        state_dict_to_load = dist_model.state_dict(mode="param")
        assert len(state_dict_to_load) == len(expected_state_dict)
        for k, v in state_dict_to_load.items():
            assert (
                k in expected_state_dict
            ), f"key {k} not in expected_state_dict:{expected_state_dict}"
            assert np.any(
                np.not_equal(
                    v._local_value().numpy(),
                    expected_state_dict[k].numpy(),
                )
            ), f"key:{k}, v:{v}, expected_state_dict[k]:{expected_state_dict[k]}"

        dist.load_state_dict(state_dict_to_load, ckpt_path)
        dist_model.set_state_dict(state_dict_to_load)

        program_state_dict = dist_model.state_dict(mode="param")
        assert len(expected_state_dict) == len(program_state_dict)
        for k, v in program_state_dict.items():
            assert (
                k in expected_state_dict
            ), f"key {k} not in expected_state_dict:{expected_state_dict}"
            np.testing.assert_equal(
                v._local_value().numpy(),
                expected_state_dict[k].numpy(),
            )

    def test_static_save_dynamic_load(self):
        paddle.disable_static()
        # set seed to promise the same input for different tp rank
        self.set_random_seed(self._seed)
        data_loader = self.create_data_loader()

        dy_layer = dist.shard_layer(
            DemoNet("dp_mp_hybrid_strategy"), self.mesh, self.shard_fn
        )

        dy2static_opt = paddle.optimizer.SGD(
            learning_rate=0.1, parameters=dy_layer.parameters()
        )
        dy2static_losses, dist_model = self.run_dy2static(
            dy_layer, dy2static_opt, data_loader
        )

        saved_static_layer_state_dict = dist_model.state_dict("param")
        ckpt_path = os.path.join(
            self._ckpt_path, "test_static_save_dynamic_load"
        )
        dist.save_state_dict(saved_static_layer_state_dict, ckpt_path)
        dist.barrier()

        paddle.disable_static()
        need_load_state_dict = {}
        expected_state_dict = {}
        with paddle.base.dygraph.guard():
            for k, v in saved_static_layer_state_dict.items():
                expected_state_dict[k] = v._local_value().clone()
                need_load_state_dict[k] = paddle.zeros_like(v)
        dy_layer.set_state_dict(need_load_state_dict)
        state_dict_to_load = dy_layer.state_dict()
        assert len(state_dict_to_load) == len(expected_state_dict)
        for k, v in state_dict_to_load.items():
            assert (
                k in expected_state_dict
            ), f"key {k} not in expected_state_dict:{expected_state_dict}"
            assert np.any(
                np.not_equal(
                    v._local_value().numpy(),
                    expected_state_dict[k].numpy(),
                )
            ), f"key:{k}, v:{v}, expected_state_dict[k]:{expected_state_dict[k]}"

        dist.load_state_dict(state_dict_to_load, ckpt_path)
        dy_layer.set_state_dict(state_dict_to_load)

        state_dict = dy_layer.state_dict()
        assert len(expected_state_dict) == len(state_dict)
        for k, v in state_dict.items():
            assert (
                k in expected_state_dict
            ), f"key {k} not in expected_state_dict:{expected_state_dict}"
            np.testing.assert_equal(
                v._local_value().numpy(),
                expected_state_dict[k].numpy(),
            )

    def run_test_case(self):
        self.test_dygraph_save_static_load()
        self.test_static_save_dynamic_load()


if __name__ == "__main__":
    TestSemiAutoParallelMutualLoadBetweenDynamicAndStatic().run_test_case()
