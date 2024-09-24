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

import paddle
import paddle.distributed as dist
from paddle import nn
from paddle.distributed import Shard
from paddle.distributed.fleet.utils import recompute
from paddle.io import DataLoader

BATCH_SIZE = 4
BATCH_NUM = 4
SEQ_LEN = 2
IMAGE_SIZE = 16
CLASS_NUM = 8


def create_numpy_like_random(name):
    return paddle.ParamAttr(
        name=name, initializer=paddle.nn.initializer.Uniform(0, 1)
    )


class RandomDataset(paddle.io.Dataset):
    def __init__(self, images, labels, num_samples, return_dict=False):
        self.images = images
        self.labels = labels
        self.num_samples = num_samples
        self.return_dict = return_dict

    def __getitem__(self, idx):
        if self.return_dict:
            return {
                "image": self.images[idx],
                "label": self.labels[idx],
            }
        else:
            return self.images[idx], self.labels[idx]

    def __len__(self):
        return self.num_samples


class DemoNet(nn.Layer):
    def __init__(
        self,
        mesh,
        param_prefix="",
        shard_weight=False,
        is_recompute=False,
    ):
        super().__init__()
        self._mesh = mesh
        self.shard_weight = shard_weight
        self.is_recompute = is_recompute
        weight_attr_0 = create_numpy_like_random(param_prefix + "_0")
        weight_attr_1 = create_numpy_like_random(param_prefix + "_1")

        self.linear_0 = nn.Linear(IMAGE_SIZE, IMAGE_SIZE, weight_attr_0)
        self.linear_1 = nn.Linear(IMAGE_SIZE, CLASS_NUM, weight_attr_1)
        if shard_weight:
            self.linear_0.weight = dist.shard_tensor(
                self.linear_0.weight,
                self._mesh,
                [Shard(1)],
                stop_gradient=False,
            )
            self.linear_1.weight = dist.shard_tensor(
                self.linear_1.weight,
                self._mesh,
                [Shard(0)],
                stop_gradient=False,
            )
        self.relu = nn.ReLU()

    def _inner_forward_fn(self, x):
        out = self.linear_0(x)
        out = self.relu(out)
        out = self.linear_1(out)
        return out

    def forward(self, x):
        if self.is_recompute:
            return recompute(self._inner_forward_fn, x)
        else:
            return self._inner_forward_fn(x)


class TestSimpleNetForSemiAutoParallel:
    def __init__(self):
        self._seed = eval(os.getenv("seed"))
        self._ckpt_path = os.getenv("ckpt_path")
        self.mesh = dist.ProcessMesh([0, 1], dim_names=["x"])

    def set_random_seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        paddle.seed(seed)

    def create_data_loader(self, return_dict=False):
        images = np.random.rand(BATCH_SIZE, IMAGE_SIZE).astype('float32')
        labels = np.random.rand(BATCH_SIZE, CLASS_NUM).astype('float32')
        dataset = RandomDataset(images, labels, BATCH_SIZE, return_dict)
        loader = DataLoader(dataset, batch_size=BATCH_SIZE)
        return loader

    def run_dy2static(self, layer, opt, dist_loader):
        # create loss
        loss_fn = nn.MSELoss()
        # static training
        dist_model = dist.to_static(layer, dist_loader, loss_fn, opt)
        loss_list = []
        dist_model.train()
        for epoch in range(5):
            for batch_id, data in enumerate(dist_loader()):
                if isinstance(data, dict):
                    image = data['image']
                    label = data['label']
                else:
                    image, label = data
                loss = dist_model(image, label)
                loss_list.append(loss)

        return np.array(loss_list), dist_model

    def run_dynamic(self, layer, opt, dist_loader, is_recompute=False):
        # create loss
        loss_fn = nn.MSELoss()
        loss_list = []
        for _ in range(5):
            for batch_id, data in enumerate(dist_loader()):
                if isinstance(data, dict):
                    image = data['image']
                    label = data['label']
                else:
                    image, label = data
                if is_recompute:
                    image.stop_gradient = False
                out = layer(image)
                loss = loss_fn(out, label)
                loss_list.append(loss.numpy())
                loss.backward()

                opt.step()
                opt.clear_grad()
        return np.array(loss_list)

    def test_dp_demo_net(self, is_dataloader_output_dict=False):
        paddle.disable_static()
        self.set_random_seed(self._seed)
        data_loader = self.create_data_loader(
            return_dict=is_dataloader_output_dict
        )

        self.set_random_seed(self._seed)
        dy_layer = DemoNet(self.mesh, "dy_dp_demonet")
        dy_opt = paddle.optimizer.SGD(
            learning_rate=0.1, parameters=dy_layer.parameters()
        )

        self.set_random_seed(self._seed)
        dy2static_layer = DemoNet(self.mesh, "dy2static_dp_demonet")
        dy2static_opt = paddle.optimizer.SGD(
            learning_rate=0.1, parameters=dy2static_layer.parameters()
        )
        dist_dataloader = dist.shard_dataloader(
            dataloader=data_loader,
            meshes=[self.mesh],
            input_keys=["image", "label"],
            shard_dims=['x'],
        )

        dy_losses = self.run_dynamic(dy_layer, dy_opt, dist_dataloader)
        dy2static_losses, _ = self.run_dy2static(
            dy2static_layer, dy2static_opt, dist_dataloader
        )

        # Check the loss values. Different from dygraph mode, when
        # the model is trained in dy2static mode, the loss values
        # are not the average of the losses of all processes, so
        # we should get the average loss first.
        paddle.disable_static()
        pd_partial_loss = paddle.to_tensor(dy2static_losses)
        pd_loss_list = []
        dist.all_gather(pd_loss_list, pd_partial_loss)
        np_dy2static_loss_list = [loss.numpy() for loss in pd_loss_list]
        np_dy2static_loss = np.array(np_dy2static_loss_list)
        np_dy2static_loss = np.mean(np_dy2static_loss, axis=0)

        np.testing.assert_allclose(dy_losses, np_dy2static_loss, rtol=1e-6)

    def test_mp_demo_net(self):
        paddle.disable_static()
        self.set_random_seed(self._seed)
        data_loader = self.create_data_loader()

        self.set_random_seed(self._seed)
        dy_layer = DemoNet(self.mesh, "dy_mp_demonet", shard_weight=True)
        dy_opt = paddle.optimizer.SGD(
            learning_rate=0.1, parameters=dy_layer.parameters()
        )

        self.set_random_seed(self._seed)
        dy2static_layer = DemoNet(
            self.mesh, "dy2static_mp_demonet", shard_weight=True
        )
        dy2static_opt = paddle.optimizer.SGD(
            learning_rate=0.1, parameters=dy2static_layer.parameters()
        )
        dist_dataloader = dist.shard_dataloader(
            dataloader=data_loader,
            meshes=[self.mesh],
        )
        dy_losses = self.run_dynamic(dy_layer, dy_opt, dist_dataloader)
        dy2static_losses, dist_model = self.run_dy2static(
            dy2static_layer, dy2static_opt, dist_dataloader
        )
        np.testing.assert_allclose(dy_losses, dy2static_losses, rtol=1e-6)

        # TODO(cql) FIX set_state_dict in PIR
        # # save load
        # state_dict_to_save = dist_model.state_dict()
        # dist.save_state_dict(state_dict_to_save, self._ckpt_path)
        # dist.barrier()
        # expected_local_state_dict = {}
        # need_load_state_dict = {}
        # with paddle.base.dygraph.guard():
        #     for k, v in state_dict_to_save.items():
        #         local_value = v._local_value()
        #         expected_local_state_dict[k] = local_value.clone()
        #         need_load_state_dict[k] = paddle.zeros_like(v)
        # dist_model.set_state_dict(need_load_state_dict)
        # program_state_dict = dist_model.state_dict()
        # for k, v in program_state_dict.items():
        #     assert v.numpy().sum() == 0, f"key {k} is not zero: {v}"
        #     assert k in expected_local_state_dict
        #     assert (
        #         need_load_state_dict[k].numpy().sum() == 0
        #     ), f"key {k} is not zero: {need_load_state_dict[k]}"
        # dist.load_state_dict(need_load_state_dict, self._ckpt_path)
        # dist_model.set_state_dict(need_load_state_dict)
        # program_state_dict = dist_model.state_dict()
        # for k, v in program_state_dict.items():
        #     local_tensor = v._local_value()
        #     assert (
        #         k in expected_local_state_dict
        #     ), f"key {k} not in expected_local_state_dict:{expected_local_state_dict}"
        #     np.testing.assert_equal(
        #         local_tensor.numpy(),
        #         expected_local_state_dict[k].numpy(),
        #     )

    def test_recompute(self):
        paddle.disable_static()
        self.set_random_seed(self._seed)
        data_loader = self.create_data_loader()

        self.set_random_seed(self._seed)
        dy_layer = DemoNet(
            self.mesh,
            "dy_mp_demonet_recompute",
            shard_weight=True,
            is_recompute=True,
        )
        dy_opt = paddle.optimizer.SGD(
            learning_rate=0.1, parameters=dy_layer.parameters()
        )

        self.set_random_seed(self._seed)
        dy2static_layer = DemoNet(
            self.mesh,
            "dy2static_mp_demonet_recompute",
            shard_weight=True,
            is_recompute=True,
        )
        dy2static_opt = paddle.optimizer.SGD(
            learning_rate=0.1, parameters=dy2static_layer.parameters()
        )

        dist_dataloader = dist.shard_dataloader(
            dataloader=data_loader,
            meshes=[self.mesh],
        )
        dy_losses = self.run_dynamic(
            dy_layer, dy_opt, dist_dataloader, is_recompute=True
        )
        dy2static_losses, dist_model = self.run_dy2static(
            dy2static_layer, dy2static_opt, dist_dataloader
        )

        # check recompute op num
        ops = dist_model.dist_main_program().block(0).ops
        ops_names = [op.type for op in ops]
        assert ops_names.count('matmul_v2') == 4
        assert ops_names.count('relu') == 2

        np.testing.assert_allclose(dy_losses, dy2static_losses, rtol=1e-6)

    def run_test_case(self):
        self.test_dp_demo_net(False)
        self.test_dp_demo_net(True)
        self.test_mp_demo_net()
        # self.test_recompute()


if __name__ == '__main__':
    TestSimpleNetForSemiAutoParallel().run_test_case()
