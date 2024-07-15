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
import random

import numpy as np
from mlp_demo import DPDemoNet, PPDemoNet
from test_to_static_pir_program import DemoNet

import paddle
import paddle.distributed as dist
from paddle import nn
from paddle.framework import _current_expected_place
from paddle.io import DataLoader

BATCH_SIZE = 4
BATCH_NUM = 4
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


class TestSimpleNetForSemiAutoParallel:
    def __init__(self):
        self._seed = eval(os.getenv("seed"))
        self._ckpt_path = os.getenv("ckpt_path")
        self.mesh = dist.ProcessMesh([0, 1], dim_names=["x"])
        self._in_pir_mode = paddle.base.framework.get_flags(
            "FLAGS_enable_pir_api"
        )["FLAGS_enable_pir_api"]

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
        layer, opt = paddle.amp.decorate(
            models=layer,
            optimizers=opt,
            level='O2',
            master_weight=False,
            master_grad=False,
        )
        # static training
        dist_model = dist.to_static(layer, dist_loader, loss_fn, opt)
        loss_list = []
        with paddle.amp.auto_cast(level='O2', dtype='float16'):
            dist_model.train()

        if self._in_pir_mode:
            mode = "train"

            dist_model._engine._has_prepared[mode] = True
            dist_model._mode = mode
            dist_model._engine._mode = mode
            paddle.disable_static()
            dist_model._engine._initialize(mode)
            dist_model._engine._executor = paddle.static.Executor(
                _current_expected_place()
            )
            dist_model._engine._init_comm()

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
        layer, opt = paddle.amp.decorate(
            models=layer,
            optimizers=opt,
            level='O2',
            master_weight=False,
            master_grad=False,
        )
        loss_list = []
        for epoch in range(5):
            for batch_id, data in enumerate(dist_loader()):
                if isinstance(data, dict):
                    image = data['image']
                    label = data['label']
                else:
                    image, label = data
                if is_recompute:
                    image.stop_gradient = False
                with paddle.amp.auto_cast(level='O2', dtype='float16'):
                    out = layer(image)
                    loss = loss_fn(out, label)
                loss_list.append(loss.numpy())
                loss.backward()
                opt.step()
                opt.clear_grad()
        return np.array(loss_list)

    def test_mp_demo_net(self):
        paddle.disable_static()
        self.set_random_seed(self._seed)
        data_loader = self.create_data_loader()

        self.set_random_seed(self._seed)
        dy_layer = DemoNet(self.mesh)
        dy_opt = paddle.optimizer.SGD(
            learning_rate=0.1, parameters=dy_layer.parameters()
        )

        paddle.base.set_flags({'FLAGS_enable_pir_api': 1})
        self.set_random_seed(self._seed)
        dy2static_layer = DemoNet(self.mesh)
        dy2static_opt = paddle.optimizer.SGD(
            learning_rate=0.1, parameters=dy2static_layer.parameters()
        )
        dist_dataloader = dist.shard_dataloader(
            dataloader=data_loader,
            meshes=[self.mesh],
        )
        dy2static_losses, dist_model = self.run_dy2static(
            dy2static_layer, dy2static_opt, dist_dataloader
        )

        dy_losses = self.run_dynamic(dy_layer, dy_opt, dist_dataloader)
        print('loss', dy_losses, dy2static_losses)
        np.testing.assert_array_equal(dy_losses, dy2static_losses)

    def test_dp_demo_net(self):
        paddle.disable_static()
        self.set_random_seed(self._seed)
        data_loader = self.create_data_loader()

        self.set_random_seed(self._seed)
        dy_layer = DPDemoNet(self.mesh)
        dy_opt = paddle.optimizer.SGD(
            learning_rate=0.1, parameters=dy_layer.parameters()
        )

        paddle.base.set_flags({'FLAGS_enable_pir_api': 1})
        self.set_random_seed(self._seed)
        dy2static_layer = DPDemoNet(self.mesh)
        dy2static_opt = paddle.optimizer.SGD(
            learning_rate=0.1, parameters=dy2static_layer.parameters()
        )
        dist_dataloader = dist.shard_dataloader(
            dataloader=data_loader,
            meshes=[self.mesh],
            input_keys=["image", "label"],
            shard_dims=['x'],
        )
        dy2static_losses, _ = self.run_dy2static(
            dy2static_layer, dy2static_opt, dist_dataloader
        )

        dy_losses = self.run_dynamic(dy_layer, dy_opt, dist_dataloader)
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
        np.testing.assert_array_equal(dy_losses, np_dy2static_loss)

    def test_pp_demo_net(self):
        paddle.disable_static()
        self.set_random_seed(self._seed)
        mesh1 = dist.ProcessMesh([0], dim_names=["x"])
        mesh2 = dist.ProcessMesh([1], dim_names=["x"])
        data_loader = self.create_data_loader()

        self.set_random_seed(self._seed)
        dy_layer = PPDemoNet(mesh1, mesh2)
        dy_opt = paddle.optimizer.SGD(
            learning_rate=0.1, parameters=dy_layer.parameters()
        )

        paddle.base.set_flags({'FLAGS_enable_pir_api': 1})
        self.set_random_seed(self._seed)
        dy2static_layer = PPDemoNet(mesh1, mesh2)
        dy2static_opt = paddle.optimizer.SGD(
            learning_rate=0.1, parameters=dy2static_layer.parameters()
        )
        dist_dataloader = dist.shard_dataloader(
            dataloader=data_loader,
            meshes=[mesh1, mesh2],
        )
        dy2static_losses, dist_model = self.run_dy2static(
            dy2static_layer, dy2static_opt, dist_dataloader
        )

        dy_losses = self.run_dynamic(dy_layer, dy_opt, dist_dataloader)
        if paddle.distributed.get_rank() == 1:
            np.testing.assert_array_equal(dy_losses, dy2static_losses)

    def run_test_case(self):
        self.test_mp_demo_net()
        self.test_pp_demo_net()
        self.test_dp_demo_net()


if __name__ == '__main__':
    TestSimpleNetForSemiAutoParallel().run_test_case()
