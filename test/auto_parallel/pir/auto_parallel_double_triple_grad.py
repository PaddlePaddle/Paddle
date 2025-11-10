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
import random
import unittest

import numpy as np

import paddle
import paddle.distributed as dist
from paddle import nn
from paddle.distributed import Shard
from paddle.io import DataLoader

BATCH_SIZE = 4
BATCH_NUM = 5
IMAGE_SIZE = 8
CLASS_NUM = 8


class RandomDataset(paddle.io.Dataset):
    def __init__(self, images, labels, num_samples):
        self.images = images
        self.labels = labels
        self.num_samples = num_samples

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]

    def __len__(self):
        return self.num_samples


def create_data_loader(
    batch_size=BATCH_SIZE,
    batch_num=BATCH_NUM,
    image_size=IMAGE_SIZE,
    class_num=CLASS_NUM,
):
    nsamples = batch_size * batch_num
    images = np.random.rand(nsamples, image_size).astype('float32')
    labels = np.random.rand(nsamples, class_num).astype('float32')
    dataset = RandomDataset(images, labels, nsamples)
    loader = DataLoader(dataset, batch_size=batch_size)
    return loader


class DemoNet(nn.Layer):
    def __init__(self, mesh, shard_type="no_shard", test_prim=False):
        super().__init__()
        self._mesh = mesh
        self._test_prim = test_prim
        self.shard_type = shard_type
        self.linear_0 = nn.Linear(IMAGE_SIZE, CLASS_NUM, bias_attr=False)
        self.linear_1 = nn.Linear(CLASS_NUM, CLASS_NUM, bias_attr=False)
        if self.shard_type == "tp":
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
        elif self.shard_type == "dp":
            pass
        else:
            raise ValueError(
                "Only support `shard_type` is one of `dp` and `tp`."
            )

    def forward(self, x):
        x.stop_gradient = False
        y = paddle.tanh(x)
        y = self.linear_0(y)
        y = self.linear_1(y)
        y = paddle.cast(y, 'float32')
        if self._test_prim:
            y = y.unsqueeze(1)
            # `p_norm_grad` needs prim_eager=True.
            y = paddle.linalg.norm(y, p=2, axis=-1)
        return y


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    paddle.seed(seed)


class TestMLPTensorParallel(unittest.TestCase):
    def run_model(self, model, loader, loss_fn, opt):
        losses = []
        for batch_id, (image, label) in enumerate(loader()):
            y = model(image)
            image.stop_gradient = False
            dx = paddle.grad(y, image, create_graph=True)[0]
            dx.stop_gradient = False
            d2x = paddle.grad(dx, image, create_graph=False)[0]
            logit = y + dx + d2x
            loss = loss_fn(logit, label)
            loss = logit
            losses.append(loss)
            loss.backward()
            opt.step()
            opt.clear_grad()
        return losses

    def run_tp_model(self, test_prim=False):
        set_random_seed(eval(os.getenv("seed")))
        mesh = dist.ProcessMesh([0, 1], dim_names=["tp"])
        mp_layer = DemoNet(mesh=mesh, shard_type="tp", test_prim=test_prim)
        opt = paddle.optimizer.SGD(
            learning_rate=0.1, parameters=mp_layer.parameters()
        )
        opt = dist.shard_optimizer(opt)
        loss_fn = nn.MSELoss()
        loader = create_data_loader()
        dist_loader = dist.shard_dataloader(loader, meshes=[mesh])
        tp_losses = self.run_model(mp_layer, dist_loader, loss_fn, opt)
        return tp_losses

    def run_dp_model(self, test_prim=False):
        set_random_seed(eval(os.getenv("seed")))
        mesh = dist.ProcessMesh([0, 1], dim_names=["dp"])
        dp_layer = DemoNet(mesh=mesh, shard_type="dp", test_prim=test_prim)
        opt = paddle.optimizer.SGD(
            learning_rate=0.1, parameters=dp_layer.parameters()
        )
        opt = dist.shard_optimizer(opt)
        loss_fn = nn.MSELoss()
        loader = create_data_loader()
        dist_loader = dist.shard_dataloader(
            loader, meshes=[mesh], shard_dims="dp"
        )
        dp_losses = self.run_model(dp_layer, dist_loader, loss_fn, opt)
        return dp_losses

    def run_pp_model(self, test_prim=False):
        set_random_seed(eval(os.getenv("seed")))
        mesh_1 = dist.ProcessMesh([0], dim_names=["pp1"])
        mesh_2 = dist.ProcessMesh([1], dim_names=["pp2"])
        pp_layer = DemoNet(
            mesh=[mesh_1, mesh_2], shard_type="pp", test_prim=test_prim
        )
        opt = paddle.optimizer.SGD(
            learning_rate=0.1, parameters=pp_layer.parameters()
        )
        opt = dist.shard_optimizer(opt)
        loss_fn = nn.MSELoss()
        loader = create_data_loader()
        dist_loader = dist.shard_dataloader(loader, meshes=[mesh_1, mesh_2])
        pp_losses = self.run_model(pp_layer, dist_loader, loss_fn, opt)
        return pp_losses

    def test_auto_parallel(self):
        rtol = 1e-5
        dp_losses = self.run_dp_model()
        tp_losses = self.run_tp_model()
        np.testing.assert_allclose(
            dp_losses,
            tp_losses,
            rtol=rtol,
        )

    def test_prim_eager_auto_parallel(self):
        rtol = 1e-5
        paddle.framework.core.set_prim_eager_enabled(True)
        dp_losses = self.run_dp_model(test_prim=True)
        tp_losses = self.run_tp_model(test_prim=True)
        np.testing.assert_allclose(
            dp_losses,
            tp_losses,
            rtol=rtol,
        )


if __name__ == "__main__":
    unittest.main()
