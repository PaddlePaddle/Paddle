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

import paddle
import paddle.distributed as dist
from paddle import nn
from paddle.distributed import fleet
from paddle.framework import _current_expected_place
from paddle.io import DataLoader

BATCH_SIZE = 1
IMAGE_SIZE = 6
CLASS_NUM = 2
VOCAB_SIZE = 10
HIDDEN_SIZE = 8


class CEmbeddingNet(nn.Layer):
    def __init__(self, mesh):
        super().__init__()
        self.embedding = fleet.meta_parallel.VocabParallelEmbedding(
            VOCAB_SIZE,
            HIDDEN_SIZE,
            weight_attr=paddle.nn.initializer.Constant(value=0.5),
        )

    def forward(self, x):
        x = paddle.to_tensor(x, dtype="int32")
        out = self.embedding(x)
        t = paddle.ones([1, 6, 8])
        out = out * t
        return out


class EmbeddingNet(nn.Layer):
    def __init__(self, mesh):
        super().__init__()
        self.embedding = paddle.nn.Embedding(
            VOCAB_SIZE,
            HIDDEN_SIZE,
            weight_attr=paddle.nn.initializer.Constant(value=0.5),
        )
        self.mesh_ = mesh
        self.embedding.weight = dist.shard_tensor(
            self.embedding.weight,
            mesh,
            [dist.Replicate(), dist.Shard(1)],
            stop_gradient=False,
        )

    def forward(self, x):
        out = self.embedding(x)
        out = dist.reshard(
            out, self.mesh_, [dist.Replicate(), dist.Replicate()]
        )
        t = paddle.ones([1, 6, 8])
        out = out * t
        return out


def create_numpy_like_random(name):
    return paddle.ParamAttr(
        name=name, initializer=paddle.nn.initializer.Uniform(0, 1)
    )


class RandomDataset(paddle.io.Dataset):
    def __init__(self, images, labels, num_samples):
        self.images = images
        self.labels = labels
        self.num_samples = num_samples

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]

    def __len__(self):
        return self.num_samples


class TestSimpleNetForSemiAutoParallel:
    def __init__(self):
        self._seed = eval(os.getenv("seed"))
        self._ckpt_path = os.getenv("ckpt_path")
        self.mesh = dist.ProcessMesh([[0, 1]])
        self._in_pir_mode = paddle.base.framework.get_flags(
            "FLAGS_enable_pir_api"
        )["FLAGS_enable_pir_api"]
        strategy = fleet.DistributedStrategy()
        self.model_parallel_size = 2
        strategy.hybrid_configs = {
            "dp_degree": 1,
            "mp_degree": self.model_parallel_size,
            "pp_degree": 1,
        }
        fleet.init(is_collective=True, strategy=strategy)

    def set_random_seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        paddle.seed(seed)

    def create_data_loader(self):
        images = np.random.randint(0, VOCAB_SIZE, (BATCH_SIZE, IMAGE_SIZE))
        labels = np.random.rand(BATCH_SIZE, IMAGE_SIZE, HIDDEN_SIZE).astype(
            'float32'
        )
        dataset = RandomDataset(images, labels, BATCH_SIZE)
        loader = DataLoader(dataset, batch_size=BATCH_SIZE)
        return loader

    def run_dy2static(self, layer, opt, dist_loader):
        loss_fn = nn.MSELoss()
        dist_model = dist.to_static(layer, dist_loader, loss_fn, opt)
        loss_list = []
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

        for epoch in range(8):
            for batch_id, data in enumerate(dist_loader()):
                image, label = data
                loss = dist_model(image, label)
                loss_list.append(loss)
        return np.array(loss_list), dist_model

    def run_dynamic(self, layer, opt, dist_loader):
        loss_fn = nn.MSELoss()
        loss_list = []
        for epoch in range(8):
            for batch_id, data in enumerate(dist_loader()):
                image, label = data
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
        dy_layer = CEmbeddingNet(self.mesh)
        dy_opt = paddle.optimizer.AdamW(
            learning_rate=0.1, parameters=dy_layer.parameters()
        )

        paddle.base.set_flags({'FLAGS_enable_pir_api': 1})
        self.set_random_seed(self._seed)
        dy2static_layer = EmbeddingNet(self.mesh)
        dy2static_opt = paddle.optimizer.AdamW(
            learning_rate=0.1, parameters=dy2static_layer.parameters()
        )
        dist_dataloader = dist.shard_dataloader(
            dataloader=data_loader,
            meshes=[self.mesh],
        )
        dy2static_losses, dist_model = self.run_dy2static(
            dy2static_layer, dy2static_opt, dist_dataloader
        )
        dy_losses = self.run_dynamic(dy_layer, dy_opt, data_loader)
        print("dy2static_losses:", dy2static_losses)
        print("dy_losses:", dy_losses)
        np.testing.assert_allclose(dy_losses, dy2static_losses, atol=1e-7)

    def run_test_case(self):
        self.test_mp_demo_net()


if __name__ == '__main__':
    TestSimpleNetForSemiAutoParallel().run_test_case()
