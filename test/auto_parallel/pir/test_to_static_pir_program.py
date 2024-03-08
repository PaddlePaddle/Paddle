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

import unittest

import numpy as np

import paddle
import paddle.distributed as dist
from paddle import nn
from paddle.distributed import Shard
from paddle.io import DataLoader

BATCH_SIZE = 4
BATCH_NUM = 4
IMAGE_SIZE = 16
CLASS_NUM = 8
np.random.seed(2024)


class RandomDataset(paddle.io.Dataset):
    def __init__(self, images, labels, num_samples):
        self.images = images
        self.labels = labels
        self.num_samples = num_samples

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]

    def __len__(self):
        return self.num_samples


class DemoNet(nn.Layer):
    def __init__(self, mesh):
        super().__init__()
        self._mesh = mesh
        self.linear_0 = nn.Linear(IMAGE_SIZE, IMAGE_SIZE)
        self.linear_1 = nn.Linear(IMAGE_SIZE, CLASS_NUM)
        self.relu = nn.ReLU()
        # shard the weights of this layer
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

    def forward(self, x):
        out = self.linear_0(x)
        out = self.relu(out)
        out = self.linear_1(out)
        return out


def create_data_loader():
    images = np.random.rand(BATCH_SIZE, IMAGE_SIZE).astype('float32')
    labels = np.random.rand(BATCH_SIZE, CLASS_NUM).astype('float32')
    dataset = RandomDataset(images, labels, BATCH_SIZE)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE)
    return loader


class TestToStaticPirProgram(unittest.TestCase):
    def test_to_static_program(self):
        paddle.base.set_flags({'FLAGS_enable_pir_api': 1})
        mesh = dist.ProcessMesh([0, 1], dim_names=["x"])
        layer = DemoNet(mesh)
        opt = paddle.optimizer.SGD(
            learning_rate=0.1, parameters=layer.parameters()
        )
        loss_fn = nn.MSELoss()
        loader = create_data_loader()
        dist_loader = dist.shard_dataloader(loader, meshes=[mesh])
        dist_model = dist.to_static(layer, dist_loader, loss_fn, opt)

        dist_model.train()
        main_program = dist_model._engine._fwd_main_progs["train"]
        for op in main_program.global_block().ops:
            tensor = op.result(0)
            if op.name() == 'pd_op.data':
                self.assertTrue(tensor.is_dist_dense_tensor_type())
                self.assertEqual(tensor.process_mesh.shape, [2])
                self.assertEqual(tensor.process_mesh.process_ids, [0, 1])
                self.assertEqual(tensor.dims_mapping, [-1, -1])
                self.assertEqual(tensor.partial_dims, set())
            else:
                self.assertTrue(tensor.is_dense_tensor_type())
                self.assertFalse(tensor.is_dist_dense_tensor_type())

        # training
        # dist_model.train()
        # for batch_id, (image, label) in enumerate(dist_loader()):
        #     loss = dist_model(image, label)


if __name__ == "__main__":
    unittest.main()
