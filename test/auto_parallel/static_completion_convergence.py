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

import random
import unittest

import numpy as np

import paddle
import paddle.distributed as dist
from paddle import nn
from paddle.distributed import Shard
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
    def __init__(self, images, labels, num_samples):
        self.images = images
        self.labels = labels
        self.num_samples = num_samples

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]

    def __len__(self):
        return self.num_samples


class MLP(nn.Layer):
    def __init__(self, mesh, shard_weight=True, param_prefix=""):
        super().__init__()
        self._mesh = mesh
        self.shard_weight = shard_weight
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

    def forward(self, x):
        out = self.linear_0(x)
        out = dist.reshard(
            out, self._mesh, [Shard(1)]
        )  # trigger infinite propagation
        out = self.linear_1(out)

        return out


class TestStaticReshard(unittest.TestCase):
    def __init__(self):
        self._seed = 1234
        self.set_random_seed(self._seed)

    def set_random_seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)

    def create_data_loader(self):
        images = np.random.rand(BATCH_SIZE, BATCH_SIZE, IMAGE_SIZE).astype(
            'float32'
        )
        labels = np.random.rand(BATCH_SIZE, BATCH_SIZE, CLASS_NUM).astype(
            'float32'
        )
        dataset = RandomDataset(images, labels, BATCH_SIZE)
        loader = DataLoader(dataset, batch_size=BATCH_SIZE)
        return loader

    def test_reshard_mesh(self):
        mesh0 = dist.ProcessMesh([0, 1], dim_names=["x"])

        dy2static_layer = MLP(mesh0)

        dy2static_opt = paddle.optimizer.SGD(
            learning_rate=0.1, parameters=dy2static_layer.parameters()
        )

        loss_fn = nn.MSELoss()

        # static training
        data_loader = self.create_data_loader()
        dist_loader = dist.shard_dataloader(data_loader, [mesh0])
        dist_model = dist.to_static(
            dy2static_layer, dist_loader, loss_fn, dy2static_opt
        )

        program = dist_model._engine._dist_contexts["train"].dist_main_programs[
            dist_model._engine._cur_rank
        ]
        ops = program.global_block().ops
        check_ops = [op.type for op in ops[:]]
        assert check_ops == [
            'matmul_v2',
            'elementwise_add',
            'all_gather',
            'split',
            'concat',
            'split',
            'assign',
            'all_gather',
            'matmul_v2',
            'elementwise_add',
            'elementwise_sub',
            'all_gather',
            'split',
            'concat',
            'assign',
            'square',
            'reduce_mean',
            'fill_constant',
            'reduce_mean_grad',
            'square_grad',
            'split',
            'elementwise_sub_grad',
            'elementwise_add_grad',
            'c_allreduce_sum',
            'scale',
            'matmul_v2_grad',
            'c_allreduce_sum',
            'scale',
            'assign',
            'all_gather',
            'split',
            'concat',
            'split',
            'elementwise_add_grad',
            'matmul_v2_grad',
            'c_allreduce_sum',
            'sgd',
            'sgd',
            'split',
            'sgd',
            'sgd',
        ]

    def run_test_case(self):
        self.test_reshard_mesh()


if __name__ == '__main__':
    TestStaticReshard().run_test_case()
