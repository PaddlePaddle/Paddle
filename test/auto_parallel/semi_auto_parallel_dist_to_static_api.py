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


def create_data_loader():
    images = np.random.rand(BATCH_SIZE, IMAGE_SIZE).astype('float32')
    labels = np.random.rand(BATCH_SIZE, CLASS_NUM).astype('float32')
    dataset = RandomDataset(images, labels, BATCH_SIZE)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE)
    return loader


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
    def __init__(
        self,
        mesh,
        param_prefix="",
        shard_input=False,
        shard_weight=False,
    ):
        super().__init__()
        self._mesh = mesh
        self.shard_input = shard_input
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
        self.relu = nn.ReLU()

    def _inner_forward_fn(self, x):
        out = self.linear_0(x)
        out = self.relu(out)
        out = self.linear_1(out)
        return out

    def forward(self, x):
        return self._inner_forward_fn(x)


class TestSimpleNetForSemiAutoParallel(unittest.TestCase):
    def __init__(self):
        self._seed = eval(os.getenv("seed"))
        self.set_random_seed(self._seed)
        self.data_loader = create_data_loader()

    def set_random_seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        paddle.seed(seed)

    def get_program_test(self, dist_model):
        with self.assertRaises(ValueError):
            main_program = dist_model.dist_main_program()

        dist_model.train()
        main_program = dist_model.dist_main_program()
        startup_program = dist_model.dist_startup_program()
        self.assertNotEqual(main_program, None)
        self.assertNotEqual(startup_program, None)

        dist_model.eval()
        main_program = dist_model.dist_main_program("eval")
        startup_program = dist_model.dist_startup_program("eval")
        self.assertNotEqual(main_program, None)
        self.assertNotEqual(startup_program, None)

        serial_main_program = dist_model.serial_main_program()
        serial_startup_program = dist_model.serial_startup_program()
        self.assertNotEqual(serial_main_program, None)
        self.assertNotEqual(serial_startup_program, None)

        serial_main_program = dist_model.serial_main_program("eval")
        serial_startup_program = dist_model.serial_startup_program("eval")
        self.assertNotEqual(serial_main_program, None)
        self.assertNotEqual(serial_startup_program, None)

        with self.assertRaises(ValueError):
            main_program = dist_model.dist_main_program("prediction")

    def run_test(self):
        mesh = dist.ProcessMesh([0, 1], dim_names=["x"])

        layer = DemoNet(mesh, "dp_mp_demonet", shard_weight=True)
        opt = paddle.optimizer.SGD(
            learning_rate=0.1, parameters=layer.parameters()
        )

        # create loss
        loss_fn = nn.MSELoss()

        # static training
        strategy = dist.Strategy()

        dist_loader = dist.shard_dataloader(
            dataloader=self.data_loader,
            meshes=[mesh],
        )

        dist_model = dist.to_static(
            layer, dist_loader, loss_fn, opt, strategy=strategy
        )

        dist_model._mode = None

        with self.assertRaises(ValueError):
            for batch_id, (image, label) in enumerate(dist_loader()):
                loss = dist_model(image, label)

        self.get_program_test(dist_model)

        dist_model.train()
        for epoch in range(2):
            for batch_id, (image, label) in enumerate(dist_loader()):
                loss = dist_model(image, label)

        dist_model.eval()
        for batch_id, (image, label) in enumerate(dist_loader()):
            loss = dist_model(image, label)

        # FIXME(ljz) enable predict mode for PIR in future
        # dist_model.predict()
        # for batch_id, (image, label) in enumerate(dist_loader()):
        #     loss = dist_model(image)

        # with self.assertRaises(ValueError):
        #     for batch_id, (image, label) in enumerate(dist_loader()):
        #         loss = dist_model(image, label)

        # lack loss function and optimizer
        # currently it will raise an error when generating another
        # static model, so instead the loss and optimizer are directly
        # set to None for the unit test.
        loss_tmp = dist_model._engine._loss
        opt_tmp = dist_model._engine._optimizer
        dist_model._engine._loss = None
        dist_model._engine._optimizer = None
        with self.assertRaises(ValueError):
            dist_model.train()
            for batch_id, (image, label) in enumerate(dist_loader()):
                loss = dist_model(image, label)
        with self.assertRaises(ValueError):
            dist_model.eval()
            for batch_id, (image, label) in enumerate(dist_loader()):
                loss = dist_model(image, label)
        dist_model._engine._loss = loss_tmp
        dist_model._engine._optimizer = opt_tmp

    def run_test_case(self):
        self.run_test()


if __name__ == '__main__':
    TestSimpleNetForSemiAutoParallel().run_test_case()
