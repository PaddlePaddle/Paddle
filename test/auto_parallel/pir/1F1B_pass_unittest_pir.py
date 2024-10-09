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

import random
from functools import reduce

import numpy as np
from mlp_demo import PPDemoNet

import paddle
import paddle.distributed as dist
from paddle import nn
from paddle.io import BatchSampler, DataLoader

BATCH_SIZE = 4
BATCH_NUM = 4
IMAGE_SIZE = 16
CLASS_NUM = 8


def apply_pass(use_1f1b=False):
    strategy = dist.Strategy()
    strategy.auto_mode = "semi"
    strategy.reinit = True

    if use_1f1b:
        pipeline = strategy.pipeline
        pipeline.enable = True
        pipeline.schedule_mode = "1F1B"
        pipeline.pp_degree = 2
        pipeline.accumulate_steps = 2
    else:
        pipeline = strategy.pipeline
        pipeline.enable = True
        pipeline.schedule_mode = "FThenB"
        pipeline.pp_degree = 2
        pipeline.accumulate_steps = 2

    return strategy


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


class TestSimpleNet1F1BPass:
    def init_dist_env(self):
        order = ["dp", "pp", "mp"]
        dp_degree = 1
        mp_degree = 1
        pp_degree = 2
        degree = [dp_degree, pp_degree, mp_degree]
        mesh_dims = list(filter(lambda x: x[1] > 1, list(zip(order, degree))))
        if not mesh_dims:
            mesh_dims = [("dp", 1)]
        dim_names = [mesh_dim[0] for mesh_dim in mesh_dims]
        mesh_shape = [mesh_dim[1] for mesh_dim in mesh_dims]
        mesh_arr = np.arange(
            0, reduce(lambda x, y: x * y, mesh_shape, 1)
        ).reshape(mesh_shape)
        global_mesh = dist.ProcessMesh(mesh_arr, dim_names)
        dist.auto_parallel.set_mesh(global_mesh)
        paddle.seed(2024)
        np.random.seed(2024)
        random.seed(2024)

    def create_data_loader(
        self,
        batch_size=BATCH_SIZE,
        batch_num=BATCH_NUM,
        image_size=IMAGE_SIZE,
        class_num=CLASS_NUM,
    ):
        nsamples = batch_size * batch_num
        images = np.random.rand(nsamples, image_size).astype('float32')
        labels = np.random.rand(nsamples, class_num).astype('float32')
        dataset = RandomDataset(images, labels, nsamples)
        data_sampler = BatchSampler(
            dataset, batch_size=batch_size, shuffle=False, drop_last=True
        )
        loader = DataLoader(dataset, batch_sampler=data_sampler, num_workers=0)
        return loader

    def test_pp_1f1b(self):
        self.init_dist_env()
        strategy_fthenb = apply_pass(use_1f1b=False)
        loss_fthenb = self.run_pipeline(strategy_fthenb)

        self.init_dist_env()
        strategy_1f1b = apply_pass(use_1f1b=True)
        loss_1f1b = self.run_pipeline(strategy_1f1b)

        cur_rank = paddle.distributed.get_rank()
        if cur_rank == 1:
            self.check_result(loss_1f1b, loss_fthenb)

    def run_pipeline(self, strategy):
        mesh1 = dist.ProcessMesh([0], dim_names=["pp"])
        mesh2 = dist.ProcessMesh([1], dim_names=["pp"])
        model = PPDemoNet(mesh1, mesh2)

        lr_scheduler = paddle.optimizer.lr.LinearWarmup(
            learning_rate=0.1, warmup_steps=2, start_lr=0, end_lr=0.0001
        )

        opt = paddle.optimizer.SGD(
            learning_rate=lr_scheduler, parameters=model.parameters()
        )

        loader = self.create_data_loader()
        dist_loader = dist.shard_dataloader(loader, meshes=[mesh1, mesh2])

        loss_fn = nn.MSELoss()

        dist_model = dist.to_static(model, dist_loader, loss_fn, opt, strategy)

        dist_model.train()
        loss = None
        for _, (image, label) in enumerate(dist_loader()):
            loss = dist_model(image, label)
            lr_scheduler.step()
            if int(dist.get_rank()) == 1:
                assert loss is not None
            else:
                assert loss is None

        return np.array(loss)

    def check_result(self, loss1, loss2):
        return np.array_equal(loss1, loss2)


if __name__ == '__main__':
    TestSimpleNet1F1BPass().test_pp_1f1b()
