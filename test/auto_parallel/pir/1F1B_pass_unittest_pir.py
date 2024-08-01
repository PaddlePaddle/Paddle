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
import unittest

import numpy as np
from mlp_demo import PPDemoNet

import paddle
import paddle.distributed as dist
from paddle import nn
from paddle.io import DataLoader

BATCH_SIZE = 4
BATCH_NUM = 4
IMAGE_SIZE = 16
CLASS_NUM = 8
np.random.seed(2024)
paddle.seed(2024)


def apply_pass(use_1f1b=False):
    strategy = dist.Strategy()
    strategy.auto_mode = "semi"
    strategy.reinit = True

    if use_1f1b:
        pipeline = strategy.pipeline
        pipeline.enable = True
        pipeline.schedule_mode = "1F1B"
        pipeline.accumulate_steps = 2
    else:
        pipeline = strategy.pipeline
        pipeline.enable = True
        pipeline.schedule_mode = "FThenB"
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


class TestSimpleNet1F1BPass(unittest.TestCase):
    def init(self):
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
        loader = DataLoader(dataset, batch_size=batch_size)
        return loader

    def test_pp_1f1b(self):
        paddle.base.set_flags({'FLAGS_enable_pir_api': 1})

        strategy_fthenb = apply_pass(use_1f1b=False)
        loss_fthenb = self.run_pipeline(strategy_fthenb)

        strategy_1f1b = apply_pass(use_1f1b=True)
        loss_1f1b = self.run_pipeline(strategy_1f1b)

        cur_rank = paddle.distributed.get_rank()
        if cur_rank == 1:
            for loss1, loss2 in zip(loss_1f1b, loss_fthenb):
                self.assertAlmostEqual(loss1, loss2)

    def run_pipeline(self, strategy):
        self.init()
        mesh1 = dist.ProcessMesh([0], dim_names=["pp"])
        mesh2 = dist.ProcessMesh([1], dim_names=["pp"])
        pp_layer = PPDemoNet(mesh1, mesh2)
        opt = paddle.optimizer.SGD(
            learning_rate=0.1, parameters=pp_layer.parameters()
        )
        loss_fn = nn.MSELoss()
        loader = self.create_data_loader()
        dist_loader = dist.shard_dataloader(loader, meshes=[mesh1, mesh2])
        dist_model = dist.to_static(
            pp_layer, dist_loader, loss_fn, opt, strategy
        )
        dist_model.train()

        loss_list = []
        for _, (image, label) in enumerate(dist_loader):
            loss = dist_model(image, label)
            if loss is not None:
                loss_list.append(np.mean(loss))

        return loss_list


if __name__ == '__main__':
    unittest.main()
