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
import unittest

import numpy as np

import paddle
import paddle.distributed as dist
from paddle import nn
from paddle.distributed import fleet
from paddle.io import DataLoader

BATCH_SIZE = 4
IMAGE_SIZE = 8
ignore_index = -100


class DemoNet(nn.Layer):
    def __init__(self, mesh):
        super().__init__()
        self.linear = nn.Linear(
            IMAGE_SIZE,
            IMAGE_SIZE,
            bias_attr=False,
            weight_attr=paddle.framework.ParamAttr(
                initializer=paddle.nn.initializer.Constant(0.0)
            ),
        )
        self.linear.weight = dist.shard_tensor(
            self.linear.weight,
            mesh,
            [dist.Replicate(), dist.Shard(1)],
            stop_gradient=False,
        )
        self.soft = paddle.nn.CrossEntropyLoss(
            reduction="none", ignore_index=ignore_index
        )

    def forward(self, x):
        out = self.linear(x)
        y = paddle.ones(shape=[BATCH_SIZE, 1], dtype='int64')
        out = paddle.cast(out, 'float32')
        out = self.soft(out, y)
        return out


class RandomDataset(paddle.io.Dataset):
    def __init__(self, images, labels, num_samples):
        self.images = images
        self.labels = labels
        self.num_samples = num_samples

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]

    def __len__(self):
        return self.num_samples


class TestMPReplaceWithParallelCrossEntropy(unittest.TestCase):
    def setUp(self):
        self.atol = 1e-5
        self.set_random_seed(eval(os.getenv("seed")))
        self.mesh = dist.ProcessMesh([[0, 1]], dim_names=["x", "y"])
        paddle.base.set_flags({'FLAGS_enable_pir_api': 1})

        self.init_dist()

    def init_dist(self):
        self.data_loader = self.create_data_loader()
        self.dist_loader = dist.shard_dataloader(
            dataloader=self.data_loader,
            meshes=[self.mesh],
        )
        self.loss_fn = nn.MSELoss()

        strategy = fleet.DistributedStrategy()
        strategy.hybrid_configs = {
            "dp_degree": 1,
            "mp_degree": 2,
            "pp_degree": 1,
        }
        fleet.init(is_collective=True, strategy=strategy)

    def set_random_seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        paddle.seed(seed)

    def create_data_loader(self):
        images = np.random.rand(BATCH_SIZE, IMAGE_SIZE).astype('float32')
        labels = np.random.rand(IMAGE_SIZE, 1).astype('float32')
        dataset = RandomDataset(images, labels, BATCH_SIZE)
        loader = DataLoader(dataset, batch_size=BATCH_SIZE)
        return loader

    def run_dy2static(self, dist_model):
        loss_list = []
        dist_model._engine._mode = "train"
        dist_model.train()

        for epoch in range(10):
            for batch_id, data in enumerate(self.dist_loader()):
                if isinstance(data, dict):
                    image = data['image']
                    label = data['label']
                else:
                    image, label = data
                loss = dist_model(image, label)
                loss_list += [loss]

        return loss_list, dist_model

    def run_mp(self, use_pass):
        paddle.disable_static()

        model = DemoNet(self.mesh)
        opt = paddle.optimizer.SGD(
            learning_rate=0.1, parameters=model.parameters()
        )
        strategy = dist.Strategy()
        strategy._mp_optimization.replace_with_parallel_cross_entropy = use_pass
        dist_model = dist.to_static(
            model, self.dist_loader, self.loss_fn, opt, strategy
        )
        losses, dist_model = self.run_dy2static(dist_model)
        return losses, dist_model

    def check_results(self, check_losses, ref_losses):
        assert len(ref_losses) == len(check_losses)
        for i in range(len(ref_losses)):
            np.testing.assert_allclose(
                ref_losses[i], check_losses[i], self.atol
            )

    def check_program(
        self, prog_with_pass, prog_without_pass, rtol=None, atol=None
    ):
        ops_with_pass = [op.name() for op in prog_with_pass.global_block().ops]
        ops_without_pass = [
            op.name() for op in prog_without_pass.global_block().ops
        ]

        self.assertIn('pd_op.c_softmax_with_cross_entropy', ops_with_pass)
        self.assertIn('pd_op.cross_entropy_with_softmax', ops_without_pass)

    def test_mp_replace_with_parallel_cross_entropy_pass(self):
        losses_with_pass, dist_model_with_pass = self.run_mp(True)
        losses_without_pass, dist_model_without_pass = self.run_mp(False)
        self.check_results(losses_with_pass, losses_without_pass)
        prog_with_pass = dist_model_with_pass.dist_main_program()
        prog_without_pass = dist_model_without_pass.dist_main_program()
        self.check_program(prog_with_pass, prog_without_pass)


if __name__ == "__main__":
    # NOTE: due to the incompatibility between Llama and GPT models
    # in the PIR model, UnitTest was completed using ‘DemoNet’.
    # Model testing can be supplemented later.
    unittest.main()
