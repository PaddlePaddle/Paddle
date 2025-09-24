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

import hashlib
import random
import unittest

import numpy as np

import paddle
import paddle.distributed as dist
from paddle import nn
from paddle.io import DataLoader
from paddle.static.pir_io import get_pir_parameters

BATCH_SIZE = 2
BATCH_NUM = 4
IMAGE_SIZE = 4
CLASS_NUM = 2


class PPDemoNet(nn.Layer):
    def __init__(self, mesh1, mesh2):
        super().__init__()
        self._mesh1 = mesh1
        self._mesh2 = mesh2
        self.linear_0 = nn.Linear(IMAGE_SIZE, IMAGE_SIZE, bias_attr=False)
        self.linear_1 = nn.Linear(IMAGE_SIZE, CLASS_NUM, bias_attr=False)
        self.linear_2 = nn.Linear(IMAGE_SIZE, IMAGE_SIZE, bias_attr=False)
        self.relu_0 = nn.ReLU()
        self.relu_1 = nn.ReLU()
        self.relu_2 = nn.ReLU()
        # shard the weights of this layer
        self.linear_0.weight = dist.shard_tensor(
            self.linear_0.weight,
            self._mesh1,
            [dist.Replicate()],
            stop_gradient=False,
        )
        self.linear_1.weight = dist.shard_tensor(
            self.linear_1.weight,
            self._mesh2,
            [dist.Replicate()],
            stop_gradient=False,
        )
        self.shared_weight = self.linear_0.weight

    def forward(self, x):
        x.stop_gradient = False
        out = self.relu_0(x)
        out = self.linear_0(out)
        out = self.relu_1(out)
        out = dist.reshard(out, self._mesh2, [dist.Replicate()])
        out = self.linear_2(out)

        shared_weight = dist.reshard(
            self.shared_weight, self._mesh2, [dist.Replicate()]
        )
        out = paddle.matmul(out, shared_weight)
        out = self.linear_1(out)
        out = self.relu_2(out)
        out = paddle.cast(out, 'float32')
        return out


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


class TestSimpleNetForSharedParameter(unittest.TestCase):
    def __init__(self):
        self._seed = 1024
        self._init_loss_scaling = 1024.0
        self.mesh = dist.ProcessMesh([0, 1], dim_names=["x"])
        self.num_batch = 3

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

    def init_strategy(self, need_apply_sync_weight_pass=False):
        strategy = dist.Strategy()
        strategy.pipeline.enable = True
        strategy.pipeline.pp_degree = 2
        strategy.pipeline.accumulate_steps = 2
        if need_apply_sync_weight_pass:
            strategy.pipeline.auto_parallel_sync_shared_params = True
        return strategy

    def run_dy2static(self, layer, opt, dist_loader, strategy):
        loss_fn = nn.MSELoss()
        dist_model = dist.to_static(
            layer, dist_loader, loss_fn, opt, strategy=strategy
        )
        loss_list = []
        dist_model.train()

        for epoch in range(self.num_batch):
            for batch_id, data in enumerate(dist_loader()):
                if isinstance(data, dict):
                    image = data['image']
                    label = data['label']
                else:
                    image, label = data
                loss = dist_model(image, label)
                if paddle.distributed.get_rank() == 1:
                    md5_loss = hashlib.md5(np.array(loss).tobytes()).hexdigest()
                    loss_list.append(md5_loss)

        return loss_list, dist_model

    def run_pp_demo_net(self, strategy):
        paddle.disable_static()
        self.set_random_seed(self._seed)
        mesh1 = dist.ProcessMesh([0], dim_names=["x"])
        mesh2 = dist.ProcessMesh([1], dim_names=["x"])
        data_loader = self.create_data_loader()

        self.set_random_seed(self._seed)

        paddle.base.set_flags({'FLAGS_enable_pir_api': 1})
        self.set_random_seed(self._seed)
        layer = PPDemoNet(mesh1, mesh2)
        opt = paddle.optimizer.Adam(
            learning_rate=0.01, parameters=layer.parameters()
        )
        dist_dataloader = dist.shard_dataloader(
            dataloader=data_loader,
            meshes=[mesh1, mesh2],
        )

        losses, dist_model = self.run_dy2static(
            layer, opt, dist_dataloader, strategy
        )
        return losses, dist_model.dist_main_program()

    def get_shared_params_count(self, program):
        shared_param_count = 0
        allreduce_count = 0

        params, _ = get_pir_parameters(program)
        for param in params:
            param_name = param.get_defining_op().str_attr('parameter_name')
            if "shared_" in param_name:
                shared_param_count += 1

        for op in program.global_block().ops:
            if op.name() == "pd_op.all_reduce":
                allreduce_count += 1

        return shared_param_count, allreduce_count

    def run_test_case(self):
        sync_strategy = self.init_strategy(need_apply_sync_weight_pass=True)
        sync_loss, sync_program = self.run_pp_demo_net(sync_strategy)

        ori_strategy = self.init_strategy(need_apply_sync_weight_pass=False)
        ori_loss, ori_program = self.run_pp_demo_net(ori_strategy)

        if paddle.distributed.get_rank() == 1:
            ori_shared_param_count, ori_allreduce_count = (
                self.get_shared_params_count(ori_program)
            )
            sync_shared_param_count, sync_allreduce_count = (
                self.get_shared_params_count(sync_program)
            )

            # Check shared parameter count.
            self.assertTrue(ori_shared_param_count == 0)
            self.assertTrue(sync_shared_param_count == 1)

            # Check allreduce shared parameter gradient count.
            self.assertTrue(ori_allreduce_count == 0)
            self.assertTrue(sync_allreduce_count == 1)

            for idx in range(self.num_batch):
                self.assertTrue(sync_loss[idx] == ori_loss[idx])


if __name__ == '__main__':
    TestSimpleNetForSharedParameter().run_test_case()
