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
from test_to_static_pir_program import (
    DemoNet,
    create_data_loader,
)

import paddle
import paddle.distributed as dist
from paddle import nn

BATCH_SIZE = 4
BATCH_NUM = 40
IMAGE_SIZE = 16
CLASS_NUM = 8
np.random.seed(2024)
paddle.seed(2024)


class PPDemoNet(nn.Layer):
    def __init__(self, mesh1, mesh2):
        super().__init__()
        self._mesh1 = mesh1
        self._mesh2 = mesh2
        self.linear_0 = nn.Linear(IMAGE_SIZE, IMAGE_SIZE, bias_attr=False)
        self.linear_1 = nn.Linear(IMAGE_SIZE, CLASS_NUM, bias_attr=False)
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

    def forward(self, x):
        x.stop_gradient = False
        out = self.relu_0(x)  # triggle backward partial allreduce
        out = self.linear_0(out)
        out = self.relu_1(out)
        out = dist.reshard(out, self._mesh2, [dist.Replicate()])
        out = self.linear_1(out)
        out = self.relu_2(out)  # triggle forward partial allreduce
        out = paddle.cast(out, 'float32')
        return out


class DPDemoNet(nn.Layer):
    def __init__(
        self,
        mesh,
    ):
        super().__init__()
        self._mesh = mesh
        self.linear_0 = nn.Linear(IMAGE_SIZE, IMAGE_SIZE, bias_attr=False)
        self.linear_1 = nn.Linear(IMAGE_SIZE, CLASS_NUM, bias_attr=False)
        self.linear_0.weight = dist.shard_tensor(
            self.linear_0.weight,
            self._mesh,
            [dist.Replicate()],
            stop_gradient=False,
        )
        self.linear_1.weight = dist.shard_tensor(
            self.linear_1.weight,
            self._mesh,
            [dist.Replicate()],
            stop_gradient=False,
        )
        self.relu_0 = nn.ReLU()
        self.relu_1 = nn.ReLU()
        self.relu_2 = nn.ReLU()

    def forward(self, x):
        out = self.relu_0(x)
        out = self.linear_0(out)
        out = self.relu_1(out)
        out = self.linear_1(out)
        out = self.relu_2(out)
        out = paddle.cast(out, 'float32')
        return out


class TestMLPTensorParallel(unittest.TestCase):
    def test_to_static_program(self):
        paddle.base.set_flags({'FLAGS_enable_pir_api': 1})
        mesh = dist.ProcessMesh([0, 1], dim_names=["x"])
        mp_layer = DemoNet(mesh, True)
        opt = paddle.optimizer.SGD(
            learning_rate=0.1, parameters=mp_layer.parameters()
        )
        loss_fn = nn.MSELoss()
        loader = create_data_loader()
        dist_loader = dist.shard_dataloader(loader, meshes=[mesh])
        dist_model = dist.to_static(mp_layer, dist_loader, loss_fn, opt)

        dist_model.train()
        for batch_id, (image, label) in enumerate(dist_loader()):
            loss = dist_model(image, label)


class TestMLPReplicated(unittest.TestCase):
    def test_to_static_program(self):
        paddle.base.set_flags({'FLAGS_enable_pir_api': 1})
        mesh = dist.ProcessMesh([0, 1], dim_names=["x"])
        replicated_layer = DemoNet(mesh, False)
        opt = paddle.optimizer.SGD(
            learning_rate=0.1, parameters=replicated_layer.parameters()
        )
        loss_fn = nn.MSELoss()
        loader = create_data_loader()
        dist_loader = dist.shard_dataloader(loader, meshes=[mesh])
        dist_model = dist.to_static(replicated_layer, dist_loader, loss_fn, opt)

        dist_model.train()
        for batch_id, (image, label) in enumerate(dist_loader()):
            loss = dist_model(image, label)


class TestMLPPipelineParallel(unittest.TestCase):
    def init_env(self):
        paddle.seed(1024)
        np.random.seed(1024)
        random.seed(1024)

    def test_to_static_program(self):
        paddle.base.set_flags({'FLAGS_enable_pir_api': 1})
        mesh1 = dist.ProcessMesh([0], dim_names=["x"])
        mesh2 = dist.ProcessMesh([1], dim_names=["y"])
        pp_layer = PPDemoNet(mesh1, mesh2)
        opt = paddle.optimizer.SGD(
            learning_rate=0.1, parameters=pp_layer.parameters()
        )
        loss_fn = nn.MSELoss()
        loader = create_data_loader()
        dist_loader = dist.shard_dataloader(loader, meshes=[mesh1, mesh2])
        dist_model = dist.to_static(pp_layer, dist_loader, loss_fn, opt)
        dist_model.train()
        mode = "train"

        for batch_id, (image, label) in enumerate(dist_loader()):
            loss = dist_model(image, label)

    def _pipeline_schedule(
        self,
        enable_schedule=False,
        schedule_mode="FThenB",
        accumulate_steps=1,
        grad_merge=False,
        enable_amp=True,
    ):
        self.init_env()
        paddle.set_flags({'FLAGS_enable_pir_api': 1})
        mesh1 = dist.ProcessMesh([0], dim_names=["x"])
        mesh2 = dist.ProcessMesh([1], dim_names=["x"])
        pp_layer = PPDemoNet(mesh1, mesh2)
        opt = paddle.optimizer.SGD(
            learning_rate=0.1, parameters=pp_layer.parameters()
        )
        loss_fn = nn.MSELoss()
        loader = create_data_loader(
            BATCH_SIZE, BATCH_NUM, IMAGE_SIZE, CLASS_NUM
        )
        strategy = dist.Strategy()
        strategy.pipeline.enable = enable_schedule
        strategy.pipeline.schedule_mode = schedule_mode
        strategy.pipeline.accumulate_steps = accumulate_steps

        if enable_amp:
            amp = strategy.amp
            amp.enable = True
            amp.dtype = 'float16'
            amp.level = 'O2'
            amp.use_master_weight = True
            amp.use_master_grad = True
            amp.use_promote = True
            amp.init_loss_scaling = 1024.0

        if grad_merge:
            gradient_merge = strategy.gradient_merge
            gradient_merge.enable = True
            gradient_merge.k_steps = accumulate_steps
            gradient_merge.avg = True

        dist_loader = dist.shard_dataloader(loader, meshes=[mesh1, mesh2])
        dist_model = dist.to_static(
            pp_layer, dist_loader, loss_fn, opt, strategy
        )
        dist_model.train()

        loss = None
        for batch_id, (image, label) in enumerate(dist_loader()):
            loss = dist_model(image, label)

            if accumulate_steps > 1 and loss is not None:
                loss = np.mean(loss)
        return loss

    def test_pp_pass(self):
        ref_loss = self._pipeline_schedule()
        # only split_program
        loss_split_prog_acc1 = self._pipeline_schedule(
            enable_schedule=False, schedule_mode="FThenB", accumulate_steps=1
        )
        self.assertEqual(ref_loss, loss_split_prog_acc1)

        loss_split_prog_acc4 = self._pipeline_schedule(
            enable_schedule=True,
            schedule_mode="FThenB",
            accumulate_steps=4,
            grad_merge=True,
        )

        if ref_loss is None:
            self.assertEqual(ref_loss, loss_split_prog_acc4)
        else:
            ret_1 = np.allclose(
                loss_split_prog_acc4,
                ref_loss,
                rtol=1e-3,
                atol=1e-2,
                equal_nan=True,
            )
            self.assertEqual(ret_1, True)

    def test_pp_pass_amp(self):
        loss_split_prog_acc1 = self._pipeline_schedule(
            enable_schedule=False,
            schedule_mode="FThenB",
            accumulate_steps=1,
            enable_amp=True,
        )
        loss_split_prog_acc4 = self._pipeline_schedule(
            enable_schedule=True,
            schedule_mode="FThenB",
            accumulate_steps=4,
            grad_merge=True,
            enable_amp=True,
        )

        cur_rank = paddle.distributed.get_rank()
        if cur_rank == 1:
            ret_1 = np.allclose(
                loss_split_prog_acc4,
                loss_split_prog_acc1,
                rtol=1e-3,
                atol=1e-2,
                equal_nan=True,
            )
            self.assertEqual(ret_1, True)


if __name__ == "__main__":
    unittest.main()
