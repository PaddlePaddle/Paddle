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
from test_to_static_pir_program import DemoNet, create_data_loader

import paddle
import paddle.distributed as dist
from paddle import nn

BATCH_SIZE = 4
BATCH_NUM = 10
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
    def test_to_static_program(self):
        paddle.base.set_flags({'FLAGS_enable_pir_api': 1})
        mesh1 = dist.ProcessMesh([0], dim_names=["x"])
        mesh2 = dist.ProcessMesh([1], dim_names=["y"])
        pp_layer = PPDemoNet(mesh1, mesh2)
        opt = paddle.optimizer.SGD(
            learning_rate=0.1, parameters=pp_layer.parameters()
        )
        loss_fn = nn.MSELoss()
        loader = create_data_loader(
            BATCH_SIZE, BATCH_NUM, IMAGE_SIZE, CLASS_NUM
        )
        dist_loader = dist.shard_dataloader(loader, meshes=[mesh1, mesh2])
        dist_model = dist.to_static(pp_layer, dist_loader, loss_fn, opt)
        dist_model.train()
        mode = "train"

        for batch_id, (image, label) in enumerate(dist_loader()):
            loss = dist_model(image, label)
            print(batch_id, loss)

    def test_split_program(self):
        paddle.set_flags({'FLAGS_enable_pir_api': 1})
        mesh1 = dist.ProcessMesh([0], dim_names=["x"])
        mesh2 = dist.ProcessMesh([1], dim_names=["y"])
        pp_layer = PPDemoNet(mesh1, mesh2)
        opt = paddle.optimizer.SGD(
            learning_rate=0.1, parameters=pp_layer.parameters()
        )
        loss_fn = nn.MSELoss()
        loader = create_data_loader(
            BATCH_SIZE, BATCH_NUM, IMAGE_SIZE, CLASS_NUM
        )
        strategy = dist.Strategy()
        strategy.pipeline.enable = True
        dist_loader = dist.shard_dataloader(loader, meshes=[mesh1, mesh2])
        dist_model = dist.to_static(
            pp_layer, dist_loader, loss_fn, opt, strategy
        )
        dist_model.train()
        mode = "train"

        # if strategy.pipeline.enable:
        #     print("==== whole program ====")
        #     print(dist_model._engine._pir_dense_main_progs[mode])
        #     print("==== fwd program ====")
        #     print(dist_model._engine._pir_dense_fwd_progs[mode])
        #     print("==== bwd program ====")
        #     print(dist_model._engine._pir_dense_bwd_progs[mode])
        #     print("==== opt program ====")
        #     print(dist_model._engine._pir_dense_opt_progs[mode])

        # if dist.get_rank() == 1:
        #     loss_in_fwd = dist_model._engine._pir_dense_fwd_progs[mode].global_block().ops[-1].result(0)
        # else:
        #     loss_in_fwd = None
        for batch_id, (image, label) in enumerate(dist_loader()):
            loss = dist_model(image, label)
            print("===== step: %d =====" % batch_id)
            print(batch_id, loss)


if __name__ == "__main__":
    # unittest.main()
    # TestMLPPipelineParallel().test_to_static_program()
    TestMLPPipelineParallel().test_split_program()
