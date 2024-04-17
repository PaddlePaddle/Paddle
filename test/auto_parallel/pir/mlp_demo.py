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

from test_to_static_pir_program import DemoNet, create_data_loader

import paddle
import paddle.distributed as dist
from paddle import nn
from paddle.framework import _current_expected_place


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
        mode = "train"

        # TODO(2024-Q2) hack for engine api
        dist_model._engine._has_prepared[mode] = True
        dist_model._mode = mode
        dist_model._engine._mode = mode
        paddle.disable_static()
        dist_model._engine._initialize(mode)
        dist_model._engine._executor = paddle.static.Executor(
            _current_expected_place()
        )
        dist_model._engine._init_comm()

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
        mode = "train"
        # dist_model._engine._build(mode)
        # main_program = dist_model._engine._pir_main_progs[mode]

        # TODO(2024-Q2) hack for engine api
        # main_program = dist_model._engine._pir_main_progs[mode]
        dist_model._engine._has_prepared[mode] = True
        dist_model._mode = mode
        dist_model._engine._mode = mode
        paddle.disable_static()
        dist_model._engine._initialize(mode)
        dist_model._engine._executor = paddle.static.Executor(
            _current_expected_place()
        )

        for batch_id, (image, label) in enumerate(dist_loader()):
            loss = dist_model(image, label)


if __name__ == "__main__":
    unittest.main()
