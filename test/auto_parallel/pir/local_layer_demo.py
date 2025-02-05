# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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
np.random.seed(2025)
paddle.seed(2025)


class LocalLossLayer(dist.LocalLayer):
    def __init__(self, mesh):
        super().__init__(
            out_dist_attrs=[(mesh, [dist.Partial(dist.ReduceType.kRedSum)])]
        )
        self.loss = nn.MSELoss()

    def forward(self, input, label):
        return self.loss(input, label)


class TestMLPTensorParallel(unittest.TestCase):
    def test_to_static_program(self):
        mesh = dist.ProcessMesh([0, 1], dim_names=["x"])
        mp_layer = DemoNet(mesh)
        opt = paddle.optimizer.SGD(
            learning_rate=0.1, parameters=mp_layer.parameters()
        )
        loss_fn = LocalLossLayer(mesh)
        loader = create_data_loader()
        dist_loader = dist.shard_dataloader(loader, meshes=[mesh])
        dist_model = dist.to_static(mp_layer, dist_loader, loss_fn, opt)

        dist_model.train()
        for batch_id, (image, label) in enumerate(dist_loader()):
            loss = dist_model(image, label)


if __name__ == "__main__":
    unittest.main()
