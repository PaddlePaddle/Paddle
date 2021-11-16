# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

import paddle
from paddle.distributed.passes import new_pass, PassManager
import paddle.distributed.fleet as fleet
from paddle.vision.models import resnet50 as resnet
import unittest
from dist_pass_test_base import DistPassTestBase
import paddle.nn as nn
import numpy as np


class TestFuseAllReducePass(DistPassTestBase):
    def init(self):
        if paddle.is_compiled_with_cuda():
            paddle.set_flags({'FLAGS_cudnn_deterministic': 1})
        self.atol = 0.0
        self.rtol = 0.0

    def apply_passes(self, main_prog, startup_prog):
        pass_manager = PassManager([
            new_pass("fuse_elewise_add_act"),
            new_pass("fuse_all_reduce", {"max_memory_size": 1024 * 1024})
        ])
        pass_manager.apply([main_prog], [startup_prog])

    def test_bs_32(self):
        self.check_main(batch_size=32)

    def get_model(self, place, batch_size):
        image = paddle.static.data(
            shape=[batch_size, 3, 224, 224], dtype='float32', name='image')
        label = paddle.static.data(
            shape=[batch_size, 1], dtype='int64', name='label')
        model = resnet(pretrained=False)
        loss_fn = nn.loss.CrossEntropyLoss()
        pred_out = model(image)
        loss = loss_fn(pred_out, label)
        optimizer = paddle.optimizer.Adam(learning_rate=1e-3)

        dist_strategy = fleet.DistributedStrategy()
        dist_strategy.fuse_all_reduce_ops = False
        dist_strategy.without_graph_optimization = True
        fleet.init(is_collective=True, strategy=dist_strategy)
        optimizer = fleet.distributed_optimizer(optimizer)
        optimizer.minimize(loss)

        rank = paddle.distributed.get_rank()

        def reader():
            np.random.seed(self.seed + rank)
            for _ in range(10):
                image_np = np.random.random(size=image.shape).astype('float32')
                label_np = np.random.randint(
                    low=0, high=1000, size=label.shape).astype('int64')
                yield image_np, label_np

        main_program = paddle.static.default_main_program()
        startup_program = paddle.static.default_startup_program()
        return main_program, startup_program, [image, label], [loss], reader


if __name__ == "__main__":
    unittest.main()
