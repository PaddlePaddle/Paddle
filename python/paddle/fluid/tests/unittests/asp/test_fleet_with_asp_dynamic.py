# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
# Copyright (c) 2022 NVIDIA Corporation.  All rights reserved.
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

import paddle.distributed.fleet as fleet
import unittest
import paddle
import paddle.fluid.core as core
import os
from paddle.fluid.contrib.sparsity.asp import ASPHelper
import numpy as np

cuda_visible_devices = os.getenv('CUDA_VISIBLE_DEVICES')
if cuda_visible_devices is None or cuda_visible_devices == "":
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
else:
    os.environ['CUDA_VISIBLE_DEVICES'] = cuda_visible_devices.split(',')[0]


class MyLayer(paddle.nn.Layer):

    def __init__(self):
        super(MyLayer, self).__init__()
        self.linear1 = paddle.nn.Linear(32, 32)
        self.linear2 = paddle.nn.Linear(32, 10)

    def forward(self, x):
        hidden = self.linear1(x)
        prediction = self.linear2(hidden)
        return prediction


class TestFleetWithASPDynamic(unittest.TestCase):

    def setUp(self):
        os.environ["PADDLE_TRAINER_ENDPOINTS"] = "127.0.0.1:36213"
        os.environ["PADDLE_CURRENT_ENDPOINTS"] = "127.0.0.1:36213"
        os.environ["PADDLE_TRAINERS_NUM"] = "1"
        os.environ["PADDLE_TRAINER_ID"] = "0"

        self.layer = MyLayer()

        self.place = paddle.CPUPlace()
        if core.is_compiled_with_cuda():
            self.place = paddle.CUDAPlace(0)

        self.optimizer = paddle.optimizer.SGD(
            learning_rate=0.01, parameters=self.layer.parameters())

    def test_with_asp(self):
        fleet.init(is_collective=True)

        self.optimizer = paddle.incubate.asp.decorate(self.optimizer)
        paddle.incubate.asp.prune_model(self.layer)

        self.optimizer = fleet.distributed_optimizer(self.optimizer)
        self.layer = fleet.distributed_model(self.layer)

        imgs = paddle.to_tensor(np.random.randn(64, 32),
                                dtype='float32',
                                place=self.place,
                                stop_gradient=False)
        labels = paddle.to_tensor(np.random.randint(10, size=(64, 1)),
                                  dtype='float32',
                                  place=self.place,
                                  stop_gradient=False)

        loss_fn = paddle.nn.MSELoss(reduction='mean')

        output = self.layer(imgs)
        loss = loss_fn(output, labels)
        loss.backward()
        self.optimizer.step()
        self.optimizer.clear_grad()

        for param in self.layer.parameters():
            if ASPHelper._is_supported_layer(
                    paddle.static.default_main_program(), param.name):
                mat = param.numpy()
                if (len(param.shape) == 4
                        and param.shape[1] < 4) or (len(param.shape) == 2
                                                    and param.shape[0] < 4):
                    self.assertFalse(
                        paddle.fluid.contrib.sparsity.check_sparsity(mat.T,
                                                                     n=2,
                                                                     m=4))
                else:
                    self.assertTrue(
                        paddle.fluid.contrib.sparsity.check_sparsity(mat.T,
                                                                     n=2,
                                                                     m=4))


class TestFleetWithASPAMPDynamic(unittest.TestCase):

    def setUp(self):
        os.environ["PADDLE_TRAINER_ENDPOINTS"] = "127.0.0.1:36213"
        os.environ["PADDLE_CURRENT_ENDPOINTS"] = "127.0.0.1:36213"
        os.environ["PADDLE_TRAINERS_NUM"] = "1"
        os.environ["PADDLE_TRAINER_ID"] = "0"

        self.layer = MyLayer()

        self.place = paddle.CPUPlace()
        if core.is_compiled_with_cuda():
            self.place = paddle.CUDAPlace(0)

        self.optimizer = paddle.optimizer.SGD(
            learning_rate=0.01, parameters=self.layer.parameters())

    def test_with_asp(self):
        fleet.init(is_collective=True)

        self.optimizer = paddle.incubate.asp.decorate(self.optimizer)
        paddle.incubate.asp.prune_model(self.layer)

        self.optimizer = fleet.distributed_optimizer(self.optimizer)
        self.layer = fleet.distributed_model(self.layer)

        imgs = paddle.to_tensor(np.random.randn(64, 32),
                                dtype='float32',
                                place=self.place,
                                stop_gradient=False)
        labels = paddle.to_tensor(np.random.randint(10, size=(64, 1)),
                                  dtype='float32',
                                  place=self.place,
                                  stop_gradient=False)

        loss_fn = paddle.nn.MSELoss(reduction='mean')
        scaler = paddle.amp.GradScaler(init_loss_scaling=1024)

        with paddle.amp.auto_cast(enable=True):
            output = self.layer(imgs)
            loss = loss_fn(output, labels)
        scaled = scaler.scale(loss)
        scaled.backward()
        scaler.minimize(self.optimizer, scaled)
        self.optimizer.clear_grad()

        for param in self.layer.parameters():
            if ASPHelper._is_supported_layer(
                    paddle.static.default_main_program(), param.name):
                mat = param.numpy()
                if (len(param.shape) == 4
                        and param.shape[1] < 4) or (len(param.shape) == 2
                                                    and param.shape[0] < 4):
                    self.assertFalse(
                        paddle.fluid.contrib.sparsity.check_sparsity(mat.T,
                                                                     n=2,
                                                                     m=4))
                else:
                    self.assertTrue(
                        paddle.fluid.contrib.sparsity.check_sparsity(mat.T,
                                                                     n=2,
                                                                     m=4))


if __name__ == "__main__":
    unittest.main()
