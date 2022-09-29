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

import unittest
import paddle
import paddle.fluid.core as core
from paddle.fluid.contrib.sparsity.asp import ASPHelper
import numpy as np


class MyLayer(paddle.nn.Layer):

    def __init__(self):
        super(MyLayer, self).__init__()
        self.conv1 = paddle.nn.Conv2D(in_channels=3,
                                      out_channels=2,
                                      kernel_size=3,
                                      padding=2)
        self.linear1 = paddle.nn.Linear(1352, 32)
        self.linear2 = paddle.nn.Linear(32, 32)
        self.linear3 = paddle.nn.Linear(32, 10)

    def forward(self, img):
        hidden = self.conv1(img)
        hidden = paddle.flatten(hidden, start_axis=1)
        hidden = self.linear1(hidden)
        hidden = self.linear2(hidden)
        prediction = self.linear3(hidden)
        return prediction


class TestASPDynamicOptimize(unittest.TestCase):

    def setUp(self):

        self.layer = MyLayer()

        self.place = paddle.CPUPlace()
        if core.is_compiled_with_cuda():
            self.place = paddle.CUDAPlace(0)

        self.optimizer = paddle.optimizer.SGD(
            learning_rate=0.01, parameters=self.layer.parameters())

    def test_is_supported_layers(self):
        program = paddle.static.default_main_program()

        names = [
            'embedding_0.w_0', 'fack_layer_0.w_0', 'conv2d_0.w_0',
            'conv2d_0.b_0', 'conv2d_1.w_0', 'conv2d_1.b_0', 'fc_0.w_0',
            'fc_0.b_0', 'fc_1.w_0', 'fc_1.b_0', 'linear_2.w_0', 'linear_2.b_0'
        ]
        ref = [
            False, False, True, False, True, False, True, False, True, False,
            True, False
        ]
        for i, name in enumerate(names):
            self.assertTrue(
                ref[i] == ASPHelper._is_supported_layer(program, name))

        paddle.incubate.asp.set_excluded_layers(['fc_1', 'conv2d_0'])
        ref = [
            False, False, False, False, True, False, True, False, False, False,
            True, False
        ]
        for i, name in enumerate(names):
            self.assertTrue(
                ref[i] == ASPHelper._is_supported_layer(program, name))

        paddle.incubate.asp.reset_excluded_layers()
        ref = [
            False, False, True, False, True, False, True, False, True, False,
            True, False
        ]
        for i, name in enumerate(names):
            self.assertTrue(
                ref[i] == ASPHelper._is_supported_layer(program, name))

    def test_decorate(self):
        param_names = [param.name for param in self.layer.parameters()]
        self.optimizer = paddle.incubate.asp.decorate(self.optimizer)

        program = paddle.static.default_main_program()

        for name in param_names:
            mask_var = ASPHelper._get_program_asp_info(program).mask_vars.get(
                name, None)
            if ASPHelper._is_supported_layer(program, name):
                self.assertTrue(mask_var is not None)
            else:
                self.assertTrue(mask_var is None)

    def test_asp_training(self):
        self.optimizer = paddle.incubate.asp.decorate(self.optimizer)

        paddle.incubate.asp.prune_model(self.layer)

        imgs = paddle.to_tensor(np.random.randn(32, 3, 24, 24),
                                dtype='float32',
                                place=self.place,
                                stop_gradient=False)
        labels = paddle.to_tensor(np.random.randint(10, size=(32, 1)),
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

    def test_asp_training_with_amp(self):
        self.optimizer = paddle.incubate.asp.decorate(self.optimizer)

        paddle.incubate.asp.prune_model(self.layer)

        imgs = paddle.to_tensor(np.random.randn(32, 3, 24, 24),
                                dtype='float32',
                                place=self.place,
                                stop_gradient=False)
        labels = paddle.to_tensor(np.random.randint(10, size=(32, 1)),
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


if __name__ == '__main__':
    unittest.main()
