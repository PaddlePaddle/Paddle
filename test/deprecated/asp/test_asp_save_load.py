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

import numpy as np

import paddle
from paddle.base import core
from paddle.incubate.asp import ASPHelper


class MyLayer(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.conv1 = paddle.nn.Conv2D(
            in_channels=3, out_channels=4, kernel_size=3, padding=2
        )
        self.linear1 = paddle.nn.Linear(4624, 32)
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
        paddle.disable_static()

        self.layer = MyLayer()

        self.place = paddle.CPUPlace()
        if core.is_compiled_with_cuda():
            self.place = paddle.CUDAPlace(0)

        self.optimizer = paddle.optimizer.SGD(
            learning_rate=0.01, parameters=self.layer.parameters()
        )
        self.optimizer = paddle.incubate.asp.decorate(self.optimizer)
        paddle.incubate.asp.prune_model(self.layer)

    def test_save_and_load(self):
        path = "/tmp/paddle_asp_save_dy/"
        net_path = path + "asp_net.pdparams"
        opt_path = path + "asp_opt.pdopt"

        paddle.save(self.layer.state_dict(), net_path)
        paddle.save(self.optimizer.state_dict(), opt_path)

        asp_info = ASPHelper._get_program_asp_info(
            paddle.static.default_main_program()
        )
        for param_name in asp_info.mask_vars:
            mask = asp_info.mask_vars[param_name]
            asp_info.update_mask_vars(
                param_name, paddle.ones(shape=mask.shape, dtype=mask.dtype)
            )
            asp_info.update_masks(param_name, np.ones(shape=mask.shape))

        net_state_dict = paddle.load(net_path)
        opt_state_dict = paddle.load(opt_path)

        self.layer.set_state_dict(net_state_dict)
        self.optimizer.set_state_dict(opt_state_dict)

        imgs = paddle.to_tensor(
            np.random.randn(64, 3, 32, 32),
            dtype='float32',
            place=self.place,
            stop_gradient=False,
        )
        labels = paddle.to_tensor(
            np.random.randint(10, size=(64, 1)),
            dtype='float32',
            place=self.place,
            stop_gradient=False,
        )

        loss_fn = paddle.nn.MSELoss(reduction='mean')

        output = self.layer(imgs)
        loss = loss_fn(output, labels)
        loss.backward()
        self.optimizer.step()
        self.optimizer.clear_grad()

        for param in self.layer.parameters():
            if ASPHelper._is_supported_layer(
                paddle.static.default_main_program(), param.name
            ):
                mat = param.numpy()
                if (len(param.shape) == 4 and param.shape[1] < 4) or (
                    len(param.shape) == 2 and param.shape[0] < 4
                ):
                    self.assertFalse(
                        paddle.incubate.asp.check_sparsity(mat.T, n=2, m=4)
                    )
                else:
                    self.assertTrue(
                        paddle.incubate.asp.check_sparsity(mat.T, n=2, m=4)
                    )


if __name__ == '__main__':
    unittest.main()
