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

import paddle


class BatchNormNet(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.batch_norm = paddle.nn.layer.norm.BatchNorm2D(10, 10)

    def forward(self, x):
        y = paddle.nn.functional.relu(x)
        z = self.batch_norm(y)
        return paddle.mean(z)


class TestBNAMP(unittest.TestCase):
    def train(self, use_cinn):
        paddle.seed(2024)
        net = BatchNormNet()
        optimizer = paddle.optimizer.SGD(
            learning_rate=0.0001, parameters=net.parameters()
        )
        scaler = paddle.amp.GradScaler(init_loss_scaling=1024)
        if use_cinn:
            net = paddle.jit.to_static(net, backend='CINN', full_graph=True)

        x = paddle.randn([4, 10, 10, 10], dtype='float16')
        x.stop_gradient = False
        with paddle.amp.auto_cast(level='O2'):
            loss = net(x)
        scaled = scaler.scale(loss)
        scaled.backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.clear_grad(set_to_zero=False)

        return x.grad.numpy()

    def test_amp_train(self):
        dy_x_grad = self.train(use_cinn=False)
        cinn_x_grad = self.train(use_cinn=True)
        np.testing.assert_allclose(dy_x_grad, cinn_x_grad, atol=1e-6, rtol=1e-6)


if __name__ == '__main__':
    unittest.main()
