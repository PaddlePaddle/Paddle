# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
from paddle.incubate.optimizer.functional.lbfgs_v2 import LBFGS

np.random.seed(123)


class TestLbfgs(unittest.TestCase):
    def test_function_fix(self):

        paddle.disable_static()
        np_w = np.random.rand(1).astype(np.float32)
        np_x = np.random.rand(1).astype(np.float32)

        input = np.random.rand(1).astype(np.float32)
        weights = [np.random.rand(1).astype(np.float32) for i in range(5)]
        targets = [weights[i] * input for i in range(5)]

        class Net(paddle.nn.Layer):
            def __init__(self):
                super(Net, self).__init__()
                w = paddle.to_tensor(np_w)
                self.w = paddle.create_parameter(
                    shape=w.shape,
                    dtype=w.dtype,
                    default_initializer=paddle.nn.initializer.Assign(w),
                )

            def forward(self, x):
                return self.w * x

        net = Net()
        opt = LBFGS(
            lr=1,
            max_iter=1,
            max_eval=None,
            tolerance_grad=1e-07,
            tolerance_change=1e-09,
            history_size=100,
            line_search_fn='strong_wolfe',
            parameters=net.parameters(),
        )

        def train_step(inputs, targets):
            def closure():
                outputs = net(inputs)
                loss = paddle.nn.functional.mse_loss(outputs, targets)
                opt.clear_grad()
                loss.backward()
                return loss

            loss = opt.step(closure)
            return loss

        for weight, target in zip(weights, targets):
            input = paddle.to_tensor(input)
            target = paddle.to_tensor(target)
            loss = 1
            while loss > 1e-4:
                loss = train_step(input, target)
            np.testing.assert_allclose(net.w, weight, rtol=1e-05)


if __name__ == '__main__':
    unittest.main()
