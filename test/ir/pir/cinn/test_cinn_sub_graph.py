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
import utils

import paddle


class CINNSoftmaxSubGraphNet(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.fn = paddle.nn.functional.softmax

    def forward(self, x, axis=-1):
        out = self.fn(x, axis=axis)
        return out


class TestCinnSubGraphBase(unittest.TestCase):
    """
    Test Pir API + @to_static + CINN.
    """

    def setUp(self):
        paddle.seed(2022)
        self.prepare_data()

    def prepare_data(self):
        self.shape = [1, 2, 1024 * 16, 1024]
        self.axis = -1
        # self.x = paddle.uniform(self.shape, dtype="float32", min=-0.5, max=0.5)
        self.x = paddle.rand(self.shape, dtype=paddle.float32)
        self.x.stop_gradient = False

    def check_jit_kernel_info(self, static_fn):
        utils.check_jit_kernel_number(static_fn, 1)
        utils.check_jit_kernel_structure(static_fn, {utils.JIT_KERNEL_NAME: 1})


class TestCinnSoftmax(TestCinnSubGraphBase):
    def train(self, use_cinn):
        paddle.seed(2022)
        net = CINNSoftmaxSubGraphNet()
        net.eval()
        net = utils.apply_to_static(net, use_cinn)
        input_specs = [
            paddle.static.InputSpec(
                shape=[1, 2, 16384, -1],
                dtype=paddle.float32,
                name="x",
                stop_gradient=False,
            )
        ]
        net = utils.apply_to_static(net, use_cinn, input_spec=input_specs)

        out = net(self.x, self.axis)

        return out, self.x.gradient()

    def test_forward(self):
        cinn_out, cinn_grad = self.train(use_cinn=True)
        dy_out, dy_grad = self.train(use_cinn=False)

        a = cinn_out.numpy().flatten()
        b = dy_out.numpy().flatten()

        # print( np.allclose( a[:, 0, 1], b[:, 0, 1], atol=1e-8))

        print(a[1024:2048])
        print(b[1024:2048])

        np.testing.assert_allclose(cinn_out.numpy(), dy_out.numpy(), atol=1e-8)


if __name__ == '__main__':
    unittest.main()
