# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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


class Net(paddle.nn.Layer):
    def __init__(self) -> None:
        super().__init__()
        self.linear = paddle.nn.Linear(10, 10)

    @paddle.jit.to_static
    def forward(self, x):
        x = self.linear(x)
        y = 0
        for i in range(paddle.shape(x)[0]):
            y = y + x[i]
            for j in range(paddle.shape(x)[0]):
                y = y + x[j]
        return y


class TestMNIST(unittest.TestCase):
    def test_run(self):
        if paddle.fluid.is_compiled_with_cuda():
            x = paddle.ones((10, 10))
            net = Net()
            net.eval()
            net = paddle.amp.decorate(
                models=net, optimizers=None, level='O2', save_dtype='float32'
            )
            with paddle.amp.auto_cast(
                enable=True,
                custom_white_list=None,
                custom_black_list=None,
                level='O2',
            ):
                out = net(x)
                np.testing.assert_equal(out.shape, (10,))


if __name__ == '__main__':
    unittest.main()
