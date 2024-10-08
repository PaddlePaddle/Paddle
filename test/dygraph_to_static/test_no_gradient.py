#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

import numpy
from dygraph_to_static_utils import Dy2StTestBase

import paddle


def static_func(x, no_grad_x):
    tx = 2 * no_grad_x
    tx.stop_gradient = True
    return 2 * x


def main_func(x, index):
    tmp = paddle.gather(x, index)
    out = paddle.jit.to_static(static_func)(x, tmp)
    return out


class TestNoGradientCase(Dy2StTestBase):
    def test_no_gradient(self):
        paddle.disable_static()
        x = paddle.randn([10, 3])
        index = paddle.arange(0, 10, 1, dtype='int32')
        x.stop_gradient = False
        index.stop_gradient = True

        func = main_func
        output = func(x, index).mean()
        output.backward()

        self.assertTrue(x.grad is not None)
        self.assertTrue(
            numpy.all(x.grad.numpy() == paddle.full([10, 3], 2.0 / 30).numpy())
        )
        self.assertTrue(index.grad is None)


if __name__ == '__main__':
    unittest.main()
