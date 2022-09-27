#   Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
from paddle import fluid
import numpy as np


class TestCpuCuda(unittest.TestCase):

    def test_cpu_cuda(self):

        def func(x):
            x = paddle.to_tensor([1, 2, 3, 4])
            x = x.cuda()
            x = x.cpu()
            return x

        x = paddle.to_tensor([3])
        print(paddle.jit.to_static(func).code)
        print(paddle.jit.to_static(func)(x))


class TestToTensor(unittest.TestCase):

    def test_to_tensor_with_variable_list(self):

        def func(x):
            ones = paddle.to_tensor([1])
            twos = paddle.to_tensor([2])
            x = paddle.to_tensor([ones, twos, 3, 4])
            return x

        x = paddle.to_tensor([3])
        print(paddle.jit.to_static(func).code)
        np.testing.assert_allclose(paddle.jit.to_static(func)(x).numpy(),
                                   np.array([1, 2, 3, 4]),
                                   rtol=1e-05)


class TestToTensor1(unittest.TestCase):

    def test_to_tensor_with_variable_list(self):

        def func(x):
            ones = paddle.to_tensor([1])
            twos = paddle.to_tensor([2])
            """ we ignore the [3] and [4], they will be assign to a variable, and is regard as scalar.
                TODO: deal with this case after 0-dim tensor is developed.
            """
            x = paddle.to_tensor([ones, twos, [3], [4]])
            return x

        x = paddle.to_tensor([3])
        print(paddle.jit.to_static(func).code)
        np.testing.assert_allclose(paddle.jit.to_static(func)(x).numpy(),
                                   np.array([1, 2, 3, 4]),
                                   rtol=1e-05)


class TestToTensor2(unittest.TestCase):

    def test_to_tensor_with_variable_list(self):

        def func(x):
            x = paddle.to_tensor([[1], [2], [3], [4]])
            return x

        x = paddle.to_tensor([3])
        print(paddle.jit.to_static(func).code)
        np.testing.assert_allclose(paddle.jit.to_static(func)(x).numpy(),
                                   np.array([[1], [2], [3], [4]]),
                                   rtol=1e-05)


if __name__ == '__main__':
    unittest.main()
