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

import numpy as np

import paddle
import paddle.nn.functional as F
from paddle import base


def p_normalize(x, axis=1, p=2, epsilon=1e-12, keepdims=True):
    xp = np.power(np.abs(x), p)
    s = np.sum(xp, axis=axis, keepdims=keepdims)
    r = np.maximum(np.power(s, 1.0 / p), epsilon)
    return x / r


class TestNNFunctionalNormalize(unittest.TestCase):
    def setUp(self):
        self.input_np = np.random.random(size=(10, 10)).astype(np.float32)
        self.input_np2 = np.array([0.0, 0.0]).astype(np.float32)
        self.expected0 = p_normalize(self.input_np)
        self.expected1 = p_normalize(self.input_np, p=1.5)
        self.expected2 = p_normalize(self.input_np, axis=0)
        self.expected3 = p_normalize(self.input_np2, axis=0)

    def run_imperative(self):
        x = paddle.to_tensor(self.input_np)
        y = F.normalize(x)
        np.testing.assert_allclose(y.numpy(), self.expected0, rtol=1e-05)

        y = F.normalize(x, p=1.5)
        np.testing.assert_allclose(y.numpy(), self.expected1, rtol=1e-05)

        y = F.normalize(x, axis=0)
        np.testing.assert_allclose(y.numpy(), self.expected2, rtol=1e-05)

        x = paddle.to_tensor(self.input_np2)
        y = F.normalize(x, axis=0)
        np.testing.assert_allclose(y.numpy(), self.expected3, rtol=1e-05)

        self.assertRaises(BaseException, F.normalize, x)

    def run_static(self, use_gpu=False):
        x = paddle.static.data(name='input', shape=[10, 10], dtype='float32')
        x2 = paddle.static.data(name='input2', shape=[2], dtype='float32')
        result0 = F.normalize(x)
        result1 = F.normalize(x, p=1.5)
        result2 = F.normalize(x, axis=0)
        result3 = F.normalize(x, name='aaa')
        result4 = F.normalize(x2, axis=0)

        place = base.CUDAPlace(0) if use_gpu else base.CPUPlace()
        exe = base.Executor(place)
        exe.run(paddle.static.default_startup_program())
        static_result = exe.run(
            feed={"input": self.input_np, "input2": self.input_np2},
            fetch_list=[result0, result1, result2, result4],
        )

        np.testing.assert_allclose(static_result[0], self.expected0, rtol=1e-05)
        np.testing.assert_allclose(static_result[1], self.expected1, rtol=1e-05)
        np.testing.assert_allclose(static_result[2], self.expected2, rtol=1e-05)
        np.testing.assert_allclose(static_result[3], self.expected3, rtol=1e-05)
        self.assertRaises(ValueError, F.normalize, x2)

    def test_cpu(self):
        paddle.disable_static(place=paddle.base.CPUPlace())
        self.run_imperative()
        paddle.enable_static()

        with paddle.static.program_guard(paddle.static.Program()):
            self.run_static()

    def test_gpu(self):
        if not base.core.is_compiled_with_cuda():
            return

        paddle.disable_static(place=paddle.base.CUDAPlace(0))
        self.run_imperative()
        paddle.enable_static()

        with paddle.static.program_guard(paddle.static.Program()):
            self.run_static(use_gpu=True)

    def test_errors(self):
        with base.dygraph.guard():
            # The size of input in Normalize should not be 0.
            def test_0_size():
                array = np.array([], dtype=np.float32)
                x = paddle.to_tensor(
                    np.reshape(array, [1, 1, 0]), dtype='float32'
                )
                paddle.nn.functional.normalize(x)

            self.assertRaises(ValueError, test_0_size)


if __name__ == "__main__":
    unittest.main()
