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

import numpy as np

import paddle
from paddle.base import core

np.random.seed(10)


class TestCountNonzeroAPI(unittest.TestCase):
    # test paddle.tensor.math.count_nonzero

    def setUp(self):
        self.x_shape = [2, 3, 4, 5]
        self.x = np.random.uniform(-1, 1, self.x_shape).astype(np.float32)
        self.place = (
            paddle.CUDAPlace(0)
            if core.is_compiled_with_cuda()
            else paddle.CPUPlace()
        )

    def test_api_static(self):
        paddle.enable_static()
        with paddle.static.program_guard(paddle.static.Program()):
            x = paddle.static.data('X', self.x_shape)
            out1 = paddle.count_nonzero(x)
            out2 = paddle.tensor.count_nonzero(x)
            out3 = paddle.tensor.math.count_nonzero(x)
            axis = np.arange(len(self.x_shape)).tolist()
            out4 = paddle.count_nonzero(x, axis)
            out5 = paddle.count_nonzero(x, tuple(axis))
            exe = paddle.static.Executor(self.place)
            res = exe.run(
                feed={'X': self.x}, fetch_list=[out1, out2, out3, out4, out5]
            )
        out_ref = np.count_nonzero(self.x)
        for out in res:
            np.testing.assert_allclose(out, out_ref, rtol=1e-05)

    def test_api_dygraph(self):
        paddle.disable_static(self.place)

        def test_case(x, axis=None, keepdim=False):
            x_tensor = paddle.to_tensor(x)
            out = paddle.count_nonzero(x_tensor, axis=axis, keepdim=keepdim)
            if isinstance(axis, list):
                axis = tuple(axis)
                if len(axis) == 0:
                    axis = None

            out_ref = np.count_nonzero(x, axis, keepdims=keepdim)
            np.testing.assert_allclose(out.numpy(), out_ref, rtol=1e-05)

        test_case(self.x)
        test_case(self.x, None)
        test_case(self.x, -1)
        test_case(self.x, keepdim=True)
        test_case(self.x, 2, keepdim=True)
        test_case(self.x, [0, 2])
        test_case(self.x, (0, 2))
        test_case(self.x, (0, 1, 3))
        test_case(self.x, [0, 1, 2, 3])
        paddle.enable_static()

    def test_errors(self):
        paddle.enable_static()
        with paddle.static.program_guard(paddle.static.Program()):
            x = paddle.static.data('X', [10, 12], 'int32')
            self.assertRaises(ValueError, paddle.count_nonzero, x, axis=10)


if __name__ == "__main__":
    unittest.main()
