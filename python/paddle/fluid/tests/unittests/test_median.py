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
from paddle.static import Program, program_guard

DELTA = 1e-6


class TestMedian(unittest.TestCase):

    def check_numpy_res(self, np1, np2):
        self.assertEqual(np1.shape, np2.shape)
        mismatch = np.sum((np1 - np2) * (np1 - np2))
        self.assertAlmostEqual(mismatch, 0, DELTA)

    def static_single_test_median(self, lis_test):
        paddle.enable_static()
        x, axis, keepdims = lis_test
        res_np = np.median(x, axis=axis, keepdims=keepdims)
        if not isinstance(res_np, np.ndarray):
            res_np = np.array([res_np])
        main_program = Program()
        startup_program = Program()
        exe = paddle.static.Executor()
        with program_guard(main_program, startup_program):
            x_in = paddle.fluid.data(shape=x.shape, dtype=x.dtype, name='x')
            y = paddle.median(x_in, axis, keepdims)
            [res_pd] = exe.run(feed={'x': x}, fetch_list=[y])
            self.check_numpy_res(res_pd, res_np)
        paddle.disable_static()

    def dygraph_single_test_median(self, lis_test):
        x, axis, keepdims = lis_test
        res_np = np.median(x, axis=axis, keepdims=keepdims)
        if not isinstance(res_np, np.ndarray):
            res_np = np.array([res_np])
        res_pd = paddle.median(paddle.to_tensor(x), axis, keepdims)
        self.check_numpy_res(res_pd.numpy(), res_np)

    def test_median_static(self):
        h = 3
        w = 4
        l = 2
        x = np.arange(h * w * l).reshape([h, w, l])
        lis_tests = [[x, axis, keepdims] for axis in [-1, 0, 1, 2, None]
                     for keepdims in [False, True]]
        for lis_test in lis_tests:
            self.static_single_test_median(lis_test)

    def test_median_dygraph(self):
        paddle.disable_static()
        h = 3
        w = 4
        l = 2
        x = np.arange(h * w * l).reshape([h, w, l])
        lis_tests = [[x, axis, keepdims] for axis in [-1, 0, 1, 2, None]
                     for keepdims in [False, True]]
        for lis_test in lis_tests:
            self.dygraph_single_test_median(lis_test)

    def test_median_exception(self):
        paddle.disable_static()
        x = [1, 2, 3, 4]
        self.assertRaises(TypeError, paddle.median, x)
        x = paddle.arange(12).reshape([3, 4])
        self.assertRaises(ValueError, paddle.median, x, 1.0)
        self.assertRaises(ValueError, paddle.median, x, 2)


if __name__ == '__main__':
    unittest.main()
