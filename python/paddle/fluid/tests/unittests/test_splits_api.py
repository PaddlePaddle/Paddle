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
import paddle.fluid as fluid
import paddle.fluid.core as core


def func_ref(func, x, num_or_sections):
    # Convert the num_or_sections in paddle to indices_or_sections in numpy
    # Do not support -1
    if isinstance(num_or_sections, int):
        indices_or_sections = num_or_sections
    else:
        indices_or_sections = np.cumsum(num_or_sections)[:-1]
    return func(x, indices_or_sections)


# TODO: add other split API, such as dsplit、hsplit
test_list = [
    (paddle.vsplit, np.vsplit),
]


class TestSplitsAPI(unittest.TestCase):

    def setUp(self):
        self.rtol = 1e-5
        self.atol = 1e-8
        self.set_input()

    def set_input(self):
        self.shape = [4, 5, 2]
        self.num_or_sections = 2
        self.x_np = np.random.uniform(-1, 1, self.shape).astype('float64')
        self.place = paddle.CUDAPlace(0) if core.is_compiled_with_cuda() \
            else paddle.CPUPlace()

    def test_static_api(self):
        paddle.enable_static()
        for func, func_type in test_list:
            with paddle.static.program_guard(paddle.static.Program()):
                x = paddle.fluid.data('X', self.x_np.shape, self.x_np.dtype)
                out = func(x, self.num_or_sections)
                exe = paddle.static.Executor(self.place)
                res = exe.run(feed={'X': self.x_np}, fetch_list=[out])
            out_ref = func_ref(func_type, self.x_np, self.num_or_sections)
            for n, p in zip(out_ref, res):
                np.testing.assert_allclose(n, p, rtol=self.rtol, atol=self.atol)

    def test_dygraph_api(self):
        paddle.disable_static(self.place)
        x = paddle.to_tensor(self.x_np)
        for func, func_type in test_list:
            out = func(x, self.num_or_sections)
            out_ref = func_ref(func_type, self.x_np, self.num_or_sections)
            for n, p in zip(out_ref, out):
                np.testing.assert_allclose(n,
                                           p.numpy(),
                                           rtol=self.rtol,
                                           atol=self.atol)
        paddle.enable_static()


class TestSplitsSections(TestSplitsAPI):
    """
        Test num_or_sections which is a list and date type is float64.
    """

    def set_input(self):
        self.shape = [6, 2, 4]
        self.num_or_sections = [2, 1, 3]
        self.x_np = np.random.uniform(-1, 1, self.shape).astype('float64')
        self.place = paddle.CUDAPlace(0) if core.is_compiled_with_cuda() \
            else paddle.CPUPlace()


class TestSplitsFloat32(TestSplitsAPI):
    """
        Test num_or_sections which is an integer and data type is float32.
    """

    def set_input(self):
        self.shape = [2, 3, 4]
        self.num_or_sections = 2
        self.x_np = np.random.uniform(-1, 1, self.shape).astype('float32')
        self.place = paddle.CUDAPlace(0) if core.is_compiled_with_cuda() \
            else paddle.CPUPlace()


class TestSplitsInt32(TestSplitsAPI):
    """
        Test data type int32.
    """

    def set_input(self):
        self.shape = [5, 1, 2]
        self.num_or_sections = 5
        self.x_np = np.random.uniform(-1, 1, self.shape).astype('int32')
        self.place = paddle.CUDAPlace(0) if core.is_compiled_with_cuda() \
            else paddle.CPUPlace()


class TestSplitsInt64(TestSplitsAPI):
    """
        Test data type int64.
    """

    def set_input(self):
        self.shape = [4, 3, 2]
        self.num_or_sections = 2
        self.x_np = np.random.uniform(-1, 1, self.shape).astype('int64')
        self.place = paddle.CUDAPlace(0) if core.is_compiled_with_cuda() \
            else paddle.CPUPlace()


class TestSplitsCPU(TestSplitsAPI):
    """
        Test cpu place and num_or_sections which is a tuple.
    """

    def set_input(self):
        self.shape = [8, 2, 3, 5]
        self.num_or_sections = (2, 3, 3)
        self.x_np = np.random.uniform(-1, 1, self.shape).astype('float64')
        self.place = paddle.CPUPlace()


class TestSplitsError(unittest.TestCase):
    """
        Test the situation that input shape less than 2.
    """

    def setUp(self):
        self.num_or_sections = 1
        self.place = paddle.CUDAPlace(0) if core.is_compiled_with_cuda() \
            else paddle.CPUPlace()

    def test_static_error(self):
        paddle.enable_static()
        for func, _ in test_list:
            with paddle.static.program_guard(paddle.static.Program()):
                x = paddle.fluid.data('X', [5], 'float32')
                self.assertRaises(ValueError, func, x, self.num_or_sections)

    def test_dygraph_error(self):
        paddle.disable_static(self.place)
        for func, _ in test_list:
            x_np = np.random.randn(2)
            x = paddle.to_tensor(x_np, dtype='float64')
            self.assertRaises(ValueError, func, x, self.num_or_sections)


if __name__ == '__main__':
    unittest.main()
