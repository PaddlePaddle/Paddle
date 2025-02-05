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

import sys
import unittest

import numpy as np

sys.path.append("../deprecated/legacy_test")
from test_softmax_op import ref_softmax

import paddle
from paddle.base import core


class TestSoftmax2DAPI(unittest.TestCase):
    def setUp(self):
        self.shape = [2, 6, 5, 4]
        self.x_np = np.random.uniform(-1, 1, self.shape).astype('float64')
        self.axis = -3
        self.place = (
            paddle.CUDAPlace(0)
            if core.is_compiled_with_cuda()
            else paddle.CPUPlace()
        )

    def test_static_api(self):
        paddle.enable_static()
        with paddle.static.program_guard(paddle.static.Program()):
            x = paddle.static.data('X', self.x_np.shape, self.x_np.dtype)
            m = paddle.nn.Softmax2D()
            out = m(x)
            exe = paddle.static.Executor(self.place)
            (res,) = exe.run(feed={'X': self.x_np}, fetch_list=[out])
        out_ref = ref_softmax(self.x_np, self.axis)
        np.testing.assert_allclose(out_ref, res, rtol=1e-05)

    def test_dygraph_api(self):
        paddle.disable_static(self.place)
        x = paddle.to_tensor(self.x_np)
        m = paddle.nn.Softmax2D()
        out = m(x)
        out_ref = ref_softmax(self.x_np, self.axis)
        np.testing.assert_allclose(out_ref, out.numpy(), rtol=1e-05)
        paddle.enable_static()


class TestSoftmax2DShape(TestSoftmax2DAPI):
    def setUp(self):
        self.shape = [2, 6, 4]
        self.x_np = np.random.uniform(-1, 1, self.shape).astype('float64')
        self.axis = -3
        self.place = (
            paddle.CUDAPlace(0)
            if core.is_compiled_with_cuda()
            else paddle.CPUPlace()
        )


class TestSoftmax2DFloat32(TestSoftmax2DAPI):
    def setUp(self):
        self.shape = [2, 3, 4]
        self.x_np = np.random.uniform(-1, 1, self.shape).astype('float32')
        self.axis = -3
        self.place = (
            paddle.CUDAPlace(0)
            if core.is_compiled_with_cuda()
            else paddle.CPUPlace()
        )


class TestSoftmax2DCPU(TestSoftmax2DAPI):
    def setUp(self):
        self.shape = [2, 6, 4]
        self.x_np = np.random.uniform(-1, 1, self.shape).astype('float64')
        self.axis = -3
        self.place = paddle.CPUPlace()


class TestSoftmax2DRepr(unittest.TestCase):
    def setUp(self):
        self.place = (
            paddle.CUDAPlace(0)
            if core.is_compiled_with_cuda()
            else paddle.CPUPlace()
        )

    def test_extra_repr(self):
        paddle.disable_static(self.place)
        m = paddle.nn.Softmax2D(name='test')
        self.assertTrue(m.extra_repr() == 'name=test')
        paddle.enable_static()


class TestSoftmax2DError(unittest.TestCase):
    def setUp(self):
        self.place = (
            paddle.CUDAPlace(0)
            if core.is_compiled_with_cuda()
            else paddle.CPUPlace()
        )

    def test_static_error(self):
        paddle.enable_static()
        with paddle.static.program_guard(paddle.static.Program()):
            x = paddle.static.data('X', [5, 5], 'float32')
            m = paddle.nn.Softmax2D()
            self.assertRaises(AssertionError, m, x)

    def test_dygraph_error(self):
        paddle.disable_static(self.place)
        x_np = np.random.randn(2, 3, 4, 2, 3)
        x = paddle.to_tensor(x_np, dtype='float64')
        m = paddle.nn.Softmax2D()
        self.assertRaises(AssertionError, m, x)


if __name__ == '__main__':
    unittest.main()
