# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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
from test_cdist import TestCdistAPI

import paddle


# test for 0-size tensor: r2 == 0, 2D
class TestCdistZeroSizeR2(TestCdistAPI):
    def init_input(self):
        self.x = np.random.rand(900, 4).astype('float32')
        self.y = np.empty((0, 4), dtype='float32')
        self.p = 1.0

    def test_dygraph_api(self):
        paddle.disable_static(self.place)
        x = paddle.to_tensor(self.x)
        y = paddle.to_tensor(self.y)
        out = paddle.cdist(x, y, self.p, self.compute_mode)
        self.assertEqual(list(out.shape), [900, 0])
        paddle.enable_static()

    def test_static_api(self):
        paddle.enable_static()
        with paddle.static.program_guard(paddle.static.Program()):
            x = paddle.static.data('x', self.x.shape, dtype=self.x.dtype)
            y = paddle.static.data('y', self.y.shape, dtype=self.y.dtype)
            out = paddle.cdist(x, y, self.p, self.compute_mode)
            exe = paddle.static.Executor(self.place)
            res = exe.run(feed={'x': self.x, 'y': self.y}, fetch_list=[out])
            self.assertEqual(list(res[0].shape), [900, 0])


# test for 0-size tensor: r1 == 0, 2D
class TestCdistZeroSizeR1(TestCdistAPI):
    def init_input(self):
        self.x = np.empty((0, 4), dtype='float32')
        self.y = np.random.rand(900, 4).astype('float32')
        self.p = 1.0

    def test_dygraph_api(self):
        paddle.disable_static(self.place)
        x = paddle.to_tensor(self.x)
        y = paddle.to_tensor(self.y)
        out = paddle.cdist(x, y, self.p, self.compute_mode)
        self.assertEqual(list(out.shape), [0, 900])
        paddle.enable_static()

    def test_static_api(self):
        paddle.enable_static()
        with paddle.static.program_guard(paddle.static.Program()):
            x = paddle.static.data('x', self.x.shape, dtype=self.x.dtype)
            y = paddle.static.data('y', self.y.shape, dtype=self.y.dtype)
            out = paddle.cdist(x, y, self.p, self.compute_mode)
            exe = paddle.static.Executor(self.place)
            res = exe.run(feed={'x': self.x, 'y': self.y}, fetch_list=[out])
            self.assertEqual(list(res[0].shape), [0, 900])


# test for 0-size tensor: c1 == 0 (feature dim is 0, distance should be 0)
class TestCdistZeroSizeC1(TestCdistAPI):
    def init_input(self):
        self.x = np.empty((5, 0), dtype='float32')
        self.y = np.empty((3, 0), dtype='float32')
        self.p = 2.0

    def test_dygraph_api(self):
        paddle.disable_static(self.place)
        x = paddle.to_tensor(self.x)
        y = paddle.to_tensor(self.y)
        out = paddle.cdist(x, y, self.p, self.compute_mode)
        self.assertEqual(list(out.shape), [5, 3])
        np.testing.assert_allclose(
            out.numpy(), np.zeros((5, 3), dtype='float32')
        )
        paddle.enable_static()

    def test_static_api(self):
        paddle.enable_static()
        with paddle.static.program_guard(paddle.static.Program()):
            x = paddle.static.data('x', self.x.shape, dtype=self.x.dtype)
            y = paddle.static.data('y', self.y.shape, dtype=self.y.dtype)
            out = paddle.cdist(x, y, self.p, self.compute_mode)
            exe = paddle.static.Executor(self.place)
            res = exe.run(feed={'x': self.x, 'y': self.y}, fetch_list=[out])
            self.assertEqual(list(res[0].shape), [5, 3])
            np.testing.assert_allclose(
                res[0], np.zeros((5, 3), dtype='float32')
            )


# test for 0-size tensor: r2 == 0, 3D batched
class TestCdistZeroSizeBatch3D(TestCdistAPI):
    def init_input(self):
        self.x = np.random.rand(3, 5, 4).astype('float32')
        self.y = np.empty((3, 0, 4), dtype='float32')
        self.p = 1.0

    def test_dygraph_api(self):
        paddle.disable_static(self.place)
        x = paddle.to_tensor(self.x)
        y = paddle.to_tensor(self.y)
        out = paddle.cdist(x, y, self.p, self.compute_mode)
        self.assertEqual(list(out.shape), [3, 5, 0])
        paddle.enable_static()

    def test_static_api(self):
        paddle.enable_static()
        with paddle.static.program_guard(paddle.static.Program()):
            x = paddle.static.data('x', self.x.shape, dtype=self.x.dtype)
            y = paddle.static.data('y', self.y.shape, dtype=self.y.dtype)
            out = paddle.cdist(x, y, self.p, self.compute_mode)
            exe = paddle.static.Executor(self.place)
            res = exe.run(feed={'x': self.x, 'y': self.y}, fetch_list=[out])
            self.assertEqual(list(res[0].shape), [3, 5, 0])


# test for 0-size tensor: r2 == 0, 4D batched
class TestCdistZeroSizeBatch4D(TestCdistAPI):
    def init_input(self):
        self.x = np.random.rand(2, 3, 5, 4).astype('float32')
        self.y = np.empty((2, 3, 0, 4), dtype='float32')
        self.p = 2.0

    def test_dygraph_api(self):
        paddle.disable_static(self.place)
        x = paddle.to_tensor(self.x)
        y = paddle.to_tensor(self.y)
        out = paddle.cdist(x, y, self.p, self.compute_mode)
        self.assertEqual(list(out.shape), [2, 3, 5, 0])
        paddle.enable_static()

    def test_static_api(self):
        paddle.enable_static()
        with paddle.static.program_guard(paddle.static.Program()):
            x = paddle.static.data('x', self.x.shape, dtype=self.x.dtype)
            y = paddle.static.data('y', self.y.shape, dtype=self.y.dtype)
            out = paddle.cdist(x, y, self.p, self.compute_mode)
            exe = paddle.static.Executor(self.place)
            res = exe.run(feed={'x': self.x, 'y': self.y}, fetch_list=[out])
            self.assertEqual(list(res[0].shape), [2, 3, 5, 0])


# test for 0-size tensor: stop_gradient propagation
class TestCdistZeroSizeGrad(unittest.TestCase):
    def test_stop_gradient_false(self):
        paddle.disable_static()
        x = paddle.randn([10, 4], dtype='float32')
        x.stop_gradient = False
        y = paddle.to_tensor(np.empty((0, 4), dtype='float32'))
        y.stop_gradient = False
        out = paddle.cdist(x, y, p=1.0)
        self.assertEqual(out.stop_gradient, False)
        grads = paddle.grad(
            [out],
            [x, y],
            grad_outputs=[paddle.ones_like(out)],
            allow_unused=True,
        )
        self.assertIsNotNone(grads)
        paddle.enable_static()

    def test_stop_gradient_true(self):
        paddle.disable_static()
        x = paddle.randn([10, 4], dtype='float32')
        y = paddle.to_tensor(np.empty((0, 4), dtype='float32'))
        out = paddle.cdist(x, y, p=1.0)
        self.assertEqual(out.stop_gradient, True)
        paddle.enable_static()


if __name__ == '__main__':
    paddle.enable_static()
    unittest.main()
