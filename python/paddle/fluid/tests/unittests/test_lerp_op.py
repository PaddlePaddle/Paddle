# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import print_function

import unittest
import numpy as np
from op_test import OpTest
import paddle
import paddle.fluid.core as core

paddle.enable_static()
np.random.seed(0)


class TestLerp(OpTest):
    def setUp(self):
        self.op_type = "lerp"
        self.init_dtype()
        x = np.arange(1., 101.).astype(self.dtype)
        y = np.full(100, 10.).astype(self.dtype)
        self.attrs = {'WeightValue': 0.5}
        self.inputs = {'X': x, 'Y': y}
        self.outputs = {'Out': x + 0.5 * (y - x)}

    def init_dtype(self):
        self.dtype = np.float64

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(['X', 'Y'], 'Out')


class TestLerpAPI(unittest.TestCase):
    def init_dtype(self):
        self.dtype = 'float64'

    def setUp(self):
        self.init_dtype()
        self.x = np.arange(1., 5.).astype(self.dtype)
        self.y = np.full(4, 10.).astype(self.dtype)
        self.w = 0.5
        self.res_ref = self.x + 0.5 * (self.y - self.x)
        self.place = [paddle.CPUPlace()]
        if core.is_compiled_with_cuda():
            self.place.append(paddle.CUDAPlace(0))

    def test_static_api(self):
        paddle.enable_static()

        def run(place):
            with paddle.static.program_guard(paddle.static.Program()):
                x = paddle.fluid.data('x', [1, 4], dtype=self.dtype)
                y = paddle.fluid.data('y', [1, 4], dtype=self.dtype)
                out = paddle.lerp(x, y, 0.5)
                exe = paddle.static.Executor(place)
                res = exe.run(feed={
                    'x': self.x.reshape([1, 4]),
                    'y': self.y.reshape([1, 4])
                })
            for r in res:
                self.assertEqual(np.allclose(self.res_ref, r), True)

        for place in self.place:
            run(place)

    def test_dygraph_api(self):
        def run(place):
            paddle.disable_static(place)
            x = paddle.to_tensor(self.x)
            y = paddle.to_tensor(self.y)
            out = paddle.lerp(x, y, 0.5)
            self.assertEqual(np.allclose(self.res_ref, out.numpy()), True)
            paddle.enable_static()

        for place in self.place:
            run(place)

    def test_broadcast(self):
        paddle.disable_static()
        x = np.arange(1., 21.).astype(self.dtype).reshape([2, 2, 5])
        y = np.full(30, 10.).astype(self.dtype).reshape([3, 2, 1, 5])
        out = paddle.lerp(paddle.to_tensor(x), paddle.to_tensor(y), 0.5)
        res_ref = x + 0.5 * (y - x)
        self.assertEqual(np.allclose(res_ref, out.numpy()), True)
        paddle.enable_static()


if __name__ == "__main__":
    unittest.main()
