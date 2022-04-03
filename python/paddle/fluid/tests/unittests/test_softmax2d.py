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

from __future__ import print_function

import unittest
import numpy as np
import paddle
import paddle.fluid as fluid
import paddle.fluid.core as core
from test_softmax_op import ref_softmax

paddle.enable_static()
np.random.seed(2022)


class TestSoftmax2DAPI(unittest.TestCase):
    # test paddle.nn.Softmax2D
    def setUp(self):
        self.shape = [2, 6, 5, 4]
        self.dtype = 'float64'
        self.x_np = np.random.uniform(-1, 1, self.shape).astype('float64')
        self.axis = -3
        self.place = paddle.CUDAPlace(0) if core.is_compiled_with_cuda() \
            else paddle.CPUPlace()

    def test_static_api(self):
        with paddle.static.program_guard(paddle.static.Program()):
            x = paddle.fluid.data('X', self.x_np.shape, self.x_np.dtype)
            m = paddle.nn.Softmax2D()
            out = m(x)
            exe = paddle.static.Executor(self.place)
            res = exe.run(feed={'X': self.x_np}, fetch_list=[out])
        out_ref = ref_softmax(self.x_np, self.axis)
        self.assertTrue(np.allclose(out_ref, res))

    def test_dygraph_api(self):
        paddle.disable_static(self.place)
        x = paddle.to_tensor(self.x_np)
        m = paddle.nn.Softmax2D()
        out = m(x)
        out_ref = ref_softmax(self.x_np, self.axis)
        self.assertTrue(np.allclose(out_ref, out.numpy()))
        paddle.enable_static()


class TestSoftmax2DShape(TestSoftmax2DAPI):
    def setUp(self):
        self.shape = [2, 6, 4]
        self.dtype = 'float64'
        self.x_np = np.random.uniform(-1, 1, self.shape).astype('float64')
        self.axis = -3
        self.place = paddle.CUDAPlace(0) if core.is_compiled_with_cuda() \
            else paddle.CPUPlace()


class TestSoftmax2DCPU(TestSoftmax2DAPI):
    def setUp(self):
        self.shape = [2, 6, 4]
        self.dtype = 'float64'
        self.x_np = np.random.uniform(-1, 1, self.shape).astype('float64')
        self.axis = -3
        self.place = paddle.CPUPlace()


class TestSoftmax2DRepr(unittest.TestCase):
    def setUp(self):
        self.place = paddle.CUDAPlace(0) if core.is_compiled_with_cuda() \
            else paddle.CPUPlace()

    def test_extra_repr(self):
        paddle.disable_static(self.place)
        m = paddle.nn.Softmax2D(name='test')
        self.assertTrue(m.extra_repr() == 'name=test')
        paddle.enable_static()


if __name__ == '__main__':
    unittest.main()
