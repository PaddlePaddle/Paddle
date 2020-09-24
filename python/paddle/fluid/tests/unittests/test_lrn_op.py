#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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
import paddle.fluid as fluid
import paddle.fluid.core as core
from op_test import OpTest
from paddle.fluid import compiler, Program, program_guard


class TestLRNOp(OpTest):
    def get_input(self):
        ''' TODO(gongweibao): why it's grad diff is so large?
        x = np.ndarray(
            shape=(self.N, self.C, self.H, self.W), dtype=float, order='C')
        for m in range(0, self.N):
            for i in range(0, self.C):
                for h in range(0, self.H):
                    for w in range(0, self.W):
                        x[m][i][h][w] = m * self.C * self.H * self.W +  \
                                        i * self.H * self.W +  \
                                        h * self.W + w + 1
        '''
        x = np.random.rand(self.N, self.C, self.H, self.W).astype("float32")
        return x + 1

    def get_out(self):
        start = -(self.n - 1) // 2
        end = start + self.n

        mid = np.empty((self.N, self.C, self.H, self.W)).astype("float32")
        mid.fill(self.k)
        for m in range(0, self.N):
            for i in range(0, self.C):
                for c in range(start, end):
                    ch = i + c
                    if ch < 0 or ch >= self.C:
                        continue

                    s = mid[m][i][:][:]
                    r = self.x[m][ch][:][:]
                    s += np.square(r) * self.alpha

        mid2 = np.power(mid, -self.beta)
        return np.multiply(self.x, mid2), mid

    def get_attrs(self):
        attrs = {
            'n': self.n,
            'k': self.k,
            'alpha': self.alpha,
            'beta': self.beta,
            'data_format': self.data_format
        }
        return attrs

    def setUp(self):
        self.op_type = "lrn"
        self.init_test_case()

        self.N = 2
        self.C = 3
        self.H = 5
        self.W = 5

        self.n = 5
        self.k = 2.0
        self.alpha = 0.0001
        self.beta = 0.75
        self.x = self.get_input()
        self.out, self.mid_out = self.get_out()
        if self.data_format == 'NHWC':
            self.x = np.transpose(self.x, [0, 2, 3, 1])
            self.out = np.transpose(self.out, [0, 2, 3, 1])
            self.mid_out = np.transpose(self.mid_out, [0, 2, 3, 1])

        self.inputs = {'X': self.x}
        self.outputs = {'Out': self.out, 'MidOut': self.mid_out}
        self.attrs = self.get_attrs()

    def init_test_case(self):
        self.data_format = 'NCHW'

    def test_check_output(self):
        self.check_output()

    def test_check_grad_normal(self):
        self.check_grad(['X'], 'Out')


class TestLRNOpAttrDataFormat(TestLRNOp):
    def init_test_case(self):
        self.data_format = 'NHWC'


class TestLRNAPI(unittest.TestCase):
    def test_case(self):
        data1 = fluid.data(name='data1', shape=[2, 4, 5, 5], dtype='float32')
        data2 = fluid.data(name='data2', shape=[2, 5, 5, 4], dtype='float32')
        out1 = fluid.layers.lrn(data1, data_format='NCHW')
        out2 = fluid.layers.lrn(data2, data_format='NHWC')
        data1_np = np.random.random((2, 4, 5, 5)).astype("float32")
        data2_np = np.transpose(data1_np, [0, 2, 3, 1])

        if core.is_compiled_with_cuda():
            place = core.CUDAPlace(0)
        else:
            place = core.CPUPlace()
        exe = fluid.Executor(place)
        exe.run(fluid.default_startup_program())
        results = exe.run(fluid.default_main_program(),
                          feed={"data1": data1_np,
                                "data2": data2_np},
                          fetch_list=[out1, out2],
                          return_numpy=True)

        self.assertTrue(
            np.allclose(results[0], np.transpose(results[1], (0, 3, 1, 2))))

    def test_exception(self):
        input1 = fluid.data(name="input1", shape=[2, 4, 5, 5], dtype="float32")
        input2 = fluid.data(
            name="input2", shape=[2, 4, 5, 5, 5], dtype="float32")

        def _attr_data_fromat():
            out = fluid.layers.lrn(input1, data_format='NDHW')

        def _input_dim_size():
            out = fluid.layers.lrn(input2)

        self.assertRaises(ValueError, _attr_data_fromat)
        self.assertRaises(ValueError, _input_dim_size)


class TestLRNOpError(unittest.TestCase):
    def test_errors(self):
        with program_guard(Program(), Program()):
            # the input must be float32
            in_w = fluid.data(name="in_w", shape=[None, 3, 3, 3], dtype="int64")
            self.assertRaises(TypeError, fluid.layers.lrn, in_w)


if __name__ == "__main__":
    unittest.main()
