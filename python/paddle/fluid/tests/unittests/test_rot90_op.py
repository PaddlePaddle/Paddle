#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
from paddle.fluid import Program, program_guard
from op_test import OpTest


class TestRot90Op_API(unittest.TestCase):
    """Test rot90 api."""

    def test_static_graph(self):
        startup_program = fluid.Program()
        train_program = fluid.Program()
        with fluid.program_guard(train_program, startup_program):
            axis = [0]
            input = fluid.data(name='input', dtype='float32', shape=[2, 3])
            output = paddle.rot90(input, k=1, dims=[0, 1])
            output = paddle.rot90(output, k=1, dims=[0, 1])
            output = output.rot90(k=1, dims=[0, 1])
            place = fluid.CPUPlace()
            if fluid.core.is_compiled_with_cuda():
                place = fluid.CUDAPlace(0)
            exe = fluid.Executor(place)
            exe.run(startup_program)

            img = np.array([[1, 2, 3], [4, 5, 6]]).astype(np.float32)
            res = exe.run(train_program,
                          feed={'input': img},
                          fetch_list=[output])

            out_np = np.array(res[0])
            out_ref = np.array([[4, 1], [5, 2], [6, 3]]).astype(np.float32)

            self.assertTrue(
                (out_np == out_ref).all(),
                msg='rot90 output is wrong, out =' + str(out_np))

    def test_dygraph(self):
        img = np.array([[1, 2, 3], [4, 5, 6]]).astype(np.float32)
        with fluid.dygraph.guard():
            inputs = fluid.dygraph.to_variable(img)
            
            ret = paddle.rot90(inputs, k=1, dims=[0, 1])
            ret = ret.rot90(1, dims=[0, 1])
            ret = paddle.rot90(ret, k=1, dims=[0, 1])
            out_ref = np.array([[4, 1], [5, 2], [6, 3]]).astype(np.float32)

            self.assertTrue(
                (ret.numpy() == out_ref).all(),
                msg='rot90 output is wrong, out =' + str(ret.numpy()))


class TestRot90Op(OpTest):
    def setUp(self):
        self.op_type = 'rot90'
        self.init_test_case()
        self.inputs = {'X': np.random.random(self.in_shape).astype('float64')}
        self.init_attrs()
        self.outputs = {'Out': self.calc_ref_res()}

    def init_attrs(self):
        self.attrs = {"k": self.k, "dims": self.dims}

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(["X"], "Out")

    def init_test_case(self):
        self.in_shape = (6, 4, 2, 3)
        self.k = 1
        self.dims = [0, 1]

    def calc_ref_res(self):
        res = self.inputs['X']
        res = np.rot90(res, self.k, self.dims)
        #if isinstance(self.axis, int):
        #    return np.flip(res, self.axis)
        #for axis in self.axis:
        #    res = np.flip(res, axis)
        return res


class TestRot90OpK1(TestRot90Op):
    def init_test_case(self):
        self.in_shape = (2, 4, 4)
        self.k = 1
        self.dims = [0, 1]


class TestRot90OpK2(TestRot90Op):
    def init_test_case(self):
        self.in_shape = (4, 4, 6, 3)
        self.k = 2
        self.dims = [0, 1]


class TestRot90OpK3(TestRot90Op):
    def init_test_case(self):
        self.in_shape = (4, 3, 1)
        self.k = 3
        self.dims = [0, 1]


class TestRot90OpKNeg1(TestRot90Op):
    def init_test_case(self):
        self.in_shape = (6, 4, 2, 2)
        self.k = -1
        self.dims = [0, 1]


class TestRot90OpKNeg2(TestRot90Op):
    def init_test_case(self):
        self.in_shape = (6, 4, 2, 2)
        self.k = -2
        self.dims = [0, 1]


class TestRot90OpKNeg3(TestRot90Op):
    def init_test_case(self):
        self.in_shape = (6, 4, 2, 2)
        self.k = -3
        self.dims = [0, 1]


class TestRot90OpK0(TestRot90Op):
    def init_test_case(self):
        self.in_shape = (6, 4, 2, 2)
        self.k = 0
        self.dims = [0, 1]


if __name__ == "__main__":
    unittest.main()
