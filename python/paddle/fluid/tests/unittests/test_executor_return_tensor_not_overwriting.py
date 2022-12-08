#  Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import unittest

import numpy as np
from op_test import OpTest, skip_check_grad_ci

import paddle
import paddle.fluid as fluid


@skip_check_grad_ci(reason="Not op test but call the method of class OpTest.")
class TestExecutorReturnTensorNotOverwritingWithOptest(OpTest):
    def setUp(self):
        pass

    def calc_add_out(self, place=None, parallel=None):
        self.x = np.random.random((2, 5)).astype(np.float32)
        self.y = np.random.random((2, 5)).astype(np.float32)
        self.out = np.add(self.x, self.y)
        self.inputs = {
            'X': OpTest.np_dtype_to_fluid_dtype(self.x),
            'Y': OpTest.np_dtype_to_fluid_dtype(self.y),
        }
        self.outputs = {'Out': self.out}
        self.op_type = "elementwise_add"
        self.dtype = np.float32
        outs, fetch_list = self._calc_output(place, parallel=parallel)
        return outs

    def calc_mul_out(self, place=None, parallel=None):
        self.x = np.random.random((2, 5)).astype(np.float32)
        self.y = np.random.random((5, 2)).astype(np.float32)
        self.out = np.dot(self.x, self.y)
        self.inputs = {
            'X': OpTest.np_dtype_to_fluid_dtype(self.x),
            'Y': OpTest.np_dtype_to_fluid_dtype(self.y),
        }
        self.outputs = {'Out': self.out}
        self.op_type = "mul"
        self.dtype = np.float32
        outs, fetch_list = self._calc_output(place, parallel=parallel)
        return outs

    def test_executor_run_twice(self):
        places = [fluid.CPUPlace()]
        if fluid.is_compiled_with_cuda():
            places.append(fluid.CUDAPlace(0))

        for place in places:
            for parallel in [True, False]:
                add_out = self.calc_add_out(place, parallel)
                add_out1 = np.array(add_out[0])
                mul_out = self.calc_mul_out(place, parallel)
                add_out2 = np.array(add_out[0])
                np.testing.assert_array_equal(add_out1, add_out2)


class TestExecutorReturnTensorNotOverOverwritingWithLayers(unittest.TestCase):
    def setUp(self):
        pass

    def calc_add_out(self, place=None, parallel=None):
        x = paddle.ones(shape=[3, 3], dtype='float32')
        y = paddle.ones(shape=[3, 3], dtype='float32')
        out = paddle.add(x=x, y=y)
        program = fluid.default_main_program()
        if parallel:
            program = fluid.CompiledProgram(program).with_data_parallel(
                places=place
            )
        exe = fluid.Executor(place)
        out = exe.run(program, fetch_list=[out], return_numpy=False)
        return out

    def calc_sub_out(self, place=None, parallel=None):
        x = paddle.ones(shape=[2, 2], dtype='float32')
        y = paddle.ones(shape=[2, 2], dtype='float32')
        out = paddle.subtract(x=x, y=y)
        program = fluid.default_main_program()
        if parallel:
            program = fluid.CompiledProgram(program).with_data_parallel(
                places=place
            )
        exe = fluid.Executor(place)
        out = exe.run(program, fetch_list=[out], return_numpy=False)
        return out

    def test_executor_run_twice(self):
        places = [fluid.CPUPlace()]
        if fluid.is_compiled_with_cuda():
            places.append(fluid.CUDAPlace(0))

        for place in places:
            for parallel in [True, False]:
                add_out = self.calc_add_out(place, parallel)
                add_out1 = np.array(add_out[0])
                sub_out = self.calc_sub_out(place, parallel)
                add_out2 = np.array(add_out[0])
                np.testing.assert_array_equal(add_out1, add_out2)


if __name__ == '__main__':
    unittest.main()
