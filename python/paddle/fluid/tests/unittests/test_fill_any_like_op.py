#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

import paddle
import paddle.fluid as fluid
import paddle.fluid.core as core
from paddle.fluid import Program, program_guard
import paddle.compat as cpt
import unittest
import numpy as np
from op_test import OpTest


class TestFillAnyLikeOp(OpTest):
    def setUp(self):
        self.op_type = "fill_any_like"
        self.dtype = np.int32
        self.value = 0.0
        self.init()
        self.inputs = {'X': np.random.random((219, 232)).astype(self.dtype)}
        self.attrs = {'value': self.value}
        self.outputs = {'Out': self.value * np.ones_like(self.inputs["X"])}

    def init(self):
        pass

    def test_check_output(self):
        self.check_output()


class TestFillAnyLikeOpFloat32(TestFillAnyLikeOp):
    def init(self):
        self.dtype = np.float32
        self.value = 0.0


class TestFillAnyLikeOpValue1(TestFillAnyLikeOp):
    def init(self):
        self.value = 1.0


class TestFillAnyLikeOpValue2(TestFillAnyLikeOp):
    def init(self):
        self.value = 1e-10


class TestFillAnyLikeOpValue3(TestFillAnyLikeOp):
    def init(self):
        self.value = 1e-100


class TestFillAnyLikeOpType(TestFillAnyLikeOp):
    def setUp(self):
        self.op_type = "fill_any_like"
        self.dtype = np.int32
        self.value = 0.0
        self.init()
        self.inputs = {'X': np.random.random((219, 232)).astype(self.dtype)}
        self.attrs = {
            'value': self.value,
            'dtype': int(core.VarDesc.VarType.FP32)
        }
        self.outputs = {
            'Out':
            self.value * np.ones_like(self.inputs["X"]).astype(np.float32)
        }


class TestFillAnyLikeOpOverflow(TestFillAnyLikeOp):
    def init(self):
        self.value = 1e100

    def test_check_output(self):
        exception = None
        try:
            self.check_output(check_dygraph=False)
        except core.EnforceNotMet as ex:
            exception = ex
        self.assertIsNotNone(exception)


class TestFillAnyLikeOpFloat16(TestFillAnyLikeOp):
    def init(self):
        self.dtype = np.float16


class TestFillAnyLikeOp_attr_out(unittest.TestCase):
    """ Test fill_any_like op(whose API is full_like) for attr out. """

    def test_attr_tensor_API(self):
        startup_program = fluid.Program()
        train_program = fluid.Program()
        with fluid.program_guard(train_program, startup_program):
            fill_value = 2.0
            input = fluid.data(name='input', dtype='float32', shape=[2, 3])
            output = fluid.layers.full_like(input, fill_value)
            output_dtype = fluid.layers.full_like(
                input, fill_value, dtype='float32')

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
            self.assertTrue(
                not (out_np - np.full_like(img, fill_value)).any(),
                msg="full_like output is wrong, out = " + str(out_np))


class TestFillAnyLikeOpError(unittest.TestCase):
    def test_errors(self):
        with program_guard(Program(), Program()):
            #for ci coverage

            input_data = fluid.data(name='input', dtype='float32', shape=[2, 3])
            output = fluid.layers.full_like(input_data, 2.0)

            def test_input_dtype():
                fluid.layers.full_like

            self.assertRaises(
                ValueError,
                fluid.layers.full_like,
                input=input_data,
                fill_value=2,
                dtype='uint4')
            self.assertRaises(
                TypeError,
                fluid.layers.full_like,
                input=input_data,
                fill_value=2,
                dtype='int16')


if __name__ == "__main__":
    unittest.main()
