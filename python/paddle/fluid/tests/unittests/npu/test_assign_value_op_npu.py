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

import unittest
import numpy as np
import sys

sys.path.append("..")

import op_test
import paddle
import paddle.fluid as fluid
import paddle.fluid.framework as framework
import paddle.fluid.layers as layers

paddle.enable_static()
np.random.seed(2021)


class TestAssignValueNPUOp(op_test.OpTest):

    def setUp(self):
        self.set_npu()
        self.place = paddle.NPUPlace(0)

        self.op_type = "assign_value"
        self.inputs = {}
        self.attrs = {}
        self.init_data()

        self.attrs["shape"] = self.value.shape
        self.attrs["dtype"] = framework.convert_np_dtype_to_dtype_(
            self.value.dtype)
        self.outputs = {"Out": self.value}

    def set_npu(self):
        self.__class__.use_npu = True

    def init_data(self):
        self.value = np.random.random(size=(2, 5)).astype(np.float32)
        self.attrs["fp32_values"] = [float(v) for v in self.value.flat]

    def test_forward(self):
        self.check_output_with_place(self.place)


class TestAssignValueNPUOp2(TestAssignValueNPUOp):

    def init_data(self):
        self.value = np.random.random(size=(2, 5)).astype(np.int32)
        self.attrs["int32_values"] = [int(v) for v in self.value.flat]


class TestAssignValueNPUOp3(TestAssignValueNPUOp):

    def init_data(self):
        self.value = np.random.random(size=(2, 5)).astype(np.int64)
        self.attrs["int64_values"] = [int(v) for v in self.value.flat]


class TestAssignValueNPUOp4(TestAssignValueNPUOp):

    def init_data(self):
        self.value = np.random.choice(a=[False, True],
                                      size=(2, 5)).astype(np.bool)
        self.attrs["bool_values"] = [int(v) for v in self.value.flat]


class TestAssignApi(unittest.TestCase):

    def setUp(self):
        self.init_dtype()
        self.value = (-100 + 200 * np.random.random(size=(2, 5))).astype(
            self.dtype)
        self.place = fluid.NPUPlace(
            0) if fluid.core.is_compiled_with_npu() else fluid.CPUPlace()

    def init_dtype(self):
        self.dtype = "float32"

    def test_assign(self):
        main_program = fluid.Program()
        with fluid.program_guard(main_program):
            x = layers.create_tensor(dtype=self.dtype)
            layers.assign(input=self.value, output=x)

        exe = fluid.Executor(self.place)
        [fetched_x] = exe.run(main_program, feed={}, fetch_list=[x])
        np.testing.assert_allclose(fetched_x, self.value)
        self.assertEqual(fetched_x.dtype, self.value.dtype)


class TestAssignApi2(TestAssignApi):

    def init_dtype(self):
        self.dtype = "int32"


class TestAssignApi3(TestAssignApi):

    def init_dtype(self):
        self.dtype = "int64"


class TestAssignApi4(TestAssignApi):

    def setUp(self):
        self.init_dtype()
        self.value = np.random.choice(a=[False, True],
                                      size=(2, 5)).astype(np.bool)
        self.place = fluid.NPUPlace(
            0) if fluid.core.is_compiled_with_npu() else fluid.CPUPlace()

    def init_dtype(self):
        self.dtype = "bool"


if __name__ == '__main__':
    unittest.main()
