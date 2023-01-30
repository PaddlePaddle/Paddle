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

<<<<<<< HEAD
=======
from __future__ import print_function, division

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
import numpy as np
import unittest
import sys

sys.path.append("..")
from op_test import OpTest, skip_check_grad_ci
import paddle

paddle.enable_static()


class TestNPUReciprocal(OpTest):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def setUp(self):
        self.op_type = "reciprocal"
        self.set_npu()
        self.init_dtype()

        np.random.seed(1024)
        x = np.random.uniform(1, 2, [11, 17]).astype(self.dtype)
        out = np.reciprocal(x)

        self.inputs = {'X': OpTest.np_dtype_to_fluid_dtype(x)}
        self.outputs = {'Out': out}

    def test_check_output(self):
        self.check_output_with_place(self.place)

    def test_check_grad(self):
<<<<<<< HEAD
        self.check_grad_with_place(
            self.place, ['X'], 'Out', max_relative_error=0.01
        )
=======
        self.check_grad_with_place(self.place, ['X'],
                                   'Out',
                                   max_relative_error=0.01)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

    def set_npu(self):
        self.__class__.use_npu = True
        self.place = paddle.NPUPlace(0)

    def init_dtype(self):
        self.dtype = np.float32


class TestNPUReciprocalFp64(TestNPUReciprocal):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def set_npu(self):
        self.__class__.use_npu = True
        self.place = paddle.NPUPlace(0)

    def init_dtype(self):
        self.dtype = np.float64


@skip_check_grad_ci(
<<<<<<< HEAD
    reason="The backward test is not supported for float16 type on NPU."
)
class TestNPUReciprocalFp16(TestNPUReciprocal):
=======
    reason="The backward test is not supported for float16 type on NPU.")
class TestNPUReciprocalFp16(TestNPUReciprocal):

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def set_npu(self):
        self.__class__.use_npu = True
        self.place = paddle.NPUPlace(0)
        self.__class__.no_need_check_grad = True

    def init_dtype(self):
        self.dtype = np.float16


if __name__ == '__main__':
    unittest.main()
