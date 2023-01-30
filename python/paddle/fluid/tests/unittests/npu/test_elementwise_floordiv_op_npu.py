#  Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

<<<<<<< HEAD
=======
from __future__ import print_function

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
import numpy as np
import unittest
import sys

sys.path.append("..")
from op_test import OpTest
import paddle

paddle.enable_static()


class TestElementwiseFloorDiv(OpTest):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def setUp(self):
        self.op_type = "elementwise_floordiv"
        self.set_npu()
        self.init_dtype()
        self.init_input_output()

        self.inputs = {
            'X': OpTest.np_dtype_to_fluid_dtype(self.x),
<<<<<<< HEAD
            'Y': OpTest.np_dtype_to_fluid_dtype(self.y),
=======
            'Y': OpTest.np_dtype_to_fluid_dtype(self.y)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        }
        self.attrs = {}
        self.outputs = {'Out': self.out}

    def set_npu(self):
        self.__class__.use_npu = True
        self.place = paddle.NPUPlace(0)

    def init_input_output(self):
        self.x = np.random.uniform(1, 1000, [10, 10]).astype(self.dtype)
        self.y = np.random.uniform(1, 1000, [10, 10]).astype(self.dtype)
        self.out = np.floor_divide(self.x, self.y)

    def init_dtype(self):
        self.dtype = "int64"

    def test_check_output(self):
        self.check_output_with_place(self.place)


class TestElementwiseFloorDiv2(TestElementwiseFloorDiv):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def init_dtype(self):
        self.dtype = "int32"


if __name__ == '__main__':
    unittest.main()
