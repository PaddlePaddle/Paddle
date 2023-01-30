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

<<<<<<< HEAD
=======
from __future__ import print_function

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
import unittest
import numpy as np
import sys

sys.path.append("..")
from op_test import OpTest
import paddle
<<<<<<< HEAD

=======
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
# from paddle.fluid import core
# from paddle.fluid.op import Operator

paddle.enable_static()
SEED = 2022


class TestShape(OpTest):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def setUp(self):
        self.set_mlu()
        self.op_type = "shape"
        self.place = paddle.MLUPlace(0)

        self.init_dtype()
        np.random.seed(SEED)
        x = np.random.uniform(1, 2, [5, 10]).astype(self.dtype)
        out = np.array([5, 10])

        self.inputs = {'Input': OpTest.np_dtype_to_fluid_dtype(x)}
        self.attrs = {}
        self.outputs = {'Out': out}

    def set_mlu(self):
        self.__class__.use_mlu = True

    def init_dtype(self):
        self.dtype = np.float32

    def test_check_output(self):
        self.check_output_with_place(self.place)


class TestShape_fp16(TestShape):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def init_dtype(self):
        self.dtype = np.float16


class TestShape_double(TestShape):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def init_dtype(self):
        self.dtype = np.float64


class TestShape_int32(TestShape):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def init_dtype(self):
        self.dtype = np.int32


class TestShape_int64(TestShape):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def init_dtype(self):
        self.dtype = np.int64


class TestShape_int8(TestShape):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def init_dtype(self):
        self.dtype = np.int8


class TestShape_uint8(TestShape):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def init_dtype(self):
        self.dtype = np.uint8


class TestShape_bool(TestShape):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def init_dtype(self):
        self.dtype = bool


if __name__ == '__main__':
    unittest.main()
