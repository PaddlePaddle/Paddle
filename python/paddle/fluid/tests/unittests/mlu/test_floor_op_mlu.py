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
from __future__ import print_function
=======
>>>>>>> 43b92b633f5d2db98f45d4b9597e5389f6f9712f
import unittest

import numpy as np
import sys

sys.path.append('..')
from op_test import OpTest
import paddle

paddle.enable_static()


class TestFloor(OpTest):
<<<<<<< HEAD

=======
>>>>>>> 43b92b633f5d2db98f45d4b9597e5389f6f9712f
    def setUp(self):
        self.op_type = "floor"
        self.place = paddle.device.MLUPlace(0)
        self.__class__.use_mlu = True
        self.init_dtype()
        self.__class__.no_need_check_grad = True
        self.python_api = paddle.floor

        np.random.seed(1024)
        x = np.random.uniform(-1, 1, [10, 12]).astype(self.dtype)
        out = np.floor(x)

        self.inputs = {'X': OpTest.np_dtype_to_fluid_dtype(x)}
        self.outputs = {'Out': out}

    def test_check_output(self):
        self.check_output_with_place(self.place, check_eager=False)

    def init_dtype(self):
        self.dtype = np.float32


class TestFloorFP16(TestFloor):
<<<<<<< HEAD

=======
>>>>>>> 43b92b633f5d2db98f45d4b9597e5389f6f9712f
    def init_dtype(self):
        self.dtype = np.float16


if __name__ == '__main__':
    unittest.main()
