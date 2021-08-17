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

from __future__ import print_function

import numpy as np
import unittest
import sys
sys.path.append("..")
from op_test import OpTest
import paddle
import paddle.fluid as fluid

paddle.enable_static()
SEED = 1024


class TestAtan(OpTest):
    def setUp(self):
        self.set_npu()
        self.op_type = "atan"
        self.place = paddle.NPUPlace(0)

        self.dtype = np.float32
        np.random.seed(SEED)
        self.shape = [11, 17]
        x = np.random.uniform(0.1, 1, self.shape).astype(self.dtype)
        out = np.arctan(x)

        self.inputs = {'X': OpTest.np_dtype_to_fluid_dtype(x)}
        self.outputs = {'Out': out}

    def set_attrs(self):
        pass

    def set_npu(self):
        self.__class__.use_npu = True

    def test_check_grad(self):
        self.check_grad_with_place(self.place, ['X'], 'Out')

    def test_out_name(self):
        with fluid.program_guard(fluid.Program()):
            np_x = np.array([0.1])
            data = fluid.layers.data(name="X", shape=[1])
            out = paddle.atan(data, name='Y')
            place = paddle.NPUPlace(0)
            exe = fluid.Executor(place)
            result, = exe.run(feed={"X": np_x}, fetch_list=[out])
            expected = np.arctan(np_x)
            self.assertEqual(result, expected)

    def test_dygraph(self):
        with fluid.dygraph.guard(paddle.NPUPlace(0)):
            np_x = np.array([0.1])
            x = fluid.dygraph.to_variable(np_x)
            z = paddle.atan(x).numpy()
            z_expected = np.arctan(np_x)
            self.assertEqual(z, z_expected)

    def test_check_output(self):
        self.check_output_with_place(self.place)


class TestAtanShape(TestAtan):
    def set_attrs(self):
        self.shape = [12, 23, 10]


class TestAtanFloat16(TestAtan):
    def set_attrs(self):
        self.dtype = np.float16


if __name__ == '__main__':
    unittest.main()
