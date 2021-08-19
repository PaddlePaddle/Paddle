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

from __future__ import print_function
import unittest

import numpy as np
from scipy.special import expit, erf

from paddle.fluid.tests.unittests.op_test import OpTest, convert_float_to_uint16, skip_check_grad_ci
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import paddle.fluid as fluid
import paddle.fluid.core as core
from paddle.fluid import compiler, Program, program_guard

paddle.enable_static()


def test_class(op_type, typename):
    class TestSin(OpTest):
        def setUp(self):
            self.op_type = "sin"
            self.__class__.use_npu = True
            self.place = paddle.NPUPlace(0)
            self.__class__.no_need_check_grad = True
            np.random.seed(1024)
            x = np.random.uniform(-1, 1, [10, 12]).astype(typename)
            out = np.sin(x)

            self.inputs = {'X': OpTest.np_dtype_to_fluid_dtype(x)}
            self.outputs = {'Out': out}

        def test_check_output(self):
            self.check_output_with_place(self.place)

        def test_check_grad(self):
            pass

        def test_out_name(self):
            with fluid.program_guard(fluid.Program()):
                np_x = np.array([0.1])
                data = fluid.layers.data(name="X", shape=[1])
                out = eval("paddle.%s(data, name='Y')" % self.op_type)
                place = fluid.NPUPlace(0)
                exe = fluid.Executor(place)
                result, = exe.run(feed={"X": np_x}, fetch_list=[out])
                expected = eval("np.%s(np_x)" % self.op_type)
                self.assertEqual(result, expected)

        def test_dygraph(self):
            with fluid.dygraph.guard(paddle.NPUPlace(0)):
                np_x = np.array([0.1])
                x = fluid.dygraph.to_variable(np_x)
                z = eval("paddle.%s(x).numpy()" % self.op_type)
                z_expected = eval("np.%s(np_x)" % self.op_type)
                self.assertEqual(z, z_expected)

    cls_name = "{0}_{1}_1".format(op_type, typename)
    TestSin.__name__ = cls_name
    globals()[cls_name] = TestSin


for _typename in {'float16', 'float32', 'float64'}:
    test_class("sin", _typename)

if __name__ == "__main__":
    unittest.main()
