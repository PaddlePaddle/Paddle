#  Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

import numpy as np
import unittest
import paddle
import sys
sys.path.append("..")
from op_test import OpTest
from paddle.fluid.op import Operator
import paddle.fluid as fluid
from paddle.fluid import Program, program_guard

paddle.enable_static()


class TestWhereIndexOp(OpTest):
    def setUp(self):
        self.set_npu()
        self.op_type = "where_index"
        self.place = paddle.NPUPlace(0)
        self.init_config()

    def test_check_output(self):
        self.check_output_with_place(self.place)

    def init_config(self):
        self.inputs = {'Condition': np.array([True, False, True]), }

        self.outputs = {'Out': np.array([[0], [2]], dtype='int64')}

    def set_npu(self):
        self.__class__.use_npu = True


class TestNotBool(TestWhereIndexOp):
    def init_config(self):
        self.inputs = {'Condition': np.array([1, 0, 8]), }

        self.outputs = {'Out': np.array([[0], [2]], dtype='int64')}


class TestAllFalse(TestWhereIndexOp):
    def init_config(self):
        self.inputs = {'Condition': np.array([False, False, False]), }

        self.outputs = {'Out': np.array([], dtype='int64')}


class TestRank2(TestWhereIndexOp):
    def init_config(self):
        self.inputs = {'Condition': np.array([[True, False], [False, True]]), }

        self.outputs = {'Out': np.array([[0, 0], [1, 1]], dtype='int64')}


class TestRank3(TestWhereIndexOp):
    def init_config(self):
        self.inputs = {
            'Condition': np.array([[[True, False], [False, True]],
                                   [[False, True], [True, False]],
                                   [[False, False], [False, True]]]),
        }

        self.outputs = {
            'Out': np.array(
                [[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 0], [2, 1, 1]],
                dtype='int64')
        }


class TestWhereOpError(unittest.TestCase):
    def test_api(self):
        with program_guard(Program(), Program()):
            cond = fluid.layers.data(name='cond', shape=[4], dtype='bool')
            result = fluid.layers.where(cond)

            exe = fluid.Executor(paddle.NPUPlace(0))
            exe.run(fluid.default_startup_program())
            cond_i = np.array([True, False, False, False]).astype("bool")
            out = exe.run(fluid.default_main_program(), feed={'cond': cond_i})


class TestWhereRaiseError(unittest.TestCase):
    def test_errors(self):
        def test_type():
            fluid.layers.where([10])

        self.assertRaises(TypeError, test_type)


if __name__ == "__main__":
    unittest.main()
