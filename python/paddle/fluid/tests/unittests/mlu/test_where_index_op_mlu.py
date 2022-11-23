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

import sys

sys.path.append("..")
import unittest
import numpy as np
from op_test import OpTest
import paddle.fluid.core as core
from paddle.fluid.op import Operator
import paddle.fluid as fluid
from paddle.fluid import Program, program_guard
import paddle

paddle.enable_static()


class TestWhereIndexOp(OpTest):

    def setUp(self):
        self.op_type = "where_index"
        self.place = paddle.device.MLUPlace(0)
        self.__class__.use_mlu = True
        self.init_config()

    def test_check_output(self):
        self.check_output_with_place(self.place)

    def init_config(self):
        self.inputs = {
            'Condition': np.array([True, False, True]),
        }

        self.outputs = {'Out': np.array([[0], [2]], dtype='int64')}


class TestAllFalse(unittest.TestCase):

    def setUp(self):
        self.op_type = "where_index"
        self.place = paddle.device.MLUPlace(0)
        self.__class__.use_mlu = True
        self.init_config()

    def check_with_place(self, place):
        scope = core.Scope()
        condition = scope.var('Condition').get_tensor()
        condition.set(self.cond_data, place)

        out = scope.var("Out").get_tensor()
        out.set(np.full(self.shape, 0).astype('int64'), place)

        op = Operator("where_index", Condition="Condition", Out="Out")
        op.run(scope, place)

        out_array = np.array(out)
        self.assertTrue((out_array == self.out_data).all())

    def init_config(self):
        self.cond_data = np.array([False, False, False])
        self.shape = (3, 1)
        self.out_data = np.array([], dtype='int64')

    def test_all_false(self):
        self.check_with_place(self.place)


class TestRank2(TestWhereIndexOp):

    def init_config(self):
        self.inputs = {
            'Condition': np.array([[True, False], [False, True]]),
        }

        self.outputs = {'Out': np.array([[0, 0], [1, 1]], dtype='int64')}


class TestRank3(TestWhereIndexOp):

    def init_config(self):
        self.inputs = {
            'Condition':
            np.array([[[True, False], [False, True]],
                      [[False, True], [True, False]],
                      [[False, False], [False, True]]]),
        }

        self.outputs = {
            'Out':
            np.array([[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 0], [2, 1, 1]],
                     dtype='int64')
        }


class TestWhereOpError(unittest.TestCase):

    def test_api(self):
        with program_guard(Program(), Program()):
            cond = fluid.layers.data(name='cond', shape=[4], dtype='bool')
            result = fluid.layers.where(cond)

            exe = fluid.Executor(paddle.device.MLUPlace(0))
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
