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

<<<<<<< HEAD
import sys
import unittest

import numpy as np

sys.path.append("..")

from op_test_xpu import XPUOpTest
from xpu.get_test_cover_info import (
    XPUOpTestWrapper,
    create_test_class,
    get_xpu_op_support_types,
)

=======
from __future__ import print_function

import numpy as np
import unittest
import sys

sys.path.append("..")

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
import paddle
import paddle.fluid as fluid
from paddle.fluid import Program, program_guard

<<<<<<< HEAD
=======
from op_test_xpu import XPUOpTest
from xpu.get_test_cover_info import create_test_class, get_xpu_op_support_types, XPUOpTestWrapper

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
paddle.enable_static()


class XPUTestWhereIndexOp(XPUOpTestWrapper):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def __init__(self):
        self.op_name = 'where_index'

    class TestWhereIndexOp(XPUOpTest):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        def setUp(self):
            self.init_config()
            self.init_data()

        def test_check_output(self):
            self.check_output_with_place(self.place)

        def init_data(self):
            self.inputs = {
                'Condition': np.array([True, False, True]).astype(self.dtype),
            }
            self.outputs = {'Out': np.array([[0], [2]], dtype='int64')}

        def init_config(self):
            self.op_type = "where_index"
            self.place = paddle.XPUPlace(0)
            self.dtype = self.in_type
            self.__class__.no_need_check_grad = True

    class TestAllFalse(TestWhereIndexOp):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        def init_data(self):
            self.inputs = {
                'Condition': np.array([False, False, False]).astype(self.dtype),
            }
            self.outputs = {'Out': np.array([], dtype='int64')}

    class TestRank2(TestWhereIndexOp):
<<<<<<< HEAD
        def init_data(self):
            self.inputs = {
                'Condition': np.array([[True, False], [False, True]]).astype(
                    self.dtype
                ),
=======

        def init_data(self):
            self.inputs = {
                'Condition':
                np.array([[True, False], [False, True]]).astype(self.dtype),
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
            }
            self.outputs = {'Out': np.array([[0, 0], [1, 1]], dtype='int64')}

    class TestRank3(TestWhereIndexOp):
<<<<<<< HEAD
        def init_data(self):
            self.inputs = {
                'Condition': np.array(
                    [
                        [[True, False], [False, True]],
                        [[False, True], [True, False]],
                        [[False, False], [False, True]],
                    ]
                ).astype(self.dtype),
            }

            self.outputs = {
                'Out': np.array(
                    [[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 0], [2, 1, 1]],
                    dtype='int64',
                )
=======

        def init_data(self):
            self.inputs = {
                'Condition':
                np.array([[[True, False], [False, True]],
                          [[False, True], [True, False]],
                          [[False, False], [False, True]]]).astype(self.dtype),
            }

            self.outputs = {
                'Out':
                np.array(
                    [[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 0], [2, 1, 1]],
                    dtype='int64')
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
            }


support_types = get_xpu_op_support_types('where_index')
for stype in support_types:
    create_test_class(globals(), XPUTestWhereIndexOp, stype)


class TestWhereOpError(unittest.TestCase):
<<<<<<< HEAD
    def test_api(self):
        with program_guard(Program(), Program()):
            cond = paddle.static.data(name='cond', shape=[-1, 4], dtype='bool')
            result = paddle.nonzero(cond)
=======

    def test_api(self):
        with program_guard(Program(), Program()):
            cond = fluid.layers.data(name='cond', shape=[4], dtype='bool')
            result = fluid.layers.where(cond)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

            exe = fluid.Executor(paddle.XPUPlace(0))
            exe.run(fluid.default_startup_program())
            cond_i = np.array([True, False, False, False]).astype("bool")
            out = exe.run(fluid.default_main_program(), feed={'cond': cond_i})


class TestWhereRaiseError(unittest.TestCase):
<<<<<<< HEAD
    def test_errors(self):
        def test_type():
            paddle.nonzero([10])

        self.assertRaises(AttributeError, test_type)
=======

    def test_errors(self):

        def test_type():
            fluid.layers.where([10])

        self.assertRaises(TypeError, test_type)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81


if __name__ == "__main__":
    unittest.main()
