#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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
import sys
sys.path.append("..")
from op_test import OpTest
from op_test_xpu import XPUOpTest
import paddle
import paddle.fluid as fluid
from paddle.fluid import compiler, Program, program_guard
from paddle.fluid import core
from xpu.get_test_cover_info import create_test_class, get_xpu_op_support_types, XPUOpTestWrapper

paddle.enable_static()
np.random.seed(10)


#Situation 1: repeat_times is a list (without tensor)
class XPUTestTileOpRank1(XPUOpTestWrapper):
    def __init__(self):
        self.op_name = 'tile'
        self.use_dynamic_create_class = False

    class TestTileOpRank1(XPUOpTest):
        def setUp(self):
            self.dtype = self.in_type
            self.__class__.no_need_check_grad = True
            self.place = paddle.XPUPlace(0)
            self.op_type = "tile"
            self.init_data()
            self.inputs = {
                'X': np.random.random(self.ori_shape).astype(self.dtype)
            }
            self.attrs = {'repeat_times': self.repeat_times}
            output = np.tile(self.inputs['X'], self.repeat_times)
            self.outputs = {'Out': output}

        def init_data(self):
            self.ori_shape = [100]
            self.repeat_times = [2]

        def test_check_output(self):
            self.check_output_with_place(self.place)

    #with dimension expanding
    class TestTileOpRank2Expanding(TestTileOpRank1):
        def init_data(self):
            self.ori_shape = [120]
            self.repeat_times = [2, 2]

    class TestTileOpRank2(TestTileOpRank1):
        def init_data(self):
            self.ori_shape = [12, 14]
            self.repeat_times = [2, 3]

    class TestTileOpRank3_Corner(TestTileOpRank1):
        def init_data(self):
            self.ori_shape = (2, 10, 5)
            self.repeat_times = (1, 1, 1)

    class TestTileOpRank3_Corner2(TestTileOpRank1):
        def init_data(self):
            self.ori_shape = (2, 10, 5)
            self.repeat_times = (2, 2)

    class TestTileOpRank3(TestTileOpRank1):
        def init_data(self):
            self.ori_shape = (2, 4, 15)
            self.repeat_times = (2, 1, 4)

    class TestTileOpRank4(TestTileOpRank1):
        def init_data(self):
            self.ori_shape = (2, 4, 5, 7)
            self.repeat_times = (3, 2, 1, 2)


# Situation 2: repeat_times is a list (with tensor)
class XPUTestTileOpRank1_tensor_attr(XPUOpTestWrapper):
    def __init__(self):
        self.op_name = 'tile'
        self.use_dynamic_create_class = False

    class TestTileOpRank1_tensor_attr(XPUOpTest):
        def setUp(self):
            self.dtype = self.in_type
            self.__class__.no_need_check_grad = True
            self.place = paddle.XPUPlace(0)
            self.op_type = "tile"
            self.init_data()
            repeat_times_tensor = []
            for index, ele in enumerate(self.repeat_times):
                repeat_times_tensor.append(("x" + str(index), np.ones(
                    (1)).astype('int32') * ele))

            self.inputs = {
                'X': np.random.random(self.ori_shape).astype(self.dtype),
                'repeat_times_tensor': repeat_times_tensor,
            }
            self.attrs = {"repeat_times": self.infer_repeat_times}
            output = np.tile(self.inputs['X'], self.repeat_times)
            self.outputs = {'Out': output}

        def init_data(self):
            self.ori_shape = [100]
            self.repeat_times = [2]
            self.infer_repeat_times = [-1]

        def test_check_output(self):
            self.check_output_with_place(self.place)

    class TestTileOpRank2_Corner_tensor_attr(TestTileOpRank1_tensor_attr):
        def init_data(self):
            self.ori_shape = [12, 14]
            self.repeat_times = [1, 1]
            self.infer_repeat_times = [1, -1]

    class TestTileOpRank2_attr_tensor(TestTileOpRank1_tensor_attr):
        def init_data(self):
            self.ori_shape = [12, 14]
            self.repeat_times = [2, 3]
            self.infer_repeat_times = [-1, 3]


# Situation 3: repeat_times is a tensor
class XPUTestTileOpRank1_tensor(XPUOpTestWrapper):
    def __init__(self):
        self.op_name = 'tile'
        self.use_dynamic_create_class = False

    class TestTileOpRank1_tensor(XPUOpTest):
        def setUp(self):
            self.dtype = self.in_type
            self.__class__.no_need_check_grad = True
            self.place = paddle.XPUPlace(0)
            self.op_type = "tile"
            self.init_data()

            self.inputs = {
                'X': np.random.random(self.ori_shape).astype(self.dtype),
                'RepeatTimes': np.array(self.repeat_times).astype("int32"),
            }
            self.attrs = {}
            output = np.tile(self.inputs['X'], self.repeat_times)
            self.outputs = {'Out': output}

        def init_data(self):
            self.ori_shape = [100]
            self.repeat_times = [2]

        def test_check_output(self):
            self.check_output_with_place(self.place)

    class TestTileOpRank2_tensor(TestTileOpRank1_tensor):
        def init_data(self):
            self.ori_shape = [12, 14]
            self.repeat_times = [2, 3]


support_types = get_xpu_op_support_types('tile')
for stype in support_types:
    create_test_class(globals(), XPUTestTileOpRank1, stype)
    create_test_class(globals(), XPUTestTileOpRank1_tensor_attr, stype)
    create_test_class(globals(), XPUTestTileOpRank1_tensor, stype)


# Test python API
class TestTileAPI(unittest.TestCase):
    def test_api(self):
        with fluid.dygraph.guard(paddle.XPUPlace(0)):
            np_x = np.random.random([12, 14]).astype("float32")
            x = paddle.to_tensor(np_x)

            positive_2 = np.array([2]).astype("int32")
            positive_2 = paddle.to_tensor(positive_2)

            repeat_times = np.array([2, 3]).astype("int32")
            repeat_times = paddle.to_tensor(repeat_times)

            out_1 = paddle.tile(x, repeat_times=[2, 3])
            out_2 = paddle.tile(x, repeat_times=[positive_2, 3])
            out_3 = paddle.tile(x, repeat_times=repeat_times)

            assert np.array_equal(out_1.numpy(), np.tile(np_x, (2, 3)))
            assert np.array_equal(out_2.numpy(), np.tile(np_x, (2, 3)))
            assert np.array_equal(out_3.numpy(), np.tile(np_x, (2, 3)))


if __name__ == "__main__":
    unittest.main()
