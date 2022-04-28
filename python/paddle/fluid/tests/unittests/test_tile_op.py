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
from op_test import OpTest
import paddle
import paddle.fluid as fluid
from paddle.fluid import compiler, Program, program_guard


#Situation 1: repeat_times is a list (without tensor)
class TestTileOpRank1(OpTest):
    def setUp(self):
        self.op_type = "tile"
        self.init_data()

        self.inputs = {'X': np.random.random(self.ori_shape).astype("float64")}
        self.attrs = {'repeat_times': self.repeat_times}
        output = np.tile(self.inputs['X'], self.repeat_times)
        self.outputs = {'Out': output}

    def init_data(self):
        self.ori_shape = [100]
        self.repeat_times = [2]

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(['X'], 'Out')


# with dimension expanding
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
class TestTileOpRank1_tensor_attr(OpTest):
    def setUp(self):
        self.op_type = "tile"
        self.init_data()
        repeat_times_tensor = []
        for index, ele in enumerate(self.repeat_times):
            repeat_times_tensor.append(("x" + str(index), np.ones(
                (1)).astype('int32') * ele))

        self.inputs = {
            'X': np.random.random(self.ori_shape).astype("float64"),
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
        self.check_output()

    def test_check_grad(self):
        self.check_grad(['X'], 'Out')


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
class TestTileOpRank1_tensor(OpTest):
    def setUp(self):
        self.op_type = "tile"
        self.init_data()

        self.inputs = {
            'X': np.random.random(self.ori_shape).astype("float64"),
            'RepeatTimes': np.array(self.repeat_times).astype("int32"),
        }
        self.attrs = {}
        output = np.tile(self.inputs['X'], self.repeat_times)
        self.outputs = {'Out': output}

    def init_data(self):
        self.ori_shape = [100]
        self.repeat_times = [2]

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(['X'], 'Out')


class TestTileOpRank2_tensor(TestTileOpRank1_tensor):
    def init_data(self):
        self.ori_shape = [12, 14]
        self.repeat_times = [2, 3]


# Situation 4: input x is Integer
class TestTileOpInteger(OpTest):
    def setUp(self):
        self.op_type = "tile"
        self.inputs = {
            'X': np.random.randint(
                10, size=(4, 4, 5)).astype("int32")
        }
        self.attrs = {'repeat_times': [2, 1, 4]}
        output = np.tile(self.inputs['X'], (2, 1, 4))
        self.outputs = {'Out': output}

    def test_check_output(self):
        self.check_output()


# Situation 5: input x is Bool
class TestTileOpBoolean(OpTest):
    def setUp(self):
        self.op_type = "tile"
        self.inputs = {'X': np.random.randint(2, size=(2, 4, 5)).astype("bool")}
        self.attrs = {'repeat_times': [2, 1, 4]}
        output = np.tile(self.inputs['X'], (2, 1, 4))
        self.outputs = {'Out': output}

    def test_check_output(self):
        self.check_output()


# Situation 56: input x is Integer
class TestTileOpInt64_t(OpTest):
    def setUp(self):
        self.op_type = "tile"
        self.inputs = {
            'X': np.random.randint(
                10, size=(2, 4, 5)).astype("int64")
        }
        self.attrs = {'repeat_times': [2, 1, 4]}
        output = np.tile(self.inputs['X'], (2, 1, 4))
        self.outputs = {'Out': output}

    def test_check_output(self):
        self.check_output()


class TestTileError(unittest.TestCase):
    def test_errors(self):
        with program_guard(Program(), Program()):
            x1 = fluid.create_lod_tensor(
                np.array([[-1]]), [[1]], fluid.CPUPlace())
            repeat_times = [2, 2]
            self.assertRaises(TypeError, paddle.tile, x1, repeat_times)
            x2 = fluid.layers.data(name='x2', shape=[4], dtype="uint8")
            self.assertRaises(TypeError, paddle.tile, x2, repeat_times)
            x3 = fluid.layers.data(name='x3', shape=[4], dtype="bool")
            x3.stop_gradient = False
            self.assertRaises(ValueError, paddle.tile, x3, repeat_times)


class TestTileAPIStatic(unittest.TestCase):
    def test_api(self):
        with program_guard(Program(), Program()):
            repeat_times = [2, 2]
            x1 = fluid.layers.data(name='x1', shape=[4], dtype="int32")
            out = paddle.tile(x1, repeat_times)
            positive_2 = fluid.layers.fill_constant([1], dtype="int32", value=2)
            out2 = paddle.tile(x1, repeat_times=[positive_2, 2])


# Test python API
class TestTileAPI(unittest.TestCase):
    def test_api(self):
        with fluid.dygraph.guard():
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
    paddle.enable_static()
    unittest.main()
