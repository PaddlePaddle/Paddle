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
import paddle.fluid as fluid


# situation 1: have shape( list, no tensor), no actual shape(Tensor)
class TestReshapeOp(OpTest):
    def setUp(self):
        self.init_data()
        self.op_type = "reshape2"
        self.inputs = {"X": np.random.random(self.ori_shape).astype("float32")}
        self.attrs = {"shape": self.new_shape}
        self.outputs = {
            "Out": self.inputs["X"].reshape(self.infered_shape),
            'XShape': np.random.random(self.ori_shape).astype("float32")
        }

    def init_data(self):
        self.ori_shape = (2, 25)
        self.new_shape = (5, 10)
        self.infered_shape = (5, 10)

    def test_check_output(self):

        self.check_output(no_check_set=['XShape'])

    def test_check_grad(self):
        self.check_grad(["X"], "Out")


class TestReshapeOpDimInfer1(TestReshapeOp):
    def init_data(self):
        self.ori_shape = (5, 10)
        self.new_shape = (5, -1, 5)
        self.infered_shape = (5, -1, 5)


class TestReshapeOpDimInfer2(TestReshapeOp):
    def init_data(self):
        self.ori_shape = (2, 2, 6)
        self.new_shape = (2, 0, 3, -1)
        self.infered_shape = (2, 2, 3, -1)


# situation 2: have shape(list, no tensor), have actual shape(Tensor)
class TestReshapeOpWithInputShape(OpTest):
    def setUp(self):
        self.init_data()
        self.op_type = "reshape2"

        self.inputs = {
            "X": np.random.random(self.ori_shape).astype("float32"),
            "Shape": np.array(
                self.actual_shape, dtype="int32")
        }
        self.attrs = {"shape": self.new_shape}
        self.outputs = {
            "Out": self.inputs["X"].reshape(self.actual_shape),
            'XShape': np.random.random(self.ori_shape).astype("float32")
        }

    def init_data(self):
        self.ori_shape = (6, 5)
        self.new_shape = (0, -1, 5)
        self.actual_shape = (2, 3, 5)

    def test_check_output(self):
        self.check_output(no_check_set=['XShape'])

    def test_check_grad(self):
        self.check_grad(["X"], "Out")


# Situation 3: have shape(list, have tensor), no actual shape(Tensor)
class TestReshapeOp_attr_ShapeTensor(OpTest):
    def setUp(self):
        self.init_data()
        self.op_type = "reshape2"

        shape_tensor = []
        for index, ele in enumerate(self.new_shape):
            shape_tensor.append(("x" + str(index), np.ones(
                (1)).astype('int32') * ele))

        self.inputs = {
            "X": np.random.random(self.ori_shape).astype("float32"),
            'ShapeTensor': shape_tensor
        }
        self.attrs = {'shape': self.shape}
        self.outputs = {
            "Out": self.inputs["X"].reshape(self.infered_shape),
            'XShape': np.random.random(self.ori_shape).astype("float32")
        }

    def init_data(self):
        self.ori_shape = (2, 25)
        self.new_shape = (5, 10)
        self.infered_shape = (5, 10)
        self.shape = (-1, -1)

    def test_check_output(self):
        self.check_output(no_check_set=['XShape'])

    def test_check_grad(self):
        self.check_grad(["X"], "Out")


class TestReshapeOpDimInfer1_attr_ShapeTensor(TestReshapeOp_attr_ShapeTensor):
    def init_data(self):
        self.ori_shape = (5, 10)
        self.new_shape = (5, -1, 5)
        self.infered_shape = (5, -1, 5)
        self.shape = (5, -1, -1)


class TestReshapeOpDimInfer2_attr_ShapeTensor(TestReshapeOp_attr_ShapeTensor):
    def init_data(self):
        self.ori_shape = (2, 2, 6)
        self.new_shape = (2, 0, 3, -1)
        self.infered_shape = (2, 2, 3, -1)
        self.shape = (2, 0, 3, -1)


# Situation 4: have shape(Tensor), no actual shape(Tensor)
class TestReshapeOp_attr_OnlyShape(OpTest):
    def setUp(self):
        self.init_data()
        self.op_type = "reshape2"

        self.inputs = {
            "X": np.random.random(self.ori_shape).astype("float32"),
            "Shape": np.array(
                self.new_shape, dtype="int32")
        }
        self.attrs = {}
        self.outputs = {
            "Out": self.inputs["X"].reshape(self.infered_shape),
            'XShape': np.random.random(self.ori_shape).astype("float32")
        }

    def init_data(self):
        self.ori_shape = (2, 25)
        self.new_shape = (5, 10)
        self.infered_shape = (5, 10)

    def test_check_output(self):
        self.check_output(no_check_set=['XShape'])

    def test_check_grad(self):
        self.check_grad(["X"], "Out")


class TestReshapeOpDimInfer1_attr_OnlyShape(TestReshapeOp_attr_OnlyShape):
    def init_data(self):
        self.ori_shape = (5, 10)
        self.new_shape = (5, -1, 5)
        self.infered_shape = (5, -1, 5)
        self.shape = (5, -1, -1)


class TestReshapeOpDimInfer2_attr_OnlyShape(TestReshapeOp_attr_OnlyShape):
    def init_data(self):
        self.ori_shape = (2, 2, 6)
        self.new_shape = (2, 0, 3, -1)
        self.infered_shape = (2, 2, 3, -1)
        self.shape = (2, 0, 3, -1)


# Test python API
class TestReshapeAPI(OpTest):
    # situation 1: have shape( list, no tensor), no actual shape(Tensor)
    def test_1(self):
        input = np.random.random([2, 25]).astype("float32")
        shape = [2, 5, 5]
        positive_five = fluid.layers.fill_constant([1], "int32", 5)
        x = fluid.layers.data(
            name="x", shape=[2, 25], append_batch_size=False, dtype="float32")

        actual_shape = fluid.layers.data(
            name="shape",
            shape=[1, 3],
            append_batch_size=False,
            dtype="float32")

        # situation 1: have shape( list, no tensor), no actual shape(Tensor)
        out_1 = fluid.layers.reshape(x, shape)
        # situation 2: have shape(list, no tensor), have actual shape(Tensor)
        out_2 = fluid.layers.reshape(x, shape=shape, actual_shape=actual_shape)
        # Situation 3: have shape(list, have tensor), no actual shape(Tensor)
        out_3 = fluid.layers.reshape(x, shape=[positive_five, 10])
        # Situation 4: have shape(Tensor), no actual shape(Tensor)
        out_4 = fluid.layers.reshape(x, shape=actual_shape)

        exe = fluid.Executor(place=fluid.CPUPlace())
        res_1, res_2, res_3, res_4 = exe.run(
            fluid.default_main_program(),
            feed={"x": input,
                  "shape": np.array([2, 5, 5]).astype("int32")},
            fetch_list=[out_1, out_2, out_3, out_4])

        assert np.array_equal(res_1, input.reshape(shape))
        assert np.array_equal(res_2, input.reshape(shape))
        assert np.array_equal(res_3, input.reshape([5, 10]))
        assert np.array_equal(res_4, input.reshape(shape))


if __name__ == "__main__":
    unittest.main()
