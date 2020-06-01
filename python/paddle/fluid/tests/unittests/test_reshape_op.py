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
from paddle.fluid import compiler, Program, program_guard


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
        self.ori_shape = (2, 60)
        self.new_shape = (12, 10)
        self.infered_shape = (12, 10)

    def test_check_output(self):
        self.check_output(no_check_set=['XShape'])

    def test_check_grad(self):
        self.check_grad(["X"], "Out")


class TestReshapeOpDimInfer1(TestReshapeOp):
    def init_data(self):
        self.ori_shape = (5, 25)
        self.new_shape = (5, -1, 5)
        self.infered_shape = (5, -1, 5)


class TestReshapeOpDimInfer2(TestReshapeOp):
    def init_data(self):
        self.ori_shape = (10, 2, 6)
        self.new_shape = (10, 0, 3, -1)
        self.infered_shape = (10, 2, 3, -1)


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
        self.ori_shape = (6, 20)
        self.new_shape = (0, -1, 20)
        self.actual_shape = (2, 3, 20)

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
        self.ori_shape = (4, 25)
        self.new_shape = (10, 10)
        self.infered_shape = (10, 10)
        self.shape = (-1, -1)

    def test_check_output(self):
        self.check_output(no_check_set=['XShape'])

    def test_check_grad(self):
        self.check_grad(["X"], "Out")


class TestReshapeOpDimInfer1_attr_ShapeTensor(TestReshapeOp_attr_ShapeTensor):
    def init_data(self):
        self.ori_shape = (5, 20)
        self.new_shape = (5, -1, 20)
        self.infered_shape = (5, -1, 20)
        self.shape = (5, -1, -1)


class TestReshapeOpDimInfer2_attr_ShapeTensor(TestReshapeOp_attr_ShapeTensor):
    def init_data(self):
        self.ori_shape = (10, 2, 6)
        self.new_shape = (10, 0, 3, -1)
        self.infered_shape = (10, 2, 3, -1)
        self.shape = (10, 0, 3, -1)


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
        self.ori_shape = (4, 25)
        self.new_shape = (10, 10)
        self.infered_shape = (10, 10)

    def test_check_output(self):
        self.check_output(no_check_set=['XShape'])

    def test_check_grad(self):
        self.check_grad(["X"], "Out")


class TestReshapeOpDimInfer1_attr_OnlyShape(TestReshapeOp_attr_OnlyShape):
    def init_data(self):
        self.ori_shape = (5, 20)
        self.new_shape = (5, -1, 10)
        self.infered_shape = (5, -1, 10)
        self.shape = (5, -1, -1)


class TestReshapeOpDimInfer2_attr_OnlyShape(TestReshapeOp_attr_OnlyShape):
    def init_data(self):
        self.ori_shape = (10, 2, 6)
        self.new_shape = (10, 0, 3, -1)
        self.infered_shape = (10, 2, 3, -1)
        self.shape = (10, 0, 3, -1)


# test int8 data type on CPU
class TestReshapeInt8Op(OpTest):
    def setUp(self):
        self.init_dtype()
        self.init_data()
        self.use_mkldnn = True
        self._cpu_only = True
        self.op_type = "reshape2"
        input = np.random.randint(0, 127, self.ori_shape).astype(self.dtype)
        self.inputs = {'X': OpTest.np_dtype_to_fluid_dtype(input)}
        self.attrs = {
            'shape': self.new_shape,
            'use_mkldnn': self.use_mkldnn,
        }
        self.outputs = {
            "Out": self.inputs["X"].reshape(self.infered_shape),
            'XShape': np.random.random(self.ori_shape).astype(np.float32)
        }

    def init_dtype(self):
        self.dtype = np.int8

    def init_data(self):
        self.ori_shape = (10, 2, 6)
        self.new_shape = (10, 0, 3, -1)
        self.infered_shape = (10, 2, 3, -1)

    def test_check_output(self):
        self.check_output_with_place(
            fluid.core.CPUPlace(), atol=1e-5, no_check_set=['XShape'])

    def test_check_grad(self):
        pass


# test unt8 data type on CPU
class TestReshapeUint8Op(TestReshapeInt8Op):
    def init_dtype(self):
        self.dtype = np.uint8


# Test python API
class TestReshapeAPI(unittest.TestCase):
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


# Test Input Error
class TestReshapeOpError(unittest.TestCase):
    def test_errors(self):
        with program_guard(Program(), Program()):
            # The x type of reshape_op must be Variable.
            def test_x_type():
                x1 = fluid.create_lod_tensor(
                    np.array([[-1]]), [[1]], fluid.CPUPlace())
                fluid.layers.reshape(x1, shape=[1])

            self.assertRaises(TypeError, test_x_type)

            # The x dtype of reshape_op must be float16, float32, float64, int32 or int64.
            def test_x_dtype():
                x2 = fluid.layers.data(
                    name="x2",
                    shape=[2, 25],
                    append_batch_size=False,
                    dtype="bool")
                fluid.layers.reshape(x2, shape=[2, 5, 5])

            self.assertRaises(TypeError, test_x_dtype)

            def test_x_dtype_float16():
                x_float16 = fluid.layers.data(
                    name="x_float16",
                    shape=[2, 25],
                    append_batch_size=False,
                    dtype="float16")
                fluid.layers.reshape(x_float16, shape=[2, 5, 5])

            test_x_dtype_float16()

            x3 = fluid.layers.data(
                name="x3",
                shape=[2, 25],
                append_batch_size=False,
                dtype="float32")

            # The argument shape's type of reshape_op must be list, tuple or Variable.
            def test_shape_type():
                fluid.layers.reshape(x3, shape=1)

            self.assertRaises(TypeError, test_shape_type)

            # The argument actual_shape's type of reshape_op must be Variable or None.
            def test_actual_shape_type():
                fluid.layers.reshape(x3, shape=[25, 2], actual_shape=1)

            self.assertRaises(TypeError, test_actual_shape_type)

            # The argument shape have more than one -1.
            def test_shape_1():
                fluid.layers.reshape(x3, shape=[-1, -1, 5])

            self.assertRaises(AssertionError, test_shape_1)

            # The argument shape have element 0 whose index exceed the input dimension.
            def test_shape_2():
                fluid.layers.reshape(x3, [2, 5, 5, 0])

            self.assertRaises(AssertionError, test_shape_2)

            # The argument shape have more than one negative value.
            def test_shape_3():
                fluid.layers.reshape(x3, [-1, -2, 5])

            self.assertRaises(AssertionError, test_shape_3)


if __name__ == "__main__":
    unittest.main()
