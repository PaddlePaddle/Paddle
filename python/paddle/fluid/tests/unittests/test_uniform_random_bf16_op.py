#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

import unittest

import numpy as np
from op_test import OpTest, convert_uint16_to_float

import paddle
import paddle.fluid as fluid
import paddle.fluid.core as core
from paddle.fluid.op import Operator
from paddle.fluid.tests.unittests.test_uniform_random_op import (
    output_hist,
    output_hist_diag,
)
from paddle.tensor import random


class TestUniformRandomOpBF16(OpTest):
    def setUp(self):
        self.op_type = "uniform_random"
        self.dtype = "uint16"
        self.inputs = {}
        self.init_attrs()
        self.outputs = {"Out": np.zeros((1000, 784)).astype("uint16")}

    def init_attrs(self):
        self.attrs = {
            "shape": [1000, 784],
            "min": -5.0,
            "max": 10.0,
            "seed": 10,
            'dtype': int(core.VarDesc.VarType.BF16),
        }
        self.output_hist = output_hist

    def verify_output(self, outs):
        if np.array(outs[0]).dtype == np.uint16:
            result = convert_uint16_to_float(np.array(outs[0]))
        else:
            result = np.array(outs[0])

        hist, prob = self.output_hist(result)
        np.testing.assert_allclose(hist, prob, rtol=0, atol=0.01)

    def test_check_output(self):
        outs = self.calc_output(core.CPUPlace())
        outs = [np.array(out) for out in outs]
        outs.sort(key=len)
        self.verify_output(outs)


class TestUniformRandomOpBF16AttrTensorList(TestUniformRandomOpBF16):
    def setUp(self):
        self.op_type = "uniform_random"
        self.new_shape = (1000, 784)
        self.dtype = "uint16"
        shape_tensor = []
        for index, ele in enumerate(self.new_shape):
            shape_tensor.append(
                ("x" + str(index), np.ones((1)).astype("int64") * ele)
            )
        self.inputs = {'ShapeTensorList': shape_tensor}
        self.init_attrs()
        self.outputs = {"Out": np.zeros((1000, 784)).astype("uint16")}

    def init_attrs(self):
        self.attrs = {
            "min": -5.0,
            "max": 10.0,
            "seed": 10,
            'dtype': int(core.VarDesc.VarType.BF16),
        }
        self.output_hist = output_hist


class TestUniformRandomOpBF16AttrTensorInt32(
    TestUniformRandomOpBF16AttrTensorList
):
    def setUp(self):
        self.op_type = "uniform_random"
        self.dtype = "uint16"
        self.inputs = {"ShapeTensor": np.array([1000, 784]).astype("int32")}
        self.init_attrs()
        self.outputs = {"Out": np.zeros((1000, 784)).astype("uint16")}


class TestUniformRandomOpBF16WithDiagInit(TestUniformRandomOpBF16):
    def init_attrs(self):
        self.attrs = {
            "shape": [1000, 784],
            "min": -5.0,
            "max": 10.0,
            "seed": 10,
            "diag_num": 784,
            "diag_step": 784,
            "diag_val": 1.0,
            'dtype': int(core.VarDesc.VarType.BF16),
        }
        self.output_hist = output_hist_diag


class TestUniformRandomOpBF16SelectedRows(unittest.TestCase):
    def test_check_output(self):
        self.check_with_place(core.CPUPlace())

    def check_with_place(self, place):
        scope = core.Scope()
        out = scope.var("X").get_selected_rows()
        paddle.seed(10)
        op = Operator(
            "uniform_random",
            Out="X",
            shape=[1000, 784],
            min=-5.0,
            max=10.0,
            seed=10,
            dtype=int(core.VarDesc.VarType.BF16),
        )
        op.run(scope, place)
        self.assertEqual(out.get_tensor().shape(), [1000, 784])
        result = convert_uint16_to_float(np.array(out.get_tensor()))
        hist, prob = output_hist(result)
        np.testing.assert_allclose(hist, prob, rtol=0, atol=0.01)


class TestUniformRandomOpBF16SelectedRowsWithDiagInit(
    TestUniformRandomOpBF16SelectedRows
):
    def check_with_place(self, place):
        scope = core.Scope()
        out = scope.var("X").get_selected_rows()
        paddle.seed(10)
        op = Operator(
            "uniform_random",
            Out="X",
            shape=[500, 784],
            min=-5.0,
            max=10.0,
            seed=10,
            diag_num=500,
            diag_step=784,
            diag_val=1.0,
            dtype=int(core.VarDesc.VarType.BF16),
        )
        op.run(scope, place)
        self.assertEqual(out.get_tensor().shape(), [500, 784])
        result = convert_uint16_to_float(np.array(out.get_tensor()))
        hist, prob = output_hist(result)
        np.testing.assert_allclose(hist, prob, rtol=0, atol=0.01)


class TestUniformRandomOpAPISeed(unittest.TestCase):
    def test_attr_tensor_API(self):
        _seed = 10
        gen = paddle.seed(_seed)
        startup_program = fluid.Program()
        train_program = fluid.Program()
        with fluid.program_guard(train_program, startup_program):
            _min = 5
            _max = 10

            ret = paddle.uniform([2, 3, 2], min=_min, max=_max, seed=_seed)
            ret_2 = paddle.uniform([2, 3, 2], min=_min, max=_max, seed=_seed)
            res = paddle.equal(ret, ret_2)
            place = fluid.CPUPlace()
            exe = fluid.Executor(place)

            exe.run(startup_program)
            ret_value, cmp_value = exe.run(train_program, fetch_list=[ret, res])
            self.assertTrue(np.array(cmp_value).all())
            for i in ret_value.flatten():
                self.assertGreaterEqual(i, _min)
                self.assertLess(i, _max)


class TestUniformRandomOpBF16SelectedRowsShapeTensor(unittest.TestCase):
    def test_check_output(self):
        place = core.CPUPlace()
        scope = core.Scope()
        out = scope.var("X").get_selected_rows()
        shape_tensor = scope.var("Shape").get_tensor()
        shape_tensor.set(np.array([1000, 784]).astype("int64"), place)
        paddle.seed(10)
        op = Operator(
            "uniform_random",
            ShapeTensor="Shape",
            Out="X",
            min=-5.0,
            max=10.0,
            seed=10,
            dtype=int(core.VarDesc.VarType.BF16),
        )
        op.run(scope, place)
        self.assertEqual(out.get_tensor().shape(), [1000, 784])
        result = convert_uint16_to_float(np.array(out.get_tensor()))
        hist, prob = output_hist(result)
        np.testing.assert_allclose(hist, prob, rtol=0, atol=0.01)


class TestUniformRandomOpBF16SelectedRowsShapeTensorList(
    TestUniformRandomOpBF16SelectedRowsShapeTensor
):
    def test_check_output(self):
        place = core.CPUPlace()
        scope = core.Scope()
        out = scope.var("X").get_selected_rows()
        shape_1 = scope.var("shape1").get_tensor()
        shape_1.set(np.array([1000]).astype("int64"), place)
        shape_2 = scope.var("shape2").get_tensor()
        shape_2.set(np.array([784]).astype("int64"), place)
        paddle.seed(10)
        op = Operator(
            "uniform_random",
            ShapeTensorList=["shape1", "shape2"],
            Out="X",
            min=-5.0,
            max=10.0,
            seed=10,
            dtype=int(core.VarDesc.VarType.BF16),
        )
        op.run(scope, place)
        self.assertEqual(out.get_tensor().shape(), [1000, 784])
        result = convert_uint16_to_float(np.array(out.get_tensor()))
        hist, prob = output_hist(result)
        np.testing.assert_allclose(hist, prob, rtol=0, atol=0.01)


class TestUniformRandomBatchSizeLikeOpBF16API(unittest.TestCase):
    def test_attr_tensorlist_int32_API(self):
        startup_program = fluid.Program()
        train_program = fluid.Program()
        with fluid.program_guard(train_program, startup_program):
            input = fluid.data(name="input", shape=[1, 3], dtype='uint16')
            out_1 = random.uniform_random_batch_size_like(
                input, [2, 4], dtype=np.uint16
            )  # out_1.shape=[1, 4]

            place = fluid.CPUPlace()
            exe = fluid.Executor(place)

            exe.run(startup_program)
            outs = exe.run(train_program, fetch_list=[out_1])


if __name__ == "__main__":
    from paddle import enable_static

    enable_static()
    unittest.main()
