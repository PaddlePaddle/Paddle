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
import paddle.fluid.core as core
from paddle.fluid.op import Operator
import paddle.fluid as fluid
from paddle.fluid import Program, program_guard


def output_hist(out):
    hist, _ = np.histogram(out, range=(-5, 10))
    hist = hist.astype("float32")
    hist /= float(out.size)
    prob = 0.1 * np.ones((10))
    return hist, prob


def output_hist_diag(out):
    diag_num = min(out.shape)
    for i in range(diag_num):
        assert abs(out[i][i] - 1.0) < 1e-9
        # ignore diagonal elements
        out[i][i] = 100
    hist, _ = np.histogram(out, range=(-5, 10))
    hist = hist.astype("float32")
    hist /= float(out.size)
    prob = 0.1 * np.ones((10))
    return hist, prob


class TestUniformRandomOp_attr_tensorlist(OpTest):
    def setUp(self):
        self.op_type = "uniform_random"
        self.new_shape = (1000, 784)
        shape_tensor = []
        for index, ele in enumerate(self.new_shape):
            shape_tensor.append(("x" + str(index), np.ones(
                (1)).astype("int64") * ele))
        self.inputs = {'ShapeTensorList': shape_tensor}
        self.init_attrs()
        self.outputs = {"Out": np.zeros((1000, 784)).astype("float32")}

    def init_attrs(self):
        self.attrs = {"min": -5.0, "max": 10.0, "seed": 10}
        self.output_hist = output_hist

    def test_check_output(self):
        self.check_output_customized(self.verify_output)

    def verify_output(self, outs):
        hist, prob = self.output_hist(np.array(outs[0]))
        self.assertTrue(
            np.allclose(
                hist, prob, rtol=0, atol=0.01), "hist: " + str(hist))


class TestUniformRandomOp_attr_tensorlist_int32(OpTest):
    def setUp(self):
        self.op_type = "uniform_random"
        self.new_shape = (1000, 784)
        shape_tensor = []
        for index, ele in enumerate(self.new_shape):
            shape_tensor.append(("x" + str(index), np.ones(
                (1)).astype("int32") * ele))
        self.inputs = {'ShapeTensorList': shape_tensor}
        self.init_attrs()
        self.outputs = {"Out": np.zeros((1000, 784)).astype("float32")}

    def init_attrs(self):
        self.attrs = {"min": -5.0, "max": 10.0, "seed": 10}
        self.output_hist = output_hist

    def test_check_output(self):
        self.check_output_customized(self.verify_output)

    def verify_output(self, outs):
        hist, prob = self.output_hist(np.array(outs[0]))
        self.assertTrue(
            np.allclose(
                hist, prob, rtol=0, atol=0.01), "hist: " + str(hist))


class TestUniformRandomOp_attr_tensor(OpTest):
    def setUp(self):
        self.op_type = "uniform_random"
        self.inputs = {"ShapeTensor": np.array([1000, 784]).astype("int64")}
        self.init_attrs()
        self.outputs = {"Out": np.zeros((1000, 784)).astype("float32")}

    def init_attrs(self):
        self.attrs = {"min": -5.0, "max": 10.0, "seed": 10}
        self.output_hist = output_hist

    def test_check_output(self):
        self.check_output_customized(self.verify_output)

    def verify_output(self, outs):
        hist, prob = self.output_hist(np.array(outs[0]))
        self.assertTrue(
            np.allclose(
                hist, prob, rtol=0, atol=0.01), "hist: " + str(hist))


class TestUniformRandomOp_attr_tensor_int32(OpTest):
    def setUp(self):
        self.op_type = "uniform_random"
        self.inputs = {"ShapeTensor": np.array([1000, 784]).astype("int32")}
        self.init_attrs()
        self.outputs = {"Out": np.zeros((1000, 784)).astype("float32")}

    def init_attrs(self):
        self.attrs = {"min": -5.0, "max": 10.0, "seed": 10}
        self.output_hist = output_hist

    def test_check_output(self):
        self.check_output_customized(self.verify_output)

    def verify_output(self, outs):
        hist, prob = self.output_hist(np.array(outs[0]))
        self.assertTrue(
            np.allclose(
                hist, prob, rtol=0, atol=0.01), "hist: " + str(hist))


class TestUniformRandomOp(OpTest):
    def setUp(self):
        self.op_type = "uniform_random"
        self.inputs = {}
        self.init_attrs()
        self.outputs = {"Out": np.zeros((1000, 784)).astype("float32")}

    def init_attrs(self):
        self.attrs = {
            "shape": [1000, 784],
            "min": -5.0,
            "max": 10.0,
            "seed": 10
        }
        self.output_hist = output_hist

    def test_check_output(self):
        self.check_output_customized(self.verify_output)

    def verify_output(self, outs):
        hist, prob = self.output_hist(np.array(outs[0]))
        self.assertTrue(
            np.allclose(
                hist, prob, rtol=0, atol=0.01), "hist: " + str(hist))


class TestUniformRandomOpError(unittest.TestCase):
    def test_errors(self):
        main_prog = Program()
        start_prog = Program()
        with program_guard(main_prog, start_prog):

            def test_Variable():
                x1 = fluid.create_lod_tensor(
                    np.zeros((4, 784)), [[1, 1, 1, 1]], fluid.CPUPlace())
                fluid.layers.uniform_random(x1)

            self.assertRaises(TypeError, test_Variable)

            def test_dtype():
                x2 = fluid.layers.data(
                    name='x2', shape=[4, 784], dtype='float32')
                fluid.layers.uniform_random(x2, 'int32')

            self.assertRaises(TypeError, test_dtype)


class TestUniformRandomOpWithDiagInit(TestUniformRandomOp):
    def init_attrs(self):
        self.attrs = {
            "shape": [1000, 784],
            "min": -5.0,
            "max": 10.0,
            "seed": 10,
            "diag_num": 784,
            "diag_step": 784,
            "diag_val": 1.0
        }
        self.output_hist = output_hist_diag


class TestUniformRandomOpSelectedRows(unittest.TestCase):
    def get_places(self):
        places = [core.CPUPlace()]
        if core.is_compiled_with_cuda():
            places.append(core.CUDAPlace(0))
        return places

    def test_check_output(self):
        for place in self.get_places():
            self.check_with_place(place)

    def check_with_place(self, place):
        scope = core.Scope()
        out = scope.var("X").get_selected_rows()

        op = Operator(
            "uniform_random",
            Out="X",
            shape=[4, 784],
            min=-5.0,
            max=10.0,
            seed=10)
        op.run(scope, place)
        self.assertEqual(out.get_tensor().shape(), [4, 784])
        hist, prob = output_hist(np.array(out.get_tensor()))
        self.assertTrue(
            np.allclose(
                hist, prob, rtol=0, atol=0.01), "hist: " + str(hist))


class TestUniformRandomOpSelectedRowsWithDiagInit(
        TestUniformRandomOpSelectedRows):
    def check_with_place(self, place):
        scope = core.Scope()
        out = scope.var("X").get_selected_rows()

        op = Operator(
            "uniform_random",
            Out="X",
            shape=[4, 784],
            min=-5.0,
            max=10.0,
            seed=10,
            diag_num=4,
            diag_step=784,
            diag_val=1.0)
        op.run(scope, place)
        self.assertEqual(out.get_tensor().shape(), [4, 784])
        hist, prob = output_hist_diag(np.array(out.get_tensor()))
        self.assertTrue(
            np.allclose(
                hist, prob, rtol=0, atol=0.01), "hist: " + str(hist))


class TestUniformRandomOpApi(unittest.TestCase):
    def test_api(self):
        x = fluid.layers.data('x', shape=[16], dtype='float32', lod_level=1)
        y = fluid.layers.fc(x,
                            size=16,
                            param_attr=fluid.initializer.Uniform(
                                low=-0.5,
                                high=0.5,
                                seed=10,
                                diag_num=16,
                                diag_step=16,
                                diag_val=1.0))

        place = fluid.CPUPlace()
        x_tensor = fluid.create_lod_tensor(
            np.random.rand(3, 16).astype("float32"), [[1, 2]], place)
        exe = fluid.Executor(place)
        exe.run(fluid.default_startup_program())
        ret = exe.run(feed={'x': x_tensor}, fetch_list=[y], return_numpy=False)


class TestUniformRandomOp_attr_tensor_API(unittest.TestCase):
    def test_attr_tensor_API(self):
        startup_program = fluid.Program()
        train_program = fluid.Program()
        with fluid.program_guard(train_program, startup_program):
            dim_tensor = fluid.layers.fill_constant([1], "int64", 3)
            ret = fluid.layers.nn.uniform_random([1, dim_tensor, 2])

            place = fluid.CPUPlace()
            if fluid.core.is_compiled_with_cuda():
                place = fluid.CUDAPlace(0)
            exe = fluid.Executor(place)

            exe.run(startup_program)
            outs = exe.run(train_program, fetch_list=[ret])

    def test_attr_tensorlist_int32_API(self):
        startup_program = fluid.Program()
        train_program = fluid.Program()
        with fluid.program_guard(train_program, startup_program):
            dim_1 = fluid.layers.fill_constant([1], "int64", 3)
            dim_2 = fluid.layers.fill_constant([1], "int32", 2)
            ret = fluid.layers.nn.uniform_random([1, dim_1, dim_2])

            place = fluid.CPUPlace()
            if fluid.core.is_compiled_with_cuda():
                place = fluid.CUDAPlace(0)
            exe = fluid.Executor(place)

            exe.run(startup_program)
            outs = exe.run(train_program, fetch_list=[ret])

    def test_attr_tensor_int32_API(self):
        startup_program = fluid.Program()
        train_program = fluid.Program()
        with fluid.program_guard(train_program, startup_program):
            shape = fluid.data(name='shape_tensor', shape=[2], dtype="int32")
            ret = fluid.layers.nn.uniform_random(shape)

            place = fluid.CPUPlace()
            if fluid.core.is_compiled_with_cuda():
                place = fluid.CUDAPlace(0)
            exe = fluid.Executor(place)
            Shape = np.array([2, 3]).astype('int32')
            exe.run(startup_program)
            outs = exe.run(train_program,
                           feed={'shape_tensor': Shape},
                           fetch_list=[ret])


class TestUniformRandomOp_API_seed(unittest.TestCase):
    def test_attr_tensor_API(self):
        startup_program = fluid.Program()
        train_program = fluid.Program()
        with fluid.program_guard(train_program, startup_program):
            _min = 5
            _max = 10
            _seed = 10
            ret = fluid.layers.nn.uniform_random(
                [2, 3, 2], min=_min, max=_max, seed=_seed)
            ret_2 = fluid.layers.nn.uniform_random(
                [2, 3, 2], min=_min, max=_max, seed=_seed)
            res = fluid.layers.equal(ret, ret_2)
            place = fluid.CPUPlace()
            if fluid.core.is_compiled_with_cuda():
                place = fluid.CUDAPlace(0)
            exe = fluid.Executor(place)

            exe.run(startup_program)
            ret_value, cmp_value = exe.run(train_program, fetch_list=[ret, res])
            self.assertTrue(np.array(cmp_value).all())
            for i in ret_value.flatten():
                self.assertGreaterEqual(i, _min)
                self.assertLess(i, _max)


class TestUniformRandomOpSelectedRowsShapeTensor(unittest.TestCase):
    def get_places(self):
        places = [core.CPUPlace()]
        if core.is_compiled_with_cuda():
            places.append(core.CUDAPlace(0))
        return places

    def test_check_output(self):
        for place in self.get_places():
            self.check_with_place(place)

    def check_with_place(self, place):
        scope = core.Scope()
        out = scope.var("X").get_selected_rows()
        shape_tensor = scope.var("Shape").get_tensor()
        shape_tensor.set(np.array([4, 784]).astype("int64"), place)

        op = Operator(
            "uniform_random",
            ShapeTensor="Shape",
            Out="X",
            min=-5.0,
            max=10.0,
            seed=10)
        op.run(scope, place)
        self.assertEqual(out.get_tensor().shape(), [4, 784])
        hist, prob = output_hist(np.array(out.get_tensor()))
        self.assertTrue(
            np.allclose(
                hist, prob, rtol=0, atol=0.01), "hist: " + str(hist))


class TestUniformRandomOpSelectedRowsShapeTensorList(unittest.TestCase):
    def get_places(self):
        places = [core.CPUPlace()]
        if core.is_compiled_with_cuda():
            places.append(core.CUDAPlace(0))
        return places

    def test_check_output(self):
        for place in self.get_places():
            self.check_with_place(place)

    def check_with_place(self, place):
        scope = core.Scope()
        out = scope.var("X").get_selected_rows()
        shape_1 = scope.var("shape1").get_tensor()
        shape_1.set(np.array([4]).astype("int64"), place)
        shape_2 = scope.var("shape2").get_tensor()
        shape_2.set(np.array([784]).astype("int64"), place)

        op = Operator(
            "uniform_random",
            ShapeTensorList=["shape1", "shape2"],
            Out="X",
            min=-5.0,
            max=10.0,
            seed=10)
        op.run(scope, place)
        self.assertEqual(out.get_tensor().shape(), [4, 784])
        hist, prob = output_hist(np.array(out.get_tensor()))
        self.assertTrue(
            np.allclose(
                hist, prob, rtol=0, atol=0.01), "hist: " + str(hist))


class TestUniformRandomDygraphMode(unittest.TestCase):
    def test_check_output(self):
        with fluid.dygraph.guard():
            x = fluid.layers.uniform_random(
                [10], dtype="float32", min=0.0, max=1.0)
            x_np = x.numpy()
            for i in range(10):
                self.assertTrue((x_np[i] > 0 and x_np[i] < 1.0))


if __name__ == "__main__":
    unittest.main()
