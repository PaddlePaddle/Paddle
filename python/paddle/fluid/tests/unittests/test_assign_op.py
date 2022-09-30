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

import op_test
import numpy as np
import unittest
import paddle
import paddle.fluid.core as core
from paddle.fluid.op import Operator
import paddle.fluid as fluid
from paddle.fluid import compiler, Program, program_guard
from paddle.fluid.backward import append_backward
import paddle.fluid.framework as framework
import gradient_checker
from decorator_helper import prog_scope
import paddle.fluid.layers as layers


class TestAssignOp(op_test.OpTest):

    def setUp(self):
        self.python_api = paddle.assign
        self.op_type = "assign"
        x = np.random.random(size=(100, 10)).astype('float64')
        self.inputs = {'X': x}
        self.outputs = {'Out': x}

    def test_forward(self):
        paddle.enable_static()
        fluid.set_flags({"FLAGS_retain_grad_for_all_tensor": True})
        self.check_output(check_eager=True)
        fluid.set_flags({"FLAGS_retain_grad_for_all_tensor": False})
        paddle.disable_static()
        framework._disable_legacy_dygraph()

    def test_backward(self):
        paddle.enable_static()
        fluid.set_flags({"FLAGS_retain_grad_for_all_tensor": True})
        self.check_grad(['X'], 'Out', check_eager=True)
        fluid.set_flags({"FLAGS_retain_grad_for_all_tensor": False})
        paddle.disable_static()
        framework._disable_legacy_dygraph()


class TestAssignFP16Op(op_test.OpTest):

    def setUp(self):
        self.python_api = paddle.assign
        self.op_type = "assign"
        x = np.random.random(size=(100, 10)).astype('float16')
        self.inputs = {'X': x}
        self.outputs = {'Out': x}

    def test_forward(self):
        paddle.enable_static()
        fluid.set_flags({"FLAGS_retain_grad_for_all_tensor": True})
        self.check_output(check_eager=True)
        fluid.set_flags({"FLAGS_retain_grad_for_all_tensor": False})
        paddle.disable_static()
        framework._disable_legacy_dygraph()

    def test_backward(self):
        paddle.enable_static()
        fluid.set_flags({"FLAGS_retain_grad_for_all_tensor": True})
        self.check_grad(['X'], 'Out', check_eager=True)
        fluid.set_flags({"FLAGS_retain_grad_for_all_tensor": False})
        paddle.disable_static()
        framework._disable_legacy_dygraph()


class TestAssignOpWithLoDTensorArray(unittest.TestCase):

    def test_assign_LoDTensorArray(self):
        paddle.enable_static()
        fluid.set_flags({"FLAGS_retain_grad_for_all_tensor": True})
        main_program = Program()
        startup_program = Program()
        with program_guard(main_program):
            x = fluid.data(name='x', shape=[100, 10], dtype='float32')
            x.stop_gradient = False
            y = fluid.layers.fill_constant(shape=[100, 10],
                                           dtype='float32',
                                           value=1)
            z = fluid.layers.elementwise_add(x=x, y=y)
            i = fluid.layers.fill_constant(shape=[1], dtype='int64', value=0)
            init_array = fluid.layers.array_write(x=z, i=i)
            array = fluid.layers.assign(init_array)
            sums = fluid.layers.array_read(array=init_array, i=i)
            mean = paddle.mean(sums)
            append_backward(mean)
        fluid.set_flags({"FLAGS_retain_grad_for_all_tensor": False})

        place = fluid.CUDAPlace(
            0) if core.is_compiled_with_cuda() else fluid.CPUPlace()
        exe = fluid.Executor(place)
        feed_x = np.random.random(size=(100, 10)).astype('float32')
        ones = np.ones((100, 10)).astype('float32')
        feed_add = feed_x + ones
        res = exe.run(main_program,
                      feed={'x': feed_x},
                      fetch_list=[sums.name, x.grad_name])
        np.testing.assert_allclose(res[0], feed_add, rtol=1e-05)
        np.testing.assert_allclose(res[1], ones / 1000.0, rtol=1e-05)
        paddle.disable_static()


class TestAssignOpError(unittest.TestCase):

    def test_errors(self):
        paddle.enable_static()
        with program_guard(Program(), Program()):
            # The type of input must be Variable or numpy.ndarray.
            x1 = fluid.create_lod_tensor(np.array([[-1]]), [[1]],
                                         fluid.CPUPlace())
            self.assertRaises(TypeError, fluid.layers.assign, x1)
            # When the type of input is numpy.ndarray, the dtype of input must be float32, int32.
            x2 = np.array([[2.5, 2.5]], dtype='uint8')
            self.assertRaises(TypeError, fluid.layers.assign, x2)
        paddle.disable_static()


class TestAssignOApi(unittest.TestCase):

    def test_assign_LoDTensorArray(self):
        paddle.enable_static()
        main_program = Program()
        startup_program = Program()
        with program_guard(main_program):
            x = fluid.data(name='x', shape=[100, 10], dtype='float32')
            x.stop_gradient = False
            y = fluid.layers.fill_constant(shape=[100, 10],
                                           dtype='float32',
                                           value=1)
            z = fluid.layers.elementwise_add(x=x, y=y)
            i = fluid.layers.fill_constant(shape=[1], dtype='int64', value=0)
            init_array = fluid.layers.array_write(x=z, i=i)
            array = paddle.assign(init_array)
            sums = fluid.layers.array_read(array=init_array, i=i)
            mean = paddle.mean(sums)
            append_backward(mean)

        place = fluid.CUDAPlace(
            0) if core.is_compiled_with_cuda() else fluid.CPUPlace()
        exe = fluid.Executor(place)
        feed_x = np.random.random(size=(100, 10)).astype('float32')
        ones = np.ones((100, 10)).astype('float32')
        feed_add = feed_x + ones
        res = exe.run(main_program,
                      feed={'x': feed_x},
                      fetch_list=[sums.name, x.grad_name])
        np.testing.assert_allclose(res[0], feed_add, rtol=1e-05)
        np.testing.assert_allclose(res[1], ones / 1000.0, rtol=1e-05)
        paddle.disable_static()

    def test_assign_NumpyArray(self):
        with fluid.dygraph.guard():
            array = np.random.random(size=(100, 10)).astype(np.bool_)
            result1 = paddle.zeros(shape=[3, 3], dtype='float32')
            paddle.assign(array, result1)
        np.testing.assert_allclose(result1.numpy(), array, rtol=1e-05)

    def test_assign_NumpyArray1(self):
        with fluid.dygraph.guard():
            array = np.random.random(size=(100, 10)).astype(np.float32)
            result1 = paddle.zeros(shape=[3, 3], dtype='float32')
            paddle.assign(array, result1)
        np.testing.assert_allclose(result1.numpy(), array, rtol=1e-05)

    def test_assign_NumpyArray2(self):
        with fluid.dygraph.guard():
            array = np.random.random(size=(100, 10)).astype(np.int32)
            result1 = paddle.zeros(shape=[3, 3], dtype='float32')
            paddle.assign(array, result1)
        np.testing.assert_allclose(result1.numpy(), array, rtol=1e-05)

    def test_assign_NumpyArray3(self):
        with fluid.dygraph.guard():
            array = np.random.random(size=(100, 10)).astype(np.int64)
            result1 = paddle.zeros(shape=[3, 3], dtype='float32')
            paddle.assign(array, result1)
        np.testing.assert_allclose(result1.numpy(), array, rtol=1e-05)

    def test_assign_List(self):
        l = [1, 2, 3]
        result = paddle.assign(l)
        np.testing.assert_allclose(result.numpy(), np.array(l), rtol=1e-05)

    def test_assign_BasicTypes(self):
        result1 = paddle.assign(2)
        result2 = paddle.assign(3.0)
        result3 = paddle.assign(True)
        np.testing.assert_allclose(result1.numpy(), np.array([2]), rtol=1e-05)
        np.testing.assert_allclose(result2.numpy(), np.array([3.0]), rtol=1e-05)
        np.testing.assert_allclose(result3.numpy(), np.array([1]), rtol=1e-05)

    def test_clone(self):
        fluid.set_flags({"FLAGS_retain_grad_for_all_tensor": True})
        self.python_api = paddle.clone

        x = paddle.ones([2])
        x.stop_gradient = False
        clone_x = paddle.clone(x)

        y = clone_x**3
        y.backward()

        np.testing.assert_array_equal(x, [1, 1])
        np.testing.assert_array_equal(clone_x.grad.numpy(), [3, 3])
        np.testing.assert_array_equal(x.grad.numpy(), [3, 3])
        fluid.set_flags({"FLAGS_retain_grad_for_all_tensor": False})
        paddle.enable_static()

        with program_guard(Program(), Program()):
            x_np = np.random.randn(2, 3).astype('float32')
            x = paddle.static.data("X", shape=[2, 3])
            clone_x = paddle.clone(x)
            exe = paddle.static.Executor()
            y_np = exe.run(paddle.static.default_main_program(),
                           feed={'X': x_np},
                           fetch_list=[clone_x])[0]

        np.testing.assert_array_equal(y_np, x_np)
        paddle.disable_static()


class TestAssignOpErrorApi(unittest.TestCase):

    def test_errors(self):
        paddle.enable_static()
        fluid.set_flags({"FLAGS_retain_grad_for_all_tensor": True})
        with program_guard(Program(), Program()):
            # The type of input must be Variable or numpy.ndarray.
            x1 = fluid.create_lod_tensor(np.array([[-1]]), [[1]],
                                         fluid.CPUPlace())
            self.assertRaises(TypeError, paddle.assign, x1)
            # When the type of input is numpy.ndarray, the dtype of input must be float32, int32.
            x2 = np.array([[2.5, 2.5]], dtype='uint8')
            self.assertRaises(TypeError, paddle.assign, x2)
        fluid.set_flags({"FLAGS_retain_grad_for_all_tensor": False})
        paddle.disable_static()

    def test_type_error(self):
        paddle.enable_static()
        with program_guard(Program(), Program()):
            x = [paddle.randn([3, 3]), paddle.randn([3, 3])]
            # not support to assign list(var)
            self.assertRaises(TypeError, paddle.assign, x)
        paddle.disable_static()


class TestAssignDoubleGradCheck(unittest.TestCase):

    def assign_wrapper(self, x):
        return paddle.fluid.layers.assign(x[0])

    @prog_scope()
    def func(self, place):
        # the shape of input variable should be clearly specified, not inlcude -1.
        eps = 0.005
        dtype = np.float32

        data = layers.data('data', [3, 4, 5], False, dtype)
        data.persistable = True
        out = paddle.fluid.layers.assign(data)
        data_arr = np.random.uniform(-1, 1, data.shape).astype(dtype)

        gradient_checker.double_grad_check([data],
                                           out,
                                           x_init=[data_arr],
                                           place=place,
                                           eps=eps)
        fluid.set_flags({"FLAGS_retain_grad_for_all_tensor": True})
        gradient_checker.double_grad_check_for_dygraph(self.assign_wrapper,
                                                       [data],
                                                       out,
                                                       x_init=[data_arr],
                                                       place=place)

    def test_grad(self):
        paddle.enable_static()
        places = [fluid.CPUPlace()]
        if core.is_compiled_with_cuda():
            places.append(fluid.CUDAPlace(0))
        for p in places:
            self.func(p)


class TestAssignTripleGradCheck(unittest.TestCase):

    def assign_wrapper(self, x):
        return paddle.fluid.layers.assign(x[0])

    @prog_scope()
    def func(self, place):
        # the shape of input variable should be clearly specified, not inlcude -1.
        eps = 0.005
        dtype = np.float32

        data = layers.data('data', [3, 4, 5], False, dtype)
        data.persistable = True
        out = paddle.fluid.layers.assign(data)
        data_arr = np.random.uniform(-1, 1, data.shape).astype(dtype)

        gradient_checker.triple_grad_check([data],
                                           out,
                                           x_init=[data_arr],
                                           place=place,
                                           eps=eps)
        fluid.set_flags({"FLAGS_retain_grad_for_all_tensor": True})
        gradient_checker.triple_grad_check_for_dygraph(self.assign_wrapper,
                                                       [data],
                                                       out,
                                                       x_init=[data_arr],
                                                       place=place)

    def test_grad(self):
        paddle.enable_static()
        places = [fluid.CPUPlace()]
        if core.is_compiled_with_cuda():
            places.append(fluid.CUDAPlace(0))
        for p in places:
            self.func(p)


if __name__ == '__main__':
    unittest.main()
