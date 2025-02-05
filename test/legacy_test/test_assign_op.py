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

import os
import unittest

import gradient_checker
import numpy as np
import op_test
from decorator_helper import prog_scope
from op_test import convert_float_to_uint16, convert_uint16_to_float

import paddle
from paddle import base
from paddle.base import Program, core, program_guard
from paddle.base.backward import append_backward


class TestAssignOp(op_test.OpTest):
    def setUp(self):
        self.python_api = paddle.assign
        self.public_python_api = paddle.assign
        self.op_type = "assign"
        self.prim_op_type = "prim"
        self.init_input_configs()
        x = np.random.random(size=self.shape).astype('float64')
        self.inputs = {'X': x}
        self.outputs = {'Out': x}

    def init_input_configs(self):
        self.shape = (100, 10)

    def test_forward(self):
        paddle.enable_static()
        self.check_output(check_pir=True)
        paddle.disable_static()

    def test_backward(self):
        paddle.enable_static()
        self.check_grad(
            ['X'], 'Out', check_prim=True, check_pir=True, check_prim_pir=True
        )
        paddle.disable_static()


class TestAssignOp_ZeroDim(TestAssignOp):
    def init_input_configs(self):
        self.shape = ()


@unittest.skipIf(
    not paddle.is_compiled_with_cuda(), "FP16 test runs only on GPU"
)
class TestAssignFP16Op(op_test.OpTest):
    def setUp(self):
        self.python_api = paddle.assign
        self.public_python_api = paddle.assign
        self.op_type = "assign"
        self.prim_op_type = "prim"
        x = np.random.random(size=(100, 10)).astype('float16')
        self.inputs = {'X': x}
        self.outputs = {'Out': x}

    def test_forward(self):
        paddle.enable_static()
        self.check_output(check_pir=True)
        paddle.disable_static()

    def test_backward(self):
        paddle.enable_static()
        self.check_grad(
            ['X'], 'Out', check_prim=True, check_pir=True, check_prim_pir=True
        )
        paddle.disable_static()


@unittest.skipIf(
    not paddle.is_compiled_with_cuda() or paddle.is_compiled_with_rocm(),
    "BFP16 test runs only on CUDA",
)
class TestAssignBFP16Op(op_test.OpTest):
    def setUp(self):
        self.python_api = paddle.assign
        self.public_python_api = paddle.assign
        self.op_type = "assign"
        self.prim_op_type = "prim"
        x = np.random.uniform(0, 1, [100, 10]).astype(np.float32)
        x = convert_float_to_uint16(x)
        self.inputs = {'X': x}
        self.outputs = {'Out': x}

    def test_forward(self):
        paddle.enable_static()
        self.check_output(check_pir=True)
        paddle.disable_static()

    def test_backward(self):
        paddle.enable_static()
        self.check_grad(
            ['X'], 'Out', check_prim=True, check_pir=True, check_prim_pir=True
        )
        paddle.disable_static()


class TestAssignOpWithTensorArray(unittest.TestCase):

    def test_assign_tensor_array(self):
        paddle.enable_static()
        main_program = paddle.static.Program()
        startup_program = paddle.static.Program()
        with paddle.static.program_guard(main_program, startup_program):
            x = paddle.static.data(name='x', shape=[100, 10], dtype='float32')
            x.stop_gradient = False
            y = paddle.tensor.fill_constant(
                shape=[100, 10], dtype='float32', value=1
            )
            z = paddle.add(x=x, y=y)
            i = paddle.tensor.fill_constant(shape=[1], dtype='int64', value=0)
            init_array = paddle.tensor.array_write(x=z, i=i)
            array = paddle.assign(init_array)
            sums = paddle.tensor.array_read(array=init_array, i=i)
            mean = paddle.mean(sums)
            [(_, x_grad)] = append_backward(mean, parameter_list=[x])

        place = (
            paddle.CUDAPlace(0)
            if paddle.is_compiled_with_cuda()
            else paddle.CPUPlace()
        )
        exe = paddle.static.Executor(place)
        feed_x = np.random.random(size=(100, 10)).astype('float32')
        ones = np.ones((100, 10)).astype('float32')
        feed_add = feed_x + ones
        res = exe.run(
            main_program,
            feed={'x': feed_x},
            fetch_list=[sums, x_grad],
        )
        np.testing.assert_allclose(res[0], feed_add, rtol=1e-05)
        np.testing.assert_allclose(res[1], ones / 1000.0, rtol=1e-05)
        paddle.disable_static()


class TestAssignOpError(unittest.TestCase):

    def test_errors(self):
        paddle.enable_static()
        with program_guard(Program(), Program()):
            # The type of input must be Variable or numpy.ndarray.
            x1 = base.create_lod_tensor(
                np.array([[-1]]), [[1]], base.CPUPlace()
            )
            self.assertRaises(TypeError, paddle.assign, x1)
            # When the type of input is numpy.ndarray, the dtype of input must be float32, int32.
            x2 = np.array([[2.5, 2.5]], dtype='uint8')
            self.assertRaises(TypeError, paddle.assign, x2)
        paddle.disable_static()


class TestAssignOpApi(unittest.TestCase):
    def test_assign_numpy_array(self):
        for dtype in [np.bool_, np.float32, np.int32, np.int64]:
            with base.dygraph.guard():
                array = np.random.random(size=(100, 10)).astype(dtype)
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
        self.python_api = paddle.clone

        x = paddle.ones([2])
        x.stop_gradient = False
        x.retain_grads()
        clone_x = paddle.clone(x)
        clone_x.retain_grads()

        y = clone_x**3
        y.backward()

        np.testing.assert_array_equal(x, [1, 1])
        np.testing.assert_array_equal(clone_x.grad.numpy(), [3, 3])
        np.testing.assert_array_equal(x.grad.numpy(), [3, 3])
        paddle.enable_static()

        with program_guard(Program(), Program()):
            x_np = np.random.randn(2, 3).astype('float32')
            x = paddle.static.data("X", shape=[2, 3])
            clone_x = paddle.clone(x)
            exe = paddle.static.Executor()
            y_np = exe.run(
                paddle.static.default_main_program(),
                feed={'X': x_np},
                fetch_list=[clone_x],
            )[0]

        np.testing.assert_array_equal(y_np, x_np)
        paddle.disable_static()


@unittest.skipIf(
    not paddle.is_compiled_with_cuda(), "FP16 test runs only on GPU"
)
class TestAssignOpApiFP16(unittest.TestCase):
    def test_assign_fp16(self):
        x = np.random.uniform(0, 10, [3, 3]).astype(np.float16)
        x = paddle.to_tensor(x)
        result = paddle.zeros(shape=[3, 3], dtype='float16')
        paddle.assign(x, result)
        np.testing.assert_equal(result.numpy(), x.numpy())

    def test_assign_bfp16(self):
        x_f = np.random.uniform(0, 10, [3, 3]).astype(np.float32)
        x = convert_float_to_uint16(x_f)
        x = paddle.to_tensor(x)
        result = paddle.zeros(shape=[3, 3], dtype='bfloat16')
        paddle.assign(x, result)
        np.testing.assert_allclose(
            convert_uint16_to_float(result.numpy()), x_f, rtol=1e-02
        )
        np.testing.assert_equal(
            convert_uint16_to_float(result.numpy()), convert_uint16_to_float(x)
        )


class TestAssignOut_(unittest.TestCase):
    def test_pir_assign_out_(self):
        with paddle.pir_utils.IrGuard():
            main_program = base.Program()
            startup_program = base.Program()
            with base.program_guard(main_program, startup_program):
                out = paddle.tensor.fill_constant(
                    [2, 2], dtype='float32', value=0.0
                )
                tmp = paddle.tensor.fill_constant(
                    [2, 2], dtype='float32', value=1.0
                )
                tmp.stop_gradient = False
                x = paddle.add(tmp, tmp)
                paddle.assign(x, out)
                loss = paddle.mean(out)
                dx = paddle.autograd.ir_backward.grad(loss, tmp)

                exe = paddle.static.Executor()
                dx_out = exe.run(
                    paddle.static.default_main_program(),
                    feed={},
                    fetch_list=[dx],
                )[0]

        np.testing.assert_array_equal(dx_out, 0.5 * np.ones((2, 2)))


class TestAssignOpErrorApi(unittest.TestCase):

    def test_errors(self):
        paddle.enable_static()
        with paddle.static.program_guard(
            paddle.static.Program(), paddle.static.Program()
        ):
            # The type of input must be Variable or numpy.ndarray.
            x1 = base.create_lod_tensor(
                np.array([[-1]]), [[1]], base.CPUPlace()
            )
            self.assertRaises(TypeError, paddle.assign, x1)
            # When the type of input is numpy.ndarray, the dtype of input must be float32, int32.
            x2 = np.array([[2.5, 2.5]], dtype='uint8')
            self.assertRaises(TypeError, paddle.assign, x2)
        paddle.disable_static()

    def test_type_error(self):
        paddle.enable_static()
        with paddle.static.program_guard(
            paddle.static.Program(), paddle.static.Program()
        ):
            x = [paddle.randn([3, 3]), paddle.randn([3, 3])]
            # not support to assign list(var)
            self.assertRaises(TypeError, paddle.assign, x)
        paddle.disable_static()


class TestAssignDoubleGradCheck(unittest.TestCase):
    def assign_wrapper(self, x):
        return paddle.assign(x[0])

    @prog_scope()
    def func(self, place):
        # the shape of input variable should be clearly specified, not include -1.
        eps = 0.005
        dtype = np.float32

        data = paddle.static.data('data', [3, 4, 5], dtype)
        data.persistable = True
        out = paddle.assign(data)
        data_arr = np.random.uniform(-1, 1, data.shape).astype(dtype)

        gradient_checker.double_grad_check(
            [data], out, x_init=[data_arr], place=place, eps=eps
        )
        gradient_checker.double_grad_check_for_dygraph(
            self.assign_wrapper, [data], out, x_init=[data_arr], place=place
        )

    def test_grad(self):
        paddle.enable_static()
        places = []
        if (
            os.environ.get('FLAGS_CI_both_cpu_and_gpu', 'False').lower()
            in ['1', 'true', 'on']
            or not core.is_compiled_with_cuda()
        ):
            places.append(base.CPUPlace())
        if core.is_compiled_with_cuda():
            places.append(base.CUDAPlace(0))
        for p in places:
            self.func(p)
        paddle.disable_static()


class TestAssignTripleGradCheck(unittest.TestCase):
    def assign_wrapper(self, x):
        return paddle.assign(x[0])

    @prog_scope()
    def func(self, place):
        # the shape of input variable should be clearly specified, not include -1.
        eps = 0.005
        dtype = np.float32

        data = paddle.static.data('data', [3, 4, 5], dtype)
        data.persistable = True
        out = paddle.assign(data)
        data_arr = np.random.uniform(-1, 1, data.shape).astype(dtype)

        gradient_checker.triple_grad_check(
            [data], out, x_init=[data_arr], place=place, eps=eps
        )
        gradient_checker.triple_grad_check_for_dygraph(
            self.assign_wrapper, [data], out, x_init=[data_arr], place=place
        )

    def test_grad(self):
        paddle.enable_static()
        places = []
        if (
            os.environ.get('FLAGS_CI_both_cpu_and_gpu', 'False').lower()
            in ['1', 'true', 'on']
            or not core.is_compiled_with_cuda()
        ):
            places.append(base.CPUPlace())
        if core.is_compiled_with_cuda():
            places.append(base.CUDAPlace(0))
        for p in places:
            self.func(p)
        paddle.disable_static()


if __name__ == '__main__':
    unittest.main()
