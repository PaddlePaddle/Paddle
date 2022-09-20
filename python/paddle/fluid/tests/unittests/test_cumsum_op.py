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

import os
import unittest
import tempfile
import numpy as np
from op_test import OpTest
import paddle
import paddle.fluid.core as core
import paddle.fluid as fluid
from paddle.fluid import compiler, Program, program_guard
import paddle.inference as paddle_infer
import gradient_checker
from decorator_helper import prog_scope
import paddle.fluid.layers as layers


class TestCumsumOp(unittest.TestCase):

    def run_cases(self):
        data_np = np.arange(12).reshape(3, 4)
        data = paddle.to_tensor(data_np)

        y = paddle.cumsum(data)
        z = np.cumsum(data_np)
        np.testing.assert_array_equal(z, y.numpy())

        y = paddle.cumsum(data, axis=0)
        z = np.cumsum(data_np, axis=0)
        np.testing.assert_array_equal(z, y.numpy())

        y = paddle.cumsum(data, axis=-1)
        z = np.cumsum(data_np, axis=-1)
        np.testing.assert_array_equal(z, y.numpy())

        y = paddle.cumsum(data, dtype='float64')
        self.assertTrue(y.dtype == core.VarDesc.VarType.FP64)

        y = paddle.cumsum(data, dtype=np.int32)
        self.assertTrue(y.dtype == core.VarDesc.VarType.INT32)

        y = paddle.cumsum(data, axis=-2)
        z = np.cumsum(data_np, axis=-2)
        np.testing.assert_array_equal(z, y.numpy())

    def run_static(self, use_gpu=False):
        with fluid.program_guard(fluid.Program()):
            data_np = np.random.random((100, 100)).astype(np.float32)
            x = paddle.static.data('X', [100, 100])
            y = paddle.cumsum(x)
            y2 = paddle.cumsum(x, axis=0)
            y3 = paddle.cumsum(x, axis=-1)
            y4 = paddle.cumsum(x, dtype='float64')
            y5 = paddle.cumsum(x, dtype=np.int32)
            y6 = paddle.cumsum(x, axis=-2)

            place = fluid.CUDAPlace(0) if use_gpu else fluid.CPUPlace()
            exe = fluid.Executor(place)
            exe.run(fluid.default_startup_program())
            out = exe.run(feed={'X': data_np},
                          fetch_list=[
                              y.name, y2.name, y3.name, y4.name, y5.name,
                              y6.name
                          ])

            z = np.cumsum(data_np)
            np.testing.assert_allclose(z, out[0], rtol=1e-05)
            z = np.cumsum(data_np, axis=0)
            np.testing.assert_allclose(z, out[1], rtol=1e-05)
            z = np.cumsum(data_np, axis=-1)
            np.testing.assert_allclose(z, out[2], rtol=1e-05)
            self.assertTrue(out[3].dtype == np.float64)
            self.assertTrue(out[4].dtype == np.int32)
            z = np.cumsum(data_np, axis=-2)
            np.testing.assert_allclose(z, out[5], rtol=1e-05)

    def test_cpu(self):
        paddle.disable_static(paddle.fluid.CPUPlace())
        self.run_cases()
        paddle.enable_static()

        self.run_static()

    def test_gpu(self):
        if not fluid.core.is_compiled_with_cuda():
            return
        paddle.disable_static(paddle.fluid.CUDAPlace(0))
        self.run_cases()
        paddle.enable_static()

        self.run_static(use_gpu=True)

    def test_name(self):
        with fluid.program_guard(fluid.Program()):
            x = paddle.static.data('x', [3, 4])
            y = paddle.cumsum(x, name='out')
            self.assertTrue('out' in y.name)


class TestSumOp1(OpTest):

    def setUp(self):
        self.op_type = "cumsum"
        self.attrs = {'axis': 2}
        self.inputs = {'X': np.random.random((5, 6, 10)).astype("float64")}
        self.outputs = {'Out': self.inputs['X'].cumsum(axis=2)}

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(['X'], 'Out')


class TestSumOp2(OpTest):

    def setUp(self):
        self.op_type = "cumsum"
        self.attrs = {'axis': -1, 'reverse': True}
        self.inputs = {'X': np.random.random((5, 6, 10)).astype("float64")}
        self.outputs = {
            'Out': np.flip(np.flip(self.inputs['X'], axis=2).cumsum(axis=2),
                           axis=2)
        }

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(['X'], 'Out')


class TestSumOp3(OpTest):

    def setUp(self):
        self.op_type = "cumsum"
        self.attrs = {'axis': 1}
        self.inputs = {'X': np.random.random((5, 6, 10)).astype("float64")}
        self.outputs = {'Out': self.inputs['X'].cumsum(axis=1)}

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(['X'], 'Out')


class TestSumOp4(OpTest):

    def setUp(self):
        self.op_type = "cumsum"
        self.attrs = {'axis': 0}
        self.inputs = {'X': np.random.random((5, 6, 10)).astype("float64")}
        self.outputs = {'Out': self.inputs['X'].cumsum(axis=0)}

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(['X'], 'Out')


class TestSumOp5(OpTest):

    def setUp(self):
        self.op_type = "cumsum"
        self.inputs = {'X': np.random.random((5, 20)).astype("float64")}
        self.outputs = {'Out': self.inputs['X'].cumsum(axis=1)}

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(['X'], 'Out')


class TestSumOp7(OpTest):

    def setUp(self):
        self.op_type = "cumsum"
        self.inputs = {'X': np.random.random((100)).astype("float64")}
        self.outputs = {'Out': self.inputs['X'].cumsum(axis=0)}

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(['X'], 'Out')


class TestSumOpExclusive1(OpTest):

    def setUp(self):
        self.op_type = "cumsum"
        self.attrs = {'axis': 2, "exclusive": True}
        a = np.random.random((4, 5, 65)).astype("float64")
        self.inputs = {'X': a}
        self.outputs = {
            'Out':
            np.concatenate((np.zeros(
                (4, 5, 1), dtype=np.float64), a[:, :, :-1].cumsum(axis=2)),
                           axis=2)
        }

    def test_check_output(self):
        self.check_output()


class TestSumOpExclusive2(OpTest):

    def setUp(self):
        self.op_type = "cumsum"
        self.attrs = {'axis': 2, "exclusive": True}
        a = np.random.random((1, 1, 888)).astype("float64")
        self.inputs = {'X': a}
        self.outputs = {
            'Out':
            np.concatenate((np.zeros(
                (1, 1, 1), dtype=np.float64), a[:, :, :-1].cumsum(axis=2)),
                           axis=2)
        }

    def test_check_output(self):
        self.check_output()


class TestSumOpExclusive3(OpTest):

    def setUp(self):
        self.op_type = "cumsum"
        self.attrs = {'axis': 2, "exclusive": True}
        a = np.random.random((4, 5, 888)).astype("float32")
        self.inputs = {'X': a}
        self.outputs = {
            'Out':
            np.concatenate((np.zeros(
                (4, 5, 1), dtype=np.float64), a[:, :, :-1].cumsum(axis=2)),
                           axis=2)
        }

    def test_check_output(self):
        self.check_output()


class TestSumOpExclusive4(OpTest):

    def setUp(self):
        self.op_type = "cumsum"
        self.attrs = {'axis': 2, "exclusive": True}
        a = np.random.random((1, 1, 3049)).astype("float64")
        self.inputs = {'X': a}
        self.outputs = {
            'Out':
            np.concatenate((np.zeros(
                (1, 1, 1), dtype=np.float64), a[:, :, :-1].cumsum(axis=2)),
                           axis=2)
        }

    def test_check_output(self):
        self.check_output()


class TestSumOpExclusive5(OpTest):

    def setUp(self):
        self.op_type = "cumsum"
        self.attrs = {'axis': 2, "exclusive": True}
        a = np.random.random((4, 5, 3096)).astype("float64")
        self.inputs = {'X': a}
        self.outputs = {
            'Out':
            np.concatenate((np.zeros(
                (4, 5, 1), dtype=np.float64), a[:, :, :-1].cumsum(axis=2)),
                           axis=2)
        }

    def test_check_output(self):
        self.check_output()


class TestSumOpReverseExclusive(OpTest):

    def setUp(self):
        self.op_type = "cumsum"
        self.attrs = {'axis': 2, 'reverse': True, "exclusive": True}
        a = np.random.random((4, 5, 6)).astype("float64")
        self.inputs = {'X': a}
        a = np.flip(a, axis=2)
        self.outputs = {
            'Out':
            np.concatenate(
                (np.flip(a[:, :, :-1].cumsum(axis=2),
                         axis=2), np.zeros((4, 5, 1), dtype=np.float64)),
                axis=2)
        }

    def test_check_output(self):
        self.check_output()


class BadInputTest(unittest.TestCase):

    def test_error(self):
        with fluid.program_guard(fluid.Program()):

            def test_bad_x():
                data = [1, 2, 4]
                result = fluid.layers.cumsum(data, axis=0)

            self.assertRaises(TypeError, test_bad_x)


class TestTensorAxis(unittest.TestCase):

    def setUp(self):
        paddle.seed(2022)
        self.temp_dir = tempfile.TemporaryDirectory()
        self.save_path = os.path.join(self.temp_dir.name, 'tensor_axis_cumsum')
        self.place = paddle.CUDAPlace(
            0) if paddle.is_compiled_with_cuda() else paddle.CPUPlace()

    def test_dygraph(self):
        paddle.disable_static()
        x = np.random.randn(5, 6)
        axis = 1
        np_out = np.cumsum(x, axis)
        pd_out = paddle.cumsum(paddle.to_tensor(x),
                               axis=paddle.to_tensor([axis], dtype='int32'))
        np.testing.assert_allclose(np_out, pd_out.numpy())

    def test_static_and_infer(self):
        paddle.enable_static()
        np_x = np.random.randn(9, 10, 11).astype('float32')
        main_prog = paddle.static.Program()
        starup_prog = paddle.static.Program()
        with paddle.static.program_guard(main_prog, starup_prog):
            # run static
            x = paddle.static.data(shape=np_x.shape, name='x', dtype=np_x.dtype)
            print(x)
            linear = paddle.nn.Linear(np_x.shape[-1], np_x.shape[-1])
            linear_out = linear(x)
            relu_out = paddle.nn.functional.relu(linear_out)
            axis = paddle.full([1], 2, dtype='int64')
            out = paddle.cumsum(relu_out, axis=axis)

            exe = paddle.static.Executor(self.place)
            exe.run(starup_prog)
            static_out = exe.run(feed={'x': np_x}, fetch_list=[out])

            # run infer
            paddle.static.save_inference_model(self.save_path, [x], [out], exe)
            config = paddle_infer.Config(self.save_path + '.pdmodel',
                                         self.save_path + '.pdiparams')
            if paddle.is_compiled_with_cuda():
                config.enable_use_gpu(100, 0)
            else:
                config.disable_gpu()

            predictor = paddle_infer.create_predictor(config)
            input_names = predictor.get_input_names()
            input_handle = predictor.get_input_handle(input_names[0])
            fake_input = np_x
            input_handle.reshape(np_x.shape)
            input_handle.copy_from_cpu(fake_input)
            predictor.run()
            output_names = predictor.get_output_names()
            output_handle = predictor.get_output_handle(output_names[0])
            infer_out = output_handle.copy_to_cpu()
            np.testing.assert_allclose(static_out[0], infer_out)


class TestCumsumDoubleGradCheck(unittest.TestCase):

    def cumsum_wrapper(self, x):
        return paddle.cumsum(x[0], 0)

    @prog_scope()
    def func(self, place):
        # the shape of input variable should be clearly specified, not inlcude -1.
        eps = 0.005
        dtype = np.float64

        data = layers.data('data', [3, 4], False, dtype)
        data.persistable = True
        out = paddle.cumsum(data, 0)
        data_arr = np.random.uniform(-1, 1, data.shape).astype(dtype)

        gradient_checker.double_grad_check([data],
                                           out,
                                           x_init=[data_arr],
                                           place=place,
                                           eps=eps)
        fluid.set_flags({"FLAGS_retain_grad_for_all_tensor": True})
        gradient_checker.double_grad_check_for_dygraph(self.cumsum_wrapper,
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


class TestCumsumTripleGradCheck(unittest.TestCase):

    def cumsum_wrapper(self, x):
        return paddle.cumsum(x[0], 0)

    @prog_scope()
    def func(self, place):
        # the shape of input variable should be clearly specified, not inlcude -1.
        eps = 0.005
        dtype = np.float32

        data = layers.data('data', [2, 3], False, dtype)
        data.persistable = True
        out = paddle.cumsum(data, 0)
        data_arr = np.random.uniform(-1, 1, data.shape).astype(dtype)

        gradient_checker.triple_grad_check([data],
                                           out,
                                           x_init=[data_arr],
                                           place=place,
                                           eps=eps)
        fluid.set_flags({"FLAGS_retain_grad_for_all_tensor": True})
        gradient_checker.triple_grad_check_for_dygraph(self.cumsum_wrapper,
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
