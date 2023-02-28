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
import tempfile
import unittest

import numpy as np
from op_test import OpTest

import paddle
import paddle.fluid as fluid
import paddle.fluid.core as core
import paddle.inference as paddle_infer


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
            out = exe.run(
                feed={'X': data_np},
                fetch_list=[
                    y.name,
                    y2.name,
                    y3.name,
                    y4.name,
                    y5.name,
                    y6.name,
                ],
            )

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
            'Out': np.flip(
                np.flip(self.inputs['X'], axis=2).cumsum(axis=2), axis=2
            )
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


class TestCumsumFP16(unittest.TestCase):
    def check_main(self, x_np, dtype):
        paddle.disable_static()
        x = paddle.to_tensor(x_np.astype(dtype))
        x.stop_gradient = False
        y = paddle.cumsum(x, dtype=dtype)
        x_g = paddle.grad(y, [x])
        y_np = y.numpy().astype('float32')
        x_g_np = x_g[0].numpy().astype('float32')
        paddle.enable_static()
        return y_np, x_g_np

    def test_main(self):
        if not paddle.is_compiled_with_cuda():
            return

        np.random.seed(20)
        x_np = np.random.random([10, 12])
        y_np_1, x_g_np_1 = self.check_main(x_np, 'float16')
        y_np_2, x_g_np_2 = self.check_main(x_np, 'float32')

        np.testing.assert_allclose(y_np_1, y_np_2, rtol=1e-03)
        np.testing.assert_allclose(x_g_np_1, x_g_np_2, rtol=1e-03)


class TestSumOpExclusive1(OpTest):
    def setUp(self):
        self.op_type = "cumsum"
        self.attrs = {'axis': 2, "exclusive": True}
        a = np.random.random((4, 5, 20)).astype("float64")
        self.inputs = {'X': a}
        self.outputs = {
            'Out': np.concatenate(
                (
                    np.zeros((4, 5, 1), dtype=np.float64),
                    a[:, :, :-1].cumsum(axis=2),
                ),
                axis=2,
            )
        }

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(['X'], 'Out')


class TestSumOpExclusive2(OpTest):
    def setUp(self):
        self.op_type = "cumsum"
        self.attrs = {'axis': 2, "exclusive": True}
        a = np.random.random((1, 1, 100)).astype("float64")
        self.inputs = {'X': a}
        self.outputs = {
            'Out': np.concatenate(
                (
                    np.zeros((1, 1, 1), dtype=np.float64),
                    a[:, :, :-1].cumsum(axis=2),
                ),
                axis=2,
            )
        }

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(['X'], 'Out')


class TestSumOpExclusive3(OpTest):
    def setUp(self):
        self.op_type = "cumsum"
        self.attrs = {'axis': 2, "exclusive": True}
        a = np.random.random((4, 5, 20)).astype("float64")
        self.inputs = {'X': a}
        self.outputs = {
            'Out': np.concatenate(
                (
                    np.zeros((4, 5, 1), dtype=np.float64),
                    a[:, :, :-1].cumsum(axis=2),
                ),
                axis=2,
            )
        }

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(['X'], 'Out')


class TestSumOpExclusive4(OpTest):
    def setUp(self):
        self.op_type = "cumsum"
        self.attrs = {'axis': 2, "exclusive": True}
        a = np.random.random((1, 1, 100)).astype("float64")
        self.inputs = {'X': a}
        self.outputs = {
            'Out': np.concatenate(
                (
                    np.zeros((1, 1, 1), dtype=np.float64),
                    a[:, :, :-1].cumsum(axis=2),
                ),
                axis=2,
            )
        }

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(['X'], 'Out')


class TestSumOpExclusive5(OpTest):
    def setUp(self):
        self.op_type = "cumsum"
        self.attrs = {'axis': 2, "exclusive": True}
        a = np.random.random((4, 5, 40)).astype("float64")
        self.inputs = {'X': a}
        self.outputs = {
            'Out': np.concatenate(
                (
                    np.zeros((4, 5, 1), dtype=np.float64),
                    a[:, :, :-1].cumsum(axis=2),
                ),
                axis=2,
            )
        }

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(['X'], 'Out')


class TestSumOpExclusiveFP16(OpTest):
    def setUp(self):
        self.op_type = "cumsum"
        self.attrs = {'axis': 2, "exclusive": True, "dtype": "float16"}
        a = np.random.random((4, 5, 20)).astype("float64")
        self.inputs = {'X': a}
        self.outputs = {
            'Out': np.concatenate(
                (
                    np.zeros((4, 5, 1), dtype=np.float64),
                    a[:, :, :-1].cumsum(axis=2),
                ),
                axis=2,
            )
        }

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(['X'], 'Out')


class TestSumOpReverseExclusive(OpTest):
    def setUp(self):
        self.op_type = "cumsum"
        self.attrs = {'axis': 2, 'reverse': True, "exclusive": True}
        a = np.random.random((4, 5, 6)).astype("float64")
        self.inputs = {'X': a}
        a = np.flip(a, axis=2)
        self.outputs = {
            'Out': np.concatenate(
                (
                    np.flip(a[:, :, :-1].cumsum(axis=2), axis=2),
                    np.zeros((4, 5, 1), dtype=np.float64),
                ),
                axis=2,
            )
        }

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(['X'], 'Out')


class BadInputTest(unittest.TestCase):
    def test_error(self):
        with fluid.program_guard(fluid.Program()):

            def test_bad_x():
                data = [1, 2, 4]
                result = paddle.cumsum(data, axis=0)

            self.assertRaises(TypeError, test_bad_x)


class TestTensorAxis(unittest.TestCase):
    def setUp(self):
        paddle.seed(2022)
        self.temp_dir = tempfile.TemporaryDirectory()
        self.save_path = os.path.join(self.temp_dir.name, 'tensor_axis_cumsum')
        self.place = (
            paddle.CUDAPlace(0)
            if paddle.is_compiled_with_cuda()
            else paddle.CPUPlace()
        )

    def test_dygraph(self):
        paddle.disable_static()
        x = np.random.randn(5, 6)
        axis = 1
        np_out = np.cumsum(x, axis)
        pd_out = paddle.cumsum(
            paddle.to_tensor(x), axis=paddle.to_tensor([axis], dtype='int32')
        )
        np.testing.assert_allclose(np_out, pd_out.numpy())

    def test_static_and_infer(self):
        paddle.enable_static()
        np_x = np.random.randn(9, 10, 11).astype('float32')
        main_prog = paddle.static.Program()
        starup_prog = paddle.static.Program()
        with paddle.static.program_guard(main_prog, starup_prog):
            # run static
            x = paddle.static.data(shape=np_x.shape, name='x', dtype=np_x.dtype)
            linear = paddle.nn.Linear(np_x.shape[-1], np_x.shape[-1])
            linear_out = linear(x)
            relu_out = paddle.nn.functional.relu(linear_out)
            axis = paddle.full([1], 2, dtype='int64')
            out = paddle.cumsum(relu_out, axis=axis)
            loss = paddle.mean(out)
            sgd = paddle.optimizer.SGD(learning_rate=0.0)
            sgd.minimize(paddle.mean(out))

            exe = paddle.static.Executor(self.place)
            exe.run(starup_prog)
            static_out = exe.run(feed={'x': np_x}, fetch_list=[out])

            # run infer
            paddle.static.save_inference_model(self.save_path, [x], [out], exe)
            config = paddle_infer.Config(
                self.save_path + '.pdmodel', self.save_path + '.pdiparams'
            )
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


class TestCumSumOpFp16(unittest.TestCase):
    def test_fp16(self):
        x_np = np.random.random((100, 100)).astype('float16')
        with paddle.static.program_guard(paddle.static.Program()):
            x = paddle.static.data(shape=[100, 100], name='x', dtype='float16')
            y1 = paddle.cumsum(x)
            y2 = paddle.cumsum(x, axis=0)
            y3 = paddle.cumsum(x, axis=-1)
            y4 = paddle.cumsum(x, axis=-2)
            if core.is_compiled_with_cuda():
                place = paddle.CUDAPlace(0)
                exe = paddle.static.Executor(place)
                exe.run(paddle.static.default_startup_program())
                out = exe.run(feed={'x': x_np}, fetch_list=[y1, y2, y3, y4])


if __name__ == '__main__':
    unittest.main()
