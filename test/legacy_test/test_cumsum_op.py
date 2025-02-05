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
import sys
import tempfile
import unittest

from paddle.framework import use_pir_api

sys.path.append("../../legacy_test")

import numpy as np
from op_test import OpTest, convert_float_to_uint16

import paddle
import paddle.inference as paddle_infer
from paddle import base
from paddle.base import core


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
        self.assertTrue(y.dtype == paddle.float64)

        y = paddle.cumsum(data, dtype=np.int32)
        self.assertTrue(y.dtype == paddle.int32)

        y = paddle.cumsum(data, axis=-2)
        z = np.cumsum(data_np, axis=-2)
        np.testing.assert_array_equal(z, y.numpy())

    def run_static(self, use_gpu=False):
        with paddle.static.program_guard(paddle.static.Program()):
            data_np = np.random.random((100, 100)).astype(np.float32)
            x = paddle.static.data('X', [100, 100])
            y = paddle.cumsum(x)
            y2 = paddle.cumsum(x, axis=0)
            y3 = paddle.cumsum(x, axis=-1)
            y4 = paddle.cumsum(x, dtype='float64')
            y5 = paddle.cumsum(x, dtype=np.int32)
            y6 = paddle.cumsum(x, axis=-2)

            place = base.CUDAPlace(0) if use_gpu else base.CPUPlace()
            exe = base.Executor(place)
            exe.run(paddle.static.default_startup_program())
            out = exe.run(
                feed={'X': data_np},
                fetch_list=[
                    y,
                    y2,
                    y3,
                    y4,
                    y5,
                    y6,
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

    def test_cpu_dygraph(self):
        paddle.disable_static(paddle.base.CPUPlace())
        self.run_cases()
        paddle.enable_static()

    def test_cpu_static(self):
        self.run_static()

    def test_gpu_dygraph(self):
        if not base.core.is_compiled_with_cuda():
            return
        paddle.disable_static(paddle.base.CUDAPlace(0))
        self.run_cases()
        paddle.enable_static()

    def test_gpu_static(self):
        if not base.core.is_compiled_with_cuda():
            return
        self.run_static(use_gpu=True)

    def test_name(self):
        with paddle.pir_utils.OldIrGuard():
            with base.program_guard(base.Program()):
                x = paddle.static.data('x', [3, 4])
                y = paddle.cumsum(x, name='out')
                self.assertTrue('out' in y.name)


def cumsum_wrapper(x, axis=-1, flatten=False, exclusive=False, reverse=False):
    return paddle._C_ops.cumsum(x, axis, flatten, exclusive, reverse)


class TestSumOp1(OpTest):
    def setUp(self):
        self.op_type = "cumsum"
        self.prim_op_type = "prim"
        self.python_api = cumsum_wrapper
        self.public_python_api = paddle.cumsum
        self.if_enable_cinn()
        self.init_dtype()
        self.set_attrs_input_output()
        if self.dtype == np.uint16:
            self.inputs = {'X': convert_float_to_uint16(self.x)}
            self.outputs = {'Out': convert_float_to_uint16(self.out)}
        else:
            self.inputs = {'X': self.x}
            self.outputs = {'Out': self.out}

    def test_check_output(self):
        self.check_output(check_pir=True)

    def test_check_grad(self):
        self.check_grad(
            ['X'], 'Out', check_prim=True, check_pir=True, check_prim_pir=True
        )

    def init_dtype(self):
        self.dtype = self.dtype_ = np.float64

    def if_enable_cinn(self):
        pass

    def set_attrs_input_output(self):
        self.attrs = {'axis': 2}
        self.x = np.random.random((5, 6, 10)).astype(self.dtype_)
        self.out = self.x.cumsum(axis=2)


class TestSumOp1_ZeroDim(TestSumOp1):
    def set_attrs_input_output(self):
        self.attrs = {'axis': 0}
        self.x = np.random.random(()).astype(self.dtype_)
        self.out = self.x

    def if_enable_cinn(self):
        self.enable_cinn = False


class TestSumOp2(TestSumOp1):
    def set_attrs_input_output(self):
        self.attrs = {'axis': -1, 'reverse': True}
        self.x = np.random.random((5, 6, 10)).astype(self.dtype_)
        self.out = np.flip(np.flip(self.x, axis=2).cumsum(axis=2), axis=2)


class TestSumOp3(TestSumOp1):
    def set_attrs_input_output(self):
        self.attrs = {'axis': 1}
        self.x = np.random.random((5, 6, 10)).astype(self.dtype_)
        self.out = self.x.cumsum(axis=1)


class TestSumOp4(TestSumOp1):
    def set_attrs_input_output(self):
        self.attrs = {'axis': 0}
        self.x = np.random.random((5, 6, 10)).astype(self.dtype_)
        self.out = self.x.cumsum(axis=0)


class TestSumOp5(TestSumOp1):
    def set_attrs_input_output(self):
        self.x = np.random.random((5, 20)).astype(self.dtype_)
        self.out = self.x.cumsum(axis=1)


class TestSumOp6(TestSumOp1):
    def set_attrs_input_output(self):
        self.attrs = {'axis': -1, 'flatten': True}
        self.x = np.random.random((5, 6, 5)).astype(self.dtype_)
        self.out = self.x.cumsum()


class TestSumOp7(TestSumOp1):
    def set_attrs_input_output(self):
        self.x = np.random.random(100).astype(self.dtype_)
        self.out = self.x.cumsum(axis=0)


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
        self.prim_op_type = "prim"
        self.python_api = cumsum_wrapper
        self.public_python_api = paddle.cumsum
        self.if_enable_cinn()
        self.init_dtype()
        self.set_attrs_input_output()
        if self.dtype == np.uint16:
            self.inputs = {'X': convert_float_to_uint16(self.x)}
            self.outputs = {'Out': convert_float_to_uint16(self.out)}
        else:
            self.inputs = {'X': self.x}
            self.outputs = {'Out': self.out}

    def test_check_output(self):
        self.check_output(check_pir=True)

    def test_check_grad(self):
        self.check_grad(
            ['X'], 'Out', check_prim=True, check_pir=True, check_prim_pir=True
        )

    def init_dtype(self):
        self.dtype = self.dtype_ = np.float64

    def if_enable_cinn(self):
        pass

    def set_attrs_input_output(self):
        self.attrs = {'axis': 2, 'exclusive': True}
        self.x = np.random.random((4, 5, 20)).astype(self.dtype_)
        self.out = np.concatenate(
            (
                np.zeros((4, 5, 1), dtype=self.dtype_),
                self.x[:, :, :-1].cumsum(axis=2),
            ),
            axis=2,
        )


class TestSumOpExclusive2(TestSumOpExclusive1):
    def set_attrs_input_output(self):
        self.attrs = {'axis': 2, 'exclusive': True}
        self.x = np.random.random((1, 1, 100)).astype(self.dtype_)
        self.out = np.concatenate(
            (
                np.zeros((1, 1, 1), dtype=self.dtype_),
                self.x[:, :, :-1].cumsum(axis=2),
            ),
            axis=2,
        )


class TestSumOpExclusive3(TestSumOpExclusive1):
    def set_attrs_input_output(self):
        self.attrs = {'axis': 2, 'exclusive': True}
        self.x = np.random.random((4, 5, 20)).astype(self.dtype_)
        self.out = np.concatenate(
            (
                np.zeros((4, 5, 1), dtype=self.dtype_),
                self.x[:, :, :-1].cumsum(axis=2),
            ),
            axis=2,
        )


class TestSumOpExclusive4(TestSumOpExclusive1):
    def set_attrs_input_output(self):
        self.attrs = {'axis': 2, 'exclusive': True}
        self.x = np.random.random((1, 1, 100)).astype(self.dtype_)
        self.out = np.concatenate(
            (
                np.zeros((1, 1, 1), dtype=self.dtype_),
                self.x[:, :, :-1].cumsum(axis=2),
            ),
            axis=2,
        )


class TestSumOpExclusive5(TestSumOpExclusive1):
    def set_attrs_input_output(self):
        self.attrs = {'axis': 2, 'exclusive': True}
        self.x = np.random.random((4, 5, 40)).astype(self.dtype_)
        self.out = np.concatenate(
            (
                np.zeros((4, 5, 1), dtype=self.dtype_),
                self.x[:, :, :-1].cumsum(axis=2),
            ),
            axis=2,
        )


class TestSumOpExclusiveFP16(OpTest):
    def setUp(self):
        self.op_type = "cumsum"
        self.prim_op_type = "prim"
        self.python_api = cumsum_wrapper
        self.public_python_api = paddle.cumsum
        self.init_dtype()
        self.attrs = {'axis': 2, "exclusive": True}
        self.x = np.random.random((4, 5, 20)).astype(self.dtype)
        self.out = np.concatenate(
            (
                np.zeros((4, 5, 1), dtype=self.dtype),
                self.x[:, :, :-1].cumsum(axis=2),
            ),
            axis=2,
        )
        if self.dtype == np.uint16:
            self.inputs = {'X': convert_float_to_uint16(self.x)}
            self.outputs = {'Out': convert_float_to_uint16(self.out)}
        else:
            self.inputs = {'X': self.x}
            self.outputs = {'Out': self.out}

    def test_check_output(self):
        self.check_output(check_pir=True)

    def test_check_grad(self):
        self.check_grad(
            ['X'], 'Out', check_prim=True, check_pir=True, check_prim_pir=True
        )

    def init_dtype(self):
        self.dtype = np.float16


class TestSumOpReverseExclusive(OpTest):
    def setUp(self):
        self.op_type = "cumsum"
        self.prim_op_type = "prim"
        self.python_api = cumsum_wrapper
        self.public_python_api = paddle.cumsum
        self.if_enable_cinn()
        self.init_dtype()
        self.attrs = {
            'axis': 2,
            'reverse': True,
            'exclusive': True,
        }
        self.x = np.random.random((4, 5, 6)).astype(self.dtype_)
        a = np.flip(self.x, axis=2)
        self.out = np.concatenate(
            (
                np.flip(a[:, :, :-1].cumsum(axis=2), axis=2),
                np.zeros((4, 5, 1), dtype=self.dtype_),
            ),
            axis=2,
        )
        if self.dtype == np.uint16:
            self.inputs = {'X': convert_float_to_uint16(self.x)}
            self.outputs = {'Out': convert_float_to_uint16(self.out)}
        else:
            self.inputs = {'X': self.x}
            self.outputs = {'Out': self.out}

    def test_check_output(self):
        self.check_output(check_pir=True)

    def test_check_grad(self):
        self.check_grad(
            ['X'], 'Out', check_prim=True, check_pir=True, check_prim_pir=True
        )

    def init_dtype(self):
        self.dtype = self.dtype_ = np.float64

    def if_enable_cinn(self):
        pass


def create_test_fp16_class(parent, max_relative_error=1e-2):
    class TestCumsumFP16Op(parent):
        def init_dtype(self):
            self.dtype = self.dtype_ = np.float16

        def if_enable_cinn(self):
            pass

        def test_check_output(self):
            self.check_output(check_pir=True)

        def test_check_grad(self):
            self.check_grad(
                ['X'],
                'Out',
                check_prim=True,
                check_pir=True,
                check_prim_pir=True,
            )

    cls_name = "{}_{}".format(parent.__name__, "Fp16")
    TestCumsumFP16Op.__name__ = cls_name
    globals()[cls_name] = TestCumsumFP16Op


create_test_fp16_class(TestSumOp1)
create_test_fp16_class(TestSumOp2)
create_test_fp16_class(TestSumOp3)
create_test_fp16_class(TestSumOp4)
create_test_fp16_class(TestSumOp5)
create_test_fp16_class(TestSumOp6)
create_test_fp16_class(TestSumOpExclusive1)
create_test_fp16_class(TestSumOpExclusive2)
create_test_fp16_class(TestSumOpExclusive3)
create_test_fp16_class(TestSumOpExclusive4)
create_test_fp16_class(TestSumOpExclusive5)
create_test_fp16_class(TestSumOpReverseExclusive)


def create_test_bf16_class(parent):
    @unittest.skipIf(
        not core.is_compiled_with_cuda()
        or not core.is_bfloat16_supported(core.CUDAPlace(0)),
        "core is not compiled with CUDA or not support bfloat16",
    )
    class TestCumsumBF16Op(parent):
        def init_dtype(self):
            self.dtype = np.uint16
            self.dtype_ = np.float32

        def if_enable_cinn(self):
            self.enable_cinn = False

        def test_check_output(self):
            place = paddle.CUDAPlace(0)
            self.check_output_with_place(place, check_prim=True, check_pir=True)

        def test_check_grad(self):
            place = paddle.CUDAPlace(0)
            self.check_grad_with_place(
                place,
                ["X"],
                "Out",
                check_prim=True,
                numeric_grad_delta=0.05,
                check_pir=True,
                check_prim_pir=True,
            )

    cls_name = "{}_{}".format(parent.__name__, "BF16")
    TestCumsumBF16Op.__name__ = cls_name
    globals()[cls_name] = TestCumsumBF16Op


create_test_bf16_class(TestSumOp1)
create_test_bf16_class(TestSumOp2)
create_test_bf16_class(TestSumOp3)
create_test_bf16_class(TestSumOp4)
create_test_bf16_class(TestSumOp5)
create_test_bf16_class(TestSumOp6)
create_test_bf16_class(TestSumOpExclusive1)
create_test_bf16_class(TestSumOpExclusive2)
create_test_bf16_class(TestSumOpExclusive3)
create_test_bf16_class(TestSumOpExclusive4)
create_test_bf16_class(TestSumOpExclusive5)
create_test_bf16_class(TestSumOpReverseExclusive)


class BadInputTest(unittest.TestCase):

    def test_error(self):
        paddle.enable_static()
        with paddle.static.program_guard(
            paddle.static.Program(), paddle.static.Program()
        ):

            def test_bad_x():
                data = [1, 2, 4]
                result = paddle.cumsum(data, axis=0)

            with self.assertRaises(TypeError):
                test_bad_x()
        paddle.disable_static()


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
        startup_prog = paddle.static.Program()
        with paddle.static.program_guard(main_prog, startup_prog):
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
            exe.run(startup_prog)
            static_out = exe.run(feed={'x': np_x}, fetch_list=[out])

            # run infer
            paddle.static.save_inference_model(self.save_path, [x], [out], exe)
            if use_pir_api():
                config = paddle_infer.Config(
                    self.save_path + '.json', self.save_path + '.pdiparams'
                )
                config.enable_new_ir()
                config.enable_new_executor()
                config.use_optimized_model(True)
            else:
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

    def test_static(self):
        paddle.enable_static()
        np_x = np.random.randn(9, 10, 11).astype('float32')
        with paddle.pir_utils.IrGuard():
            main_prog = paddle.static.Program()
            startup_prog = paddle.static.Program()
            with paddle.static.program_guard(main_prog, startup_prog):
                # run static
                x = paddle.static.data(
                    shape=np_x.shape, name='x', dtype=np_x.dtype
                )
                linear = paddle.nn.Linear(np_x.shape[-1], np_x.shape[-1])
                linear_out = linear(x)
                relu_out = paddle.nn.functional.relu(linear_out)
                axis = paddle.full([1], 2, dtype='int64')
                out = paddle.cumsum(relu_out, axis=axis)
                loss = paddle.mean(out)
                sgd = paddle.optimizer.SGD(learning_rate=0.0)
                sgd.minimize(paddle.mean(out))

                exe = paddle.static.Executor(self.place)
                exe.run(startup_prog)
                static_out = exe.run(feed={'x': np_x}, fetch_list=[out])
                # run infer
                paddle.static.save_inference_model(
                    self.save_path, [x], [out], exe, program=main_prog
                )

                exe = paddle.static.Executor(self.place)
                load_program, _, _ = paddle.static.load_inference_model(
                    self.save_path, exe
                )

                self.assertEqual(
                    len(load_program.global_block().ops),
                    9,
                )
                print(load_program)
                self.assertEqual(
                    load_program.global_block().ops[7].name(), 'pd_op.cumsum'
                )

                out = exe.run(
                    program=load_program,
                    feed={'x': np_x},
                    fetch_list=[],
                )
                np.testing.assert_allclose(static_out, out)


class TestCumSumOpFp16(unittest.TestCase):

    def test_fp16(self):
        if core.is_compiled_with_cuda():
            paddle.enable_static()
            x_np = np.random.random((100, 100)).astype('float16')
            with paddle.static.program_guard(paddle.static.Program()):
                x = paddle.static.data(
                    shape=[100, 100], name='x', dtype='float16'
                )
                y1 = paddle.cumsum(x)
                y2 = paddle.cumsum(x, axis=0)
                y3 = paddle.cumsum(x, axis=-1)
                y4 = paddle.cumsum(x, axis=-2)
                place = paddle.CUDAPlace(0)
                exe = paddle.static.Executor(place)
                exe.run(paddle.static.default_startup_program())
                out = exe.run(feed={'x': x_np}, fetch_list=[y1, y2, y3, y4])
            paddle.disable_static()


if __name__ == '__main__':
    unittest.main()
