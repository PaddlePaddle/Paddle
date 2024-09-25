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

from __future__ import annotations

import itertools
import unittest

import numpy as np
from op_test import OpTest, convert_float_to_uint16, convert_uint16_to_float

import paddle
from paddle import base
from paddle.base import core


def np_naive_logcumsumexp(x: np.ndarray, axis: int | None = None):
    return np.log(np.cumsum(np.exp(x), axis=axis))


def np_logcumsumexp(
    x: np.ndarray,
    axis: int | None = None,
    flatten: bool | None = None,
    reverse: bool = False,
    exclusive: bool = False,
):
    # `flatten` aligns with c++ op
    if flatten:
        assert axis in [0, None]
        axis = None

    x = np.copy(x)

    if axis is None:
        x = x.flatten()
        axis = 0

    if reverse:
        x = np.flip(x, axis)

    dimensions = [range(dim) for dim in x.shape[:axis]]

    if exclusive:
        x = np.roll(x, 1, axis)
        for prefix_dim in itertools.product(*dimensions):
            x[prefix_dim][0] = np.finfo(x.dtype).min

    for prefix_dim in itertools.product(*dimensions):
        arr = x[prefix_dim]
        for dim in range(1, arr.shape[0]):
            arr[dim] = np.logaddexp(arr[dim - 1], arr[dim])

    if reverse:
        x = np.flip(x, axis)

    return x


def np_logcumsumexp_grad(
    x: np.ndarray,
    dout: np.ndarray,
    axis: int | None = None,
    flatten: bool | None = None,
    reverse: bool = False,
    exclusive: bool = False,
):
    out = np_logcumsumexp(x, axis, flatten, reverse, exclusive)
    log_grad_positive = np.where(dout > 0, np.log(dout), np.finfo(x.dtype).min)
    log_grad_negative = np.where(dout < 0, np.log(-dout), np.finfo(x.dtype).min)

    output_pos = np.exp(
        np_logcumsumexp(
            log_grad_positive - out,
            axis=axis,
            flatten=flatten,
            reverse=not reverse,
            exclusive=exclusive,
        ).reshape(x.shape)
        + x
    )
    output_neg = np.exp(
        np_logcumsumexp(
            log_grad_negative - out,
            axis=axis,
            flatten=flatten,
            reverse=not reverse,
            exclusive=exclusive,
        ).reshape(x.shape)
        + x
    )

    return output_pos - output_neg


class TestLogcumsumexp(unittest.TestCase):
    def run_imperative(self):
        data_np = np.arange(12, dtype=np.float32).reshape(3, 4)
        data = paddle.to_tensor(data_np)

        y = paddle.logcumsumexp(data)
        z = np_logcumsumexp(data_np)
        np.testing.assert_allclose(z, y.numpy(), rtol=1e-05)

        y = paddle.logcumsumexp(data, axis=0)
        z = np_logcumsumexp(data_np, axis=0)
        np.testing.assert_allclose(z, y.numpy(), rtol=1e-05)

        y = paddle.logcumsumexp(data, axis=-1)
        z = np_logcumsumexp(data_np, axis=-1)
        np.testing.assert_allclose(z, y.numpy(), rtol=1e-05)

        y = paddle.logcumsumexp(data, dtype='float32')
        self.assertTrue(y.dtype == paddle.float32)

        y = paddle.logcumsumexp(data, axis=-2)
        z = np_logcumsumexp(data_np, axis=-2)
        np.testing.assert_allclose(z, y.numpy(), rtol=1e-05)

        with self.assertRaises(IndexError):
            y = paddle.logcumsumexp(data, axis=-3)

        with self.assertRaises(IndexError):
            y = paddle.logcumsumexp(data, axis=2)

        data_np = np.arange(10000, 10024, dtype=np.float32)
        data = paddle.to_tensor(data_np)
        y = paddle.logcumsumexp(data)
        z = np_naive_logcumsumexp(data_np)
        # check that naive algorithm overflows
        self.assertTrue(all(z == np.inf))
        z = np_logcumsumexp(data_np)
        # check that our algorithm doesn't overflow
        self.assertTrue(all(z != np.inf))
        np.testing.assert_allclose(z, y.numpy(), rtol=1e-05)

    def run_static(self, use_gpu=False):
        main = paddle.static.Program()
        startup = paddle.static.Program()
        with paddle.static.program_guard(main, startup):
            data_np = np.random.random((5, 4)).astype(np.float32)
            x = paddle.static.data('X', [5, 4])
            y = paddle.logcumsumexp(x)
            y2 = paddle.logcumsumexp(x, axis=0)
            y3 = paddle.logcumsumexp(x, axis=-1)
            y4 = paddle.logcumsumexp(x, dtype='float64')
            y5 = paddle.logcumsumexp(x, axis=-2)

            place = base.CUDAPlace(0) if use_gpu else base.CPUPlace()
            exe = base.Executor(place)
            out = exe.run(
                main,
                feed={'X': data_np},
                fetch_list=[
                    y,
                    y2,
                    y3,
                    y4,
                    y5,
                ],
            )

            z = np_logcumsumexp(data_np)
            np.testing.assert_allclose(z, out[0], rtol=1e-05)
            z = np_logcumsumexp(data_np, axis=0)
            np.testing.assert_allclose(z, out[1], rtol=1e-05)
            z = np_logcumsumexp(data_np, axis=-1)
            np.testing.assert_allclose(z, out[2], rtol=1e-05)
            self.assertTrue(out[3].dtype == np.float64)
            z = np_logcumsumexp(data_np, axis=-2)
            np.testing.assert_allclose(z, out[4], rtol=1e-05)

    def test_cpu(self):
        paddle.disable_static(paddle.base.CPUPlace())
        self.run_imperative()
        paddle.enable_static()

        self.run_static()

    def test_gpu(self):
        if not base.core.is_compiled_with_cuda():
            return
        paddle.disable_static(paddle.base.CUDAPlace(0))
        self.run_imperative()
        paddle.enable_static()

        self.run_static(use_gpu=True)

    def test_name(self):
        paddle.enable_static()
        with paddle.pir_utils.OldIrGuard():
            with base.program_guard(base.Program()):
                x = paddle.static.data('x', [3, 4])
                y = paddle.logcumsumexp(x, name='out')
                self.assertTrue('out' in y.name)
        paddle.disable_static()

    def test_type_error(self):
        main = paddle.static.Program()
        startup = paddle.static.Program()
        with paddle.static.program_guard(main, startup):
            with self.assertRaises(TypeError):
                data_np = np.random.random((100, 100), dtype=np.int32)
                x = paddle.static.data('X', [100, 100], dtype='int32')
                y = paddle.logcumsumexp(x)

                place = base.CUDAPlace(0)
                exe = base.Executor(place)
                out = exe.run(main, feed={'X': data_np}, fetch_list=[y])


def logcumsumexp_wrapper(
    x, axis=-1, flatten=False, exclusive=False, reverse=False
):
    return paddle._C_ops.logcumsumexp(x, axis, flatten, exclusive, reverse)


class BaseTestCases:
    class BaseOpTest(OpTest):
        def setUp(self):
            self.op_type = "logcumsumexp"
            self.prim_op_type = "prim"
            self.python_api = logcumsumexp_wrapper
            self.public_python_api = logcumsumexp_wrapper
            input, attrs = self.input_and_attrs()
            self.inputs = {'X': input}
            self.attrs = attrs
            if "dtype" in attrs:
                del attrs["dtype"]
            self.outputs = {'Out': np_logcumsumexp(input, **attrs)}

        def test_check_output(self):
            self.check_output(check_pir=True)

        def test_check_grad(self):
            self.check_grad(
                ['X'],
                'Out',
                user_defined_grads=[
                    np_logcumsumexp_grad(
                        self.inputs['X'],
                        1 / self.inputs['X'].size,
                        **self.attrs,
                    )
                ],
                check_pir=True,
                check_prim_pir=True,
            )

        def input_and_attrs(self):
            raise NotImplementedError


class TestLogcumsumexpOp1(BaseTestCases.BaseOpTest):
    def input_and_attrs(self):
        return np.arange(100, dtype=np.float64).reshape(10, 10), {
            'axis': 0,
            'flatten': True,
            'reverse': True,
        }


class TestLogcumsumexpOp2(BaseTestCases.BaseOpTest):
    def input_and_attrs(self):
        return np.arange(100, dtype=np.float64).reshape(10, 10), {
            'axis': 1,
            'reverse': True,
        }


class TestLogcumsumexpOp3(BaseTestCases.BaseOpTest):
    def input_and_attrs(self):
        return np.arange(100, dtype=np.float64).reshape(10, 10), {'axis': 1}


class TestLogcumsumexpOp4(BaseTestCases.BaseOpTest):
    def input_and_attrs(self):
        return np.arange(100, dtype=np.float64).reshape(10, 10), {
            'axis': 0,
            'flatten': True,
            'reverse': True,
            'exclusive': True,
        }


class TestLogcumsumexpFP16(unittest.TestCase):
    def check_main(self, x_np, dtype, axis=None):
        paddle.disable_static()
        x = paddle.to_tensor(x_np.astype(dtype))
        x.stop_gradient = False
        y = paddle.logcumsumexp(x, dtype=dtype, axis=axis)
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

        y_np_1, x_g_np_1 = self.check_main(x_np, 'float16', axis=1)
        y_np_2, x_g_np_2 = self.check_main(x_np, 'float32', axis=1)
        np.testing.assert_allclose(y_np_1, y_np_2, rtol=1e-03)
        np.testing.assert_allclose(x_g_np_1, x_g_np_2, rtol=2e-03)


@unittest.skipIf(
    not core.is_compiled_with_cuda()
    or not core.is_bfloat16_supported(core.CUDAPlace(0)),
    "core is not compiled with CUDA and not support the bfloat16",
)
class TestLogcumsumexpBF16Op(OpTest):
    def setUp(self):
        self.op_type = 'logcumsumexp'
        self.prim_op_type = 'prim'
        self.dtype = np.uint16
        self.python_api = logcumsumexp_wrapper
        self.public_python_api = logcumsumexp_wrapper
        x = np.arange(100, dtype=np.float64).reshape(10, 10)
        output = np_logcumsumexp(x)
        self.inputs = {'X': convert_float_to_uint16(x)}
        self.outputs = {'Out': convert_float_to_uint16(output)}

    def test_check_output(self):
        place = core.CUDAPlace(0)
        place = core.CUDAPlace(0)
        self.check_output_with_place_customized(
            checker=self.verify_output, place=place, check_pir=True
        )

    def verify_output(self, outs):
        outs = convert_uint16_to_float(outs)
        self.assertEqual(outs[0].shape, (10, 10))
        hist, _ = np.histogram(outs[0], range=(-3, 5))
        hist = hist.astype("float64")
        hist /= float(outs[0].size)

        x = np.arange(100, dtype=np.float64).reshape(10, 10)
        data = np_logcumsumexp(x)
        hist2, _ = np.histogram(data, range=(-3, 5))
        hist2 = hist2.astype("float64")
        hist2 /= float(outs[0].size)
        np.testing.assert_allclose(hist, hist2, rtol=0.3)

    def test_check_grad(self):
        place = core.CUDAPlace(0)
        self.check_grad_with_place(
            place,
            ['X'],
            'Out',
            numeric_grad_delta=0.5,
            max_relative_error=0.5,
            check_pir=True,
            check_prim_pir=True,
        )


if __name__ == '__main__':
    unittest.main()
