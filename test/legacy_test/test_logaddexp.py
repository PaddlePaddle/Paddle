#  Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import unittest

import numpy as np
from op_test import (
    OpTest,
    convert_float_to_uint16,
    convert_uint16_to_float,
    get_device_place,
    is_custom_device,
)

import paddle
from paddle.base import core


def ref_logaddexp_old(x, y):
    y = np.broadcast_to(y, x.shape)
    out = np.log1p(np.exp(-np.absolute(x - y))) + np.maximum(x, y)
    return out


def ref_logaddexp(x, y):
    return np.logaddexp(x, y)


class TestLogsumexpAPI(unittest.TestCase):
    def setUp(self):
        self.place = get_device_place()

    def api_case(self):
        self.x = np.random.uniform(-1, 1, self.xshape).astype(self.dtype)
        self.y = np.random.uniform(-1, 1, self.yshape).astype(self.dtype)
        out_ref = ref_logaddexp(self.x, self.y)

        # paddle.disable_static(self.place)
        x = paddle.to_tensor(self.x)
        y = paddle.to_tensor(self.y)
        out = paddle.logaddexp(x, y)
        np.testing.assert_allclose(out.numpy(), out_ref, atol=1e-06)

    def test_api(self):
        self.xshape = [1, 2, 3, 4]
        self.yshape = [1, 2, 3, 4]
        self.dtype = np.float64
        self.api_case()

    def test_api_broadcast(self):
        self.xshape = [1, 2, 3, 4]
        self.yshape = [1, 2, 3, 1]
        self.dtype = np.float32
        self.api_case()

    def test_api_bigdata(self):
        self.xshape = [10, 200, 300]
        self.yshape = [10, 200, 300]
        self.dtype = np.float32
        self.api_case()

    def test_api_int32(self):
        self.xshape = [10, 200, 300]
        self.yshape = [10, 200, 300]
        self.dtype = np.int32
        self.api_case()

    def test_api_int64(self):
        self.xshape = [10, 200, 300]
        self.yshape = [10, 200, 300]
        self.dtype = np.int64
        self.api_case()


class TestLogsumexpAPI_ZeroSize(unittest.TestCase):
    def setUp(self):
        self.place = get_device_place()

    def api_case(self):
        self.x = np.random.uniform(-1, 1, self.xshape).astype(self.dtype)
        self.y = np.random.uniform(-1, 1, self.yshape).astype(self.dtype)
        out_ref = ref_logaddexp(self.x, self.y)

        paddle.disable_static(self.place)
        x = paddle.to_tensor(self.x)
        y = paddle.to_tensor(self.y)
        x.stop_gradient = False
        y.stop_gradient = False
        out = paddle.logaddexp(x, y)
        np.testing.assert_allclose(out.numpy(), out_ref, atol=1e-06)

        loss = paddle.sum(out)
        loss.backward()
        np.testing.assert_allclose(x.grad.shape, x.shape)

    def test_api(self):
        self.xshape = [1, 2, 3, 0]
        self.yshape = [1, 2, 3, 1]
        self.dtype = np.float32
        self.api_case()


class TestLogaddexpOut(unittest.TestCase):
    def setUp(self):
        self.place = get_device_place()
        self.x = np.random.uniform(-3, 3, [2, 3]).astype(np.float32)
        self.y = np.random.uniform(-3, 3, [1, 3]).astype(np.float32)
        self.out_ref = ref_logaddexp(self.x, self.y)

    def test_dygraph_out(self):
        paddle.disable_static(self.place)
        buf = paddle.zeros([2, 3], 'float32')
        res = paddle.logaddexp(
            paddle.to_tensor(self.x), paddle.to_tensor(self.y), out=buf
        )
        np.testing.assert_allclose(res.numpy(), self.out_ref, rtol=1e-6)
        np.testing.assert_allclose(buf.numpy(), self.out_ref, rtol=1e-6)

    def test_pir_out(self):
        paddle.enable_static()
        try:
            with paddle.pir_utils.IrGuard():
                main = paddle.static.Program()
                with paddle.static.program_guard(main):
                    x = paddle.static.data('x', [2, 3], 'float32')
                    y = paddle.static.data('y', [1, 3], 'float32')
                    buf = paddle.zeros([2, 3], 'float32')
                    res = paddle.logaddexp(x, y, out=buf)
                exe = paddle.static.Executor(self.place)
                res_v, buf_v = exe.run(
                    main,
                    feed={'x': self.x, 'y': self.y},
                    fetch_list=[res, buf],
                )
        finally:
            paddle.disable_static()
        np.testing.assert_allclose(res_v, self.out_ref, rtol=1e-6)
        np.testing.assert_allclose(buf_v, self.out_ref, rtol=1e-6)


# Shapes chosen to hit both branches of the backward kernel: writing the grad
# in place (no reduce) and reducing over leading / size-1 broadcast dims.
class TestLogaddexpOp(OpTest):
    def setUp(self):
        self.op_type = "logaddexp"
        self.python_api = paddle.logaddexp
        self.public_python_api = paddle.logaddexp
        self.init_shape()
        self.init_dtype()
        x = np.random.uniform(-1, 1, self.x_shape).astype(self.np_dtype)
        y = np.random.uniform(-1, 1, self.y_shape).astype(self.np_dtype)
        self.inputs = {'x': x, 'y': y}
        self.outputs = {'out': ref_logaddexp(x, y)}

    def init_shape(self):
        self.x_shape = [13, 17]
        self.y_shape = [13, 17]

    def init_dtype(self):
        self.dtype = self.np_dtype = np.float64

    def test_check_output(self):
        self.check_output(check_pir=True)

    def test_check_grad(self):
        self.check_grad(['x', 'y'], 'out', check_pir=True)


# OpTest requires every input to have at least 100 elements.
class TestLogaddexpOpYBroadcast(TestLogaddexpOp):
    def init_shape(self):
        self.x_shape = [2, 3, 100]
        self.y_shape = [1, 1, 100]


class TestLogaddexpOpRankDiff(TestLogaddexpOp):
    def init_shape(self):
        self.x_shape = [100]
        self.y_shape = [2, 3, 100]


class TestLogaddexpOpBothBroadcast(TestLogaddexpOp):
    def init_shape(self):
        self.x_shape = [2, 1, 100]
        self.y_shape = [1, 3, 100]


@unittest.skipIf(
    not (core.is_compiled_with_cuda() or is_custom_device()),
    "float16 logaddexp is only registered on GPU",
)
class TestLogaddexpOpFP16(TestLogaddexpOpYBroadcast):
    def init_dtype(self):
        self.dtype = self.np_dtype = np.float16

    def test_check_output(self):
        self.check_output_with_place(get_device_place(), check_pir=True)

    def test_check_grad(self):
        self.check_grad_with_place(
            get_device_place(), ['x', 'y'], 'out', check_pir=True
        )


@unittest.skipIf(
    not (core.is_compiled_with_cuda() or is_custom_device())
    or not core.is_bfloat16_supported(get_device_place()),
    "bfloat16 logaddexp is only registered on GPU",
)
class TestLogaddexpOpBF16(TestLogaddexpOpFP16):
    def setUp(self):
        self.op_type = "logaddexp"
        self.python_api = paddle.logaddexp
        self.public_python_api = paddle.logaddexp
        self.dtype = np.uint16
        self.init_shape()
        x = np.random.uniform(-1, 1, self.x_shape).astype(np.float32)
        y = np.random.uniform(-1, 1, self.y_shape).astype(np.float32)
        x_bf16 = convert_float_to_uint16(x)
        y_bf16 = convert_float_to_uint16(y)
        out = ref_logaddexp(
            convert_uint16_to_float(x_bf16), convert_uint16_to_float(y_bf16)
        )
        self.inputs = {'x': x_bf16, 'y': y_bf16}
        self.outputs = {'out': convert_float_to_uint16(out)}


def ref_logaddexp_grad(self_, other, grad):
    # torch's formula grad / (1 + exp(other - self)), each step in self's
    # dtype, before reducing to self's shape.
    with np.errstate(over='ignore', invalid='ignore'):
        one = self_.dtype.type(1)
        full = grad / (one + np.exp(other - self_))
    lead = full.ndim - self_.ndim
    axes = tuple(range(lead)) + tuple(
        lead + i
        for i, n in enumerate(self_.shape)
        if n == 1 and full.shape[lead + i] != 1
    )
    return full.sum(axis=axes, keepdims=True).reshape(self_.shape)


def get_places():
    places = [paddle.CPUPlace()]
    if core.is_compiled_with_cuda() or is_custom_device():
        places.append(get_device_place())
    return places


class TestLogaddexpSpecialValues(unittest.TestCase):
    # Each column is one (x, y) pair: equal infinities (fwd special-cased,
    # bwd NaN like torch), opposite infinities, x - y overflowing to inf,
    # NaN propagation and signed zeros.
    def get_inputs(self, dtype):
        big = np.finfo(dtype).max
        inf, nan = np.inf, np.nan
        x = np.array([inf, -inf, inf, -inf, big, nan, 0.0, 1.0], dtype)
        y = np.array([inf, -inf, -inf, inf, -big, 1.0, -0.0, nan], dtype)
        return x, y

    def check(self, place, dtype):
        x_np, y_np = self.get_inputs(dtype)
        g_np = np.ones_like(x_np)
        with np.errstate(over='ignore', invalid='ignore'):
            out_ref = np.logaddexp(x_np, y_np)
        paddle.disable_static(place)
        x = paddle.to_tensor(x_np, stop_gradient=False)
        y = paddle.to_tensor(y_np, stop_gradient=False)
        out = paddle.logaddexp(x, y)
        dx, dy = paddle.grad([out], [x, y])
        # assert_allclose requires infinities and NaNs to match exactly.
        np.testing.assert_allclose(out.numpy(), out_ref, rtol=1e-6)
        np.testing.assert_allclose(
            dx.numpy(), ref_logaddexp_grad(x_np, y_np, g_np)
        )
        np.testing.assert_allclose(
            dy.numpy(), ref_logaddexp_grad(y_np, x_np, g_np)
        )

    def test_special_values(self):
        for place in get_places():
            for dtype in (np.float32, np.float64):
                with self.subTest(place=str(place), dtype=dtype):
                    self.check(place, dtype)


class TestLogaddexpSingleSideGrad(unittest.TestCase):
    # Only one operand requires grad, so the other grad output is skipped.
    def test_single_side(self):
        x_np = np.random.uniform(-1, 1, [2, 3, 8]).astype(np.float32)
        y_np = np.random.uniform(-1, 1, [1, 3, 1]).astype(np.float32)
        g_np = np.random.uniform(-1, 1, [2, 3, 8]).astype(np.float32)
        for place in get_places():
            paddle.disable_static(place)
            for need_x in (True, False):
                with self.subTest(place=str(place), need_x=need_x):
                    x = paddle.to_tensor(x_np, stop_gradient=not need_x)
                    y = paddle.to_tensor(y_np, stop_gradient=need_x)
                    out = paddle.logaddexp(x, y)
                    (grad,) = paddle.grad(
                        [out], [x if need_x else y], [paddle.to_tensor(g_np)]
                    )
                    ref = (
                        ref_logaddexp_grad(x_np, y_np, g_np)
                        if need_x
                        else ref_logaddexp_grad(y_np, x_np, g_np)
                    )
                    np.testing.assert_allclose(
                        grad.numpy(), ref, rtol=1e-5, atol=1e-6
                    )


class TestLogaddexpZeroSizeGrad(unittest.TestCase):
    def check(self, place, x_shape, y_shape):
        paddle.disable_static(place)
        x = paddle.zeros(x_shape, 'float32')
        y = paddle.zeros(y_shape, 'float32')
        x.stop_gradient = False
        y.stop_gradient = False
        out = paddle.logaddexp(x, y)
        dx, dy = paddle.grad([out], [x, y])
        self.assertEqual(dx.shape, x_shape)
        self.assertEqual(dy.shape, y_shape)
        np.testing.assert_array_equal(dx.numpy(), np.zeros(x_shape))
        np.testing.assert_array_equal(dy.numpy(), np.zeros(y_shape))

    def test_zero_size(self):
        for place in get_places():
            with self.subTest(place=str(place)):
                self.check(place, [0, 64], [0, 64])
                # The non-empty operand broadcast over an empty output gets
                # an all-zero grad.
                self.check(place, [0, 64], [1, 64])
                self.check(place, [1, 64], [0, 64])


class TestLogaddexpTypePromotion(unittest.TestCase):
    # (x dtype, x shape, y dtype, y shape, expected out dtype); expectations
    # match torch.logaddexp except int32/int64, which torch rejects.
    cases = [
        ('int64', [3], 'float16', [3], 'float16'),
        ('bool', [3], 'float32', [3], 'float32'),
        ('int64', [3], 'float32', [], 'float32'),
        ('int64', [], 'float16', [3], 'float16'),
        ('float16', [3], 'float32', [], 'float16'),
        ('bfloat16', [3], 'float16', [], 'bfloat16'),
        ('float32', [3], 'float64', [], 'float32'),
        ('float16', [], 'float32', [], 'float32'),
        ('float16', [3], 'float32', [3], 'float32'),
        ('int32', [3], 'int64', [3], 'float32'),
    ]

    def setUp(self):
        self.place = get_device_place()
        # fp16/bf16 logaddexp kernels are only registered on GPU. Dygraph runs
        # the kernel, and the PIR api checks inputs against the registered
        # kernel dtypes (CheckValueDataType), so both reject them on CPU-only
        # builds.
        self.low_precision_ok = (
            core.is_compiled_with_cuda() or is_custom_device()
        )

    def supported_cases(self, cases):
        return [
            c
            for c in cases
            if self.low_precision_ok or c[-1] not in ('float16', 'bfloat16')
        ]

    def make(self, dtype, shape):
        value = np.random.uniform(0, 2, shape)
        if dtype == 'bfloat16':
            return paddle.to_tensor(value.astype('float32')).astype(dtype)
        return paddle.to_tensor(value.astype(dtype))

    def test_c_ops_float_promotion(self):
        # Float/0-D promotion is done by the framework, so the raw _C_ops
        # entry agrees with paddle.logaddexp. Integral operands are still cast
        # in python, so only float-float cases apply here.
        floats = ('float16', 'bfloat16', 'float32', 'float64')
        float_cases = [
            c for c in self.cases if c[0] in floats and c[2] in floats
        ]
        paddle.disable_static(self.place)
        for x_dt, x_shape, y_dt, y_shape, expected in self.supported_cases(
            float_cases
        ):
            x = self.make(x_dt, x_shape)
            y = self.make(y_dt, y_shape)
            for a, b in ((x, y), (y, x)):
                with self.subTest(x=a.dtype, y=b.dtype):
                    out = paddle._C_ops.logaddexp(a, b)
                    self.assertEqual(out.dtype, getattr(paddle, expected))

    def test_dygraph(self):
        paddle.disable_static(self.place)
        for x_dt, x_shape, y_dt, y_shape, expected in self.supported_cases(
            self.cases
        ):
            x = self.make(x_dt, x_shape)
            y = self.make(y_dt, y_shape)
            for a, b in ((x, y), (y, x)):
                with self.subTest(x=a.dtype, y=b.dtype):
                    out = paddle.logaddexp(a, b)
                    self.assertEqual(out.dtype, getattr(paddle, expected))

    def test_pir(self):
        paddle.enable_static()
        try:
            with paddle.pir_utils.IrGuard():
                for (
                    x_dt,
                    x_shape,
                    y_dt,
                    y_shape,
                    expected,
                ) in self.supported_cases(self.cases):
                    main = paddle.static.Program()
                    with paddle.static.program_guard(main):
                        x = paddle.static.data('x', x_shape, x_dt)
                        y = paddle.static.data('y', y_shape, y_dt)
                        for a, b in ((x, y), (y, x)):
                            with self.subTest(x=a.dtype, y=b.dtype):
                                out = paddle.logaddexp(a, b)
                                self.assertEqual(
                                    out.dtype, getattr(paddle, expected)
                                )
        finally:
            paddle.disable_static()


if __name__ == '__main__':
    unittest.main()
