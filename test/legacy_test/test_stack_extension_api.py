# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

import itertools
import sys
import unittest

import numpy as np

import paddle
from paddle.base import core

RTOL = 1e-5
ATOL = 1e-8
DTYPE_ALL = [
    'float16',
    'float32',
    'float64',
    'int32',
    'int64',
    'bfloat16',
]

DTYPE_COLUMN_STACK = DTYPE_ALL

PLACES = [('cpu', paddle.CPUPlace())] + (
    [('gpu', paddle.CUDAPlace(0))] if core.is_compiled_with_cuda() else []
)


def rearrange_data(*inputs):
    data = list(zip(*inputs))
    return [list(itertools.chain(*data[i])) for i in range(4)]


def generate_data(shape, count=1, dtype='int32'):
    """generate test data

    Args:
        shape(list of int): shape of inputs
        count(int): input count for each dim
        dtype(str): dtype

    Returns:
        a list of data like:
            [[data, dtype, shape, name], [data, dtype, shape, name] ... ]
    """
    return list(
        zip(
            *[
                [
                    # bfloat16 convert to uint16 for numpy
                    np.random.randint(0, 255, size=shape).astype(
                        dtype if dtype != 'bfloat16' else 'uint16'
                    ),
                    dtype,
                    shape,
                    f'{shape}d_{idx}_{dtype}',
                ]
                for idx in range(count)
            ]
        )
    )


class BaseTest(unittest.TestCase):
    """Test in each `PLACES` and in `static/dygraph`"""

    def _test_static_api(
        self,
        func_paddle,
        func_numpy,
        inputs: list,
        dtypes: list,
        shapes: list,
        names: list,
    ):
        """Test `static`, convert `Tensor` to `numpy array` before feed into graph"""
        paddle.enable_static()

        for device, place in PLACES:
            paddle.set_device(device)

            exe = paddle.static.Executor(place)
            new_scope = paddle.static.Scope()
            main_program = paddle.static.Program()
            with paddle.static.scope_guard(new_scope):
                with paddle.static.program_guard(main_program):
                    x = []
                    feed = {}
                    for i in range(len(inputs)):
                        input = inputs[i]
                        shape = shapes[i]
                        dtype = dtypes[i]
                        name = names[i]

                        _x = paddle.static.data(name, shape, dtype)
                        _x.stop_gradient = False
                        x.append(_x)

                        # the data feeded should NOT be a Tensor
                        feed[name] = input

                    out = func_paddle(x)
                    out.stop_gradient = False

                    y = out * 123

                    # not check old ir
                    if paddle.framework.in_pir_mode():
                        fetch_list = [out]
                        grads = paddle.autograd.ir_backward.grad(y, x)
                        fetch_list.append(grads)

                        exe = paddle.static.Executor(place)
                        res, *res_grad = exe.run(
                            feed=feed, fetch_list=fetch_list
                        )

                        # convert grad value to bool if dtype is bool
                        grad_value = 123.0 if dtypes[0] != 'bool' else True
                        np.testing.assert_allclose(
                            res_grad[0], np.ones(x[0].shape) * grad_value
                        )

                        out_ref = func_numpy(inputs)
                        for n, p in zip(out_ref, res):
                            np.testing.assert_allclose(
                                n, p, rtol=RTOL, atol=ATOL
                            )

    def _test_dygraph_api(
        self,
        func_paddle,
        func_numpy,
        inputs: list,
        dtypes: list,
        shapes: list,
        names: list,
    ):
        """Test `dygraph`, and check grads"""
        paddle.disable_static()

        for device, place in PLACES:
            paddle.set_device(device)

            out = func_paddle(
                [
                    paddle.to_tensor(inputs[i]).astype(dtypes[i])
                    for i in range(len(inputs))
                ]
            )
            out_ref = func_numpy(inputs)

            for n, p in zip(out_ref, out):
                np.testing.assert_allclose(n, p.numpy(), rtol=RTOL, atol=ATOL)

            # check grads
            if len(inputs) == 1:
                out = [out]

            for y in out:
                y.stop_gradient = False
                z = y * 123
                grads = paddle.grad(z, y)
                self.assertTrue(len(grads), 1)
                self.assertEqual(grads[0].dtype, y.dtype)
                self.assertEqual(grads[0].shape, y.shape)

    def _test_all(self, args, dtype=''):
        self._test_dygraph_api(self.func_paddle, self.func_numpy, *args)
        self._test_static_api(self.func_paddle, self.func_numpy, *args)


class BaseCases:
    def test_0d(self):
        self._test_all(generate_data([], count=1, dtype='float64'))

    def test_1d(self):
        self._test_all(generate_data([1], count=1, dtype='float64'))

    def test_2d(self):
        self._test_all(generate_data([1, 1], count=1, dtype='float64'))
        self._test_all(generate_data([3, 2], count=1, dtype='float64'))

    def test_3d(self):
        self._test_all(generate_data([1, 1, 1], count=1, dtype='float64'))
        self._test_all(generate_data([3, 4, 2], count=1, dtype='float64'))

    def test_4d(self):
        self._test_all(generate_data([1, 1, 1, 1], count=1, dtype='float64'))
        self._test_all(generate_data([3, 4, 2, 5], count=1, dtype='float64'))

    def test_0d_more(self):
        self._test_all(generate_data([], count=3, dtype='float64'))

    def test_1d_more(self):
        self._test_all(generate_data([1], count=3, dtype='float64'))
        self._test_all(generate_data([5], count=3, dtype='float64'))

    def test_2d_more(self):
        self._test_all(generate_data([1, 1], count=3, dtype='float64'))
        self._test_all(generate_data([3, 2], count=3, dtype='float64'))

    def test_3d_more(self):
        self._test_all(generate_data([1, 1, 1], count=3, dtype='float64'))
        self._test_all(generate_data([3, 4, 2], count=3, dtype='float64'))

    def test_4d_more(self):
        self._test_all(generate_data([1, 1, 1, 1], count=3, dtype='float64'))
        self._test_all(generate_data([3, 4, 2, 5], count=3, dtype='float64'))


class TestHStack(BaseTest, BaseCases):
    def setUp(self):
        self.func_paddle = paddle.hstack
        self.func_numpy = np.hstack

    def test_mix_ndim(self):
        d0 = generate_data([], count=1, dtype='float64')
        d1 = generate_data([2], count=1, dtype='float64')
        self._test_all(rearrange_data(d0, d1))

    def test_dtype(self):
        for dtype in DTYPE_ALL:
            if dtype == 'float16' and (
                not core.is_compiled_with_cuda()
                or (
                    not core.is_float16_supported(paddle.CUDAPlace(0))
                    or sys.platform == 'win32'
                )
            ):
                continue

            if dtype == 'bfloat16' and (
                not core.is_compiled_with_cuda()
                or (
                    not core.is_bfloat16_supported(paddle.CUDAPlace(0))
                    or sys.platform == 'win32'
                )
            ):
                continue

            self._test_all(
                generate_data([], count=1, dtype=dtype),
                dtype,
            )


class TestVStack(BaseTest, BaseCases):
    def setUp(self):
        self.func_paddle = paddle.vstack
        self.func_numpy = np.vstack

    def test_mix_ndim(self):
        d0 = generate_data([2], count=1, dtype='float64')
        d1 = generate_data([1, 2], count=1, dtype='float64')
        self._test_all(rearrange_data(d0, d1))

    def test_dtype(self):
        for dtype in DTYPE_ALL:
            if dtype == 'float16' and (
                not core.is_compiled_with_cuda()
                or (
                    not core.is_float16_supported(paddle.CUDAPlace(0))
                    or sys.platform == 'win32'
                )
            ):
                continue

            if dtype == 'bfloat16' and (
                not core.is_compiled_with_cuda()
                or (
                    not core.is_bfloat16_supported(paddle.CUDAPlace(0))
                    or sys.platform == 'win32'
                )
            ):
                continue

            self._test_all(
                generate_data([], count=1, dtype=dtype),
                dtype,
            )


class TestDStack(BaseTest, BaseCases):
    def setUp(self):
        self.func_paddle = paddle.dstack
        self.func_numpy = np.dstack

    def test_mix_ndim(self):
        d0 = generate_data([2], count=1, dtype='float64')
        d1 = generate_data([1, 2], count=1, dtype='float64')
        self._test_all(rearrange_data(d0, d1))

        d0 = generate_data([2], count=1, dtype='float64')
        d1 = generate_data([1, 2, 1], count=1, dtype='float64')
        self._test_all(rearrange_data(d0, d1))

    def test_dtype(self):
        for dtype in DTYPE_ALL:
            if dtype == 'float16' and (
                not core.is_compiled_with_cuda()
                or (
                    not core.is_float16_supported(paddle.CUDAPlace(0))
                    or sys.platform == 'win32'
                )
            ):
                continue

            if dtype == 'bfloat16' and (
                not core.is_compiled_with_cuda()
                or (
                    not core.is_bfloat16_supported(paddle.CUDAPlace(0))
                    or sys.platform == 'win32'
                )
            ):
                continue

            self._test_all(
                generate_data([], count=1, dtype=dtype),
                dtype,
            )


class TestColumnStack(BaseTest, BaseCases):
    def setUp(self):
        self.func_paddle = paddle.column_stack
        self.func_numpy = np.column_stack

    def test_mix_ndim(self):
        d0 = generate_data([2], count=1, dtype='float64')
        d1 = generate_data([2, 1], count=1, dtype='float64')
        self._test_all(rearrange_data(d0, d1))

    def test_dtype(self):
        for dtype in DTYPE_COLUMN_STACK:
            if dtype == 'float16' and (
                not core.is_compiled_with_cuda()
                or (
                    not core.is_float16_supported(paddle.CUDAPlace(0))
                    or sys.platform == 'win32'
                )
            ):
                continue

            if dtype == 'bfloat16' and (
                not core.is_compiled_with_cuda()
                or (
                    not core.is_bfloat16_supported(paddle.CUDAPlace(0))
                    or sys.platform == 'win32'
                )
            ):
                continue

            self._test_all(
                generate_data([], count=1, dtype=dtype),
                dtype=dtype,
            )


class TestRowStack(BaseTest, BaseCases):
    def setUp(self):
        self.func_paddle = paddle.row_stack
        self.func_numpy = np.vstack

    def test_mix_ndim(self):
        d0 = generate_data([2], count=1, dtype='float64')
        d1 = generate_data([1, 2], count=1, dtype='float64')
        self._test_all(rearrange_data(d0, d1))

    def test_dtype(self):
        for dtype in DTYPE_ALL:
            if dtype == 'float16' and (
                not core.is_compiled_with_cuda()
                or (
                    not core.is_float16_supported(paddle.CUDAPlace(0))
                    or sys.platform == 'win32'
                )
            ):
                continue

            if dtype == 'bfloat16' and (
                not core.is_compiled_with_cuda()
                or (
                    not core.is_bfloat16_supported(paddle.CUDAPlace(0))
                    or sys.platform == 'win32'
                )
            ):
                continue

            self._test_all(
                generate_data([], count=1, dtype=dtype),
                dtype,
            )


class ErrorCases:
    def test_mix_dtype(self):
        with self.assertRaises(ValueError):
            d0 = generate_data([2], count=1, dtype='float32')
            d1 = generate_data([2], count=1, dtype='float64')
            self._test_dygraph_api(
                self.func_paddle, self.func_numpy, *rearrange_data(d0, d1)
            )

        with self.assertRaises(TypeError):
            d0 = generate_data([2], count=1, dtype='float32')
            d1 = generate_data([2], count=1, dtype='float64')
            self._test_static_api(
                self.func_paddle, self.func_numpy, *rearrange_data(d0, d1)
            )

    def test_1d_2d(self):
        with self.assertRaises(ValueError):
            d0 = generate_data([2, 1], count=1, dtype='float64')
            d1 = generate_data([3], count=1, dtype='float64')
            self._test_all(rearrange_data(d0, d1))

        with self.assertRaises(ValueError):
            d0 = generate_data([1, 2], count=1, dtype='float64')
            d1 = generate_data([3], count=1, dtype='float64')
            self._test_all(rearrange_data(d0, d1))

    def test_1d_3d(self):
        with self.assertRaises(ValueError):
            d0 = generate_data([2, 3, 1], count=1, dtype='float64')
            d1 = generate_data([3], count=1, dtype='float64')
            self._test_all(rearrange_data(d0, d1))

        with self.assertRaises(ValueError):
            d0 = generate_data([1, 1, 1], count=1, dtype='float64')
            d1 = generate_data([2], count=1, dtype='float64')
            self._test_all(rearrange_data(d0, d1))

    def test_2d_3d(self):
        with self.assertRaises(ValueError):
            d0 = generate_data([2, 3, 1], count=1, dtype='float64')
            d1 = generate_data([1, 3], count=1, dtype='float64')
            self._test_all(rearrange_data(d0, d1))

        with self.assertRaises(ValueError):
            d0 = generate_data([1, 1, 1], count=1, dtype='float64')
            d1 = generate_data([1, 2], count=1, dtype='float64')
            self._test_all(rearrange_data(d0, d1))


class ErrorCases0d1d(ErrorCases):
    """hstack works fine with 0d & 1d"""

    def test_vstack_0d_1d(self):
        with self.assertRaises(ValueError):
            d0 = generate_data([], count=1, dtype='float64')
            d1 = generate_data([2], count=1, dtype='float64')
            self._test_all(rearrange_data(d0, d1))


class TestErrorHStack(BaseTest, ErrorCases):
    def setUp(self):
        self.func_paddle = paddle.hstack
        self.func_numpy = np.hstack


class TestErrorVStack(BaseTest, ErrorCases0d1d):
    def setUp(self):
        self.func_paddle = paddle.vstack
        self.func_numpy = np.vstack


class TestErrorDStack(BaseTest, ErrorCases0d1d):
    def setUp(self):
        self.func_paddle = paddle.dstack
        self.func_numpy = np.dstack


class TestErrorColumnStack(BaseTest, ErrorCases0d1d):
    def setUp(self):
        self.func_paddle = paddle.column_stack
        self.func_numpy = np.column_stack


class TestErrorRowStack(BaseTest, ErrorCases0d1d):
    def setUp(self):
        self.func_paddle = paddle.row_stack
        self.func_numpy = np.vstack


if __name__ == '__main__':
    unittest.main()
