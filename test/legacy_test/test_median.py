#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

import copy
import unittest

import numpy as np

import paddle
from paddle.base import core

DELTA = 1e-6


def np_medain_min(data, keepdims=False):
    shape = data.shape
    data_flat = data.flatten()
    data_cnt = len(data_flat)

    if data.dtype != 'int32' and data.dtype != 'int64':
        data_flat[np.isnan(data_flat)] = np.inf
    data_sort = np.sort(data_flat)
    if data.dtype != 'int32' and data.dtype != 'int64':
        data_sort[np.isinf(data_sort)] = np.nan

    if data_cnt % 2:
        is_odd = False
    else:
        is_odd = True

    i = int(data_cnt / 2)
    if is_odd:
        np_res = min(data_sort[i - 1], data_sort[i])
    else:
        np_res = data_sort[i]
    if keepdims:
        new_shape = [1] * len(shape)
        np_res = np_res.reshape(new_shape)
    return np_res + np.sum(np.isnan(data).astype(data.dtype) * data)


def np_median_min_axis(data, axis=None, keepdims=False):
    data = copy.deepcopy(data)
    if axis is None:
        return np_medain_min(data, keepdims)

    axis = axis + len(data.shape) if axis < 0 else axis
    trans_shape = []
    reshape = []
    for i in range(len(data.shape)):
        if i != axis:
            trans_shape.append(i)
            reshape.append(data.shape[i])
    trans_shape.append(axis)
    last_shape = data.shape[axis]
    reshape.append(last_shape)

    data_flat = np.transpose(data, trans_shape)

    data_flat = np.reshape(data_flat, (-1, reshape[-1]))

    data_cnt = np.full(
        shape=data_flat.shape[:-1], fill_value=data_flat.shape[-1]
    )

    if data.dtype != 'int32' and data.dtype != 'int64':
        data_flat[np.isnan(data_flat)] = np.inf
    data_sort = np.sort(data_flat, axis=-1)
    if data.dtype != 'int32' and data.dtype != 'int64':
        data_sort[np.isinf(data_sort)] = np.nan

    is_odd = data_cnt % 2

    np_res = np.zeros(len(is_odd), dtype=data.dtype)

    for j in range(len(is_odd)):
        if data_cnt[j] == 0:
            np_res[j] = np.nan
            continue

        i = int(data_cnt[j] / 2)
        if is_odd[j]:
            np_res[j] = data_sort[j, i]
        else:
            np_res[j] = min(data_sort[j, i - 1], data_sort[j, i])

    if keepdims:
        shape = list(data.shape)
        shape[axis] = 1
        np_res = np.reshape(np_res, shape)
    else:
        np_res = np.reshape(np_res, reshape[:-1])
    return np_res + np.sum(
        np.isnan(data).astype(data.dtype) * data, axis=axis, keepdims=keepdims
    )


class TestMedianAvg(unittest.TestCase):
    def check_numpy_res(self, np1, np2):
        self.assertEqual(np1.shape, np2.shape)
        np1_isnan = np.isnan(np1)
        np2_isnan = np.isnan(np2)
        nan_mismatch = np.sum(
            (np1_isnan.astype('int32') - np2_isnan.astype('int32'))
            * (np1_isnan.astype('int32') - np2_isnan.astype('int32'))
        )
        self.assertEqual(nan_mismatch, 0)
        np1 = np.where(np.isnan(np1), 0.0, np1)
        np2 = np.where(np.isnan(np2), 0.0, np2)
        mismatch = np.sum((np1 - np2) * (np1 - np2))
        self.assertAlmostEqual(mismatch, 0, delta=DELTA)

    def static_single_test_median(self, lis_test):
        paddle.enable_static()
        x, axis, keepdims = lis_test
        res_np = np.median(x, axis=axis, keepdims=keepdims)
        main_program = paddle.static.Program()
        startup_program = paddle.static.Program()
        exe = paddle.static.Executor()
        with paddle.static.program_guard(main_program, startup_program):
            x_in = paddle.static.data(shape=x.shape, dtype=x.dtype, name='x')
            y = paddle.median(x_in, axis, keepdims)
            [res_pd] = exe.run(feed={'x': x}, fetch_list=[y])
            self.check_numpy_res(res_pd, res_np)
        paddle.disable_static()

    def dygraph_single_test_median(self, lis_test):
        x, axis, keepdims = lis_test
        res_np = np.median(x, axis=axis, keepdims=keepdims)
        res_pd = paddle.median(paddle.to_tensor(x), axis, keepdims)
        self.check_numpy_res(res_pd.numpy(False), res_np)

    def test_median_static(self):
        h = 3
        w = 4
        l = 2
        x = np.arange(h * w * l).reshape([h, w, l])
        lis_tests = [
            [x.astype(dtype), axis, keepdims]
            for axis in [-1, 0, 1, 2, None]
            for keepdims in [False, True]
            for dtype in ['float32', 'float64', 'int32', 'int64']
        ]
        for lis_test in lis_tests:
            self.static_single_test_median(lis_test)

    def test_median_dygraph(self):
        paddle.disable_static()
        h = 3
        w = 4
        l = 2
        x = np.arange(h * w * l).reshape([h, w, l])
        lis_tests = [
            [x.astype(dtype), axis, keepdims]
            for axis in [-1, 0, 1, 2, None]
            for keepdims in [False, True]
            for dtype in ['float32', 'float64', 'int32', 'int64']
        ]
        for lis_test in lis_tests:
            self.dygraph_single_test_median(lis_test)

    def test_median_exception(self):
        paddle.disable_static()
        x = [1, 2, 3, 4]
        self.assertRaises(TypeError, paddle.median, x)
        x = paddle.arange(12).reshape([3, 4])
        self.assertRaises(ValueError, paddle.median, x, 1.0)
        self.assertRaises(ValueError, paddle.median, x, 2)
        self.assertRaises(ValueError, paddle.median, x, 2, False, 'max')
        self.assertRaises(ValueError, paddle.median, paddle.to_tensor([]))

    def test_nan(self):
        paddle.disable_static()
        x = np.array(
            [[1, 2, 3, float('nan')], [1, 2, 3, 4], [float('nan'), 1, 2, 3]]
        )
        lis_tests = [
            [x.astype(dtype), axis, keepdims]
            for axis in [-1, 0, 1, None]
            for keepdims in [False, True]
            for dtype in ['float32', 'float64']
        ]
        for lis_test in lis_tests:
            self.dygraph_single_test_median(lis_test)

    @unittest.skipIf(
        not core.is_compiled_with_cuda()
        or not core.is_float16_supported(core.CUDAPlace(0)),
        "core is not compiled with CUDA and do not support float16",
    )
    def test_float16(self):
        paddle.disable_static(core.CUDAPlace(0))
        x = np.array(
            [[1, 2, 3, float('nan')], [1, 2, 3, 4], [float('nan'), 1, 2, 3]]
        ).astype('float16')
        lis_tests = [
            [axis, keepdims]
            for axis in [-1, 0, 1, None]
            for keepdims in [False, True]
        ]
        for axis, keepdims in lis_tests:
            res_np = np.median(x, axis=axis, keepdims=keepdims)
            res_pd = paddle.median(paddle.to_tensor(x), axis, keepdims)
            self.check_numpy_res(res_pd.numpy(False), res_np.astype('float64'))
            np.testing.assert_equal(res_pd.numpy(False).dtype, np.float32)

    def test_output_dtype(self):
        supported_dypes = ['float32', 'float64', 'int32', 'int64']
        for inp_dtype in supported_dypes:
            x = np.random.randint(low=-100, high=100, size=[2, 4, 5]).astype(
                inp_dtype
            )
            res = paddle.median(paddle.to_tensor(x), mode='avg')
            if inp_dtype == 'float64':
                np.testing.assert_equal(res.numpy().dtype, np.float64)
            else:
                np.testing.assert_equal(res.numpy().dtype, np.float32)


class TestMedianMin(unittest.TestCase):
    def static_single_test_median(self, lis_test):
        paddle.enable_static()
        x, axis, keepdims = lis_test
        res_np = np_median_min_axis(x, axis=axis, keepdims=keepdims)
        main_program = paddle.static.Program()
        startup_program = paddle.static.Program()
        exe = paddle.static.Executor()
        with paddle.static.program_guard(main_program, startup_program):
            x_in = paddle.static.data(shape=x.shape, dtype=x.dtype, name='x')
            y = paddle.median(x_in, axis, keepdims, mode='min')
            [res_pd, _] = exe.run(feed={'x': x}, fetch_list=[y])
            np.testing.assert_allclose(res_pd, res_np)
        paddle.disable_static()

    def dygraph_single_test_median(self, lis_test):
        x, axis, keepdims = lis_test
        res_np = np_median_min_axis(x, axis=axis, keepdims=keepdims)
        if axis is None:
            res_pd = paddle.median(
                paddle.to_tensor(x), axis, keepdims, mode='min'
            )
        else:
            res_pd, _ = paddle.median(
                paddle.to_tensor(x), axis, keepdims, mode='min'
            )
        np.testing.assert_allclose(res_pd.numpy(False), res_np)

    def test_median_static(self):
        h = 3
        w = 4
        l = 2
        x = np.arange(h * w * l).reshape([h, w, l]).astype("float32")
        lis_tests = [
            [x.astype(dtype), axis, keepdims]
            for axis in [-1, 0, 1, 2]
            for keepdims in [False, True]
            for dtype in ['float32', 'float64', 'int32', 'int64']
        ]
        for lis_test in lis_tests:
            self.static_single_test_median(lis_test)

    def test_median_dygraph(self):
        paddle.disable_static()
        h = 3
        w = 4
        l = 2
        x = np.arange(h * w * l).reshape([h, w, l]).astype("float32")
        lis_tests = [
            [x.astype(dtype), axis, keepdims]
            for axis in [-1, 0, 1, 2]
            for keepdims in [False, True]
            for dtype in ['float32', 'float64', 'int32', 'int64']
        ]
        for lis_test in lis_tests:
            self.dygraph_single_test_median(lis_test)

    def test_index_even_case(self):
        paddle.disable_static()
        x = paddle.arange(2 * 100).reshape((2, 100)).astype(paddle.float32)
        out, index = paddle.median(x, axis=1, mode='min')
        np.testing.assert_allclose(out.numpy(), [49.0, 149.0])
        np.testing.assert_equal(index.numpy(), [49, 49])

    def test_index_odd_case(self):
        paddle.disable_static()
        x = paddle.arange(30).reshape((3, 10)).astype(paddle.float32)
        out, index = paddle.median(x, axis=1, mode='min')
        np.testing.assert_allclose(out.numpy(), [4.0, 14.0, 24.0])
        np.testing.assert_equal(index.numpy(), [4, 4, 4])

    def test_nan(self):
        paddle.disable_static()
        x = np.array(
            [
                [1, 2, 3, float('nan')],
                [1, 2, 3, 4],
                [float('nan'), 1, 2, 3],
                [1, float('nan'), 3, float('nan')],
                [float('nan'), float('nan'), 3, float('nan')],
            ]
        )
        lis_tests = [
            [x.astype(dtype), axis, keepdims]
            for axis in [-1, 0, 1, None]
            for keepdims in [False, True]
            for dtype in ['float32', 'float64']
        ]
        for lis_test in lis_tests:
            self.dygraph_single_test_median(lis_test)

    @unittest.skipIf(
        not core.is_compiled_with_cuda()
        or not core.is_float16_supported(core.CUDAPlace(0)),
        "core is not compiled with CUDA and do not support float16",
    )
    def test_float16(self):
        paddle.disable_static(core.CUDAPlace(0))
        x = np.array(
            [[1, 2, 3, float('nan')], [1, 2, 3, 4], [float('nan'), 1, 2, 3]]
        ).astype('float16')
        lis_tests = [
            [axis, keepdims]
            for axis in [-1, 0, 1, None]
            for keepdims in [False, True]
        ]
        for axis, keepdims in lis_tests:
            res_np = np_median_min_axis(x, axis=axis, keepdims=keepdims)
            if axis is None:
                res_pd = paddle.median(
                    paddle.to_tensor(x), axis, keepdims, mode='min'
                )
            else:
                res_pd, _ = paddle.median(
                    paddle.to_tensor(x), axis, keepdims, mode='min'
                )
            np.testing.assert_allclose(res_pd.numpy(False), res_np)
            np.testing.assert_equal(res_pd.numpy(False).dtype, np.float16)

    def test_output_dtype(self):
        supported_dypes = ['float32', 'float64', 'int32', 'int64']
        for inp_dtype in supported_dypes:
            x = np.random.randint(low=-100, high=100, size=[2, 4, 5]).astype(
                inp_dtype
            )
            res = paddle.median(paddle.to_tensor(x), mode='min')
            np.testing.assert_equal(res.numpy().dtype, np.dtype(inp_dtype))


class TestMedianAvg_ZeroSize(unittest.TestCase):
    def dygraph_single_test_median(self, lis_test):
        x, axis, keepdims = lis_test
        res_np = np.median(x, axis=axis, keepdims=keepdims)
        x_pd = paddle.to_tensor(x)
        x_pd.stop_gradient = False
        res_pd = paddle.median(x_pd, axis, keepdims)
        np.testing.assert_allclose(res_pd.numpy(), res_np)
        paddle.sum(res_pd).backward()
        np.testing.assert_allclose(x_pd.grad.shape, x_pd.shape)

    def test_median_dygraph(self):
        paddle.disable_static()
        h = 0
        w = 4
        l = 2
        x = np.arange(h * w * l).reshape([h, w, l])
        self.dygraph_single_test_median([x, 1, False])


class TestMedianMin_ZeroSize(unittest.TestCase):
    def dygraph_single_test_median(self, lis_test):
        x, axis, keepdims = lis_test
        res_np = np_median_min_axis(x, axis=axis, keepdims=keepdims)
        x_pd = paddle.to_tensor(x)
        x_pd.stop_gradient = False
        if axis is None:
            res_pd = paddle.median(x_pd, axis, keepdims, mode='min')
        else:
            res_pd, _ = paddle.median(x_pd, axis, keepdims, mode='min')
        np.testing.assert_allclose(res_pd.numpy(), res_np)
        paddle.sum(res_pd).backward()
        np.testing.assert_allclose(x_pd.grad.shape, x_pd.shape)

    def test_median_dygraph(self):
        paddle.disable_static()
        h = 0
        w = 4
        l = 2
        x = np.arange(h * w * l).reshape([h, w, l]).astype("float32")
        self.dygraph_single_test_median([x, 1, False])


class TestMedianSort(unittest.TestCase):
    def dygraph_single_test_median(self, lis_test):
        x, axis, keepdims = lis_test
        res_np = np.median(x, axis=axis, keepdims=keepdims)
        x_pd = paddle.to_tensor(x)
        x_pd.stop_gradient = False
        res_pd = paddle.median(x_pd, axis, keepdims)
        np.testing.assert_allclose(res_pd.numpy(), res_np)

    def test_median_dygraph(self):
        paddle.disable_static()
        h = 2
        w = 20000
        l = 2
        x = np.arange(h * w * l).reshape([h, w, l])
        self.dygraph_single_test_median([x, 1, False])


class TestMedianAlias(unittest.TestCase):
    def static_single_test_median(self, lis_test):
        paddle.enable_static()
        x, axis, keepdims = lis_test
        res_np = np_median_min_axis(x, axis=axis, keepdims=keepdims)
        main_program = paddle.static.Program()
        startup_program = paddle.static.Program()
        exe = paddle.static.Executor()
        with paddle.static.program_guard(main_program, startup_program):
            x_in = paddle.static.data(shape=x.shape, dtype=x.dtype, name='x')
            y = paddle.median(x_in, dim=axis, keepdim=keepdims)
            [res_pd, _] = exe.run(feed={'x': x}, fetch_list=[y])
            np.testing.assert_allclose(res_pd, res_np)
        paddle.disable_static()

    def dygraph_single_test_median(self, lis_test):
        x, axis, keepdims = lis_test
        res_np = np_median_min_axis(x, axis=axis, keepdims=keepdims)
        if axis is None:
            res_pd = paddle.median(
                paddle.to_tensor(x), dim=axis, keepdim=keepdims
            )
        else:
            res_pd, _ = paddle.median(
                paddle.to_tensor(x), dim=axis, keepdim=keepdims
            )
        np.testing.assert_allclose(res_pd.numpy(False), res_np)

    def test_median_static(self):
        h = 3
        w = 4
        l = 2
        x = np.arange(h * w * l).reshape([h, w, l]).astype("float32")
        lis_tests = [
            [x.astype(dtype), axis, keepdims]
            for axis in [-1, 0, 1, 2]
            for keepdims in [False, True]
            for dtype in ['float32', 'float64', 'int32', 'int64']
        ]
        for lis_test in lis_tests:
            self.static_single_test_median(lis_test)

    def test_median_dygraph(self):
        paddle.disable_static()
        h = 3
        w = 4
        l = 2
        x = np.arange(h * w * l).reshape([h, w, l]).astype("float32")
        lis_tests = [
            [x.astype(dtype), axis, keepdims]
            for axis in [-1, 0, 1, 2]
            for keepdims in [False, True]
            for dtype in ['float32', 'float64', 'int32', 'int64']
        ]
        for lis_test in lis_tests:
            self.dygraph_single_test_median(lis_test)


class TestMedianOutAPI(unittest.TestCase):
    def test_out_in_dygraph(self):
        paddle.disable_static()
        np.random.seed(2024)
        x = paddle.to_tensor(
            np.random.randn(5, 7).astype('float32'), stop_gradient=False
        )

        def run_case(case_type, axis=1, mode='avg'):
            if axis is None:
                out_shape = []
            else:
                out_shape = list(x.shape)
                out_shape[axis] = 1

            out_buf = paddle.zeros(out_shape, dtype='float32')
            out_buf.stop_gradient = False

            if case_type == 'return':
                if mode == 'min' and axis is not None:
                    z, indices = paddle.median(x, axis=axis, mode=mode)
                else:
                    z = paddle.median(x, axis=axis, mode=mode)
            elif case_type == 'input_out':
                if mode == 'min' and axis is not None:
                    paddle.median(x, axis=axis, mode=mode, out=(out_buf, None))
                    z = out_buf
                else:
                    paddle.median(x, axis=axis, mode=mode, out=out_buf)
                    z = out_buf
            elif case_type == 'both_return':
                if mode == 'min' and axis is not None:
                    z, indices = paddle.median(
                        x, axis=axis, mode=mode, out=(out_buf, None)
                    )
                else:
                    z = paddle.median(x, axis=axis, mode=mode, out=out_buf)
            elif case_type == 'both_input_out':
                if mode == 'min' and axis is not None:
                    _ = paddle.median(
                        x, axis=axis, mode=mode, out=(out_buf, None)
                    )
                    z = out_buf
                else:
                    _ = paddle.median(x, axis=axis, mode=mode, out=out_buf)
                    z = out_buf
            else:
                raise AssertionError

            # Reference calculation
            if mode == 'min' and axis is not None:
                ref, _ = paddle.median(x, axis=axis, mode=mode)
            else:
                ref = paddle.median(x, axis=axis, mode=mode)

            np.testing.assert_allclose(
                z.numpy(), ref.numpy(), rtol=1e-6, atol=1e-6
            )

            loss = (z * 2).mean()
            loss.backward()
            return z.numpy(), x.grad.numpy()

        # Test mode='avg'
        z1, gx1 = run_case('return', axis=1, mode='avg')
        x.clear_gradient()
        z2, gx2 = run_case('input_out', axis=1, mode='avg')
        x.clear_gradient()
        z3, gx3 = run_case('both_return', axis=1, mode='avg')
        x.clear_gradient()
        z4, gx4 = run_case('both_input_out', axis=1, mode='avg')

        np.testing.assert_allclose(z1, z2, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(z1, z3, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(z1, z4, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(gx1, gx2, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(gx1, gx3, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(gx1, gx4, rtol=1e-6, atol=1e-6)

        # Test mode='min'
        x.clear_gradient()
        z1, gx1 = run_case('return', axis=1, mode='min')
        x.clear_gradient()
        z2, gx2 = run_case('input_out', axis=1, mode='min')
        x.clear_gradient()
        z3, gx3 = run_case('both_return', axis=1, mode='min')
        x.clear_gradient()
        z4, gx4 = run_case('both_input_out', axis=1, mode='min')

        np.testing.assert_allclose(z1, z2, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(z1, z3, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(z1, z4, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(gx1, gx2, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(gx1, gx3, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(gx1, gx4, rtol=1e-6, atol=1e-6)

        # Test global median (axis=None)
        x.clear_gradient()
        z1, gx1 = run_case('return', axis=None, mode='avg')
        x.clear_gradient()
        z2, gx2 = run_case('input_out', axis=None, mode='avg')
        x.clear_gradient()
        z3, gx3 = run_case('both_return', axis=None, mode='avg')
        x.clear_gradient()
        z4, gx4 = run_case('both_input_out', axis=None, mode='avg')

        np.testing.assert_allclose(z1, z2, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(z1, z3, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(z1, z4, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(gx1, gx2, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(gx1, gx3, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(gx1, gx4, rtol=1e-6, atol=1e-6)

        paddle.enable_static()

    def test_out_with_alias(self):
        paddle.disable_static()
        np.random.seed(2024)
        x = paddle.to_tensor(
            np.random.randn(5, 7).astype('float32'), stop_gradient=False
        )

        # Test with input alias
        out_buf = paddle.zeros([5], dtype='float32')
        out_buf.stop_gradient = False

        z1 = paddle.median(x, axis=1, out=out_buf)
        z2 = paddle.median(input=x, axis=1, out=out_buf)

        np.testing.assert_allclose(z1.numpy(), z2.numpy(), rtol=1e-6, atol=1e-6)

        # Test with dim alias
        out_buf = paddle.zeros([5], dtype='float32')
        out_buf.stop_gradient = False

        z1 = paddle.median(x, axis=1, mode='min', out=out_buf)
        z2 = paddle.median(
            x, dim=1, out=out_buf
        )  # mode='min' by default with dim alias

        np.testing.assert_allclose(z1.numpy(), z2.numpy(), rtol=1e-6, atol=1e-6)

        paddle.enable_static()

    def test_out_error_cases(self):
        paddle.disable_static()

        x = paddle.to_tensor([[1, 2, 3], [4, 5, 6]], dtype='float32')

        # Test wrong shape
        out_wrong_shape = paddle.zeros([2, 2], dtype='float32')
        with self.assertRaises(ValueError):
            paddle.median(x, axis=1, out=out_wrong_shape)

        # Test wrong dtype for mode='avg'
        out_wrong_dtype = paddle.zeros([2], dtype='int32')
        with self.assertRaises(ValueError):
            paddle.median(x, axis=1, mode='avg', out=out_wrong_dtype)

        # Test out is not a tensor
        with self.assertRaises(TypeError):
            paddle.median(x, axis=1, out=[1, 2, 3])

        paddle.enable_static()

    def test_out_with_indices(self):
        paddle.disable_static()
        np.random.seed(2024)
        x = paddle.to_tensor(
            np.random.randn(5, 7).astype('float32'), stop_gradient=False
        )

        # Test mode='min' with indices output
        out_tensor = paddle.zeros([5], dtype='float32')
        out_indices = paddle.zeros([5], dtype='int64')
        out_tensor.stop_gradient = False
        out_indices.stop_gradient = False

        # Test return mode
        z1, idx1 = paddle.median(x, axis=1, mode='min')

        # Test out mode
        paddle.median(x, axis=1, mode='min', out=(out_tensor, out_indices))
        z2, idx2 = out_tensor, out_indices

        np.testing.assert_allclose(z1.numpy(), z2.numpy(), rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(
            idx1.numpy(), idx2.numpy(), rtol=1e-6, atol=1e-6
        )

        # Test partial out (only tensor, no indices)
        out_tensor = paddle.zeros([5], dtype='float32')
        out_tensor.stop_gradient = False
        z3, idx3 = paddle.median(x, axis=1, mode='min', out=(out_tensor, None))

        np.testing.assert_allclose(z1.numpy(), z3.numpy(), rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(
            idx1.numpy(), idx3.numpy(), rtol=1e-6, atol=1e-6
        )

        # Test error case: wrong out type for mode='min'
        with self.assertRaises(ValueError):
            paddle.median(
                x, axis=1, mode='min', out=paddle.zeros([5], dtype='float32')
            )

        # Test error case: wrong out type for mode='avg'
        with self.assertRaises(ValueError):
            paddle.median(
                x,
                axis=1,
                mode='avg',
                out=(paddle.zeros([5], dtype='float32'), None),
            )

        paddle.enable_static()


if __name__ == '__main__':
    unittest.main()
