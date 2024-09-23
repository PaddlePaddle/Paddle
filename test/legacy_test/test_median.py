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


def np_medain_min_axis(data, axis=None, keepdims=False):
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
        res_np = np_medain_min_axis(x, axis=axis, keepdims=keepdims)
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
        res_np = np_medain_min_axis(x, axis=axis, keepdims=keepdims)
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
            res_np = np_medain_min_axis(x, axis=axis, keepdims=keepdims)
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


if __name__ == '__main__':
    unittest.main()
