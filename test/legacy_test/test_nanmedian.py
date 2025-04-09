#   Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
from op_test import OpTest, convert_float_to_uint16

import paddle
from paddle.base import core

np.random.seed(102)


def np_nanmedain(data):
    data_flat = data.flatten()
    data_cnt = len(data_flat)
    nan_cnt = np.isnan(data).sum()

    data_flat[np.isnan(data_flat)] = np.inf
    data_sort = np.sort(data_flat)
    data_sort[np.isinf(data_sort)] = np.nan

    valid_num = data_cnt - nan_cnt

    if valid_num % 2:
        is_odd = False
    else:
        is_odd = True

    i = int(valid_num / 2)
    if is_odd:
        np_res = min(data_sort[i - 1], data_sort[i])
    else:
        np_res = data_sort[i]
    return np_res


def np_nanmedain_axis(data, axis=None):
    data = copy.deepcopy(data)

    if axis is None:
        return np_nanmedain(data)

    if isinstance(axis, list):
        axis = axis
    elif isinstance(axis, set):
        axis = list(axis)
    else:
        axis = [axis]

    axis = [a + len(data.shape) if a < 0 else a for a in axis]

    trans_shape = []
    reshape = []
    for i in range(len(data.shape)):
        if i not in axis:
            trans_shape.append(i)
            reshape.append(data.shape[i])
    last_shape = 1
    for i in range(len(data.shape)):
        if i in axis:
            trans_shape.append(i)
            last_shape *= data.shape[i]
    reshape.append(last_shape)

    data_flat = np.transpose(data, trans_shape)

    data_flat = np.reshape(data_flat, (-1, reshape[-1]))

    data_cnt = data_flat.shape[-1]
    nan_cnt = np.isnan(data_flat).sum(-1)

    data_flat[np.isnan(data_flat)] = np.inf
    data_sort = np.sort(data_flat, axis=-1)
    data_sort[np.isinf(data_sort)] = np.nan

    valid_num = data_cnt - nan_cnt
    is_odd = valid_num % 2

    np_res = np.zeros(len(is_odd), dtype=data.dtype)
    for j in range(len(is_odd)):
        if valid_num[j] == 0:
            np_res[j] = np.nan
            continue

        i = int(valid_num[j] / 2)
        if is_odd[j]:
            np_res[j] = data_sort[j, i]
        else:
            np_res[j] = min(data_sort[j, i - 1], data_sort[j, i])

    np_res = np.reshape(np_res, reshape[:-1])
    return np_res


class TestNanmedianModeMin(unittest.TestCase):
    def setUp(self):
        single_axis_shape = 120
        multi_axis_shape = (2, 3, 4, 5)

        self.fake_data = {
            "single_axis_normal": np.random.uniform(
                -1, 1, single_axis_shape
            ).astype(np.float32),
            "multi_axis_normal": np.random.uniform(
                -1, 1, multi_axis_shape
            ).astype(np.float32),
            "single_axis_all_nan": np.full(single_axis_shape, np.nan),
            "multi_axis_all_nan": np.full(multi_axis_shape, np.nan),
        }

        single_partial_nan = self.fake_data["single_axis_normal"].copy()
        single_partial_nan[single_partial_nan > 0] = np.nan
        multi_partial_nan = self.fake_data["multi_axis_normal"].copy()
        multi_partial_nan[multi_partial_nan > 0] = np.nan
        self.fake_data["single_axis_partial_nan"] = single_partial_nan
        self.fake_data["multi_axis_partial_nan"] = multi_partial_nan

        row_data = np.random.uniform(-10, 10, multi_axis_shape)
        row_data[:, :, :, 0] = np.nan
        row_data[:, :, :2, 1] = np.nan
        row_data[:, :, 2:, 2] = np.nan
        self.fake_data["row_nan_even"] = row_data.astype(np.float32)
        self.fake_data["row_nan_float64"] = row_data.astype(np.float64)

        col_data = np.random.uniform(-10, 10, multi_axis_shape)
        col_data[:, :, 0, :] = float('nan')
        col_data[:, :, 1, :3] = np.nan
        col_data[:, :, 2, 3:] = np.nan
        self.fake_data["col_nan_odd"] = col_data.astype(np.float32)

        self.place = (
            paddle.CUDAPlace(0)
            if core.is_compiled_with_cuda()
            else paddle.CPUPlace()
        )
        self.axis_candidate_list = [
            None,
            0,
            2,
            -1,
            -2,
            (1, 2),
            [0, -1],
            [0, 1, 3],
            (1, 2, 3),
            [0, 2, 1, 3],
        ]

    def test_api_static(self):
        data = self.fake_data["col_nan_odd"]
        paddle.enable_static()
        np_res = np_nanmedain(data)
        with paddle.static.program_guard(paddle.static.Program()):
            x = paddle.static.data('X', data.shape)
            out1 = paddle.nanmedian(x, keepdim=False, mode='min')
            out2 = paddle.tensor.nanmedian(x, keepdim=False, mode='min')
            out3 = paddle.tensor.stat.nanmedian(x, keepdim=False, mode='min')
            axis = np.arange(len(data.shape)).tolist()
            out4 = paddle.nanmedian(x, axis=axis, keepdim=False, mode='min')
            out5 = paddle.nanmedian(
                x, axis=tuple(axis), keepdim=False, mode='min'
            )
            exe = paddle.static.Executor(self.place)
            res = exe.run(
                feed={'X': data}, fetch_list=[out1, out2, out3, out4, out5]
            )

        for out in res:
            np.testing.assert_allclose(np_res, out, rtol=1e-05, equal_nan=True)

    def test_api_dygraph(self):
        paddle.disable_static(self.place)

        def clean_axis_numpy(axis, shape_len):
            if isinstance(axis, tuple):
                axis = list(axis)
            if isinstance(axis, list):
                for k in range(len(axis)):
                    if axis[k] < 0:
                        axis[k] += shape_len
                axis = set(axis)
            return axis

        def test_data_case(data, name):
            for keep_dim in [False, True]:
                if np.isnan(data).all() and keep_dim:
                    np_ver = np.version.version.split('.')
                    if int(np_ver[0]) < 1 or int(np_ver[1]) <= 20:
                        print(
                            "This numpy version does not support all nan elements when keepdim is True"
                        )
                        continue

                np_res = np_nanmedain(data)
                pd_res = paddle.nanmedian(
                    paddle.to_tensor(data), keepdim=keep_dim, mode='min'
                )
                np.testing.assert_allclose(
                    np_res, pd_res.item(), rtol=1e-05, equal_nan=True
                )

        def test_axis_case(data, axis):
            if (axis is not None) and (not isinstance(axis, (list, tuple))):
                pd_res, _ = paddle.nanmedian(
                    paddle.to_tensor(data), axis=axis, keepdim=False, mode='min'
                )
            else:
                pd_res = paddle.nanmedian(
                    paddle.to_tensor(data), axis=axis, keepdim=False, mode='min'
                )
            axis = clean_axis_numpy(axis, len(data.shape))
            np_res = np_nanmedain_axis(data, axis)
            np.testing.assert_allclose(
                np_res, pd_res.numpy(), rtol=1e-05, equal_nan=True
            )

        for name, data in self.fake_data.items():
            test_data_case(data, name)

        for axis in self.axis_candidate_list:
            test_axis_case(self.fake_data["row_nan_even"], axis)
            test_axis_case(self.fake_data["col_nan_odd"], axis)

        paddle.enable_static()

    def test_errors(self):
        paddle.enable_static()
        with paddle.static.program_guard(paddle.static.Program()):
            x = paddle.static.data("X", [10, 12])

            def test_dtype():
                x2 = paddle.static.data('X2', [10, 12], 'bool')
                paddle.nanmedian(x2, mode='min')

            def test_empty_axis():
                paddle.nanmedian(x, axis=[], keepdim=True, mode='min')

            def test_axis_not_in_range():
                paddle.nanmedian(x, axis=3, keepdim=True, mode='min')

            def test_duplicated_axis():
                paddle.nanmedian(x, axis=[1, -1], keepdim=True, mode='min')

            self.assertRaises(TypeError, test_dtype)
            self.assertRaises(ValueError, test_empty_axis)
            self.assertRaises(ValueError, test_axis_not_in_range)
            self.assertRaises(ValueError, test_duplicated_axis)

    def test_dygraph(self):
        paddle.disable_static(place=self.place)
        with paddle.base.dygraph.guard():
            data = self.fake_data["col_nan_odd"]
            out = paddle.nanmedian(
                paddle.to_tensor(data), keepdim=False, mode='min'
            )
        np_res = np_nanmedain(data)
        np.testing.assert_allclose(np_res, out, rtol=1e-05, equal_nan=True)
        paddle.enable_static()

    def test_check_grad(self):
        paddle.disable_static(place=self.place)
        shape = (4, 5)
        x_np = np.arange(np.prod(shape)).reshape(shape).astype(np.float64)
        x_np[0, :] = np.nan
        x_np[1, :3] = np.nan
        x_np[2, 3:] = np.nan

        x_tensor = paddle.to_tensor(x_np, stop_gradient=False)
        y = paddle.nanmedian(x_tensor, keepdim=True, mode='min')
        dx = paddle.grad(y, x_tensor)[0].numpy()

        np_grad = np.zeros(shape)
        np_grad[2, 2] = 1.0
        np.testing.assert_allclose(np_grad, dx, rtol=1e-05, equal_nan=True)

    def test_check_grad_axis(self):
        paddle.disable_static(place=self.place)
        shape = (4, 5)
        x_np = np.random.uniform(-1, 1, shape).astype(np.float64)
        x_np[0, :] = np.nan
        x_np[1, :3] = np.nan
        x_np[2, 3:] = np.nan
        x_np_sorted = np.sort(x_np)
        nan_counts = np.count_nonzero(np.isnan(x_np).astype(np.int32), axis=1)
        np_grad = np.zeros(shape)
        for i in range(shape[0]):
            valid_cnts = shape[1] - nan_counts[i]
            if valid_cnts == 0:
                continue

            mid = int(valid_cnts / 2)
            targets = []
            is_odd = valid_cnts % 2
            if not is_odd and mid > 0:
                min_val = min(x_np_sorted[i, mid - 1], x_np_sorted[i, mid])
                targets.append(min_val)
            else:
                targets.append(x_np_sorted[i, mid])

            for j in range(shape[1]):
                if x_np[i, j] in targets:
                    np_grad[i, j] = 1 if is_odd else 1

        x_tensor = paddle.to_tensor(x_np, stop_gradient=False)
        y, _ = paddle.nanmedian(x_tensor, axis=1, mode='min')
        dx = paddle.grad(y, x_tensor)[0].numpy()
        np.testing.assert_allclose(np_grad, dx, rtol=1e-05, equal_nan=True)

    def test_mode_min_index(self):
        paddle.disable_static(place=self.place)
        x = paddle.arange(2 * 100).reshape((2, 100)).astype(paddle.float32)
        out, index = paddle.nanmedian(x, axis=1, mode='min')
        np.testing.assert_allclose(out.numpy(), [49.0, 149.0])
        np.testing.assert_equal(index.numpy(), [49, 49])

    def test_check_grad_0d(self):
        paddle.disable_static(place=self.place)
        x = paddle.rand([])
        x.stop_gradient = False
        y = paddle.nanmedian(x, mode='min')
        y.backward()
        self.assertEqual(x.grad.shape, [])
        np.testing.assert_allclose(x.grad, np.array(1.0))

        x = paddle.to_tensor(float('nan'), stop_gradient=False)
        y = paddle.nanmedian(x, mode='min')
        y.backward()
        self.assertEqual(x.grad.shape, [])
        np.testing.assert_allclose(x.grad, np.array(0.0))

    def test_dygraph_cpu(self):
        paddle.disable_static(place=paddle.CPUPlace())
        with paddle.base.dygraph.guard():
            data = np.array(
                [[1.4907, 1.0593, 1.5696], [1.4907, 1.0593, 1.5696]]
            )
            out, index = paddle.nanmedian(
                paddle.to_tensor(data), axis=1, keepdim=False, mode='min'
            )
        np_res = np_nanmedain_axis(data, axis=1)
        np.testing.assert_allclose(np_res, out, rtol=1e-05, equal_nan=True)
        np.testing.assert_allclose(
            np.array([0, 0]), index, rtol=1e-05, equal_nan=True
        )
        paddle.enable_static()


class TestNanmedianModeMean(unittest.TestCase):
    def setUp(self):
        single_axis_shape = 120
        multi_axis_shape = (2, 3, 4, 5)

        self.fake_data = {
            "single_axis_normal": np.random.uniform(
                -1, 1, single_axis_shape
            ).astype(np.float32),
            "multi_axis_normal": np.random.uniform(
                -1, 1, multi_axis_shape
            ).astype(np.float32),
            "single_axis_all_nan": np.full(single_axis_shape, np.nan),
            "multi_axis_all_nan": np.full(multi_axis_shape, np.nan),
        }

        single_partial_nan = self.fake_data["single_axis_normal"].copy()
        single_partial_nan[single_partial_nan > 0] = np.nan
        multi_partial_nan = self.fake_data["multi_axis_normal"].copy()
        multi_partial_nan[multi_partial_nan > 0] = np.nan
        self.fake_data["single_axis_partial_nan"] = single_partial_nan
        self.fake_data["multi_axis_partial_nan"] = multi_partial_nan

        row_data = np.random.uniform(-10, 10, multi_axis_shape)
        row_data[:, :, :, 0] = np.nan
        row_data[:, :, :2, 1] = np.nan
        row_data[:, :, 2:, 2] = np.nan
        self.fake_data["row_nan_even"] = row_data.astype(np.float32)
        self.fake_data["row_nan_float64"] = row_data.astype(np.float64)
        # self.fake_data["row_nan_int64"] = row_data.astype(np.int64)
        # self.fake_data["row_nan_int32"] = row_data.astype(np.int32)

        col_data = np.random.uniform(-10, 10, multi_axis_shape)
        col_data[:, :, 0, :] = float('nan')
        col_data[:, :, 1, :3] = np.nan
        col_data[:, :, 2, 3:] = np.nan
        self.fake_data["col_nan_odd"] = col_data.astype(np.float32)

        self.place = (
            paddle.CUDAPlace(0)
            if core.is_compiled_with_cuda()
            else paddle.CPUPlace()
        )
        self.axis_candidate_list = [
            None,
            0,
            2,
            -1,
            -2,
            (1, 2),
            [0, -1],
            [0, 1, 3],
            (1, 2, 3),
            [0, 2, 1, 3],
        ]

    def test_api_static(self):
        data = self.fake_data["col_nan_odd"]
        paddle.enable_static()
        np_res = np.nanmedian(data)
        with paddle.static.program_guard(paddle.static.Program()):
            x = paddle.static.data('X', data.shape)
            out1 = paddle.nanmedian(x, keepdim=False)
            out2 = paddle.tensor.nanmedian(x, keepdim=False)
            out3 = paddle.tensor.stat.nanmedian(x, keepdim=False)
            axis = np.arange(len(data.shape)).tolist()
            out4 = paddle.nanmedian(x, axis=axis, keepdim=False)
            out5 = paddle.nanmedian(x, axis=tuple(axis), keepdim=False)
            exe = paddle.static.Executor(self.place)
            res = exe.run(
                feed={'X': data}, fetch_list=[out1, out2, out3, out4, out5]
            )

        for out in res:
            np.testing.assert_allclose(np_res, out, rtol=1e-05, equal_nan=True)

    def test_api_dygraph(self):
        paddle.disable_static(self.place)

        def clean_axis_numpy(axis, shape_len):
            if isinstance(axis, tuple):
                axis = list(axis)
            if isinstance(axis, list):
                for k in range(len(axis)):
                    if axis[k] < 0:
                        axis[k] += shape_len
                axis = set(axis)
            return axis

        def test_data_case(data, name):
            for keep_dim in [False, True]:
                if np.isnan(data).all() and keep_dim:
                    np_ver = np.version.version.split('.')
                    if int(np_ver[0]) < 1 or int(np_ver[1]) <= 20:
                        print(
                            "This numpy version does not support all nan elements when keepdim is True"
                        )
                        continue

                np_res = np.nanmedian(data)
                pd_res = paddle.nanmedian(
                    paddle.to_tensor(data), keepdim=keep_dim
                )

                np.testing.assert_allclose(
                    np_res, pd_res.item(), rtol=1e-05, equal_nan=True
                )

        def test_axis_case(data, axis):
            pd_res = paddle.nanmedian(
                paddle.to_tensor(data), axis=axis, keepdim=False
            )
            axis = clean_axis_numpy(axis, len(data.shape))
            np_res = np.nanmedian(data, axis)
            np.testing.assert_allclose(
                np_res, pd_res.numpy(), rtol=1e-05, equal_nan=True
            )

        for name, data in self.fake_data.items():
            test_data_case(data, name)

        for axis in self.axis_candidate_list:
            test_axis_case(self.fake_data["row_nan_even"], axis)
            test_axis_case(self.fake_data["col_nan_odd"], axis)

        paddle.enable_static()

    def test_errors(self):
        paddle.enable_static()
        with paddle.static.program_guard(paddle.static.Program()):
            x = paddle.static.data("X", [10, 12])

            def test_dtype():
                x2 = paddle.static.data('X2', [10, 12], 'bool')
                paddle.nanmedian(x2)

            def test_empty_axis():
                paddle.nanmedian(x, axis=[], keepdim=True)

            def test_axis_not_in_range():
                paddle.nanmedian(x, axis=3, keepdim=True)

            def test_duplicated_axis():
                paddle.nanmedian(x, axis=[1, -1], keepdim=True)

            def test_mode():
                paddle.nanmedian(x, mode='max')

            self.assertRaises(TypeError, test_dtype)
            self.assertRaises(ValueError, test_empty_axis)
            self.assertRaises(ValueError, test_axis_not_in_range)
            self.assertRaises(ValueError, test_duplicated_axis)
            self.assertRaises(ValueError, test_mode)

    def test_dygraph(self):
        paddle.disable_static(place=self.place)
        with paddle.base.dygraph.guard():
            data = self.fake_data["col_nan_odd"]
            out = paddle.nanmedian(paddle.to_tensor(data), keepdim=False)
        np_res = np.nanmedian(data)
        np.testing.assert_allclose(np_res, out, rtol=1e-05, equal_nan=True)
        paddle.enable_static()

    def test_check_grad(self):
        paddle.disable_static(place=self.place)
        shape = (4, 5)
        x_np = np.arange(np.prod(shape)).reshape(shape).astype(np.float64)
        x_np[0, :] = np.nan
        x_np[1, :3] = np.nan
        x_np[2, 3:] = np.nan

        x_tensor = paddle.to_tensor(x_np, stop_gradient=False)
        y = paddle.nanmedian(x_tensor, keepdim=True)
        dx = paddle.grad(y, x_tensor)[0].numpy()

        np_grad = np.zeros(shape)
        np_grad[2, 2] = 0.5
        np_grad[3, 0] = 0.5
        np.testing.assert_allclose(np_grad, dx, rtol=1e-05, equal_nan=True)

    def test_check_grad_axis(self):
        paddle.disable_static(place=self.place)
        shape = (4, 5)
        x_np = np.random.uniform(-1, 1, shape).astype(np.float64)
        x_np[0, :] = np.nan
        x_np[1, :3] = np.nan
        x_np[2, 3:] = np.nan
        x_np_sorted = np.sort(x_np)
        nan_counts = np.count_nonzero(np.isnan(x_np).astype(np.int32), axis=1)
        np_grad = np.zeros(shape)
        for i in range(shape[0]):
            valid_cnts = shape[1] - nan_counts[i]
            if valid_cnts == 0:
                continue

            mid = int(valid_cnts / 2)
            targets = [x_np_sorted[i, mid]]
            is_odd = valid_cnts % 2
            if not is_odd and mid > 0:
                targets.append(x_np_sorted[i, mid - 1])
            for j in range(shape[1]):
                if x_np[i, j] in targets:
                    np_grad[i, j] = 1 if is_odd else 0.5

        x_tensor = paddle.to_tensor(x_np, stop_gradient=False)
        y = paddle.nanmedian(x_tensor, axis=1)
        dx = paddle.grad(y, x_tensor)[0].numpy()
        np.testing.assert_allclose(np_grad, dx, rtol=1e-05, equal_nan=True)

    def test_check_grad_0d(self):
        paddle.disable_static(place=self.place)
        x = paddle.rand([])
        x.stop_gradient = False
        y = paddle.nanmedian(x)
        y.backward()
        self.assertEqual(x.grad.shape, [])
        np.testing.assert_allclose(x.grad, np.array(1.0))

        x = paddle.to_tensor(float('nan'), stop_gradient=False)
        y = paddle.nanmedian(x)
        y.backward()
        self.assertEqual(x.grad.shape, [])
        np.testing.assert_allclose(x.grad, np.array(0.0))

    def test_dygraph_cpu(self):
        paddle.disable_static(place=paddle.CPUPlace())
        with paddle.base.dygraph.guard():
            data = np.array(
                [[1.4907, 1.0593, 1.5696], [1.4907, 1.0593, 1.5696]]
            )
            out = paddle.nanmedian(
                paddle.to_tensor(data), axis=1, keepdim=False
            )
        np_res = np.nanmedian(data, axis=1)
        np.testing.assert_allclose(np_res, out, rtol=1e-05, equal_nan=True)
        paddle.enable_static()


class TestNanmedianFP16Op(OpTest):
    def setUp(self):
        self.op_type = "nanmedian"
        self.python_api = paddle.nanmedian
        self.public_python_api = paddle.nanmedian
        self.dtype = np.float16
        self.python_out_sig = ["Out"]
        X = np.random.random((100, 100)).astype('float16')
        Out = np.nanmedian(X)
        indices = np.zeros_like(Out, dtype='int64')
        self.inputs = {'X': X}
        self.outputs = {'Out': Out, 'MedianIndex': indices}

    def test_check_output(self):
        self.check_output(check_pir=True)

    def test_check_grad(self):
        self.check_grad(['X'], 'Out', check_pir=True)


@unittest.skipIf(
    not core.is_compiled_with_cuda()
    or not core.is_bfloat16_supported(core.CUDAPlace(0)),
    "core is not compiled with CUDA and not support the bfloat16",
)
class TestNanmedianBF16Op(OpTest):
    def setUp(self):
        self.op_type = "nanmedian"
        self.python_api = paddle.nanmedian
        self.public_python_api = paddle.nanmedian
        self.dtype = np.uint16
        self.python_out_sig = ["Out"]
        X = np.random.random((100, 100)).astype('float32')
        Out = np.nanmedian(X)
        indices = np.zeros_like(Out, dtype='int64')
        self.inputs = {'X': convert_float_to_uint16(X)}
        self.outputs = {
            'Out': convert_float_to_uint16(Out),
            'MedianIndex': indices,
        }

    def test_check_output(self):
        place = core.CUDAPlace(0)
        self.check_output_with_place(place, check_pir=True)

    def test_check_grad(self):
        place = core.CUDAPlace(0)
        self.check_grad_with_place(place, ['X'], 'Out', check_pir=True)


if __name__ == "__main__":
    unittest.main()
