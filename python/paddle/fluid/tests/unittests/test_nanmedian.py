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

import unittest
import numpy as np
import paddle
import paddle.fluid.core as core

np.random.seed(102)


class TestNanmedian(unittest.TestCase):

    def setUp(self):
        single_axis_shape = (120)
        multi_axis_shape = (2, 3, 4, 5)

        self.fake_data = {
            "single_axis_normal":
            np.random.uniform(-1, 1, single_axis_shape).astype(np.float32),
            "multi_axis_normal":
            np.random.uniform(-1, 1, multi_axis_shape).astype(np.float32),
            "single_axis_all_nan":
            np.full(single_axis_shape, np.nan),
            "multi_axis_all_nan":
            np.full(multi_axis_shape, np.nan),
        }

        single_partial_nan = self.fake_data["single_axis_normal"].copy()
        single_partial_nan[single_partial_nan > 0] = np.nan
        multi_partial_nan = self.fake_data["multi_axis_normal"].copy()
        multi_partial_nan[multi_partial_nan > 0] = np.nan
        self.fake_data["single_axis_partial_nan"] = single_partial_nan
        self.fake_data["multi_axis_partial_nan"] = multi_partial_nan

        row_data = np.random.uniform(-1, 1, multi_axis_shape).astype(np.float32)
        row_data[:, :, :, 0] = np.nan
        row_data[:, :, :2, 1] = np.nan
        row_data[:, :, 2:, 2] = np.nan
        self.fake_data["row_nan_even"] = row_data
        self.fake_data["row_nan_float64"] = row_data.astype(np.float64)
        self.fake_data["row_nan_int64"] = row_data.astype(np.int64)
        self.fake_data["row_nan_int32"] = row_data.astype(np.int32)

        col_data = np.random.uniform(-1, 1, multi_axis_shape).astype(np.float32)
        col_data[:, :, 0, :] = np.nan
        col_data[:, :, 1, :3] = np.nan
        col_data[:, :, 2, 3:] = np.nan
        self.fake_data["col_nan_odd"] = col_data

        self.place = paddle.CUDAPlace(0) if core.is_compiled_with_cuda() \
            else paddle.CPUPlace()
        self.axis_candiate_list = [
            None, 0, 2, -1, -2, (1, 2), [0, -1], [0, 1, 3], (1, 2, 3),
            [0, 2, 1, 3]
        ]

    def test_api_static(self):
        data = self.fake_data["col_nan_odd"]
        paddle.enable_static()
        np_res = np.nanmedian(data, keepdims=True)
        with paddle.static.program_guard(paddle.static.Program()):
            x = paddle.fluid.data('X', data.shape)
            out1 = paddle.nanmedian(x, keepdim=True)
            out2 = paddle.tensor.nanmedian(x, keepdim=True)
            out3 = paddle.tensor.stat.nanmedian(x, keepdim=True)
            axis = np.arange(len(data.shape)).tolist()
            out4 = paddle.nanmedian(x, axis=axis, keepdim=True)
            out5 = paddle.nanmedian(x, axis=tuple(axis), keepdim=True)
            exe = paddle.static.Executor(self.place)
            res = exe.run(feed={'X': data},
                          fetch_list=[out1, out2, out3, out4, out5])

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

        def test_data_case(data):
            for keep_dim in [False, True]:
                if np.isnan(data).all() and keep_dim:
                    np_ver = np.version.version.split('.')
                    if int(np_ver[0]) < 1 or int(np_ver[1]) <= 20:
                        print(
                            "This numpy version does not support all nan elements when keepdim is True"
                        )
                        continue

                np_res = np.nanmedian(data, keepdims=keep_dim)
                pd_res = paddle.nanmedian(paddle.to_tensor(data),
                                          keepdim=keep_dim)
                np.testing.assert_allclose(np_res,
                                           pd_res.numpy(),
                                           rtol=1e-05,
                                           equal_nan=True)

        def test_axis_case(data, axis):
            pd_res = paddle.nanmedian(paddle.to_tensor(data),
                                      axis=axis,
                                      keepdim=False)
            axis = clean_axis_numpy(axis, len(data.shape))
            np_res = np.nanmedian(data, axis=axis, keepdims=False)
            np.testing.assert_allclose(np_res,
                                       pd_res.numpy(),
                                       rtol=1e-05,
                                       equal_nan=True)

        for name, data in self.fake_data.items():
            test_data_case(data)

        for axis in self.axis_candiate_list:
            test_axis_case(self.fake_data["row_nan_even"], axis)
            test_axis_case(self.fake_data["col_nan_odd"], axis)

        paddle.enable_static()

    def test_errors(self):
        paddle.enable_static()
        with paddle.static.program_guard(paddle.static.Program()):
            x = paddle.fluid.data("X", [10, 12])

            def test_dtype():
                x2 = paddle.fluid.data('X2', [10, 12], 'bool')
                paddle.nanmedian(x2)

            def test_empty_axis():
                paddle.nanmedian(x, axis=[], keepdim=True)

            def test_axis_not_in_range():
                paddle.nanmedian(x, axis=3, keepdim=True)

            def test_duplicated_axis():
                paddle.nanmedian(x, axis=[1, -1], keepdim=True)

            self.assertRaises(TypeError, test_dtype)
            self.assertRaises(ValueError, test_empty_axis)
            self.assertRaises(ValueError, test_axis_not_in_range)
            self.assertRaises(ValueError, test_duplicated_axis)

    def test_dygraph(self):
        paddle.disable_static(place=self.place)
        with paddle.fluid.dygraph.guard():
            data = self.fake_data["col_nan_odd"]
            out = paddle.nanmedian(paddle.to_tensor(data), keepdim=True)
        np_res = np.nanmedian(data, keepdims=True)
        np.testing.assert_allclose(np_res, out, rtol=1e-05, equal_nan=True)
        paddle.enable_static()

    def test_check_grad(self):
        paddle.disable_static(place=self.place)
        shape = (4, 5)
        x_np = np.random.uniform(-1, 1, shape).astype(np.float64)
        x_np[0, :] = np.nan
        x_np[1, :3] = np.nan
        x_np[2, 3:] = np.nan
        x_np_sorted = np.sort(x_np)
        nan_counts = np.count_nonzero(np.isnan(x_np).astype(np.int32), axis=1)
        np_grad = np.zeros((shape))
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
        y = paddle.nanmedian(x_tensor, axis=1, keepdim=True)
        dx = paddle.grad(y, x_tensor)[0].numpy()
        np.testing.assert_allclose(np_grad, dx, rtol=1e-05, equal_nan=True)


if __name__ == "__main__":
    unittest.main()
