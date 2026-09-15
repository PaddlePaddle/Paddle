#   Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

import sys
import unittest

import numpy as np
from op_test import OpTest, get_device_place, is_custom_device

import paddle
from paddle import base


def cummin_dim2(arr, axis=None):
    if axis is None:
        arr = arr.flatten()
        cummin = np.minimum.accumulate(arr)
        shape = arr.shape
        indices = np.zeros(shape).astype('int32')
        min_val = sys.maxsize
        min_ind = 0
        for i in range(shape[0]):
            if arr[i] <= min_val:
                min_val = min(arr[i], min_val)
                min_ind = i
                indices[i] = i
            else:
                indices[i] = min_ind
    else:
        cummin = np.minimum.accumulate(arr, axis)
        shape = arr.shape
        indices = np.zeros(shape).astype('int32')
        if axis < 0:
            axis = axis + len(shape)
        if axis == 0:
            for j in range(shape[1]):
                min_ind = 0
                min_val = sys.maxsize
                for i in range(shape[0]):
                    if arr[i][j] <= min_val:
                        min_val = arr[i][j]
                        min_ind = i
                        indices[i][j] = i
                    else:
                        indices[i][j] = min_ind
        elif axis == 1:
            for i in range(shape[0]):
                min_ind = 0
                min_val = sys.maxsize
                for j in range(shape[1]):
                    if arr[i][j] <= min_val:
                        min_val = arr[i][j]
                        min_ind = j
                        indices[i][j] = j
                    else:
                        indices[i][j] = min_ind
        else:
            raise Exception("unfeasible axis")
    return cummin, indices


class TestCumminOp(OpTest):
    def setUp(self):
        self.op_type = "cummin"
        self.python_api = paddle.cummin
        self.dtype = np.float64
        self.axis = -1
        self.indices_type = paddle.int64
        self.init_shape()
        self.input_data = np.random.random(self.shape).astype(self.dtype)
        self.set_attrs()

        self.inputs = {'x': self.input_data}
        self.attrs = {'axis': self.axis, 'dtype': self.indices_type}
        self.np_res, self.np_ind = cummin_dim2(self.input_data, axis=self.axis)
        self.outputs = {'out': self.np_res, 'indices': self.np_ind}

    def set_attrs(self):
        pass

    def init_shape(self):
        self.shape = (10, 10)

    def test_check_output(self):
        paddle.enable_static()
        self.check_output(check_pir=True)

    def test_check_grad(self):
        paddle.enable_static()
        self.check_grad(['x'], 'out', check_pir=True)


class TestCuinOpAxis1(TestCumminOp):
    def set_attrs(self):
        self.axis = 0


class TestCumminOpAxis2(TestCumminOp):
    def set_attrs(self):
        self.axis = -2


class TestCumminOpIndexType(TestCumminOp):
    def set_attrs(self):
        self.indices_type = paddle.int32


class TestCumminOp_ZeroSize(TestCumminOp):
    def init_shape(self):
        self.shape = (10, 0)


class TestCumminAPI(unittest.TestCase):
    def run_cases(self):
        data_np = np.random.random((100, 100)).astype(np.float32)
        data = paddle.to_tensor(data_np)

        y, indices = paddle.cummin(data)
        z, ind = cummin_dim2(data_np)
        np.testing.assert_array_equal(z, y.numpy())
        np.testing.assert_array_equal(ind, indices.numpy())

        y, indices = paddle.cummin(data, axis=0)
        z, ind = cummin_dim2(data_np, axis=0)
        np.testing.assert_array_equal(z, y.numpy())
        np.testing.assert_array_equal(ind, indices.numpy())

        y, indices = paddle.cummin(data, axis=-1)
        z, ind = cummin_dim2(data_np, axis=-1)
        np.testing.assert_array_equal(z, y.numpy())
        np.testing.assert_array_equal(ind, indices.numpy())

        y, indices = paddle.cummin(data, axis=-2)
        z, ind = cummin_dim2(data_np, axis=-2)
        np.testing.assert_array_equal(z, y.numpy())
        np.testing.assert_array_equal(ind, indices.numpy())

        y, indices = paddle.cummin(data, axis=-2, dtype='int32')
        z, ind = cummin_dim2(data_np, axis=-2)
        np.testing.assert_array_equal(z, y.numpy())
        np.testing.assert_array_equal(ind, indices.numpy())
        self.assertTrue(indices.dtype == paddle.int32)

        data_np = np.random.randint(0, 10, size=(100, 100)).astype(np.int32)
        data = paddle.to_tensor(data_np)
        y, indices = paddle.cummin(data, axis=0)
        z, ind = cummin_dim2(data_np, axis=0)
        np.testing.assert_array_equal(z, y.numpy())
        np.testing.assert_array_equal(ind, indices.numpy())

    def run_static(self, use_gpu=False):
        with base.program_guard(base.Program()):
            data_np = np.random.random((100, 100)).astype(np.float32)
            x = paddle.static.data('x', [100, 100])
            y1, indices1 = paddle.cummin(x)
            y2, indices2 = paddle.cummin(x, axis=0)
            y3, indices3 = paddle.cummin(x, axis=-1)
            y4, indices4 = paddle.cummin(x, axis=-2)
            y5, indices5 = paddle.cummin(x, axis=-2, dtype=np.int32)

            place = get_device_place() if use_gpu else base.CPUPlace()
            exe = base.Executor(place)
            out = exe.run(
                feed={'x': data_np},
                fetch_list=[
                    y1,
                    indices1,
                    y2,
                    indices2,
                    y3,
                    indices3,
                    y4,
                    indices4,
                    y5,
                    indices5,
                ],
            )

            z, ind = cummin_dim2(data_np)
            np.testing.assert_allclose(z, out[0], rtol=1e-05)
            np.testing.assert_allclose(ind, out[1], rtol=1e-05)

            z, ind = cummin_dim2(data_np, axis=0)
            np.testing.assert_allclose(z, out[2], rtol=1e-05)
            np.testing.assert_allclose(ind, out[3], rtol=1e-05)

            z, ind = cummin_dim2(data_np, axis=-1)
            np.testing.assert_allclose(z, out[4], rtol=1e-05)
            np.testing.assert_allclose(ind, out[5], rtol=1e-05)

            z, ind = cummin_dim2(data_np, axis=-2)
            np.testing.assert_allclose(z, out[6], rtol=1e-05)
            np.testing.assert_allclose(ind, out[7], rtol=1e-05)

            z, ind = cummin_dim2(data_np, axis=-2)
            np.testing.assert_allclose(z, out[8], rtol=1e-05)
            np.testing.assert_allclose(ind, out[9], rtol=1e-05)

    def test_cpu(self):
        paddle.disable_static(paddle.base.CPUPlace())
        self.run_cases()
        paddle.enable_static()
        self.run_static()

    def test_gpu(self):
        if not (base.core.is_compiled_with_cuda() or is_custom_device()):
            return
        paddle.disable_static(get_device_place())
        self.run_cases()
        paddle.enable_static()
        self.run_static(use_gpu=True)

    def test_errors(self):
        paddle.enable_static()
        with base.program_guard(base.Program()):

            def test_x_type():
                data = [1, 2, 3]
                y, indices = paddle.cummin(data, axis=0)

            self.assertRaises(TypeError, test_x_type)

        paddle.disable_static()

        def test_indices_type():
            data_np = np.random.random((10, 10)).astype(np.float32)
            data = paddle.to_tensor(data_np)
            y, indices = paddle.cummin(data, dtype='float32')

        self.assertRaises(ValueError, test_indices_type)

        def test_axis_outrange():
            data_np = np.random.random(100).astype(np.float32)
            data = paddle.to_tensor(data_np)
            y, indices = paddle.cummin(data, axis=-2)

        self.assertRaises(IndexError, test_axis_outrange)


def cum_scatter_add_ref(indices, out_grad, axis):
    """Reference for the cummax/cummin gradient.
    The gradient scatter-adds every upstream gradient element into ``x_grad`` at
    the argmax/argmin source position recorded in ``indices`` along ``axis``.
    Duplicate targets (a run of equal extrema) accumulate together.
    """
    ndim = out_grad.ndim
    axis = axis % ndim
    idx = np.moveaxis(indices, axis, 0).copy()
    g = np.moveaxis(out_grad, axis, 0).copy()
    row = idx.shape[0]
    idx2 = idx.reshape(row, -1)
    g2 = g.reshape(row, -1)
    grad2 = np.zeros_like(g2)
    for c in range(idx2.shape[1]):
        for r in range(row):
            t = int(idx2[r, c])
            t = 0 if t < 0 else (row - 1 if t >= row else t)
            grad2[t, c] += g2[r, c]
    return np.moveaxis(grad2.reshape(g.shape), 0, axis)


class TestCumminGradDeterministicGPU(unittest.TestCase):
    """Regression tests for the deterministic cummin gradient path.
    ``ScatterAddDeterministic`` in ``cum_maxmin_grad_kernel.cu``
    is only reached when ``FLAGS_cudnn_deterministic`` are on.
    These tests cover forward+backward, duplicate-minimum run merging,
    several axes (including negative), both ``int32``/``int64`` index
    dtypes, runs longer than a warp, and run-to-run stability.
    """

    def setUp(self):
        if not (base.core.is_compiled_with_cuda() or is_custom_device()):
            self.skipTest("deterministic cummin grad path is GPU only")
        paddle.disable_static(get_device_place())
        self._old_flags = paddle.get_flags(["FLAGS_cudnn_deterministic"])
        paddle.set_flags({"FLAGS_cudnn_deterministic": True})

    def tearDown(self):
        if hasattr(self, "_old_flags"):
            paddle.set_flags(self._old_flags)
        paddle.enable_static()

    def _grad(self, x_np, g_np, axis, indices_dtype):
        xt = paddle.to_tensor(x_np)
        xt.stop_gradient = False
        out, ind = paddle.cummin(xt, axis=axis, dtype=indices_dtype)
        gt = paddle.to_tensor(g_np)
        dx = paddle.grad(out, xt, grad_outputs=gt)[0]
        return dx.numpy(), ind.numpy()

    def test_grad_matches_reference(self):
        rng = np.random.RandomState(20)
        # Integer valued input creates many ties, hence duplicate indices, so
        # the run-merging branch of the deterministic scatter-add is exercised.
        cases = [((64, 80), -1), ((64, 80), 0), ((5, 6, 7), 1), ((5, 6, 7), -2)]
        for shape, axis in cases:
            x_np = rng.randint(-3, 4, size=shape).astype("float32")
            g_np = rng.randn(*shape).astype("float32")
            for idtype in ["int32", "int64"]:
                dx, ind = self._grad(x_np, g_np, axis, idtype)
                ref = cum_scatter_add_ref(ind, g_np, axis)
                np.testing.assert_allclose(dx, ref, rtol=1e-4, atol=1e-4)

    def test_long_run_merge(self):
        # A dominant first element makes the whole axis a single run of length
        # 100 (> warp size), exercising the multi-pass warp reduction + tail.
        x_np = np.random.RandomState(3).randn(8, 100).astype("float32")
        x_np[:, 0] = -1e4  # global minimum at position 0 of every row
        g_np = np.random.RandomState(4).randn(8, 100).astype("float32")
        dx, ind = self._grad(x_np, g_np, -1, "int64")
        ref = cum_scatter_add_ref(ind, g_np, -1)
        np.testing.assert_allclose(dx, ref, rtol=1e-4, atol=1e-3)
        # the entire row gradient must collapse onto column 0.
        np.testing.assert_allclose(
            dx[:, 0], g_np.sum(axis=1), rtol=1e-4, atol=1e-3
        )
        np.testing.assert_allclose(dx[:, 1:], 0.0, atol=1e-6)

    # especially for deterministic path to test the results stability
    def test_run_to_run_stable(self):
        rng = np.random.RandomState(9)
        x_np = rng.randint(-3, 4, size=(64, 80)).astype("float32")
        g_np = rng.randn(64, 80).astype("float32")
        first, _ = self._grad(x_np, g_np, -1, "int64")
        for _ in range(3):
            again, _ = self._grad(x_np, g_np, -1, "int64")
            np.testing.assert_array_equal(first, again)


if __name__ == '__main__':
    unittest.main()
