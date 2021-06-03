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

import paddle
import unittest
import numpy as np
import paddle.fluid.core as core
from op_test import OpTest, skip_check_grad_ci
paddle.enable_static()


def AffineGrid(theta, grid_shape):
    n = grid_shape[0]
    h = grid_shape[1]
    w = grid_shape[2]
    h_idx = np.repeat(
        np.linspace(-1, 1, h)[np.newaxis, :], w, axis=0).T[:, :, np.newaxis]
    w_idx = np.repeat(
        np.linspace(-1, 1, w)[np.newaxis, :], h, axis=0)[:, :, np.newaxis]
    grid = np.concatenate(
        [w_idx, h_idx, np.ones([h, w, 1])], axis=2)  # h * w * 3
    grid = np.repeat(grid[np.newaxis, :], n, axis=0)  # n * h * w *3

    ret = np.zeros([n, h * w, 2])
    theta = theta.transpose([0, 2, 1])
    for i in range(len(theta)):
        ret[i] = np.dot(grid[i].reshape([h * w, 3]), theta[i])

    return ret.reshape([n, h, w, 2]).astype("float64")


def getGridPointValue(data, x, y):
    data_shape = data.shape
    N = data_shape[0]
    C = data_shape[1]
    in_H = data_shape[2]
    in_W = data_shape[3]
    out_H = x.shape[1]
    out_W = x.shape[2]

    #out = np.zeros(data_shape, dtype='float64')
    out = np.zeros([N, C, out_H, out_W], dtype='float64')
    for i in range(N):
        for j in range(out_H):
            for k in range(out_W):
                if y[i, j, k] < 0 or y[i, j, k] > in_H - 1 or x[
                        i, j, k] < 0 or x[i, j, k] > in_W - 1:
                    out[i, :, j, k] = 0
                else:
                    out[i, :, j, k] = data[i, :, y[i, j, k], x[i, j, k]]

    return out


def clip(x, min_n, max_n):
    return np.maximum(np.minimum(x, max_n), min_n)


def unnormalizeAndClip(grid_slice, max_val, align_corners, padding_mode):
    if align_corners:
        grid_slice = 0.5 * ((grid_slice.astype('float64') + 1.0) * max_val)
    else:
        grid_slice = 0.5 * (
            (grid_slice.astype('float64') + 1.0) * (max_val + 1)) - 0.5

    if padding_mode == "border":
        grid_slice = clip(grid_slice, 0, max_val)
    elif padding_mode == "reflection":
        double_range = 2 * max_val if align_corners else (max_val + 1) * 2
        grid_abs = np.abs(grid_slice) if align_corners else np.abs(grid_slice +
                                                                   0.5)
        extra = grid_abs - np.floor(grid_abs / double_range) * double_range
        grid_slice = np.minimum(extra, double_range - extra)
        grid_slice = grid_slice if align_corners else clip(grid_slice - 0.5, 0,
                                                           max_val)
    return grid_slice


def GridSampler(data,
                grid,
                align_corners=True,
                mode="bilinear",
                padding_mode="zeros"):
    dims = data.shape
    N = dims[0]
    in_C = dims[1]
    in_H = dims[2]
    in_W = dims[3]

    out_H = grid.shape[1]
    out_W = grid.shape[2]

    x = grid[:, :, :, 0]
    y = grid[:, :, :, 1]
    y_max = in_H - 1
    x_max = in_W - 1

    x = unnormalizeAndClip(x, x_max, align_corners, padding_mode)
    y = unnormalizeAndClip(y, y_max, align_corners, padding_mode)

    if mode == "bilinear":
        x0 = np.floor(x).astype('int32')
        x1 = x0 + 1
        y0 = np.floor(y).astype('int32')
        y1 = y0 + 1

        wa = np.tile(((x1 - x) * (y1 - y)).reshape((N, 1, out_H, out_W)),
                     (1, in_C, 1, 1))
        wb = np.tile(((x1 - x) * (y - y0)).reshape((N, 1, out_H, out_W)),
                     (1, in_C, 1, 1))
        wc = np.tile(((x - x0) * (y1 - y)).reshape((N, 1, out_H, out_W)),
                     (1, in_C, 1, 1))
        wd = np.tile(((x - x0) * (y - y0)).reshape((N, 1, out_H, out_W)),
                     (1, in_C, 1, 1))

        va = getGridPointValue(data, x0, y0)
        vb = getGridPointValue(data, x0, y1)
        vc = getGridPointValue(data, x1, y0)
        vd = getGridPointValue(data, x1, y1)

        out = (wa * va + wb * vb + wc * vc + wd * vd).astype('float64')
    elif mode == "nearest":
        x = np.round(x).astype('int32')
        y = np.round(y).astype('int32')
        out = getGridPointValue(data, x, y)
    return out


class TestGridSamplerOp(OpTest):
    def setUp(self):
        self.use_cudnn = False
        self.numeric_grad_delta = 0.0001
        self.op_type = 'grid_sampler'
        self.align_corners = True
        self.padding_mode = "zeros"
        self.mode = "bilinear"
        self.initTestCase()
        x = np.random.randint(0, 255, self.x_shape).astype('float64')

        theta = np.zeros(self.theta_shape).astype('float64')
        for i in range(self.theta_shape[0]):
            for j in range(2):
                for k in range(3):
                    theta[i, j, k] = np.random.rand(1)[0]
        grid = AffineGrid(theta, self.grid_shape)

        self.inputs = {'X': x, 'Grid': grid}
        self.attrs = {
            'use_cudnn': self.use_cudnn,
            "align_corners": self.align_corners,
            "padding_mode": self.padding_mode,
            "mode": self.mode
        }
        self.outputs = {
            'Output': GridSampler(x, grid, self.align_corners, self.mode,
                                  self.padding_mode)
        }

    def test_check_output(self):
        self.check_output()

    def test_check_grad_normal(self):
        self.check_grad(
            ['X', 'Grid'],
            'Output',
            max_relative_error=0.01,
            numeric_grad_delta=self.numeric_grad_delta)

    def initTestCase(self):
        self.x_shape = (2, 3, 8, 8)
        self.grid_shape = (2, 7, 9, 2)
        self.theta_shape = (2, 2, 3)
        self.align_corners = True
        self.padding_mode = "zeros"
        self.mode = "bilinear"
        self.use_cudnn = False if core.is_compiled_with_rocm() else True


class Case1(TestGridSamplerOp):
    def initTestCase(self):
        self.x_shape = (2, 3, 5, 6)
        self.grid_shape = (2, 8, 9, 2)
        self.theta_shape = (2, 2, 3)
        self.align_corners = False
        self.padding_mode = "zeros"
        self.mode = "bilinear"


class Case1(TestGridSamplerOp):
    def initTestCase(self):
        self.x_shape = (2, 3, 5, 6)
        self.grid_shape = (2, 8, 9, 2)
        self.theta_shape = (2, 2, 3)
        self.align_corners = False
        self.padding_mode = "border"
        self.mode = "bilinear"


class Case2(TestGridSamplerOp):
    def initTestCase(self):
        self.x_shape = (2, 3, 5, 6)
        self.grid_shape = (2, 8, 9, 2)
        self.theta_shape = (2, 2, 3)
        self.align_corners = False
        self.padding_mode = "reflection"
        self.mode = "bilinear"


class Case3(TestGridSamplerOp):
    def initTestCase(self):
        self.x_shape = (2, 3, 5, 6)
        self.grid_shape = (2, 8, 9, 2)
        self.theta_shape = (2, 2, 3)
        self.align_corners = True
        self.padding_mode = "reflection"
        self.mode = "bilinear"


class Case4(TestGridSamplerOp):
    def initTestCase(self):
        self.x_shape = (2, 3, 5, 6)
        self.grid_shape = (2, 8, 9, 2)
        self.theta_shape = (2, 2, 3)
        self.align_corners = False
        self.padding_mode = "reflection"
        self.mode = "nearest"
        self.numeric_grad_delta = 0.0001


@skip_check_grad_ci(reason="'check_grad' on large inputs is too slow, " +
                    "however it is desirable to cover the forward pass")
class LargeInputCase(TestGridSamplerOp):
    def get_places(self):
        places = []
        if core.is_compiled_with_cuda():
            places.append(core.CUDAPlace(0))
        return places

    def initTestCase(self):
        self.no_need_check_grad = True
        self.x_shape = (2, 3, 128, 128)
        self.grid_shape = (2, 130, 130, 2)
        self.theta_shape = (2, 2, 3)
        self.align_corners = False
        self.padding_mode = "reflection"
        self.mode = "bilinear"

    def test_check_grad_normal(self):
        pass


@skip_check_grad_ci(reason="'check_grad' on large inputs is too slow, " +
                    "however it is desirable to cover the forward pass")
class Case5(LargeInputCase):
    def initTestCase(self):
        self.no_need_check_grad = True
        self.x_shape = (2, 3, 128, 128)
        self.grid_shape = (2, 130, 130, 2)
        self.theta_shape = (2, 2, 3)
        self.align_corners = True
        self.padding_mode = "zeros"
        self.mode = "bilinear"
        self.use_cudnn = False if core.is_compiled_with_rocm() else True


if __name__ == "__main__":
    unittest.main()
