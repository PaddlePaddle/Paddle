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

import unittest
import numpy as np
from op_test import OpTest


def AffineGrid(theta, size):
    n = size[0]
    h = size[2]
    w = size[3]
    h_idx = np.repeat(
        np.linspace(-1, 1, h)[np.newaxis, :], w, axis=0).T[:, :, np.newaxis]
    w_idx = np.repeat(
        np.linspace(-1, 1, w)[np.newaxis, :], h, axis=0)[:, :, np.newaxis]
    grid = np.concatenate(
        [w_idx, h_idx, np.ones([h, w, 1])], axis=2)  # h * w * 3
    grid = np.repeat(grid[np.newaxis, :], size[0], axis=0)  # n * h * w *3

    ret = np.zeros([n, h * w, 2])
    theta = theta.transpose([0, 2, 1])
    for i in range(len(theta)):
        ret[i] = np.dot(grid[i].reshape([h * w, 3]), theta[i])

    return ret.reshape([n, h, w, 2]).astype("float64")


def getGridPointValue(data, x, y):
    data_shape = data.shape
    N = data_shape[0]
    H = data_shape[2]
    W = data_shape[3]

    out = np.zeros(data_shape, dtype='float64')
    for i in range(N):
        for j in range(H):
            for k in range(W):
                if y[i, j, k] < 0 or y[i, j, k] > H - 1 or x[i, j, k] < 0 or x[
                        i, j, k] > W - 1:
                    out[i, :, j, k] = 0
                else:
                    out[i, :, j, k] = data[i, :, y[i, j, k], x[i, j, k]]

    return out


def GridSampler(data, grid):
    dims = data.shape
    N = dims[0]
    C = dims[1]
    H = dims[2]
    W = dims[3]

    x = grid[:, :, :, 0]
    y = grid[:, :, :, 1]
    y_max = H - 1
    x_max = W - 1

    x = 0.5 * ((x.astype('float64') + 1.0) * x_max)
    y = 0.5 * ((y.astype('float64') + 1.0) * y_max)

    x0 = np.floor(x).astype('int32')
    x1 = x0 + 1
    y0 = np.floor(y).astype('int32')
    y1 = y0 + 1

    wa = np.tile(((x1 - x) * (y1 - y)).reshape((N, 1, H, W)), (1, C, 1, 1))
    wb = np.tile(((x1 - x) * (y - y0)).reshape((N, 1, H, W)), (1, C, 1, 1))
    wc = np.tile(((x - x0) * (y1 - y)).reshape((N, 1, H, W)), (1, C, 1, 1))
    wd = np.tile(((x - x0) * (y - y0)).reshape((N, 1, H, W)), (1, C, 1, 1))

    va = getGridPointValue(data, x0, y0)
    vb = getGridPointValue(data, x0, y1)
    vc = getGridPointValue(data, x1, y0)
    vd = getGridPointValue(data, x1, y1)

    out = (wa * va + wb * vb + wc * vc + wd * vd).astype('float64')
    return out


class TestGridSamplerOp(OpTest):
    def setUp(self):
        self.initTestCase()
        self.op_type = 'grid_sampler'
        x = np.random.randint(0, 255, self.x_shape).astype('float64')

        theta = np.zeros(self.theta_shape).astype('float64')
        for i in range(self.theta_shape[0]):
            for j in range(2):
                for k in range(3):
                    theta[i, j, k] = np.random.rand(1)[0]
        grid = AffineGrid(theta, self.x_shape)

        self.inputs = {'X': x, 'Grid': grid}
        self.attrs = {'use_cudnn': True}
        self.outputs = {'Output': GridSampler(x, grid)}

    def test_check_output(self):
        self.check_output()

    def test_check_grad_normal(self):
        self.check_grad(['X', 'Grid'], 'Output', max_relative_error=0.61)

    def initTestCase(self):
        self.x_shape = (2, 5, 7, 3)
        self.grid_shape = (2, 7, 3, 2)
        self.theta_shape = (2, 2, 3)


if __name__ == "__main__":
    unittest.main()
