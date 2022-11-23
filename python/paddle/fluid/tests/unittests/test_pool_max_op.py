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


def adaptive_start_index(index, input_size, output_size):
    return int(np.floor(index * input_size / output_size))


def adaptive_end_index(index, input_size, output_size):
    return int(np.ceil((index + 1) * input_size / output_size))


def max_pool3D_forward_naive(x,
                             ksize,
                             strides,
                             paddings,
                             global_pool=False,
                             adaptive=False):

    N, C, D, H, W = x.shape
    if global_pool:
        ksize = [D, H, W]
        paddings = [0, 0, 0]

    if adaptive:
        D_out, H_out, W_out = ksize
    else:
        D_out = (D - ksize[0] + 2 * paddings[0]) // strides[0] + 1
        H_out = (H - ksize[1] + 2 * paddings[1]) // strides[1] + 1
        W_out = (W - ksize[2] + 2 * paddings[2]) // strides[2] + 1
    out = np.zeros((N, C, D_out, H_out, W_out))
    mask = np.zeros((N, C, D_out, H_out, W_out))
    for k in range(D_out):
        if adaptive:
            d_start = adaptive_start_index(k, D, ksize[0])
            d_end = adaptive_end_index(k, D, ksize[0])
        else:
            d_start = np.max((k * strides[0] - paddings[0], 0))
            d_end = np.min((k * strides[0] + ksize[0] - paddings[0], D))
        for i in range(H_out):
            if adaptive:
                h_start = adaptive_start_index(i, H, ksize[1])
                h_end = adaptive_end_index(i, H, ksize[1])
            else:
                h_start = np.max((i * strides[1] - paddings[1], 0))
                h_end = np.min((i * strides[1] + ksize[1] - paddings[1], H))
            for j in range(W_out):
                if adaptive:
                    w_start = adaptive_start_index(j, W, ksize[2])
                    w_end = adaptive_end_index(j, W, ksize[2])
                else:
                    w_start = np.max((j * strides[2] - paddings[2], 0))
                    w_end = np.min((j * strides[2] + ksize[2] - paddings[2], W))
                x_masked = x[:, :, d_start:d_end, h_start:h_end, w_start:w_end]

                out[:, :, k, i, j] = np.max(x_masked, axis=(2, 3, 4))

                for n in range(N):
                    for c in range(C):
                        arr = x_masked[n, c, :, :, :]
                        index = np.where(arr == np.max(arr))
                        sub_deep = index[0][0]
                        sub_row = index[1][0]
                        sub_col = index[2][0]
                        index = ((d_start + sub_deep) * H +
                                 (h_start + sub_row)) * W + w_start + sub_col
                        mask[n, c, k, i, j] = index

    return out, mask


def max_pool2D_forward_naive(x,
                             ksize,
                             strides,
                             paddings,
                             global_pool=False,
                             adaptive=False):

    N, C, H, W = x.shape
    if global_pool:
        ksize = [H, W]
        paddings = [0, 0]

    if adaptive:
        H_out, W_out = ksize
    else:
        H_out = (H - ksize[0] + 2 * paddings[0]) // strides[0] + 1
        W_out = (W - ksize[1] + 2 * paddings[1]) // strides[1] + 1
    out = np.zeros((N, C, H_out, W_out))
    mask = np.zeros((N, C, H_out, W_out))
    for i in range(H_out):
        for j in range(W_out):
            if adaptive:
                r_start = adaptive_start_index(i, H, ksize[0])
                r_end = adaptive_end_index(i, H, ksize[0])
                c_start = adaptive_start_index(j, W, ksize[1])
                c_end = adaptive_end_index(j, W, ksize[1])
            else:
                r_start = np.max((i * strides[0] - paddings[0], 0))
                r_end = np.min((i * strides[0] + ksize[0] - paddings[0], H))
                c_start = np.max((j * strides[1] - paddings[1], 0))
                c_end = np.min((j * strides[1] + ksize[1] - paddings[1], W))
            x_masked = x[:, :, r_start:r_end, c_start:c_end]

            out[:, :, i, j] = np.max(x_masked, axis=(2, 3))

            for n in range(N):
                for c in range(C):
                    arr = x_masked[n, c, :, :]
                    index = np.where(arr == np.max(arr))
                    sub_row = index[0][0]
                    sub_col = index[1][0]
                    index = (r_start + sub_row) * W + c_start + sub_col
                    mask[n, c, i, j] = index

    return out, mask


class TestMaxPoolWithIndex_Op(OpTest):

    def setUp(self):
        self.init_test_case()
        self.init_global()
        self.init_adaptive()

        input = np.random.random(self.shape).astype("float64")
        input = np.round(input * 100., 2)
        output, mask = self.pool_forward_naive(input, self.ksize, self.strides,
                                               self.paddings, self.global_pool,
                                               self.adaptive)
        output = output.astype("float64")
        mask = mask.astype("int32")

        self.attrs = {
            'strides': self.strides,
            'paddings': self.paddings,
            'ksize': self.ksize,
            'global_pooling': self.global_pool,
            'adaptive': self.adaptive,
        }

        self.inputs = {'X': input}
        self.outputs = {'Out': output, "Mask": mask}

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(set(['X']), ['Out'])

    def init_test_case(self):
        self.op_type = "max_pool3d_with_index"
        self.pool_forward_naive = max_pool3D_forward_naive
        self.shape = [2, 3, 7, 7, 7]
        self.ksize = [3, 3, 3]
        self.strides = [2, 2, 2]
        self.paddings = [1, 1, 1]

    def init_global(self):
        self.global_pool = False

    def init_adaptive(self):
        self.adaptive = False


class TestCase1(TestMaxPoolWithIndex_Op):

    def init_global(self):
        self.global_pool = True


class TestCase2(TestMaxPoolWithIndex_Op):

    def init_test_case(self):
        self.op_type = "max_pool3d_with_index"
        self.pool_forward_naive = max_pool3D_forward_naive
        self.shape = [2, 3, 7, 7, 7]
        self.ksize = [3, 3, 3]
        self.strides = [2, 2, 2]
        self.paddings = [0, 0, 0]

    def init_global(self):
        self.global_pool = True


class TestCase3(TestCase2):

    def init_global(self):
        self.global_pool = False


#----------------max_pool2d_with_index----------------
class TestCase4(TestMaxPoolWithIndex_Op):

    def init_test_case(self):
        self.op_type = "max_pool2d_with_index"
        self.pool_forward_naive = max_pool2D_forward_naive
        self.shape = [2, 3, 7, 7]
        self.ksize = [3, 3]
        self.strides = [1, 1]
        self.paddings = [1, 1]

    def init_global(self):
        self.global_pool = True


class TestCase5(TestCase4):

    def init_global(self):
        self.global_pool = False


class TestCase6(TestMaxPoolWithIndex_Op):

    def init_test_case(self):
        self.op_type = "max_pool2d_with_index"
        self.pool_forward_naive = max_pool2D_forward_naive
        self.shape = [2, 3, 7, 7]
        self.ksize = [3, 3]
        self.strides = [2, 2]
        self.paddings = [0, 0]

    def init_global(self):
        self.global_pool = True


class TestCase7(TestCase6):

    def init_global(self):
        self.global_pool = False


class TestCastAdaptive2d(TestCase6):

    def init_adaptive(self):
        self.adaptive = True


class TestCastAdaptive3d(TestMaxPoolWithIndex_Op):

    def init_adaptive(self):
        self.adaptive = True


if __name__ == '__main__':
    unittest.main()
