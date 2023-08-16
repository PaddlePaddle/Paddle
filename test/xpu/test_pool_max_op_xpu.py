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
from get_test_cover_info import (
    XPUOpTestWrapper,
    create_test_class,
    get_xpu_op_support_types,
)
from op_test_xpu import XPUOpTest

import paddle

paddle.enable_static()


def max_pool2D_forward_naive(
    x, ksize, strides, paddings, global_pool=False, adaptive=False
):
    N, C, H, W = x.shape
    global_pool = global_pool or (adaptive or (ksize[0] * ksize[1] == 1))
    if global_pool:
        ksize = [H, W]
        paddings = [0, 0]

    H_out = (H - ksize[0] + 2 * paddings[0]) // strides[0] + 1
    W_out = (W - ksize[1] + 2 * paddings[1]) // strides[1] + 1
    out = np.zeros((N, C, H_out, W_out))
    mask = np.zeros((N, C, H_out, W_out))
    for i in range(H_out):
        for j in range(W_out):
            r0 = i * strides[0] - paddings[0]
            r1 = r0 + ksize[0]
            c0 = j * strides[1] - paddings[1]
            c1 = c0 + ksize[1]
            r_start = np.max((r0, 0))
            r_end = np.min((r1, H))
            c_start = np.max((c0, 0))
            c_end = np.min((c1, W))
            x_masked = x[:, :, r_start:r_end, c_start:c_end]

            out[:, :, i, j] = np.max(x_masked, axis=(2, 3))

            for n in range(N):
                for c in range(C):
                    arr = x_masked[n, c, :, :]
                    index = np.where(arr == np.max(arr))
                    sub_row = index[0][-1] - r0 if r0 < 0 else index[0][-1]
                    sub_col = index[1][-1] - c0 if c0 < 0 else index[1][-1]
                    index = sub_row * (r1 - r0) + sub_col
                    mask[n, c, i, j] = index

    return out, mask


class XPUTestPoolWithIndex_op(XPUOpTestWrapper):
    def __init__(self):
        self.op_name = 'max_pool2d_with_index'
        self.use_dynamic_create_class = False

    class TestMaxPoolWithIndex_Op(XPUOpTest):
        def setUp(self):
            self.op_type = 'max_pool2d_with_index'
            self.dtype = self.in_type
            self.place = paddle.XPUPlace(0)
            self.init_test_case()
            self.init_global()
            self.init_adaptive()

            input = np.random.random(self.shape).astype(self.dtype)
            input = np.round(input * 100.0, 2)
            output, mask = self.pool_forward_naive(
                input,
                self.ksize,
                self.strides,
                self.paddings,
                self.global_pool,
                self.adaptive,
            )
            output = output.astype(self.dtype)
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
            self.check_output_with_place(self.place)

        def test_check_grad(self):
            self.check_grad_with_place(self.place, {'X'}, ['Out'])

        def init_test_case(self):
            self.pool_forward_naive = max_pool2D_forward_naive
            self.shape = [2, 3, 7, 7]
            self.ksize = [3, 3]
            self.strides = [2, 2]
            self.paddings = [1, 1]

        def init_global(self):
            self.global_pool = False

        def init_adaptive(self):
            self.adaptive = False

    # TODO pool3d is not supported for now
    # ----------------max_pool2d_with_index----------------
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


support_types = get_xpu_op_support_types('max_pool2d_with_index')
for stype in support_types:
    create_test_class(globals(), XPUTestPoolWithIndex_op, stype)


if __name__ == '__main__':
    unittest.main()
