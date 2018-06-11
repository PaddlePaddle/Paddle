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


def unpool2dmax_forward_naive(input, indices, ksize, strides, paddings):
    s0, s1, s2, s3 = input.shape
    out_hsize = (s2 - 1) * strides[0] - 2 * paddings[0] + ksize[0]
    out_wsize = (s2 - 1) * strides[1] - 2 * paddings[1] + ksize[1]
    out = np.zeros((s0, s1, out_hsize, out_wsize))
    for nidx in xrange(s0):
        for cidx in xrange(s1):
            for h in xrange(s2):
                for w in xrange(s3):
                    index = indices[nidx, cidx, h, w]
                    hidx = (index - index % out_wsize) / out_wsize
                    widx = index % out_wsize
                    out[nidx, cidx, int(hidx), int(widx)] = \
                            input[nidx, cidx, h, w]

    return out


class TestUnpoolOp(OpTest):
    def setUp(self):
        self.op_type = "unpool"
        self.init_test_case()
        pre_input = np.random.random(self.shape).astype("float32")
        nsize, csize, hsize, wsize = pre_input.shape
        hsize_out = (hsize - self.ksize[0] + 2 * self.paddings[0]) / \
                self.strides[0] + 1
        wsize_out = (wsize - self.ksize[1] + 2 * self.paddings[1]) / \
                self.strides[1] + 1
        input = np.zeros((nsize, csize, hsize_out, wsize_out))
        indices = np.zeros((nsize, csize, hsize_out, wsize_out))
        for i in xrange(hsize_out):
            for j in xrange(wsize_out):
                r_start = np.max((i * self.strides[0] - self.paddings[0], 0))
                r_end = np.min((i * self.strides[0] + self.ksize[0] - \
                        self.paddings[0], hsize))
                c_start = np.max((j * self.strides[1] - self.paddings[1], 0))
                c_end = np.min((j * self.strides[1] + self.ksize[1] - \
                        self.paddings[1], wsize))
                for nidx in xrange(nsize):
                    for cidx in xrange(csize):
                        x_masked = pre_input[nidx, cidx, r_start:r_end, \
                                c_start:c_end]
                        input[nidx, cidx, i, j] = x_masked.max()
                        arg = x_masked.argmax()
                        indices[nidx, cidx, i, j] = \
                                (r_start + arg / self.ksize[1]) * wsize + \
                                c_start + arg % self.ksize[1]
        output = self.unpool2d_forward_naive(input, indices, self.ksize, \
                self.strides, self.paddings).astype("float32")
        self.inputs = {
            'X': input.astype('float32'),
            'Indices': indices.astype('int32')
        }
        self.attrs = {
            'strides': self.strides,
            'paddings': self.paddings,
            'ksize': self.ksize,
            'unpooling_type': self.unpooling_type,
        }
        self.outputs = {'Out': output.astype('float32')}

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(['X'], 'Out')

    def init_test_case(self):
        self.unpool2d_forward_naive = unpool2dmax_forward_naive
        self.unpooling_type = "max"
        self.shape = [6, 4, 5, 5]
        self.ksize = [3, 3]
        self.strides = [2, 2]
        self.paddings = [0, 0]


if __name__ == '__main__':
    unittest.main()
