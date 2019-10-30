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

from __future__ import print_function

import unittest
import numpy as np
from op_test import OpTest
import paddle.fluid as fluid


class TestMatchMatrixTensorOp(OpTest):
    def setUp(self):
        self.init_op_type()
        self.set_data()
        self.compute()

    def init_op_type(self):
        self.op_type = "match_matrix_tensor"

    def set_data(self):
        ix, iy, h, dim_t = [5, 8, 3, 4]
        x_lod = [[1, 2, 2]]
        y_lod = [[3, 1, 4]]
        self.init_data(ix, x_lod, iy, y_lod, h, dim_t)

    def init_data(self, ix, x_lod, iy, y_lod, h, dim_t):
        x_data = np.random.random((ix, h)).astype('float32')
        y_data = np.random.random((iy, h)).astype('float32')
        w_data = np.random.random((h, dim_t, h)).astype('float32')
        self.inputs = {'X': (x_data, x_lod), 'Y': (y_data, y_lod), 'W': w_data}
        self.attrs = {'dim_t': dim_t}

    def compute(self):
        x_data, x_lod = self.inputs['X']
        y_data, y_lod = self.inputs['Y']
        # [k, dim_t, k] -> [dim_t, k, k]
        w_data = self.inputs['W'].transpose(1, 0, 2)
        out = np.zeros((0, 1), dtype=x_data.dtype)
        # for x*w
        tmp = np.zeros((0, 1), dtype=x_data.dtype)
        out_lod = [[]]
        tmp_lod = [[]]

        x_offset, y_offset = 0, 0
        for idx in range(len(x_lod[0])):
            x_len = x_lod[0][idx]
            y_len = y_lod[0][idx]
            x_sub = x_data[x_offset:(x_offset + x_len), :]
            y_sub = y_data[y_offset:(y_offset + y_len), :]
            tmp_sub = np.dot(x_sub, w_data)
            tmp = np.vstack((tmp, tmp_sub.reshape(tmp_sub.size, 1)))

            out_sub = np.dot(tmp_sub, y_sub.T).transpose(1, 0, 2)
            out_lod[0].append(out_sub.size)
            out = np.vstack((out, out_sub.reshape(out_sub.size, 1)))

            x_offset += x_len
            y_offset += y_len
        self.outputs = {'Out': (out, out_lod), 'Tmp': tmp}

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(['X', 'Y'], 'Out', max_relative_error=0.005)


class TestMatchMatrixTensorOpCase1(TestMatchMatrixTensorOp):
    def set_data(self):
        ix, iy, h, dim_t = [5, 8, 16, 4]
        x_lod = [[5]]
        y_lod = [[8]]
        self.init_data(ix, x_lod, iy, y_lod, h, dim_t)


class TestMatchMatrixTensorOpCase2(TestMatchMatrixTensorOp):
    def set_data(self):
        ix, iy, h, dim_t = [7, 8, 1, 4]
        x_lod = [[2, 3, 2]]
        y_lod = [[3, 1, 4]]
        self.init_data(ix, x_lod, iy, y_lod, h, dim_t)


class TestMatchMatrixTensorOpCase3(TestMatchMatrixTensorOp):
    def set_data(self):
        ix, iy, h, dim_t = [5, 9, 32, 1]
        x_lod = [[1, 2, 2]]
        y_lod = [[3, 2, 4]]
        self.init_data(ix, x_lod, iy, y_lod, h, dim_t)


class TestMatchMatrixTensorOpCase4(TestMatchMatrixTensorOp):
    def set_data(self):
        ix, iy, h, dim_t = [8, 12, 16, 5]
        x_lod = [[1, 2, 3, 1, 1]]
        y_lod = [[3, 2, 4, 1, 2]]
        self.init_data(ix, x_lod, iy, y_lod, h, dim_t)

    def test_api(self):
        x_lod_tensor = fluid.layers.data(name='x', shape=[10], lod_level=1)
        y_lod_tensor = fluid.layers.data(name='y', shape=[10], lod_level=1)
        out, out_tmp = fluid.contrib.match_matrix_tensor(
            x=x_lod_tensor, y=y_lod_tensor, channel_num=3)

        place = fluid.CPUPlace()
        x_data = np.random.rand(7, 10).astype('float32')
        y_data = np.random.rand(9, 10).astype('float32')
        x = fluid.create_lod_tensor(x_data, [[2, 5]], place)
        y = fluid.create_lod_tensor(y_data, [[3, 6]], place)

        exe = fluid.Executor(place=place)
        exe.run(fluid.default_startup_program())
        ret = exe.run(feed={'x': x,
                            'y': y},
                      fetch_list=[out],
                      return_numpy=False)


if __name__ == '__main__':
    unittest.main()
