#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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
from copy import deepcopy
import sys

sys.path.append("../")
from op_test import OpTest


class TestSequenceTopkAvgPoolingOp(OpTest):

    def setUp(self):
        self.init_op_type()
        self.set_data()
        self.compute()

    def init_op_type(self):
        self.op_type = "sequence_topk_avg_pooling"

    def set_data(self):
        topks = [1, 3, 5]
        channel_num = 3
        dim = 10
        row = [30, 45]
        col = [25, 36]
        self.init_data(topks, channel_num, row, col, dim)
        self.init_data(topks, channel_num, row, col, dim)

    def init_data(self, topks, channel_num, row, col, dim=10):
        self.attrs = {"topks": topks, "channel_num": channel_num}
        feature = [row[i] * col[i] for i in range(len(row))]
        numel = sum(feature) * channel_num
        x_data = np.arange(numel).astype('float32')
        x_lod = [[x * channel_num for x in feature]]
        row_data = np.random.random((sum(row), dim)).astype('float32')
        col_data = np.random.random((sum(col), dim)).astype('float32')
        self.inputs = {
            'X': (x_data, x_lod),
            'ROW': (row_data, [row]),
            'COLUMN': (col_data, [col])
        }

    def calc_gradient(self, pos_data, topks, channel_num, row, col):
        max_k = topks[-1]
        pos_data = pos_data.flatten()
        in_numel = sum([row[i] * col[i] for i in range(len(row))]) * channel_num
        out_numel = sum(row) * len(topks) * channel_num
        gradient = np.zeros(shape=(in_numel), dtype="float32")
        dout_val = 1. / out_numel
        pos_offset, in_offset = 0, 0
        for bs_idx in range(len(row)):  # batch
            row_size = row[bs_idx]
            col_size = col[bs_idx]
            for ch in range(channel_num):  # channel
                for row_idx in range(row_size):  # row
                    in_idx = in_offset + row_idx * col_size
                    pos_idx = pos_offset + row_idx * max_k
                    for k_idx in range(len(topks)):
                        for k in range(topks[k_idx]):
                            if pos_data[pos_idx + k] != -1:
                                gradient[in_idx +
                                         pos_data[pos_idx +
                                                  k]] += dout_val / topks[k_idx]
                in_offset += row_size * col_size
                pos_offset += row_size * max_k
        return gradient

    def compute(self):
        topks = self.attrs['topks']
        max_k = topks[-1]
        x_data, x_lod = self.inputs['X']
        row_data, row_lod = self.inputs['ROW']
        col_data, col_lod = self.inputs['COLUMN']
        channel_num = self.attrs['channel_num']
        out = np.zeros((0, len(topks) * channel_num), dtype=x_data.dtype)
        pos = np.zeros((0, ), dtype='int32')
        out_lod = deepcopy(row_lod)

        offset = 0
        for idx in range(len(x_lod[0])):
            x_len = x_lod[0][idx]
            self.assertTrue(
                x_len == channel_num * row_lod[0][idx] * col_lod[0][idx],
                "x_len: %s can't mod channel_num: %s" % (x_len, channel_num))
            out_tmp = np.zeros((0, ), dtype=x_data.dtype)
            pos_tmp = np.zeros((0, ), dtype='int32')
            for ch in range(channel_num):
                for r_id in range(row_lod[0][idx]):
                    x_sub = x_data[offset:(offset + col_lod[0][idx])]
                    topk_val, topk_pos = self.get_topk(x_sub, max_k)
                    sum_data = self.topk_sum(topk_val, topk_pos, max_k)
                    new_feature = np.array(
                        [sum_data[topk] / topk for topk in topks])
                    out_tmp = np.hstack((out_tmp, new_feature))
                    pos_tmp = np.hstack((pos_tmp, topk_pos))

                    offset += col_lod[0][idx]

            out_tmp = out_tmp.reshape([channel_num, -1,
                                       len(topks)]).transpose(1, 0, 2)
            pos_tmp = pos_tmp.reshape([channel_num, -1,
                                       max_k]).transpose(1, 0, 2)
            out = np.vstack(
                (out, out_tmp.reshape([-1, len(topks) * channel_num])))
            pos = np.hstack((pos, pos_tmp.flatten()))

        self.outputs = {'Out': (out.astype('float32'), out_lod), 'pos': pos}
        self.gradient = self.calc_gradient(pos, topks, channel_num, row_lod[0],
                                           col_lod[0])

    def get_topk(self, x, topk):
        real_topk = topk if topk < len(x) else len(x)
        topk_pos = np.array(x).argsort()[-topk:][::-1]
        topk_val = np.array(x)[topk_pos]
        if real_topk < topk:
            topk_pos = np.hstack((topk_pos, np.full((topk - real_topk, ), -1)))
            topk_val = np.hstack((topk_val, np.full((topk - real_topk, ), 0.0)))

        return topk_val, topk_pos

    def topk_sum(self, x, pos, max_k):
        sum_data = [0.] * (max_k + 1)
        for i in range(1, max_k + 1):
            if pos[i - 1] == -1:
                sum_data[i] = sum_data[i - 1]
            else:
                sum_data[i] = sum_data[i - 1] + x[i - 1]
        return sum_data

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(['X'], 'Out', user_defined_grads=[self.gradient])


class TestSequenceTopkAvgPoolingOpCase1(TestSequenceTopkAvgPoolingOp):

    def set_data(self):
        topks = [2, 3]
        channel_num = 5
        dim = 10
        row = [36]
        col = [48]
        self.init_data(topks, channel_num, row, col, dim)

    def test_api(self):
        import paddle.fluid as fluid
        x = fluid.layers.data(name='x', shape=[1], lod_level=1)
        row = fluid.layers.data(name='row', shape=[10], lod_level=1)
        col = fluid.layers.data(name='col', shape=[10], lod_level=1)
        topk_avg = fluid.contrib.sequence_topk_avg_pooling(input=x,
                                                           row=row,
                                                           col=col,
                                                           topks=[1, 3, 5],
                                                           channel_num=5)

        place = fluid.CPUPlace()
        x_tensor = fluid.create_lod_tensor(
            np.random.rand(45, 1).astype('float32'), [[30, 15]], place)
        row_tensor = fluid.create_lod_tensor(
            np.random.rand(5, 10).astype('float32'), [[2, 3]], place)
        col_tensor = fluid.create_lod_tensor(
            np.random.rand(4, 10).astype('float32'), [[3, 1]], place)

        exe = fluid.Executor(place)
        exe.run(fluid.default_startup_program())
        ret = exe.run(feed={
            'x': x_tensor,
            'row': row_tensor,
            'col': col_tensor
        },
                      fetch_list=[topk_avg],
                      return_numpy=False)


if __name__ == '__main__':
    unittest.main()
