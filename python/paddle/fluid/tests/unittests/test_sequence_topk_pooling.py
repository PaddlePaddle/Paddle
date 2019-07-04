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


class TestSequenceTopkPoolingOp(OpTest):
    def setUp(self):
        self.init_op_type()
        self.set_data()
        self.compute()

    def init_op_type(self):
        self.op_type = "sequence_topk_pooling"

    def set_data(self):
        topk = 8
        channel_num = 5
        feature = [32, 16, 48]
        self.init_data(topk, channel_num, feature)

    def init_data(self, topk, channel_num, feature):
        self.attrs = {"topk": topk, "channel_num": channel_num}
        numel = sum(feature) * channel_num
        x_data = np.random.random((numel, )).astype('float32')
        x_lod = [[x * channel_num for x in feature]]
        self.inputs = {'X': (x_data, x_lod)}

    def compute(self):
        x_data, x_lod = self.inputs['X']
        topk = self.attrs['topk']
        channel_num = self.attrs['channel_num']
        out = np.zeros((0, ), dtype=x_data.dtype)
        pos = np.zeros((0, ), dtype='int32')
        out_lod = [[topk * channel_num] * len(x_lod[0])]

        offset = 0
        for idx in range(len(x_lod[0])):
            x_len = x_lod[0][idx]
            self.assertTrue(x_len % channel_num == 0,
                            "x_len: %s can't mod channel_num: %s" %
                            (x_len, channel_num))
            feature = x_len / channel_num
            for ch in range(channel_num):
                x_sub = x_data[offset:(offset + feature)]
                topk_val, topk_pos = self.get_topk(x_sub, topk)
                out = np.hstack((out, topk_val))
                pos = np.hstack((pos, topk_pos))

                offset += feature

        out = out.reshape([-1, topk * channel_num]).astype('float32')
        self.outputs = {'Out': (out, out_lod), 'pos': pos}

    def get_topk(self, x, topk):
        real_topk = topk if topk < len(x) else len(x)
        topk_pos = np.array(x).argsort()[-topk:][::-1]
        topk_val = np.array(x)[topk_pos]
        if real_topk < topk:
            topk_pos = np.hstack((topk_pos, np.full((topk - real_topk, ), -1)))
            topk_val = np.hstack((topk_val, np.full((topk - real_topk, ), 0.0)))

        return topk_val, topk_pos

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(['X'], 'Out', max_relative_error=0.005)


class TestSequenceTopkPoolingOpCase1(TestSequenceTopkPoolingOp):
    def set_data(self):
        topk = 48
        channel_num = 5
        feature = [32, 16, 48]
        self.init_data(topk, channel_num, feature)


if __name__ == '__main__':
    unittest.main()
