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

import sys
import unittest

import numpy as np
from op_test import OpTest

sys.path.append("../sequence")
from test_sequence_conv import seqconv


class TestSeqConvEltAddRelu(OpTest):
    def set_conf(self):
        pass

    def setUp(self):
        self.op_type = 'fusion_seqconv_eltadd_relu'
        self.lod = [[6, 4]]
        self.in_fea_size = 16
        self.out_fea_size = 8
        self.context_length = 4
        self.context_stride = 1
        self.context_start = 0
        self.set_conf()

        assert self.context_stride == 1

        T = sum(self.lod[0])
        x = np.random.uniform(-1, 1, [T, self.in_fea_size]).astype('float32')
        w = np.random.uniform(
            -1, 1, [self.in_fea_size * self.context_length, self.out_fea_size]
        ).astype('float32')
        b = np.random.uniform(-2, 1, [1, self.out_fea_size]).astype('float32')
        out = seqconv(x, self.lod, w, self.context_length, self.context_start)
        out = np.maximum(out + b, 0)

        self.inputs = {'X': (x, self.lod), 'Filter': w, 'Bias': b}
        self.attrs = {
            'contextStart': self.context_start,
            'contextLength': self.context_length,
            'contextStride': self.context_stride,
        }
        self.outputs = {'Out': out}

    def test_check_output(self):
        self.check_output(check_dygraph=False)


class TestSeqConvEltAddReluBS1(TestSeqConvEltAddRelu):
    def set_conf(self):
        self.lod = [[10]]


class TestSeqConvEltAddReluBS1Case2(TestSeqConvEltAddRelu):
    def set_conf(self):
        self.lod = [[2]]


class TestSeqConvEltAddReluCase1(TestSeqConvEltAddRelu):
    def set_conf(self):
        self.lod = [[3, 5, 1, 6]]
        self.context_length = 3
        self.context_start = -2


class TestSeqConvEltAddReluCase2(TestSeqConvEltAddRelu):
    def set_conf(self):
        self.lod = [[10, 1, 2, 4, 1, 5, 6]]
        self.in_fea_size = 2
        self.context_length = 4
        self.context_start = -1


class TestSeqConvEltAddReluCase3(TestSeqConvEltAddRelu):
    def set_conf(self):
        self.lod = [[10, 1, 2, 4, 1, 5, 6]]
        self.context_length = 5
        self.context_start = -4


if __name__ == '__main__':
    unittest.main()
