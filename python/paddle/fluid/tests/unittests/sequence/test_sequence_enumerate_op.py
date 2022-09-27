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
import sys

sys.path.append("../")
from op_test import OpTest


def sequence_enumerate(input_seq, in_lod, win_size, pad_value):
    lod0 = [0]
    for i in range(0, len(in_lod[0])):
        lod0.append(lod0[i] + in_lod[0][i])
    out_seq = []
    for i in range(0, len(lod0) - 1):
        for idx in range(lod0[i], lod0[i + 1]):
            single_seq = []
            for word_idx in range(win_size):
                word_pos = idx + word_idx
                dat = input_seq[word_pos] if word_pos < lod0[i+1] \
                    else pad_value
                single_seq.append(dat)
            out_seq.append(single_seq)
    return out_seq


class TestSequenceEnumerateOp(OpTest):

    def setUp(self):
        self.op_type = "sequence_enumerate"
        self.init_test_case()
        self.inputs = {'X': (self.in_seq, self.lod)}
        self.attrs = {'win_size': self.win_size, 'pad_value': self.pad_value}
        self.outputs = {'Out': (self.out_seq, self.lod)}

    def test_check_output(self):
        self.check_output()

    def init_test_case(self):
        self.in_seq = np.random.randint(0, 10, (30, 1)).astype("int32")
        self.lod = [[9, 4, 11, 6]]
        self.win_size = 2
        self.pad_value = 0
        out_seq = sequence_enumerate(self.in_seq, self.lod, self.win_size,
                                     self.pad_value)
        self.out_seq = np.array(out_seq).astype("int32")


class TesSequenceEnumerateOpInt64(TestSequenceEnumerateOp):

    def init_test_case(self):
        self.in_seq = np.random.randint(0, 10, (30, 1)).astype("int64")
        self.lod = [[9, 4, 11, 6]]
        self.win_size = 2
        self.pad_value = 0
        out_seq = sequence_enumerate(self.in_seq, self.lod, self.win_size,
                                     self.pad_value)
        self.out_seq = np.array(out_seq).astype("int64")


class TestSequenceEnumerateOpLargeWinSize(TestSequenceEnumerateOp):

    def init_test_case(self):
        self.in_seq = np.random.randint(0, 10, (30, 1)).astype("int32")
        self.lod = [[9, 4, 11, 6]]
        self.win_size = 5
        self.pad_value = 0
        out_seq = sequence_enumerate(self.in_seq, self.lod, self.win_size,
                                     self.pad_value)
        self.out_seq = np.array(out_seq).astype("int32")


class TestSequenceEnumerateOpMaxWinSize(TestSequenceEnumerateOp):

    def init_test_case(self):
        self.in_seq = np.random.randint(0, 10, (30, 1)).astype("int32")
        self.lod = [[9, 4, 11, 6]]
        self.win_size = 30
        self.pad_value = 0
        out_seq = sequence_enumerate(self.in_seq, self.lod, self.win_size,
                                     self.pad_value)
        self.out_seq = np.array(out_seq).astype("int32")


class TestSequenceEnumerateOpLargePadValue(TestSequenceEnumerateOp):

    def init_test_case(self):
        self.in_seq = np.random.randint(0, 10, (30, 1)).astype("int32")
        self.lod = [[9, 4, 11, 6]]
        self.win_size = 5
        self.pad_value = 5
        out_seq = sequence_enumerate(self.in_seq, self.lod, self.win_size,
                                     self.pad_value)
        self.out_seq = np.array(out_seq).astype("int32")


class TestSequenceEnumerateOpLargePadValueSeqLen0(TestSequenceEnumerateOp):

    def init_test_case(self):
        self.in_seq = np.random.randint(0, 10, (30, 1)).astype("int32")
        self.lod = [[0, 14, 0, 16, 0]]
        self.win_size = 5
        self.pad_value = 5
        out_seq = sequence_enumerate(self.in_seq, self.lod, self.win_size,
                                     self.pad_value)
        self.out_seq = np.array(out_seq).astype("int32")


if __name__ == "__main__":
    unittest.main()
