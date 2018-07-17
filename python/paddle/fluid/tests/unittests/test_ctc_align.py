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
from test_softmax_op import stable_softmax


def CTCAlign(input, lod, blank, merge_repeated):
    lod0 = lod[0]
    result = []
    cur_offset = 0
    for i in range(len(lod0)):
        prev_token = -1
        for j in range(cur_offset, cur_offset + lod0[i]):
            token = input[j][0]
            if (token != blank) and not (merge_repeated and
                                         token == prev_token):
                result.append(token)
            prev_token = token
        cur_offset += lod0[i]
    result = np.array(result).reshape([len(result), 1]).astype("int32")
    if len(result) == 0:
        result = np.array([-1])
    return result


class TestCTCAlignOp(OpTest):
    def config(self):
        self.op_type = "ctc_align"
        self.input_lod = [[11, 7]]
        self.blank = 0
        self.merge_repeated = False
        self.input = np.array(
            [0, 1, 2, 2, 0, 4, 0, 4, 5, 0, 6, 6, 0, 0, 7, 7, 7, 0]).reshape(
                [18, 1]).astype("int32")

    def setUp(self):
        self.config()
        output = CTCAlign(self.input, self.input_lod, self.blank,
                          self.merge_repeated)

        self.inputs = {"Input": (self.input, self.input_lod), }
        self.outputs = {"Output": output}
        self.attrs = {
            "blank": self.blank,
            "merge_repeated": self.merge_repeated
        }

    def test_check_output(self):
        self.check_output()
        pass


class TestCTCAlignOpCase1(TestCTCAlignOp):
    def config(self):
        self.op_type = "ctc_align"
        self.input_lod = [[11, 8]]
        self.blank = 0
        self.merge_repeated = True
        self.input = np.array(
            [0, 1, 2, 2, 0, 4, 0, 4, 5, 0, 6, 6, 0, 0, 7, 7, 7, 0, 0]).reshape(
                [19, 1]).astype("int32")


class TestCTCAlignOpCase2(TestCTCAlignOp):
    def config(self):
        self.op_type = "ctc_align"
        self.input_lod = [[4]]
        self.blank = 0
        self.merge_repeated = True
        self.input = np.array([0, 0, 0, 0]).reshape([4, 1]).astype("int32")


if __name__ == "__main__":
    unittest.main()
