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

import paddle


def CTCAlign(input, lod, blank, merge_repeated, padding=0, input_length=None):
    if input_length is None:
        lod0 = lod[0]
        result = []
        cur_offset = 0
        for i in range(len(lod0)):
            prev_token = -1
            for j in range(cur_offset, cur_offset + lod0[i]):
                token = input[j][0]
                if (token != blank) and not (
                    merge_repeated and token == prev_token
                ):
                    result.append(token)
                prev_token = token
            cur_offset += lod0[i]
        result = np.array(result).reshape([len(result), 1]).astype("int32")
        if len(result) == 0:
            result = np.array([-1])
        return result
    else:
        result = [[] for i in range(len(input))]
        output_length = []
        for i in range(len(input)):
            prev_token = -1
            for j in range(input_length[i][0]):
                token = input[i][j]
                if (token != blank) and not (
                    merge_repeated and token == prev_token
                ):
                    result[i].append(token)
                prev_token = token
            start = len(result[i])
            output_length.append([start])
            for j in range(start, len(input[i])):
                result[i].append(padding)
        result = (
            np.array(result)
            .reshape([len(input), len(input[0])])
            .astype("int32")
        )
        output_length = (
            np.array(output_length).reshape([len(input), 1]).astype("int32")
        )

    return result, output_length


class TestCTCAlignOp(OpTest):
    def config(self):
        self.op_type = "ctc_align"
        self.input_lod = [[11, 7]]
        self.blank = 0
        self.merge_repeated = False
        self.input = (
            np.array([0, 1, 2, 2, 0, 4, 0, 4, 5, 0, 6, 6, 0, 0, 7, 7, 7, 0])
            .reshape([18, 1])
            .astype("int32")
        )

    def setUp(self):
        self.config()
        output = CTCAlign(
            self.input, self.input_lod, self.blank, self.merge_repeated
        )

        self.inputs = {
            "Input": (self.input, self.input_lod),
        }
        self.outputs = {"Output": output}
        self.attrs = {
            "blank": self.blank,
            "merge_repeated": self.merge_repeated,
        }

    def test_check_output(self):
        self.check_output()


class TestCTCAlignOpCase1(TestCTCAlignOp):
    def config(self):
        self.op_type = "ctc_align"
        self.input_lod = [[11, 8]]
        self.blank = 0
        self.merge_repeated = True
        self.input = (
            np.array([0, 1, 2, 2, 0, 4, 0, 4, 5, 0, 6, 6, 0, 0, 7, 7, 7, 0, 0])
            .reshape([19, 1])
            .astype("int32")
        )


class TestCTCAlignOpCase2(TestCTCAlignOp):
    def config(self):
        self.op_type = "ctc_align"
        self.input_lod = [[4]]
        self.blank = 0
        self.merge_repeated = True
        self.input = np.array([0, 0, 0, 0]).reshape([4, 1]).astype("int32")


class TestCTCAlignPaddingOp(OpTest):
    def config(self):
        self.op_type = "ctc_align"
        self.input_lod = []
        self.blank = 0
        self.padding_value = 0
        self.merge_repeated = True
        self.input = (
            np.array(
                [
                    [0, 2, 4, 4, 0, 6, 3, 6, 6, 0, 0],
                    [1, 1, 3, 0, 0, 4, 5, 6, 0, 0, 0],
                ]
            )
            .reshape([2, 11])
            .astype("int32")
        )
        self.input_length = np.array([[9], [8]]).reshape([2, 1]).astype("int32")

    def setUp(self):
        self.config()
        output, output_length = CTCAlign(
            self.input,
            self.input_lod,
            self.blank,
            self.merge_repeated,
            self.padding_value,
            self.input_length,
        )
        self.inputs = {
            "Input": (self.input, self.input_lod),
            "InputLength": self.input_length,
        }
        self.outputs = {"Output": output, "OutputLength": output_length}
        self.attrs = {
            "blank": self.blank,
            "merge_repeated": self.merge_repeated,
            "padding_value": self.padding_value,
        }

    def test_check_output(self):
        self.check_output()


class TestCTCAlignOpCase3(TestCTCAlignPaddingOp):
    def config(self):
        self.op_type = "ctc_align"
        self.blank = 0
        self.input_lod = []
        self.merge_repeated = True
        self.padding_value = 0
        self.input = (
            np.array(
                [[0, 1, 2, 2, 0, 4], [0, 4, 5, 0, 6, 0], [0, 7, 7, 7, 0, 0]]
            )
            .reshape([3, 6])
            .astype("int32")
        )
        self.input_length = (
            np.array([[6], [5], [4]]).reshape([3, 1]).astype("int32")
        )


class TestCTCAlignOpCase4(TestCTCAlignPaddingOp):
    '''
    # test tensor input which has attr input padding_value
    '''

    def config(self):
        self.op_type = "ctc_align"
        self.blank = 0
        self.input_lod = []
        self.merge_repeated = False
        self.padding_value = 0
        self.input = (
            np.array(
                [[0, 1, 2, 2, 0, 4], [0, 4, 5, 0, 6, 0], [0, 7, 7, 7, 0, 0]]
            )
            .reshape([3, 6])
            .astype("int32")
        )
        self.input_length = (
            np.array([[6], [5], [4]]).reshape([3, 1]).astype("int32")
        )


class TestCTCAlignOpCase5(TestCTCAlignPaddingOp):
    def config(self):
        self.op_type = "ctc_align"
        self.blank = 0
        self.input_lod = []
        self.merge_repeated = False
        self.padding_value = 1
        self.input = (
            np.array(
                [[0, 1, 2, 2, 0, 4], [0, 4, 5, 0, 6, 0], [0, 7, 1, 7, 0, 0]]
            )
            .reshape([3, 6])
            .astype("int32")
        )
        self.input_length = (
            np.array([[6], [5], [4]]).reshape([3, 1]).astype("int32")
        )


if __name__ == "__main__":
    paddle.enable_static()
    unittest.main()
