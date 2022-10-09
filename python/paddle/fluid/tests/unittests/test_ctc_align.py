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
import paddle.fluid as fluid
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
                if (token != blank) and not (merge_repeated
                                             and token == prev_token):
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
                if (token != blank) and not (merge_repeated
                                             and token == prev_token):
                    result[i].append(token)
                prev_token = token
            start = len(result[i])
            output_length.append([start])
            for j in range(start, len(input[i])):
                result[i].append(padding)
        result = np.array(result).reshape([len(input),
                                           len(input[0])]).astype("int32")
        output_length = np.array(output_length).reshape([len(input),
                                                         1]).astype("int32")

    return result, output_length


class TestCTCAlignOp(OpTest):

    def config(self):
        self.op_type = "ctc_align"
        self.input_lod = [[11, 7]]
        self.blank = 0
        self.merge_repeated = False
        self.input = np.array(
            [0, 1, 2, 2, 0, 4, 0, 4, 5, 0, 6, 6, 0, 0, 7, 7, 7,
             0]).reshape([18, 1]).astype("int32")

    def setUp(self):
        self.config()
        output = CTCAlign(self.input, self.input_lod, self.blank,
                          self.merge_repeated)

        self.inputs = {
            "Input": (self.input, self.input_lod),
        }
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
            [0, 1, 2, 2, 0, 4, 0, 4, 5, 0, 6, 6, 0, 0, 7, 7, 7, 0,
             0]).reshape([19, 1]).astype("int32")


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
        self.input = np.array([[0, 2, 4, 4, 0, 6, 3, 6, 6, 0, 0],
                               [1, 1, 3, 0, 0, 4, 5, 6, 0, 0,
                                0]]).reshape([2, 11]).astype("int32")
        self.input_length = np.array([[9], [8]]).reshape([2, 1]).astype("int32")

    def setUp(self):
        self.config()
        output, output_length = CTCAlign(self.input, self.input_lod, self.blank,
                                         self.merge_repeated,
                                         self.padding_value, self.input_length)
        self.inputs = {
            "Input": (self.input, self.input_lod),
            "InputLength": self.input_length
        }
        self.outputs = {"Output": output, "OutputLength": output_length}
        self.attrs = {
            "blank": self.blank,
            "merge_repeated": self.merge_repeated,
            "padding_value": self.padding_value
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
        self.input = np.array([[0, 1, 2, 2, 0, 4], [0, 4, 5, 0, 6, 0],
                               [0, 7, 7, 7, 0, 0]]).reshape([3,
                                                             6]).astype("int32")
        self.input_length = np.array([[6], [5],
                                      [4]]).reshape([3, 1]).astype("int32")


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
        self.input = np.array([[0, 1, 2, 2, 0, 4], [0, 4, 5, 0, 6, 0],
                               [0, 7, 7, 7, 0, 0]]).reshape([3,
                                                             6]).astype("int32")
        self.input_length = np.array([[6], [5],
                                      [4]]).reshape([3, 1]).astype("int32")


class TestCTCAlignOpCase5(TestCTCAlignPaddingOp):

    def config(self):
        self.op_type = "ctc_align"
        self.blank = 0
        self.input_lod = []
        self.merge_repeated = False
        self.padding_value = 1
        self.input = np.array([[0, 1, 2, 2, 0, 4], [0, 4, 5, 0, 6, 0],
                               [0, 7, 1, 7, 0, 0]]).reshape([3,
                                                             6]).astype("int32")
        self.input_length = np.array([[6], [5],
                                      [4]]).reshape([3, 1]).astype("int32")


class TestCTCAlignOpApi(unittest.TestCase):

    def test_api(self):
        x = fluid.layers.data('x', shape=[4], dtype='float32')
        y = fluid.layers.ctc_greedy_decoder(x, blank=0)

        x_pad = fluid.layers.data('x_pad', shape=[4, 4], dtype='float32')
        x_pad_len = fluid.layers.data('x_pad_len', shape=[1], dtype='int64')
        y_pad, y_pad_len = fluid.layers.ctc_greedy_decoder(
            x_pad, blank=0, input_length=x_pad_len)

        place = fluid.CPUPlace()
        x_tensor = fluid.create_lod_tensor(
            np.random.rand(8, 4).astype("float32"), [[4, 4]], place)

        x_pad_tensor = np.random.rand(2, 4, 4).astype("float32")
        x_pad_len_tensor = np.array([[4], [4]]).reshape([2, 1]).astype("int64")

        exe = fluid.Executor(place)

        exe.run(fluid.default_startup_program())
        ret = exe.run(feed={
            'x': x_tensor,
            'x_pad': x_pad_tensor,
            'x_pad_len': x_pad_len_tensor
        },
                      fetch_list=[y, y_pad, y_pad_len],
                      return_numpy=False)


class BadInputTestCTCAlignr(unittest.TestCase):

    def test_error(self):
        with fluid.program_guard(fluid.Program()):

            def test_bad_x():
                x = fluid.layers.data(name='x', shape=[8], dtype='int64')
                cost = fluid.layers.ctc_greedy_decoder(input=x, blank=0)

            self.assertRaises(TypeError, test_bad_x)


if __name__ == "__main__":
    paddle.enable_static()
    unittest.main()
