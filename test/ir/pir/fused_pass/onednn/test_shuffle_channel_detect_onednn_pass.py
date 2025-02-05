# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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
from pass_test import PassTest

import paddle

paddle.enable_static()


class TestReshapeTranspoeMatmulFusePatternCase1(PassTest):
    r'''
        x
        |
     reshape
        |
    transpose
        |
     reshape
        |
       out
    '''

    def is_program_valid(self, program=None):
        return True

    def build_ir_program(self):
        with paddle.pir_utils.IrGuard():
            main_prog = paddle.static.Program()
            start_prog = paddle.static.Program()
            with paddle.pir.core.program_guard(main_prog, start_prog):
                x = paddle.static.data(
                    name='x', shape=[5, 64, 2, 3], dtype='float32'
                )

                reshape_out_0 = paddle.reshape(x, shape=[0, 8, -1, 2, 3])
                transpose_out = paddle.transpose(
                    reshape_out_0, perm=[0, 2, 1, 3, 4]
                )
                reshape_out_1 = paddle.reshape(
                    transpose_out, shape=[0, -1, 2, 3]
                )
                out = paddle.assign(reshape_out_1)
                self.pass_attr_list = [{'shuffle_channel_detect_pass': {}}]
                self.feeds = {
                    "x": np.random.random((5, 64, 2, 3)).astype("float32"),
                }
                self.fetch_list = [out]
                self.valid_op_map = {
                    "onednn_op.shuffle_channel": 1,
                    "pd_op.reshape": 0,
                    "pd_op.transpose": 0,
                }
                return [main_prog, start_prog]

    def sample_program(self):
        yield self.build_ir_program(), False

    def setUp(self):
        self.places.append(paddle.CPUPlace())

    def test_check_output(self):
        self.check_pass_correct()


if __name__ == "__main__":
    unittest.main()
