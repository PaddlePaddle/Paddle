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
from paddle.base import core


class TestTransposeTRTPattern(PassTest):
    def is_program_valid(self, program=None):
        return True

    def sample_program(self):
        with paddle.pir_utils.IrGuard():
            main_prog = paddle.static.Program()
            start_prog = paddle.static.Program()
            with paddle.pir.core.program_guard(main_prog, start_prog):
                x = paddle.static.data(
                    name='x', shape=[2, 3, 4], dtype='float32'
                )
                perm0 = [1, 0, 2]
                transpose_out = paddle.transpose(x, perm0)
                out = paddle.assign(transpose_out)
                self.pass_attr_list = [{'trt_op_marker_pass': {}}]
                self.feeds = {
                    "x": np.array(
                        [
                            [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]],
                            [
                                [13, 14, 15, 16],
                                [17, 18, 19, 20],
                                [21, 22, 23, 24],
                            ],
                        ]
                    ).astype("float32"),
                }
                self.fetch_list = [out]
                self.valid_op_map = {
                    "pd_op.fusion_transpose_flatten_concat": 0,
                }
                yield [main_prog, start_prog], False

    def setUp(self):
        if core.is_compiled_with_cuda():
            self.places.append(paddle.CUDAPlace(0))
        self.trt_expected_ops = {"pd_op.transpose"}

    def test_check_output(self):
        self.check_pass_correct()


if __name__ == '__main__':
    unittest.main()
