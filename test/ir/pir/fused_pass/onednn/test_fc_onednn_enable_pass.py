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


class TestFcOneDNNEnablePattern(PassTest):
    r"""
    x     w
     \   /
     matmul  y
        \   /
         add
          |
        [relu]
          |
         out
    """

    def is_program_valid(self, program=None):
        return True

    def sample_program(self):
        for x_shape in [[3, 2]]:
            for w_shape in [[2, 3]]:
                for y_shape in [[3], [1, 3]]:
                    for with_relu in [False, True]:
                        with paddle.pir_utils.IrGuard():
                            start_prog = paddle.static.Program()
                            main_prog = paddle.static.Program()
                            with paddle.pir.core.program_guard(
                                main_prog, start_prog
                            ):
                                x = paddle.static.data(
                                    name='x', shape=x_shape, dtype='float32'
                                )
                                w = paddle.static.data(
                                    name='w', shape=w_shape, dtype='float32'
                                )
                                y = paddle.static.data(
                                    name='y', shape=y_shape, dtype='float32'
                                )
                                if with_relu:
                                    relu_op = paddle.nn.ReLU()
                                    out = relu_op(
                                        paddle.add(paddle.matmul(x, w), y)
                                    )
                                else:
                                    out = paddle.add(paddle.matmul(x, w), y)
                                out = paddle.assign(out)
                                self.pass_attr_list = [
                                    {'matmul_add_act_fuse_pass': {}},
                                    {'fc_onednn_enable_pass': {}},
                                ]
                                self.feeds = {
                                    "x": np.random.random(x_shape).astype(
                                        "float32"
                                    ),
                                    "w": np.random.random(w_shape).astype(
                                        "float32"
                                    ),
                                    "y": np.random.random(y_shape).astype(
                                        "float32"
                                    ),
                                }
                                self.fetch_list = [out]
                                self.valid_op_map = {
                                    "pd_op.add": 0,
                                    "pd_op.relu": 0,
                                    "pd_op.matmul": 0,
                                    "pd_op.fc": 0,
                                    "onednn_op.fc": 1,
                                }

                                yield [main_prog, start_prog], False

    def setUp(self):
        self.places.append(paddle.CPUPlace())

    def test_check_output(self):
        self.check_pass_correct()


if __name__ == "__main__":
    unittest.main()
