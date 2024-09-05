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

paddle.enable_static()


class TestAddLayernormXpuFusePattern(PassTest):
    def is_program_valid(self, program):
        return True

    def build_ir_program(self):
        with paddle.pir_utils.IrGuard():
            main_prog = paddle.static.Program()
            start_prog = paddle.static.Program()
            with paddle.pir.core.program_guard(main_prog, start_prog):
                q = paddle.static.data(
                    name='q', shape=[16, 64, 256], dtype='float32'
                )
                k = paddle.static.data(
                    name='k', shape=[16, 64, 256], dtype='float32'
                )
                v = paddle.static.data(
                    name='v', shape=[16, 64, 256], dtype='float32'
                )
                reshape_q = paddle.reshape(q, shape=[0, 0, 8, 32])
                reshape_k = paddle.reshape(k, shape=[0, 0, 8, 32])
                reshape_v = paddle.reshape(v, shape=[0, 0, 8, 32])
                transpos_q = paddle.transpose(reshape_q, perm=[0, 2, 1, 3])
                transpos_k = paddle.transpose(reshape_k, perm=[0, 2, 1, 3])
                transpos_v = paddle.transpose(reshape_v, perm=[0, 2, 1, 3])
                qk_matmul = paddle.matmul(
                    transpos_q, transpos_k, transpose_x=False, transpose_y=True
                )
                qk_scale = paddle.scale(qk_matmul, scale=1.0 / np.sqrt(32))
                qk_softmax = paddle.nn.functional.softmax(qk_scale, axis=-1)
                qkv_matmul = paddle.matmul(
                    qk_softmax, transpos_v, transpose_x=False, transpose_y=False
                )
                qkv_transpose = paddle.transpose(qkv_matmul, perm=[0, 2, 1, 3])
                qkv_out = paddle.reshape(qkv_transpose, shape=[0, 0, 256])

                qkv_out = paddle.assign(qkv_out)

                self.pass_attr_list = [{'decoder_attention_xpu_fuse_pass': {}}]
                self.feeds = {
                    "q": np.random.random((16, 64, 256)).astype("float32"),
                    "k": np.random.random((16, 64, 256)).astype("float32"),
                    "v": np.random.random((16, 64, 256)).astype("float32"),
                }
                self.fetch_list = [qkv_out]
                self.valid_op_map = {
                    "pd_op.reshape": 0,
                    "pd_op.transpose": 0,
                    "pd_op.matmul": 0,
                    "pd_op.scale": 0,
                    "pd_op.softmax": 0,
                    "pd_op.qkv_attention_xpu": 1,
                }
                return [main_prog, start_prog]

    def setUp(self):
        if core.is_compiled_with_xpu():
            self.places.append(paddle.XPUPlace(0))
        self.skip_accuracy_verification = False

    def sample_program(self):
        yield self.build_ir_program(), False

    def test_check_output(self):
        self.check_pass_correct()


if __name__ == "__main__":
    unittest.main()
