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


class TestQKVFusePattern(PassTest):
    r""" """

    def is_program_valid(self, program=None):
        return True

    # def sample_program(self):
    #   bsz = 2
    #   seq_len = 16
    #   num_head = 16
    #   head_dim = 64
    #   dim = num_head * head_dim
    #   x_shape = [bsz*seq_len, dim]
    #   wq_shape = [dim, num_head*head_dim]

    #   with paddle.pir_utils.IrGuard():
    #     start_prog = paddle.static.Program()
    #     main_prog = paddle.static.Program()
    #     with paddle.pir.core.program_guard(
    #         main_prog, start_prog
    #     ):
    #         x = paddle.static.data(
    #             name='x', shape=x_shape, dtype='float16'
    #         )
    #         wq = paddle.static.data(
    #             name='wq', shape=wq_shape, dtype='float16'
    #         )
    #         wk = paddle.static.data(
    #             name='wk', shape=wq_shape, dtype='float16'
    #         )
    #         wv = paddle.static.data(
    #             name='wv', shape=wq_shape, dtype='float16'
    #         )

    #         w_qkv = paddle.concat([wq, wk, wv], axis=1)
    #         x_qkv = paddle.matmul(x, w_qkv)
    #         xq, xk, xv = paddle.split(x_qkv, num_or_sections=3, axis=-1)
    #         xq = paddle.assign(xq)
    #         xk = paddle.assign(xk)
    #         xv = paddle.assign(xv)
    #         self.pass_attr_list = [
    #             {
    #                 'qkv_fuse_pass': {}
    #             }
    #         ]
    #         self.feeds = {
    #             "x": np.random.random(x_shape).astype(
    #                 "float16"
    #             ),
    #             "wq": np.random.random(wq_shape).astype(
    #                 "float16"
    #             ),
    #             "wk": np.random.random(wq_shape).astype(
    #                 "float16"
    #             ),
    #             "wv": np.random.random(wq_shape).astype(
    #                 "float16"
    #             ),
    #         }
    #         self.fetch_list = [xq, xk, xv]
    #         self.valid_op_map = {
    #             "pd_op.concat": 1,
    #             "pd_op.matmul": 1,
    #             "pd_op.split": 0,
    #         }

    #         yield [main_prog, start_prog], False

    def sample_program(self):
        bsz = 2
        seq_len = 16
        num_head = 16
        head_dim = 64
        dim = num_head * head_dim
        x_shape = [bsz * seq_len, dim]
        wq_shape = [dim, num_head * head_dim]

        with paddle.pir_utils.IrGuard():
            start_prog = paddle.static.Program()
            main_prog = paddle.static.Program()
            with paddle.pir.core.program_guard(main_prog, start_prog):
                x = paddle.static.data(name="x", shape=x_shape, dtype="float16")
                wq = paddle.static.data(name="wq", shape=wq_shape, dtype="float16")
                wk = paddle.static.data(name="wk", shape=wq_shape, dtype="float16")
                wv = paddle.static.data(name="wv", shape=wq_shape, dtype="float16")

                xq = paddle.matmul(x, wq)
                xk = paddle.matmul(x, wk)
                xv = paddle.matmul(x, wv)
                xq = paddle.assign(xq)
                xk = paddle.assign(xk)
                xv = paddle.assign(xv)
                self.pass_attr_list = [{"matmul_horizontal_fuse_pass": {}}]
                self.feeds = {
                    "x": np.random.random(x_shape).astype("float16"),
                    "wq": np.random.random(wq_shape).astype("float16"),
                    "wk": np.random.random(wq_shape).astype("float16"),
                    "wv": np.random.random(wq_shape).astype("float16"),
                }
                self.fetch_list = [xq, xk, xv]
                self.valid_op_map = {
                    "pd_op.concat": 1,
                    "pd_op.matmul": 1,
                    "pd_op.split": 1,
                }

                yield [main_prog, start_prog], False

    # def sample_program(self):
    #   bsz = 2
    #   seq_len = 16
    #   num_head = 16
    #   head_dim = 64
    #   dim = num_head * head_dim
    #   x_shape = [bsz*seq_len, dim]
    #   wq_shape = [dim, num_head*head_dim]

    #   with paddle.pir_utils.IrGuard():
    #     start_prog = paddle.static.Program()
    #     main_prog = paddle.static.Program()
    #     with paddle.pir.core.program_guard(
    #         main_prog, start_prog
    #     ):
    #         x = paddle.static.data(
    #             name='x', shape=x_shape, dtype='float16'
    #         )
    #         wq = paddle.static.data(
    #             name='wq', shape=wq_shape, dtype='float16'
    #         )
    #         wk = paddle.static.data(
    #             name='wk', shape=wq_shape, dtype='float16'
    #         )
    #         wv = paddle.static.data(
    #             name='wv', shape=wq_shape, dtype='float16'
    #         )

    #         xq = paddle.matmul(x, wq)
    #         xk = paddle.matmul(x, wk)
    #         xv = paddle.matmul(x, wv)
    #         x_qkv = paddle.concat([xq, xk, xv], axis=1)
    #         x_qkv = paddle.assign(x_qkv)
    #         # xq = paddle.assign(xq)
    #         # xk = paddle.assign(xk)
    #         # xv = paddle.assign(xv)
    #         self.pass_attr_list = [
    #             {
    #                 'qkv_fuse_pass': {}
    #             }
    #         ]
    #         self.feeds = {
    #             "x": np.random.random(x_shape).astype(
    #                 "float16"
    #             ),
    #             "wq": np.random.random(wq_shape).astype(
    #                 "float16"
    #             ),
    #             "wk": np.random.random(wq_shape).astype(
    #                 "float16"
    #             ),
    #             "wv": np.random.random(wq_shape).astype(
    #                 "float16"
    #             ),
    #         }
    #         self.fetch_list = [x_qkv]  # [xq, xk, xv]
    #         self.valid_op_map = {
    #             "pd_op.concat": 1,
    #             "pd_op.matmul": 1,
    #             "pd_op.split": 0,
    #         }

    #         yield [main_prog, start_prog], False

    def setUp(self):
        if core.is_compiled_with_cuda():
            self.places.append(paddle.CUDAPlace(0))

    def test_check_output(self):
        self.check_pass_correct(atol=2e-3, rtol=2e-3)


if __name__ == "__main__":
    unittest.main()
