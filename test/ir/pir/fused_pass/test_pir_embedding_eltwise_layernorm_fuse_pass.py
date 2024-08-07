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


class TestFused3EmbeddingEltwiseLayernormPattern(PassTest):
    r'''
    in_var1  emb_var   in_var2   emb_var   in_var3   emb_var
      |        |        |         |        |         |
     lookup_table      lookup_table       lookup_table
          |                 |                  |
       lkt_var           lkt_var            lkt_var
          \                 /                  |
            elementwise_add                    |
                   \                          /
                         elementwise_add
                                 |
                               layer_norm
    '''

    def is_program_valid(self, program):
        return True

    def sample_program(self):
        with paddle.pir_utils.IrGuard():
            main_prog = paddle.static.Program()
            start_prog = paddle.static.Program()
            with paddle.pir.core.program_guard(main_prog, start_prog):
                x1 = paddle.static.data(name='x1', shape=[1, 30], dtype='int64')
                x2 = paddle.static.data(name='x2', shape=[1, 30], dtype='int64')

                embedding1 = paddle.nn.Embedding(512, 768)
                embedding2 = paddle.nn.Embedding(30522, 768)
                embedding3 = paddle.nn.Embedding(2, 768)
                add_out1 = paddle.add(embedding1(x1), embedding2(x1))
                add_out2 = paddle.add(add_out1, embedding3(x2))
                layer_norm = paddle.nn.LayerNorm(add_out2.shape[-1:])
                out = layer_norm(add_out2)
                out = paddle.assign(out)
                self.pass_attr_list = [
                    {'embedding_eltwise_layernorm_fuse_pass': {}}
                ]
                self.feeds = {
                    "x1": np.random.random((1, 30)).astype("int64"),
                    "x2": np.random.random((1, 30)).astype("int64"),
                }
                self.fetch_list = [out]
                self.valid_op_map = {
                    "pd_op.add": 0,
                    "pd_op.layer_norm": 0,
                    "pd_op.embedding": 0,
                    "pd_op.fused_embedding_eltwise_layernorm": 1,
                }
                yield [main_prog, start_prog], False

    def setUp(self):
        if core.is_compiled_with_cuda():
            self.places.append(paddle.CUDAPlace(0))

    def test_check_output(self):
        self.check_pass_correct(atol=1e-3, rtol=1e-3)


class TestFused2EmbeddingEltwiseLayernormPattern(PassTest):
    r'''
    in_var1  emb_var   in_var2   emb_var
      |        |        |         |
     lookup_table      lookup_table
          |                 |
       lkt_var           lkt_var
          \                 /
            elementwise_add
                   |
                layer_norm
    '''

    def is_program_valid(self, program):
        return True

    def sample_program(self):
        with paddle.pir_utils.IrGuard():
            main_prog = paddle.static.Program()
            start_prog = paddle.static.Program()
            with paddle.pir.core.program_guard(main_prog, start_prog):
                x1 = paddle.static.data(name='x1', shape=[1, 30], dtype='int64')

                embedding1 = paddle.nn.Embedding(512, 768)
                embedding2 = paddle.nn.Embedding(30522, 768)

                add_out1 = paddle.add(embedding1(x1), embedding2(x1))
                layer_norm = paddle.nn.LayerNorm(add_out1.shape[-1:])
                out = layer_norm(add_out1)
                out = paddle.assign(out)
                self.pass_attr_list = [
                    {'embedding_eltwise_layernorm_fuse_pass': {}}
                ]
                self.feeds = {
                    "x1": np.random.random((1, 30)).astype("int64"),
                }
                self.fetch_list = [out]
                self.valid_op_map = {
                    "pd_op.add": 0,
                    "pd_op.layer_norm": 0,
                    "pd_op.embedding": 0,
                    "pd_op.fused_embedding_eltwise_layernorm": 1,
                }
                yield [main_prog, start_prog], False

    def setUp(self):
        if core.is_compiled_with_cuda():
            self.places.append(paddle.CUDAPlace(0))

    def test_check_output(self):
        self.check_pass_correct(atol=1e-3, rtol=1e-3)


if __name__ == "__main__":
    unittest.main()
