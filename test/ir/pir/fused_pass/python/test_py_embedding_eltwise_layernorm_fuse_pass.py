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
from paddle import pir
from paddle.base import core

paddle.enable_static()


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

    def fused_2embedding_eltwise_layernorm_pattern(self):
        def cons_function(match_ctx):
            w1_dtype = match_ctx.Tensor("w1").dtype
            w2_dtype = match_ctx.Tensor("w2").dtype
            if w1_dtype != w2_dtype or (
                not w1_dtype == core.DataType.FLOAT16
                and not w1_dtype == core.DataType.FLOAT32
            ):
                return False
            x1_shape = match_ctx.Tensor("x1").shape
            x2_shape = match_ctx.Tensor("x2").shape
            if len(x1_shape) != len(x2_shape):
                return False
            for i in range(len(x1_shape)):
                if x1_shape[i] != x2_shape[i]:
                    return False
            return True

        ctx = pir.DrrPatternContext()
        pat = ctx.SourcePattern()

        embedding1_op = pat.Op("pd_op.embedding")
        embedding2_op = pat.Op("pd_op.embedding")
        add_op = pat.Op("pd_op.add")

        layer_norm_op = pat.Op(
            "pd_op.layer_norm", {"epsilon": pat.Attr("epsilon")}
        )

        embedding1_op(
            [pat.Tensor("x1"), pat.Tensor("w1")],
            [pat.Tensor("embedding_1_out")],
        )
        embedding2_op(
            [pat.Tensor("x2"), pat.Tensor("w2")],
            [pat.Tensor("embedding_2_out")],
        )

        add_op(
            [
                pat.Tensor("embedding_1_out"),
                pat.Tensor("embedding_2_out"),
            ],
            [pat.Tensor("add_out")],
        )
        layer_norm_op(
            [
                pat.Tensor("add_out"),
                pat.Tensor("scale"),
                pat.Tensor("bias"),
            ],
            [
                pat.Tensor("layernorm_out"),
                pat.Tensor("layernorm_mean"),
                pat.Tensor("layernorm_variance"),
            ],
        )

        pat.AddConstraint(cons_function)

        # res pattern
        res = pat.ResultPattern()

        combine_op_1 = res.Op("builtin.combine")
        combine_op_1(
            [res.Tensor("x1"), res.Tensor("x2")],
            [res.Tensor("combine1_out")],
        )

        combine_op_2 = res.Op("builtin.combine")
        combine_op_2(
            [res.Tensor("w1"), res.Tensor("w2")],
            [res.Tensor("combine2_out")],
        )

        def compute_dtype(match_ctx):
            return (match_ctx.Tensor("w1").dtype, "datatype")

        cast_op_dtype = res.ComputeAttr(compute_dtype)

        cast_op_1 = res.Op("pd_op.cast", {"dtype": cast_op_dtype})
        cast_op_2 = res.Op("pd_op.cast", {"dtype": cast_op_dtype})
        fused_embedding_eltwise_layernorm_op = res.Op(
            "pd_op.fused_embedding_eltwise_layernorm",
            {"epsilon": pat.Attr("epsilon")},
        )

        # op forward
        cast_op_1([res.Tensor("bias")], [res.Tensor("casted_bias")])
        cast_op_2([res.Tensor("scale")], [res.Tensor("casted_scale")])
        fused_embedding_eltwise_layernorm_op(
            [
                res.Tensor("combine1_out"),
                res.Tensor("combine2_out"),
                res.Tensor("casted_bias"),
                res.Tensor("casted_scale"),
            ],
            [res.Tensor("layernorm_out")],
        )

        return ctx

    def is_program_valid(self, program):
        return True

    def sample_program(self):
        fused_embedding_ctx = self.fused_2embedding_eltwise_layernorm_pattern()
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
                    {
                        'py_embedding_eltwise_layernorm_fuse_pass': fused_embedding_ctx
                    }
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
