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
from paddle.pir.core import create_parameter

np.random.seed(42)
paddle.enable_static()


class TestVitAttentionPattern(PassTest):
    r'''
    x        w
    |        |
      matmul       bias
        |           |
        elementwise_add
             |
            reshape
              |
           transpose
      /       |       \
   slice    slice     slice
     |        |         |
     |        |      transpose
     |        |         |
     |           matmul
     |             |
     |           scale
     |             |
     |           softmax
     \             /
       \         /
          matmul
            |
         transpose
            |
          reshape
    '''

    def vit_attention_fuse_pattern(self):
        ctx = pir.DrrPatternContext()
        pat = ctx.SourcePattern()

        def constraint_function(match_ctx):
            softmax_axis = match_ctx.Int32Attr("axis")
            if softmax_axis != -1 and softmax_axis != 3:
                return False

            if len(match_ctx.Tensor("matmul_out_1").shape) != 3:
                return False

            matmul_1_transpose_x_1 = match_ctx.BoolAttr("transpose_x_1")
            matmul_1_transpose_y_1 = match_ctx.BoolAttr("transpose_y_1")
            if matmul_1_transpose_x_1 or matmul_1_transpose_y_1:
                return False

            matmul_1_transpose_x_2 = match_ctx.BoolAttr("transpose_x_2")
            matmul_1_transpose_y_2 = match_ctx.BoolAttr("transpose_y_2")
            if matmul_1_transpose_x_2 or matmul_1_transpose_y_2:
                return False

            matmul_1_transpose_x_3 = match_ctx.BoolAttr("transpose_x_3")
            matmul_1_transpose_y_3 = match_ctx.BoolAttr("transpose_y_3")
            if matmul_1_transpose_x_3 or matmul_1_transpose_y_3:
                return False

            return True

        # Source Pattern
        matmul_1 = pat.Op(
            "pd_op.matmul",
            {
                "transpose_x": pat.Attr("transpose_x_1"),
                "transpose_y": pat.Attr("transpose_y_1"),
            },
        )
        matmul_2 = pat.Op(
            "pd_op.matmul",
            {
                "transpose_x": pat.Attr("transpose_x_2"),
                "transpose_y": pat.Attr("transpose_y_2"),
            },
        )
        matmul_3 = pat.Op(
            "pd_op.matmul",
            {
                "transpose_x": pat.Attr("transpose_x_3"),
                "transpose_y": pat.Attr("transpose_y_3"),
            },
        )

        add = pat.Op("pd_op.add")

        full_int_array_1 = pat.Op(
            "pd_op.full_int_array",
            {"value": pat.Attr("full_int_array_value_1")},
        )
        reshape_1 = pat.Op("pd_op.reshape")

        full_int_array_2 = pat.Op(
            "pd_op.full_int_array",
            {"value": pat.Attr("full_int_array_value_2")},
        )
        reshape_2 = pat.Op("pd_op.reshape")

        transpose_1 = pat.Op("pd_op.transpose", {"perm": pat.Attr("perm_1")})
        transpose_2 = pat.Op("pd_op.transpose", {"perm": pat.Attr("perm_2")})
        transpose_3 = pat.Op("pd_op.transpose", {"perm": pat.Attr("perm_3")})

        full_int_array_3 = pat.Op(
            "pd_op.full_int_array",
            {"value": pat.Attr("full_int_array_value_3")},
        )
        full_int_array_4 = pat.Op(
            "pd_op.full_int_array",
            {"value": pat.Attr("full_int_array_value_4")},
        )

        slice_1 = pat.Op(
            "pd_op.slice",
            {
                "axes": pat.Attr("axes_1"),
                "infer_flags": pat.Attr("infer_flags_1"),
                "decrease_axis": pat.Attr("decrease_axis_1"),
            },
        )

        full_int_array_5 = pat.Op(
            "pd_op.full_int_array",
            {"value": pat.Attr("full_int_array_value_5")},
        )
        full_int_array_6 = pat.Op(
            "pd_op.full_int_array",
            {"value": pat.Attr("full_int_array_value_6")},
        )

        slice_2 = pat.Op(
            "pd_op.slice",
            {
                "axes": pat.Attr("axes_2"),
                "infer_flags": pat.Attr("infer_flags_2"),
                "decrease_axis": pat.Attr("decrease_axis_2"),
            },
        )

        full_int_array_7 = pat.Op(
            "pd_op.full_int_array",
            {"value": pat.Attr("full_int_array_value_7")},
        )
        full_int_array_8 = pat.Op(
            "pd_op.full_int_array",
            {"value": pat.Attr("full_int_array_value_8")},
        )

        slice_3 = pat.Op(
            "pd_op.slice",
            {
                "axes": pat.Attr("axes_3"),
                "infer_flags": pat.Attr("infer_flags_3"),
                "decrease_axis": pat.Attr("decrease_axis_3"),
            },
        )

        full_1 = pat.Op("pd_op.full", {"value": pat.Attr("full_1_value")})

        scale = pat.Op(
            "pd_op.scale",
            {
                "bias": pat.Attr("scale_bias"),
                "bias_after_scale": pat.Attr("bias_after_scale"),
            },
        )

        softmax = pat.Op("pd_op.softmax", {"axis": pat.Attr("axis")})

        matmul_1(
            [pat.Tensor("x1"), pat.Tensor("w1")],
            [pat.Tensor("matmul_out_1")],
        )

        add(
            [pat.Tensor("matmul_out_1"), pat.Tensor("bias")],
            [pat.Tensor("add_1_out")],
        )

        full_int_array_1([], [pat.Tensor("full_int_array_1")])
        reshape_1(
            [
                pat.Tensor("add_1_out"),
                pat.Tensor("full_int_array_1"),
            ],
            [pat.Tensor("reshape_1_out")],
        )

        transpose_1(
            [pat.Tensor("reshape_1_out")],
            [pat.Tensor("transpose_1_out")],
        )

        full_int_array_3([], [pat.Tensor("full_int_array_3")])
        full_int_array_4([], [pat.Tensor("full_int_array_4")])
        slice_1(
            [
                pat.Tensor("transpose_1_out"),
                pat.Tensor("full_int_array_3"),
                pat.Tensor("full_int_array_4"),
            ],
            [pat.Tensor("slice_out_1")],
        )

        full_int_array_5([], [pat.Tensor("full_int_array_5")])
        full_int_array_6([], [pat.Tensor("full_int_array_6")])
        slice_2(
            [
                pat.Tensor("transpose_1_out"),
                pat.Tensor("full_int_array_5"),
                pat.Tensor("full_int_array_6"),
            ],
            [pat.Tensor("slice_out_2")],
        )

        full_int_array_7([], [pat.Tensor("full_int_array_7")])
        full_int_array_8([], [pat.Tensor("full_int_array_8")])
        slice_3(
            [
                pat.Tensor("transpose_1_out"),
                pat.Tensor("full_int_array_7"),
                pat.Tensor("full_int_array_8"),
            ],
            [pat.Tensor("slice_out_3")],
        )

        transpose_2(
            [pat.Tensor("slice_out_3")],
            [pat.Tensor("transpose_2_out")],
        )

        matmul_2(
            [
                pat.Tensor("slice_out_2"),
                pat.Tensor("transpose_2_out"),
            ],
            [pat.Tensor("matmul_out_2")],
        )

        full_1([], [pat.Tensor("full_1")])
        scale(
            [pat.Tensor("matmul_out_2"), pat.Tensor("full_1")],
            [pat.Tensor("scale_out")],
        )

        softmax([pat.Tensor("scale_out")], [pat.Tensor("softmax_out")])

        matmul_3(
            [
                pat.Tensor("softmax_out"),
                pat.Tensor("slice_out_1"),
            ],
            [pat.Tensor("matmul_out_3")],
        )

        transpose_3(
            [pat.Tensor("matmul_out_3")],
            [pat.Tensor("transpose_3_out")],
        )

        full_int_array_2([], [pat.Tensor("full_int_array_2")])
        reshape_2(
            [
                pat.Tensor("transpose_3_out"),
                pat.Tensor("full_int_array_2"),
            ],
            [pat.Tensor("reshape_2_out")],
        )

        pat.AddConstraint(constraint_function)

        # Result Pattern
        res = pat.ResultPattern()

        def compute_reshape_w_shape(match_ctx):
            w1_shape = match_ctx.Tensor("w1").shape
            dim_0 = w1_shape[0]
            dim_2 = w1_shape[1] // 3
            return ([dim_0, 3, dim_2], "vector<int64>")

        reshape_w_shape_attr = res.ComputeAttr(compute_reshape_w_shape)

        res_reshape1 = res.Op("pd_op.reshape", {"shape": reshape_w_shape_attr})
        res_reshape1(
            [res.Tensor("w1")],
            [res.Tensor("reshape_w_out"), res.OutputNoneTensor()],
        )

        def compute_reshape_b_shape(match_ctx):
            bias_shape = match_ctx.Tensor("bias").shape
            dim = bias_shape[0] // 3
            return ([3, dim], "vector<int64>")

        reshape_b_shape_attr = res.ComputeAttr(compute_reshape_b_shape)

        res_reshape2 = res.Op("pd_op.reshape", {"shape": reshape_b_shape_attr})
        res_reshape2(
            [res.Tensor("bias")],
            [
                res.Tensor("reshape_bias_out"),
                res.OutputNoneTensor(),
            ],
        )

        def compute_head_number(match_ctx):
            return (
                match_ctx.Tensor("softmax_out").shape[1],
                "int32",
            )

        head_number = res.ComputeAttr(compute_head_number)

        def compute_alpha(match_ctx):
            return (match_ctx.DoubleAttr("full_1_value"), "float")

        alpha = res.ComputeAttr(compute_alpha)

        multihead_matmul_op = res.Op(
            "pd_op.multihead_matmul",
            {
                "transpose_q": res.BoolAttr(False),
                "transpose_k": res.BoolAttr(False),
                "transpose_v": res.BoolAttr(False),
                "alpha": alpha,
                "head_number": head_number,
            },
        )

        multihead_matmul_op(
            [
                res.Tensor("x1"),
                res.Tensor("reshape_w_out"),
                res.Tensor("reshape_bias_out"),
                res.InputNoneTensor(),
            ],
            [res.Tensor("reshape_2_out")],
        )

        return ctx

    def is_program_valid(self, program):
        return True

    def build_ir_program(self):
        vit_attention_fuse_ctx = self.vit_attention_fuse_pattern()
        for bs in [1]:
            for seq_len in [128]:
                for head_dim in [64]:
                    for num_heads in [12]:
                        with paddle.pir_utils.IrGuard():
                            main_prog = paddle.static.Program()
                            start_prog = paddle.static.Program()
                            with paddle.pir.core.program_guard(
                                main_prog, start_prog
                            ):
                                hidden_dim = head_dim * num_heads
                                x = paddle.static.data(
                                    name='x',
                                    shape=[bs, seq_len, hidden_dim],
                                    dtype='float32',
                                )
                                bias = paddle.static.data(
                                    name='bias',
                                    shape=[3 * hidden_dim],
                                    dtype='float32',
                                )

                                w = create_parameter(
                                    name="w",
                                    shape=[hidden_dim, 3 * hidden_dim],
                                    dtype='float32',
                                    initializer=paddle.nn.initializer.Assign(
                                        np.random.rand(
                                            hidden_dim, 3 * hidden_dim
                                        ).astype(np.float32)
                                    ),
                                )
                                matmul_out_1 = paddle.matmul(x, w)
                                add_out = paddle.add(matmul_out_1, bias)
                                # bs,seq_len,num_heads,3,head_dim
                                reshape_out_1 = paddle.reshape(
                                    add_out,
                                    shape=[bs, seq_len, 3, num_heads, head_dim],
                                )
                                transpose_out_1 = paddle.transpose(
                                    reshape_out_1, perm=[2, 0, 3, 1, 4]
                                )
                                # bs,num_heads,seq_len,head_dim
                                q = transpose_out_1[0, :, :, :, :]
                                k = transpose_out_1[1, :, :, :, :]
                                v = transpose_out_1[2, :, :, :, :]
                                matmul_out_2 = paddle.matmul(
                                    q, paddle.transpose(k, perm=[0, 1, 3, 2])
                                )
                                scale_out = paddle.scale(
                                    matmul_out_2,
                                    scale=0.125,
                                    bias=0.0,
                                )
                                softmax_out = paddle.nn.functional.softmax(
                                    scale_out
                                )
                                # bs,num_head,seq_len,head_dim
                                matmul_out_3 = paddle.matmul(softmax_out, v)
                                transpose_out_2 = paddle.transpose(
                                    matmul_out_3, perm=[0, 2, 1, 3]
                                )
                                reshape_out_2 = paddle.reshape(
                                    transpose_out_2,
                                    shape=[bs, seq_len, num_heads * head_dim],
                                )
                                out = paddle.assign(reshape_out_2)
                                self.pass_attr_list = [
                                    {
                                        'py_multihead_matmul_fuse_pass': vit_attention_fuse_ctx
                                    }
                                ]
                                self.feeds = {
                                    "x": np.random.random(
                                        (bs, seq_len, hidden_dim)
                                    ).astype("float32")
                                    - 0.5,
                                    "bias": np.random.random(
                                        3 * hidden_dim
                                    ).astype("float32"),
                                }
                                self.fetch_list = [out]
                                self.valid_op_map = {
                                    "pd_op.multihead_matmul": 1,
                                }
                                return [main_prog, start_prog]

    def sample_program(self):
        yield self.build_ir_program(), False

    def setUp(self):
        if core.is_compiled_with_cuda():
            self.places.append(paddle.CUDAPlace(0))

    def test_check_output(self):
        self.check_pass_correct(atol=1e-3, rtol=1e-3)


if __name__ == "__main__":
    unittest.main()
