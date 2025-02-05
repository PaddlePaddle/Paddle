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


class TestFusedRotaryPositionEmbeddingPass(PassTest):
    r"""
    k                        k                       cos       position_ids               sin        position_ids                   q             q
   /  \                       \                         |             |                      |              |                      /             / \
slice slice                    \                   squeeze       unsqueeze              squeeze       unsqueeze                   /          slice slice
  |     |                       \                      \             /                      \             /                      /             |     |
  |   scale                      \                        gather_nd                            gather_nd                        /              |   scale
   \     /                        \                           |                                    |                           /               \     /
   concat                          \                      unsqueeze                            unsqueeze                      /                 concat
      \                              \                        / \                                  /\                         /                   /
       \                              \                      /   \                                /  \                       /                   /
        \                              \                    /     \                              /    \                    /                    /
         \                              \                  /       \                            /      \                   /                   /
          \                              \                /         \                          /        \                 /                   /
           \                              \              /           \                        /          \               /                   /
            \                                 multiply                \                      /            \             /                   /
             \                                     \                   \                    /              \           /                   /
              \                                     \                   \                  /                \         /                   /
               \                                     \                   \                /                  \       /                   /
                \                                     \                   \              /                    \     /                   /
                 \                                     \                   \            /                      \   /                   /
                  \                                     \                   \          /                        \ /                   /
                   \                                     \                   \        /                         /\                   /
                    \                                     \                   \      /                         /  \                 /
                     \                                     \                   \    /                         /    \               /
                      \                                     \                   \  /                         /      \             /
                       \                                     \                   \/                         /          multiply
                        \                                     \                  /\                        /              /
                         \                                     \                /  \                      /              /
                          \                                     \              /    \                    /              /
                           \                                     \            /      \                  /              /
                            \                                     \          /        \                /              /
                             \                                     \        /          \              /              /
                              \                                     \      /              multiply                  /
                               \                                     \    /                    \                   /
                                \                                     \  /                      \                 /
                                 \                                     \/                        \               /
                                  \                                    /\                         \             /
                                   \                                  /  \                         \           /
                                    \                                /    \                            add
                                     \                              /      \
                                      \                            /        \
                                       \                          /          \
                                        \                        /            /
                                         \                      /            /
                                          \                    /            /
                                           \                 /             /
                                                multiply                  /
                                                       \                 /
                                                        \               /
                                                         \             /
                                                          \           /
                                                           \         /
                                                               add
  """

    def fused_rotary_position_embedding_pattern(self):
        ctx = pir.DrrPatternContext()
        pat = ctx.SourcePattern()

        def check_axes(axes):
            expected_axes = [0, 2]
            if len(axes) != len(expected_axes):
                return False
            for i in range(len(axes)):
                if axes[i] != expected_axes[i]:
                    return False
            return True

        def check_unsqueeze_axes(axes):
            expected_axes_1 = [-1]
            expected_axes_2 = [2]
            if len(axes) != 1:
                return False
            if axes[0] == expected_axes_1[0] or axes[0] == expected_axes_2[0]:
                return True
            return False

        def constraint_function(match_ctx):
            check_passes = True
            axis = match_ctx.VectorInt64Attr("full_13_value")
            axis_2 = match_ctx.VectorInt64Attr("full_12_value")
            check_passes = (
                check_passes and check_axes(axis) and check_axes(axis_2)
            )

            unsqueeze_axis = match_ctx.VectorInt64Attr("full_11_value")
            unsqueeze_axis_1 = match_ctx.VectorInt64Attr("full_10_value")
            unsqueeze_axis_2 = match_ctx.VectorInt64Attr("full_8_value")
            unsqueeze_axis_3 = match_ctx.VectorInt64Attr("full_9_value")

            check_passes = (
                check_passes
                and check_unsqueeze_axes(unsqueeze_axis)
                and check_unsqueeze_axes(unsqueeze_axis_1)
                and check_unsqueeze_axes(unsqueeze_axis_2)
                and check_unsqueeze_axes(unsqueeze_axis_3)
            )

            return check_passes

        # Source Pattern
        squeeze = pat.Op("pd_op.squeeze")
        squeeze_1 = pat.Op("pd_op.squeeze")
        gather_nd = pat.Op("pd_op.gather_nd")
        gather_nd_1 = pat.Op("pd_op.gather_nd")
        unsqueeze = pat.Op("pd_op.unsqueeze")
        unsqueeze_1 = pat.Op("pd_op.unsqueeze")
        unsqueeze_2 = pat.Op("pd_op.unsqueeze")
        unsqueeze_4 = pat.Op("pd_op.unsqueeze")

        add = pat.Op("pd_op.add")
        add_1 = pat.Op("pd_op.add")
        multiply1 = pat.Op("pd_op.multiply")
        multiply2 = pat.Op("pd_op.multiply")
        multiply3 = pat.Op("pd_op.multiply")
        multiply4 = pat.Op("pd_op.multiply")

        slice_q = pat.Op("pd_op.slice")
        slice_q_1 = pat.Op("pd_op.slice")
        slice_k = pat.Op("pd_op.slice")
        slice_k_1 = pat.Op("pd_op.slice")

        full_op = pat.Op(
            "pd_op.full",
            {
                "shape": pat.Attr("shape"),
                "value": pat.Attr("value"),
                "dtype": pat.Attr("dtype"),
                "place": pat.Attr("place"),
            },
        )
        full_op_1 = pat.Op("pd_op.full", {"value": pat.Attr("full_op_1")})
        full_op_2 = pat.Op("pd_op.full")
        full_op_3 = pat.Op("pd_op.full")

        scale_op = pat.Op("pd_op.scale")
        scale_op_k = pat.Op("pd_op.scale")

        full_1 = pat.Op("pd_op.full_int_array")
        full_2 = pat.Op("pd_op.full_int_array")
        full_3 = pat.Op("pd_op.full_int_array")
        full_4 = pat.Op("pd_op.full_int_array")
        full_5 = pat.Op("pd_op.full_int_array")
        full_6 = pat.Op("pd_op.full_int_array")
        full_7 = pat.Op("pd_op.full_int_array")
        full_8 = pat.Op(
            "pd_op.full_int_array", {"value": pat.Attr("full_8_value")}
        )
        full_9 = pat.Op(
            "pd_op.full_int_array", {"value": pat.Attr("full_9_value")}
        )
        full_10 = pat.Op(
            "pd_op.full_int_array", {"value": pat.Attr("full_10_value")}
        )
        full_11 = pat.Op(
            "pd_op.full_int_array", {"value": pat.Attr("full_11_value")}
        )
        full_12 = pat.Op(
            "pd_op.full_int_array", {"value": pat.Attr("full_12_value")}
        )
        full_13 = pat.Op(
            "pd_op.full_int_array", {"value": pat.Attr("full_13_value")}
        )
        full_14 = pat.Op("pd_op.full_int_array")

        concat_op = pat.Op("pd_op.concat")
        combine = pat.Op("builtin.combine")
        concat_op_k = pat.Op("pd_op.concat")
        combine_k = pat.Op("builtin.combine")

        full_13([], [pat.Tensor("full_13")])
        squeeze(
            [pat.Tensor("cos"), pat.Tensor("full_13")],
            [pat.Tensor("squeeze_out_cos")],
        )

        full_12([], [pat.Tensor("full_12")])
        squeeze_1(
            [pat.Tensor("sin"), pat.Tensor("full_12")],
            [pat.Tensor("squeeze_out_sin")],
        )

        full_11([], [pat.Tensor("full_11")])
        unsqueeze(
            [pat.Tensor("position_ids"), pat.Tensor("full_11")],
            [pat.Tensor("unsqueeze_s_out_cos")],
        )

        gather_nd(
            [
                pat.Tensor("squeeze_out_cos"),
                pat.Tensor("unsqueeze_s_out_cos"),
            ],
            [pat.Tensor("gather_nd_out_cos")],
        )

        gather_nd_1(
            [
                pat.Tensor("squeeze_out_sin"),
                pat.Tensor("unsqueeze_s_out_sin"),
            ],
            [pat.Tensor("gather_nd_out_sin")],
        )

        full_10([], [pat.Tensor("full_10")])
        unsqueeze_1(
            [
                pat.Tensor("gather_nd_out_cos"),
                pat.Tensor("full_10"),
            ],
            [pat.Tensor("unsqueeze_out_cos")],
        )

        full_8([], [pat.Tensor("full_8")])
        unsqueeze_4(
            [pat.Tensor("position_ids"), pat.Tensor("full_8")],
            [pat.Tensor("unsqueeze_s_out_sin")],
        )

        full_9([], [pat.Tensor("full_9")])
        unsqueeze_2(
            [
                pat.Tensor("gather_nd_out_sin"),
                pat.Tensor("full_9"),
            ],
            [pat.Tensor("unsqueeze_out_sin")],
        )

        multiply1(
            [pat.Tensor("q"), pat.Tensor("unsqueeze_out_cos")],
            [pat.Tensor("tmp_25")],
        )

        full_1([], [pat.Tensor("full_1")])
        full_2([], [pat.Tensor("full_2")])
        slice_q(
            [
                pat.Tensor("q"),
                pat.Tensor("full_1"),
                pat.Tensor("full_2"),
            ],
            [pat.Tensor("q_slice_out1")],
        )

        full_3([], [pat.Tensor("full_3")])
        full_4([], [pat.Tensor("full_4")])
        slice_q_1(
            [
                pat.Tensor("q"),
                pat.Tensor("full_3"),
                pat.Tensor("full_4"),
            ],
            [pat.Tensor("q_slice_out2")],
        )

        full_op([], [pat.Tensor("full_op")])
        scale_op(
            [pat.Tensor("q_slice_out2"), pat.Tensor("full_op")],
            [pat.Tensor("scale_out")],
        )

        combine(
            [pat.Tensor("scale_out"), pat.Tensor("q_slice_out1")],
            [pat.Tensor("combine_out")],
        )

        full_op_3([], [pat.Tensor("full_op_3")])
        concat_op(
            [pat.Tensor("combine_out"), pat.Tensor("full_op_3")],
            [pat.Tensor("concat_out")],
        )

        multiply3(
            [
                pat.Tensor("concat_out"),
                pat.Tensor("unsqueeze_out_sin"),
            ],
            [pat.Tensor("tmp_27")],
        )

        add(
            [pat.Tensor("tmp_25"), pat.Tensor("tmp_27")],
            [pat.Tensor("out_q")],
        )

        multiply2(
            [pat.Tensor("k"), pat.Tensor("unsqueeze_out_cos")],
            [pat.Tensor("tmp_29")],
        )

        full_5([], [pat.Tensor("full_5")])
        full_6([], [pat.Tensor("full_6")])
        slice_k(
            [
                pat.Tensor("k"),
                pat.Tensor("full_5"),
                pat.Tensor("full_6"),
            ],
            [pat.Tensor("k_slice_out1")],
        )

        full_7([], [pat.Tensor("full_7")])
        full_14([], [pat.Tensor("full_14")])
        slice_k_1(
            [
                pat.Tensor("k"),
                pat.Tensor("full_7"),
                pat.Tensor("full_14"),
            ],
            [pat.Tensor("k_slice_out2")],
        )

        full_op_1([], [pat.Tensor("full_op_1")])
        scale_op_k(
            [pat.Tensor("k_slice_out2"), pat.Tensor("full_op_1")],
            [pat.Tensor("scale_out_k")],
        )

        combine_k(
            [
                pat.Tensor("scale_out_k"),
                pat.Tensor("k_slice_out1"),
            ],
            [pat.Tensor("combine_out_k")],
        )

        full_op_2([], [pat.Tensor("full_op_2")])
        concat_op_k(
            [
                pat.Tensor("combine_out_k"),
                pat.Tensor("full_op_2"),
            ],
            [pat.Tensor("concat_out_k")],
        )

        multiply4(
            [
                pat.Tensor("concat_out_k"),
                pat.Tensor("unsqueeze_out_sin"),
            ],
            [pat.Tensor("tmp_31")],
        )

        add_1(
            [pat.Tensor("tmp_29"), pat.Tensor("tmp_31")],
            [pat.Tensor("out_k")],
        )

        pat.AddConstraint(constraint_function)

        # Result Pattern
        res = pat.ResultPattern()

        fused_rotary_position_embedding = res.Op(
            "pd_op.fused_rotary_position_embedding",
            {
                "use_neox_rotary_style": res.BoolAttr(False),
                "time_major": res.BoolAttr(False),
                "rotary_emb_base": res.Float32Attr(10000.0),
            },
        )

        fused_rotary_position_embedding(
            [
                res.Tensor("q"),
                res.Tensor("k"),
                res.InputNoneTensor(),
                res.Tensor("sin"),
                res.Tensor("cos"),
                res.Tensor("position_ids"),
            ],
            [
                res.Tensor("out_q"),
                res.Tensor("out_k"),
                res.OutputNoneTensor(),
            ],
        )

        return ctx

    def is_program_valid(self, program=None):
        return True

    def rotate_half(self, x):
        """Rotates half the hidden dims of the input."""
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        x3 = paddle.concat([-x2, x1], axis=-1)
        return x3

    def sample_program(self):
        rope_ctx = self.fused_rotary_position_embedding_pattern()

        q_shape = [2, 8, 2, 16]
        k_shape = [2, 8, 2, 16]
        cos_shape = [1, 8, 1, 16]
        sin_shape = [1, 8, 1, 16]
        position_ids_shape = [2, 8]

        with paddle.pir_utils.IrGuard():
            start_prog = paddle.static.Program()
            main_prog = paddle.static.Program()

            with paddle.pir.core.program_guard(main_prog, start_prog):
                q = paddle.static.data(name="q", shape=q_shape, dtype='float32')
                k = paddle.static.data(name="k", shape=k_shape, dtype='float32')
                cos = paddle.static.data(
                    name="cos", shape=cos_shape, dtype='float32'
                )
                sin = paddle.static.data(
                    name="sin", shape=sin_shape, dtype='float32'
                )
                position_ids = paddle.static.data(
                    name="position_ids", shape=position_ids_shape, dtype='int64'
                )

                cos = cos.squeeze(axis=[0, 2])
                sin = sin.squeeze(axis=[0, 2])
                cos = cos[position_ids].unsqueeze(2)
                sin = sin[position_ids].unsqueeze(2)

                q_embed = (q * cos) + (self.rotate_half(q) * sin)
                k_embed = (k * cos) + (self.rotate_half(k) * sin)

                out1 = paddle.assign(q_embed)
                out2 = paddle.assign(k_embed)

                self.pass_attr_list = [
                    {'py_fused_rotary_position_embedding_pass': rope_ctx}
                ]

                self.feeds = {
                    "q": np.random.random(q_shape).astype("float32"),
                    "k": np.random.random(k_shape).astype("float32"),
                    "cos": np.random.random(cos_shape).astype("float32"),
                    "sin": np.random.random(sin_shape).astype("float32"),
                    "position_ids": np.arange(8, dtype='int64')
                    .reshape(2, 4)
                    .repeat(2, axis=1),
                }

                self.fetch_list = [out1, out2]
                self.valid_op_map = {
                    "pd_op.squeeze": 0,
                    "pd_op.unsqueeze": 0,
                    "pd_op.concat": 0,
                    "pd_op.multiply": 0,
                    "pd_op.add": 0,
                    "pd_op.slice": 0,
                    "pd_op.scale": 0,
                    "builtin.combine": 0,
                    "pd_op.gather_nd": 0,
                    "pd_op.fused_rotary_position_embedding": 1,
                }

                yield [main_prog, start_prog], False

    def setUp(self):
        if core.is_compiled_with_cuda():
            self.places.append(paddle.CUDAPlace(0))

    def test_check_output(self):
        self.check_pass_correct()


if __name__ == "__main__":
    unittest.main()
