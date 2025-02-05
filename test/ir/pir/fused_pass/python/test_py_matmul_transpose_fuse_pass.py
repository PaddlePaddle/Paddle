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

import os
import unittest

import numpy as np
from pass_test import PassTest

import paddle
from paddle import pir
from paddle.base import core

paddle.enable_static()


class TestMatmulOutTransposeFusePattern(PassTest):
    r"""
    x_var     y_var
       \       /
        \     /
         matmul
           |
       transpose
           |
          out

    x_var   y_var
      \       /
     matmul(tans)
          |
         out

    """

    def matmul_transpose_fuse_pattern(self):
        def cons_function(match_ctx):
            x_shape = match_ctx.Tensor("a").shape
            y_shape = match_ctx.Tensor("b").shape
            if len(x_shape) < 2 or len(y_shape) < 2:
                return False
            perm = match_ctx.VectorInt32Attr("perm")
            perm_size = len(perm)
            for i in range(perm_size - 2):
                if perm[i] != i:
                    return False
            if (perm[perm_size - 1] != perm_size - 2) and (
                perm[perm_size - 2] != perm_size - 1
            ):
                return False
            return True

        ctx = pir.DrrPatternContext()
        pat = ctx.SourcePattern()

        matmul_op = pat.Op(
            "pd_op.matmul",
            {
                "transpose_x": pat.Attr("transpose_x"),
                "transpose_y": pat.Attr("transpose_y"),
            },
        )
        transpose_op = pat.Op("pd_op.transpose", {"perm": pat.Attr("perm")})

        matmul_op(
            [pat.Tensor("a"), pat.Tensor("b")],
            [pat.Tensor("matmul_op_out")],
        )
        transpose_op(
            [pat.Tensor("matmul_op_out")],
            [pat.Tensor("transpose_op_out")],
        )

        pat.AddConstraint(cons_function)

        res = pat.ResultPattern()

        def res_transpose_x(match_ctx):
            return (not match_ctx.BoolAttr("transpose_x"), "bool")

        transpose_x = res.ComputeAttr(res_transpose_x)

        def res_transpose_y(match_ctx):
            return (not match_ctx.BoolAttr("transpose_y"), "bool")

        transpose_y = res.ComputeAttr(res_transpose_y)

        fused_matmul_transpose_op = res.Op(
            "pd_op.matmul",
            {"transpose_x": transpose_y, "transpose_y": transpose_x},
        )

        fused_matmul_transpose_op(
            [res.Tensor("b"), res.Tensor("a")],
            [res.Tensor("transpose_op_out")],
        )

        return ctx

    def is_program_valid(self, program=None):
        return True

    def sample_program(self):
        matmul_transpose_fuse_ctx = self.matmul_transpose_fuse_pattern()
        with paddle.pir_utils.IrGuard():
            main_program = paddle.static.Program()
            start_prog = paddle.static.Program()
            with paddle.static.program_guard(main_program, start_prog):
                perm = [0, 2, 1]
                a = paddle.static.data(
                    name="a", shape=[1, 2, 3], dtype="float32"
                )
                b = paddle.static.data(
                    name="b", shape=[1, 3, 2], dtype="float32"
                )

                matmul_out = paddle.matmul(a, b, name='matmul_out')
                out = paddle.transpose(matmul_out, perm=perm)
                out = paddle.assign(out)

                self.pass_attr_list = [
                    {"py_matmul_transpose_fuse_pass": matmul_transpose_fuse_ctx}
                ]

                self.feeds = {
                    "a": np.random.random([1, 2, 3]).astype("float32"),
                    "b": np.random.random([1, 3, 2]).astype("float32"),
                }
                self.fetch_list = [out]
                self.valid_op_map = {
                    "pd_op.matmul": 1,
                    "pd_op.transpose": 0,
                }
                yield [main_program, start_prog], False

    def setUp(self):
        if (
            os.environ.get('FLAGS_CI_both_cpu_and_gpu', 'False').lower()
            in ['1', 'true', 'on']
            or not core.is_compiled_with_cuda()
        ):
            self.places.append(paddle.CPUPlace())
        if core.is_compiled_with_cuda():
            self.places.append(paddle.CUDAPlace(0))

    def test_check_output(self):
        self.check_pass_correct()


if __name__ == "__main__":
    unittest.main()
