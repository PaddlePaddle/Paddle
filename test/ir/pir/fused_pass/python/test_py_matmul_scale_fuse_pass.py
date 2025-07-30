# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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


class TestMatmulScaleFusePattern(PassTest):
    r"""
    x_var   f_var
      \       /
         matmul
           |
          scale
    """

    def matmul_scale_fuse_pass(self):
        ctx = pir.DrrPatternContext()
        pat = ctx.SourcePattern()

        def constraint_function(match_ctx):
            if not pir.value_is_persistable(match_ctx.Tensor("w")):
                return False
            return abs(match_ctx.Attr("bias")) <= 1e-6

        # Source Pattern
        matmul_op = pat.Op(
            "pd_op.matmul",
            {
                "transpose_x": pat.Attr("transpose_x"),
                "transpose_y": pat.Attr("transpose_y"),
            },
        )

        matmul_op(
            [pat.Tensor("x"), pat.Tensor("w")],
            [pat.Tensor("matmul_out")],
        )

        full_op = pat.Op(
            "pd_op.full",
            {
                "shape": pat.Attr("shape"),
                "value": pat.Attr("value"),
                "dtype": pat.Attr("dtype"),
                "place": pat.Attr("place"),
            },
        )

        scale_op = pat.Op(
            "pd_op.scale",
            {
                "bias": pat.Attr("bias"),
                "bias_after_scale": pat.Attr("bias_after_scale"),
            },
        )

        full_op([], [pat.Tensor("full_op")])
        scale_op(
            [pat.Tensor("matmul_out"), pat.Tensor("full_op")],
            [pat.Tensor("scale_out")],
        )

        pat.AddConstraint(constraint_function)

        # Result Pattern
        res = pat.ResultPattern()

        full_op_res = res.Op(
            "pd_op.full",
            {
                "shape": pat.Attr("shape"),
                "value": pat.Attr("value"),
                "dtype": pat.Attr("dtype"),
                "place": pat.Attr("place"),
            },
        )

        scale_op_res = res.Op(
            "pd_op.scale",
            {
                "bias": res.Float32Attr(0.0),
                "bias_after_scale": pat.Attr("bias_after_scale"),
            },
        )

        matmul_op_res = res.Op(
            "pd_op.matmul",
            {
                "transpose_x": pat.Attr("transpose_x"),
                "transpose_y": pat.Attr("transpose_y"),
            },
        )

        full_op_res([], [res.Tensor("full_op_res")])
        scale_op_res(
            [res.Tensor("w"), res.Tensor("full_op_res")],
            [res.Tensor("scale_res_out")],
        )

        matmul_op_res(
            [res.Tensor("x"), res.Tensor("scale_res_out")],
            [res.Tensor("scale_out")],
        )

        return ctx

    def is_program_valid(self, program=None):
        return True

    def sample_program(self):
        matmul_scale_fuse_ctx = self.matmul_scale_fuse_pass()
        for x_shape in [[3, 2]]:
            for w_shape in [[2, 3]]:
                for scale_bias in [1e-7]:
                    for scale_value in [2.0]:
                        for bias_after_scale in [True]:
                            with paddle.pir_utils.IrGuard():
                                main_prog = paddle.static.Program()
                                start_prog = paddle.static.Program()
                                with paddle.static.program_guard(
                                    main_prog, start_prog
                                ):
                                    x = paddle.static.data(
                                        name='x', shape=x_shape, dtype='float32'
                                    )
                                    w = paddle.static.data(
                                        name='w', shape=w_shape, dtype='float32'
                                    )
                                    out = paddle.scale(
                                        paddle.matmul(x, w),
                                        scale=scale_value,
                                        bias=scale_bias,
                                        bias_after_scale=bias_after_scale,
                                    )
                                    out = paddle.assign(out)
                                    self.pass_attr_list = [
                                        {
                                            'py_matmul_scale_fuse_pass': matmul_scale_fuse_ctx
                                        }
                                    ]
                                    self.feeds = {
                                        "x": np.random.random(x_shape).astype(
                                            "float32"
                                        ),
                                        "w": np.random.random(w_shape).astype(
                                            "float32"
                                        ),
                                    }
                                    self.fetch_list = [out]
                                    self.valid_op_map = {
                                        "pd_op.scale": 1,
                                        "pd_op.matmul": 1,
                                    }
                                    yield [main_prog, start_prog], False

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
