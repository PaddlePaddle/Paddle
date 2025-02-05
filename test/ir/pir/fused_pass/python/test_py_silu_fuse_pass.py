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


class TestSiluFusePass(PassTest):
    r""" """

    def fused_silu_pass(self):
        ctx = pir.DrrPatternContext()
        pat = ctx.SourcePattern()
        sigmoid_op = pat.Op("pd_op.sigmoid")
        multiply_op = pat.Op("pd_op.multiply")

        sigmoid_op(
            [pat.Tensor("sigmoid_in")],
            [pat.Tensor("sigmoid_out")],
        )
        multiply_op(
            [pat.Tensor("sigmoid_in"), pat.Tensor("sigmoid_out")],
            [pat.Tensor("multiply_out")],
        )

        # res pattern
        res = pat.ResultPattern()
        swish_op = res.Op("pd_op.swish")
        swish_op(
            [res.Tensor("sigmoid_in")],
            [res.Tensor("multiply_out")],
        )

        return ctx

    def is_program_valid(self, program=None):
        return True

    def build_ir_program(self):
        fused_sliu_ctx = self.fused_silu_pass()
        with paddle.pir_utils.IrGuard():
            main_prog = paddle.static.Program()
            start_prog = paddle.static.Program()
            with paddle.pir.core.program_guard(main_prog, start_prog):
                x = paddle.static.data(
                    name='x', shape=[3, 1, 28, 28], dtype='float32'
                )

                sigmoid_op = paddle.nn.Sigmoid()
                out = paddle.multiply(x, sigmoid_op(x))
                out = paddle.assign(out)
                self.pass_attr_list = [{'py_silu_fuse_pass': fused_sliu_ctx}]
                self.feeds = {
                    "x": np.random.random((3, 1, 28, 28)).astype("float32"),
                }
                self.fetch_list = [out]
                self.valid_op_map = {"pd_op.swish": 1}
                return [main_prog, start_prog]

    def sample_program(self):
        yield self.build_ir_program(), False

    def setUp(self):
        if core.is_compiled_with_cuda():
            self.places.append(paddle.CUDAPlace(0))

    def test_check_output(self):
        self.check_pass_correct()


if __name__ == "__main__":
    unittest.main()
