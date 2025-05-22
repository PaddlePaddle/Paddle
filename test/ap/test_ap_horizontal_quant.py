# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

import utils

import paddle
import paddle.incubate.cc as pcc
import paddle.incubate.cc.typing as pct


class TestQuantHorizontalFusion(unittest.TestCase):
    def setUp(self):
        paddle.seed(2022)
        self.prepare_data()

    def prepare_data(self):
        self.dtype = "float32"

        self.x_shape = [4, 32, 128]
        self.x = paddle.randn(self.x_shape, dtype=self.dtype)
        self.x.stop_gradient = False

        self.y_shape = [128, 64]
        self.y = paddle.randn(self.y_shape, dtype=self.dtype)
        self.y.stop_gradient = False

    def run_with_pcc(self):
        B = pct.DimVar(self.x_shape[0])
        M = pct.DimVar(self.x_shape[1])
        N = pct.DimVar(self.y_shape[1])
        K = pct.DimVar(self.x_shape[2])
        DType = pct.DTypeVar("T", self.dtype)

        def horizontal_quant_func(
            x: pct.Tensor([B, M, K], DType),
            y: pct.Tensor([K, N], DType),
        ):
            tie_op = pcc.ap.TieOp()
            quant_x_op = pcc.ap.FacadeQuantOp()
            quant_y_op = pcc.ap.FacadeQuantOp()

            tie_out0, tie_out1 = tie_op([x, y])
            x_quanted, x_scale = quant_x_op([tie_out0])
            y_quanted, y_scale = quant_y_op([tie_out1])

            with pcc.fuse.horizontal_component():
                output0 = paddle.nn.functional.relu(x_quanted)

            with pcc.fuse.horizontal_component():
                output1 = paddle.nn.functional.relu(y_quanted)
            return output0, output1, x_scale, y_scale

        fused = pcc.compile(
            horizontal_quant_func,
            ap_path=f"{os.path.dirname(paddle.__file__)}/apy/quant",
        )
        ap_pir_program = utils.get_pir_program(fused, [self.x, self.y])
        self.assertTrue('pd_op.ap_variadic' in ap_pir_program, "fusion failed")
        outs = fused(self.x, self.y)
        return outs

    def test_horizontal_quant(self):
        pcc_outs = self.run_with_pcc()


if __name__ == "__main__":
    unittest.main()
