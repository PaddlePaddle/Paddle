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


import unittest

from amp_base_models import (
    AmpTestBase,
    build_embedding_model,
)

import paddle
from paddle.static import amp

paddle.enable_static()


class TestStaticMasterGradProgramFP16(AmpTestBase):
    def _check_optimizer(self, program, expected_num_mp):
        optimizers = []
        for block in program.blocks:
            for op in block.ops:
                if "Param" in op.input_names and "Grad" in op.input_names:
                    optimizers.append(op)

        actual_num_mp = 0
        for op in optimizers:
            if op.has_attr("multi_precision") and op.attr("multi_precision"):
                actual_num_mp += 1
        self.assertEqual(
            actual_num_mp,
            expected_num_mp,
            f"The number of optimizers with multi_precision = True is expected to be {expected_num_mp}, but received {actual_num_mp}.",
        )

    def amp_fp16_o2(self, use_master_grad):
        main_program, _, _, _, _ = build_embedding_model(
            True, "float16", "O2", use_master_grad=use_master_grad
        )
        self.assertEqual(main_program.num_blocks, 1)

        amp.debugging.collect_operator_stats(main_program)
        op_stats_list = amp.debugging._get_op_stats_list(main_program)
        expected_fp32_calls = {"lookup_table_v2": 1}
        if use_master_grad:
            expected_fp16_calls = {
                "matmul_v2": 1,
                "elementwise_add": 1,
                "dropout": 1,
                "lookup_table_v2": 0,
                "squared_l2_norm": 0,
                "adamw": 3,
            }
        else:
            expected_fp16_calls = {
                "matmul_v2": 1,
                "elementwise_add": 1,
                "dropout": 1,
                "lookup_table_v2": 0,
                "squared_l2_norm": 3,
                "adamw": 3,
            }
        self._check_optimizer(
            main_program,
            expected_fp16_calls["matmul_v2"]
            + expected_fp16_calls["elementwise_add"]
            + expected_fp32_calls["lookup_table_v2"],
        )
        self._check_op_calls(
            op_stats_list[0], expected_fp16_calls=expected_fp16_calls
        )

    def test_amp_fp16_o2(self):
        with paddle.pir_utils.OldIrGuard():
            use_master_grad_list = [False, True]
            for master_grad in use_master_grad_list:
                self.amp_fp16_o2(master_grad)


if __name__ == '__main__':
    unittest.main()
