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


import random
import unittest

import numpy as np
from amp_base_models import AmpTestBase, build_embedding_model

import paddle
from paddle.fluid import core
from paddle.static import amp

paddle.enable_static()


@unittest.skipIf(
    not core.supports_bfloat16(), "place does not support BF16 evaluation"
)
class TestStaticMasterGradProgramBF16(AmpTestBase):
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
            f"The number of optimizers with multi_precison = True is expected to be {expected_num_mp}, but recieved {actual_num_mp}.",
        )

    def test_amp_bf16_o1(self):
        main_program, startup_program, _, _, _ = build_embedding_model(
            True, "bfloat16", "O1"
        )
        self.assertEqual(main_program.num_blocks, 1)
        self._check_optimizer(main_program, 0)

        amp.debugging.collect_operator_stats(main_program)
        op_stats_list = amp.debugging._get_op_stats_list(main_program)
        expected_bf16_calls = {
            "matmul_v2": 1,
            "elementwise_add": 1,
            "dropout": 1,
            "lookup_table_v2": 0,
            "squared_l2_norm": 0,
            "adamw": 0,
        }
        self._check_op_calls(op_stats_list[0], expected_bf16_calls)

    def amp_bf16_o2(self, use_master_grad):
        main_program, startup_program, _, _, _ = build_embedding_model(
            True, "bfloat16", "O2", use_master_grad=use_master_grad
        )
        self.assertEqual(main_program.num_blocks, 1)

        amp.debugging.collect_operator_stats(main_program)
        op_stats_list = amp.debugging._get_op_stats_list(main_program)
        if use_master_grad:
            expected_bf16_calls = {
                "matmul_v2": 1,
                "elementwise_add": 1,
                "dropout": 1,
                "lookup_table_v2": 0,
                "squared_l2_norm": 0,
                "adamw": 2,
            }
        else:
            expected_bf16_calls = {
                "matmul_v2": 1,
                "elementwise_add": 1,
                "dropout": 1,
                "lookup_table_v2": 0,
                "squared_l2_norm": 2,
                "adamw": 2,
            }
        self._check_optimizer(
            main_program,
            expected_bf16_calls["matmul_v2"]
            + expected_bf16_calls["elementwise_add"],
        )
        self._check_op_calls(op_stats_list[0], expected_bf16_calls)

    def test_amp_bf16_o2(self):
        use_master_grad_list = [False, True]
        for master_grad in use_master_grad_list:
            self.amp_bf16_o2(master_grad)

    def _generate_feed_x(self):
        seed = 0
        paddle.seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        x = np.random.randint(1, 128, size=[1, 32]).astype("int64")
        return x

    def test_compare_o1_and_o2_master_grad(self):
        def _run(place, exe, x_np, max_iters, level, use_master_grad=False):
            (
                main_program,
                startup_program,
                optimizer,
                feed_vars,
                fetch_vars,
            ) = build_embedding_model(
                True, "bfloat16", level, use_master_grad=use_master_grad
            )

            seed = 0
            paddle.seed(seed)
            np.random.seed(seed)
            random.seed(seed)

            losses = self.run_program(
                main_program,
                startup_program,
                optimizer,
                feed_vars,
                fetch_vars,
                place,
                exe,
                x_np,
                max_iters,
                level,
            )
            return losses

        max_iters = 40
        x = self._generate_feed_x()
        place = paddle.CUDAPlace(0)
        exe = paddle.static.Executor(place)
        losses_o1 = _run(place, exe, x, max_iters, 'O1')
        losses_o2_no_master_grad = _run(place, exe, x, max_iters, 'O2')
        losses_o2_master_grad = _run(place, exe, x, max_iters, 'O2', True)

        self.assertNotEqual(
            losses_o1,
            losses_o2_no_master_grad,
            f"loss of o1 and o2 should be equal, but recieved loss o1: {losses_o1}, loss o2: {losses_o2_no_master_grad}",
        )

        self.assertEqual(
            losses_o1,
            losses_o2_master_grad,
            f"loss of o1 and o2 should be equal, but recieved loss o1: {losses_o1}, loss o2: {losses_o2_master_grad}",
        )


if __name__ == '__main__':
    unittest.main()
