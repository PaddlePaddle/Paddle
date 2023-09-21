# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

from test_distributed_fused_lamb_op_with_clip import run_test


class TestDistributedFusedLambGradientMerge(unittest.TestCase):
    def test_gm(self):
        run_test(
            clip_after_allreduce=True,
            max_global_norm=-1.0,
            gradient_merge_steps=2,
        )

    def test_gm_with_fp16_acc_grad(self):
        run_test(
            clip_after_allreduce=True,
            max_global_norm=-1.0,
            gradient_merge_steps=2,
            use_master_acc_grad=False,
        )

    def test_gm_new_comm(self):
        run_test(
            clip_after_allreduce=True,
            max_global_norm=-1.0,
            gradient_merge_steps=2,
            need_env={"FLAGS_dynamic_static_unified_comm": "1"},
        )

    def test_gm_with_fp16_acc_grad_new_comm(self):
        run_test(
            clip_after_allreduce=True,
            max_global_norm=-1.0,
            gradient_merge_steps=2,
            use_master_acc_grad=False,
            need_env={"FLAGS_dynamic_static_unified_comm": "1"},
        )


if __name__ == "__main__":
    unittest.main()
