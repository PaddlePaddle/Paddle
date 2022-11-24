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

from test_distributed_fused_lamb_op_with_clip import run_test
import unittest


class TestDistributedFusedLambGradientMerge(unittest.TestCase):

    def test_gm(self):
<<<<<<< HEAD
        run_test(clip_after_allreduce=True,
                 max_global_norm=-1.0,
                 gradient_merge_steps=2)

    def test_gm_with_fp16_acc_grad(self):
        run_test(clip_after_allreduce=True,
                 max_global_norm=-1.0,
                 gradient_merge_steps=2,
                 use_master_acc_grad=False)
=======
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
>>>>>>> 43b92b633f5d2db98f45d4b9597e5389f6f9712f


if __name__ == "__main__":
    unittest.main()
