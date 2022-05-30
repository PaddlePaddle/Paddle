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


class TestDistributedFusedLambWithoutClip(unittest.TestCase):
    def test_1(self):
        run_test(clip_after_allreduce=True, max_global_norm=-1.0)

    def test_2(self):
        run_test(clip_after_allreduce=False, max_global_norm=-1.0)


if __name__ == "__main__":
    unittest.main()
