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

import os
import subprocess
import sys
import unittest


def strategy_test(saving, loading, gather_to):
    cmd = f"{sys.executable} dygraph_dist_save_load.py --test_case {saving}:{loading} --gather_to {gather_to}"
    p = subprocess.Popen(cmd.split())
    p.communicate()
    assert p.poll() == 0


class TestDistSaveLoad(unittest.TestCase):
    def test_dygraph_save_load_dp_sharding_stage2(self):
        strategy_test("dp", "sharding_stage2", 0)
        strategy_test("dp", "sharding_stage2", 1)
        strategy_test("sharding_stage2", "dp", 1)


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
    unittest.main()
