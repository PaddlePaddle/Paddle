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


def strategy_test(saving, seed=1024, loading="static"):
    cmd = f"{sys.executable} dygraph_save_for_auto_infer.py --test_case {saving}:{loading} --cmd main --seed {seed}"
    p = subprocess.Popen(cmd.split())
    p.communicate()
    assert p.poll() == 0


class TestHybrid(unittest.TestCase):
    def test_dygraph_save_load_dp_sharding_stage2(self):
        strategy_test("dp")
        strategy_test("mp")
        strategy_test("pp")


class TestSharding(unittest.TestCase):
    def test_dygraph_save_load_dp_sharding_stage2(self):
        strategy_test("sharding_stage2")
        strategy_test("sharding_stage3")


class TestSingleCard(unittest.TestCase):
    def test_dygraph_save_load_dp_sharding_stage2(self):
        strategy_test("single")


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
    unittest.main()
