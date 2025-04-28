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

os.environ["PARALLEL_CROSS_ENTROPY"] = "true"

import os
import unittest

import collective.test_communication_api_base as test_base

os.environ['FLAGS_enable_pir_api'] = '0'


class TestParallelCrossEntropy(test_base.CommunicationTestDistBase):
    def setUp(self):
        super().setUp(num_of_devices=2, timeout=200, nnode=1)

    def test_dp(self):
        self.run_test_case(
            "semi_auto_parallel_c_cross_entropy_dp.py",
        )

    def test_mp(self):
        self.run_test_case(
            "semi_auto_parallel_c_cross_entropy_mp.py",
        )

    def test_mp_pir(self):
        os.environ["FLAGS_enable_pir_in_executor"] = "True"
        self.test_mp()
        os.environ["FLAGS_enable_pir_in_executor"] = "False"


class TestParallelCrossEntropyHybrid(test_base.CommunicationTestDistBase):
    def setUp(self):
        super().setUp(num_of_devices=4, timeout=200, nnode=1)

    def test_hybrid(self):
        self.run_test_case(
            "semi_auto_parallel_c_cross_entropy_hybrid.py",
        )


if __name__ == "__main__":
    unittest.main()
