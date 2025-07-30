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

import collective.test_communication_api_base as test_base


class TestSemiAutoParallelLlamaACCTest(test_base.CommunicationTestDistBase):
    def setUp(self):
        super().setUp(num_of_devices=2, timeout=200, nnode=1)

    def test_simple_net_dp_strategy_recompute(self):
        _default_envs = {
            "dp": "2",
            "mp": "1",
            "pp": "1",
            "FLAGS_embedding_deterministic": "1",
            "FLAGS_cudnn_deterministic": "1",
        }
        _changeable_envs = {
            "backend": ["gpu"],
        }
        envs_list = test_base.gen_product_envs_list(
            _default_envs, _changeable_envs
        )
        for envs in envs_list:
            self.run_test_case(
                "auto_parallel_recompute_pir_pass_unittest.py",
                user_defined_envs=envs,
            )

    def test_simple_net_dp_strategy_refined_recompute(self):
        _default_envs = {
            "dp": "2",
            "mp": "1",
            "pp": "1",
            "FLAGS_embedding_deterministic": "1",
            "FLAGS_cudnn_deterministic": "1",
        }
        _changeable_envs = {
            "backend": ["gpu"],
        }
        envs_list = test_base.gen_product_envs_list(
            _default_envs, _changeable_envs
        )
        for envs in envs_list:
            self.run_test_case(
                "auto_parallel_refined_recompute_pir_pass_unittest.py",
                user_defined_envs=envs,
            )


if __name__ == "__main__":
    unittest.main()
