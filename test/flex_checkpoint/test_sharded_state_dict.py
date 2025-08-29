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

import unittest

import collective.test_communication_api_base as test_base

TEST_CONFIGS = {
    "2_card_tests": [
        {
            "test_type": "layer",
            "layer_type": "ColumnParallelLinear",
            "world_size": 2,
            "tp": 2,
            "dp": 1,
            "has_bias": "True",
        },
        {
            "test_type": "layer",
            "layer_type": "RowParallelLinear",
            "world_size": 2,
            "tp": 2,
            "dp": 1,
            "has_bias": "True",
        },
        {
            "test_type": "layer",
            "layer_type": "VocabParallelEmbedding",
            "world_size": 2,
            "tp": 2,
            "dp": 1,
            "has_bias": "False",
        },
        {
            "test_type": "layer",
            "layer_type": "ColumnParallelLinear",
            "world_size": 2,
            "tp": 2,
            "dp": 1,
            "has_bias": "False",
        },
        {
            "test_type": "layer",
            "layer_type": "RowParallelLinear",
            "world_size": 2,
            "tp": 2,
            "dp": 1,
            "has_bias": "False",
        },
        {
            "test_type": "layer",
            "layer_type": "ColumnSequenceParallelLinear",
            "world_size": 2,
            "tp": 2,
            "dp": 1,
            "has_bias": "True",
        },
        {
            "test_type": "layer",
            "layer_type": "RowSequenceParallelLinear",
            "world_size": 2,
            "tp": 2,
            "dp": 1,
            "has_bias": "True",
        },
        # {"test_type": "optimizer", "layer_type": "DygraphShardingOptimizer", "world_size": 2, "tp": 1, "dp": 2},
        # {"test_type": "optimizer", "layer_type": "DygraphShardingOptimizerV2", "world_size": 2, "tp": 1, "dp": 2},
    ],
    "4_card_tests": [
        {
            "test_type": "layer",
            "layer_type": "ColumnParallelLinear",
            "world_size": 4,
            "tp": 4,
            "dp": 1,
            "has_bias": "True",
        },
        {
            "test_type": "layer",
            "layer_type": "RowParallelLinear",
            "world_size": 4,
            "tp": 4,
            "dp": 1,
            "has_bias": "True",
        },
        {
            "test_type": "layer",
            "layer_type": "ColumnParallelLinear",
            "world_size": 4,
            "tp": 2,
            "dp": 2,
            "has_bias": "True",
        },
        {
            "test_type": "layer",
            "layer_type": "RowParallelLinear",
            "world_size": 4,
            "tp": 2,
            "dp": 2,
            "has_bias": "True",
        },
    ],
}


class TestParallelLayersWith2Devices(test_base.CommunicationTestDistBase):
    def setUp(self):
        super().setUp(num_of_devices=2, timeout=240)

    def test_metadata(self):
        for config in TEST_CONFIGS["2_card_tests"]:
            envs = {k: str(v) for k, v in config.items()}
            self.run_test_case(
                "sharded_state_dict_logic.py",
                user_defined_envs=envs,
            )


class TestParallelLayersWith4Devices(test_base.CommunicationTestDistBase):
    def setUp(self):
        super().setUp(num_of_devices=4, timeout=240)

    def test_metadata(self):
        for config in TEST_CONFIGS["4_card_tests"]:
            envs = {k: str(v) for k, v in config.items()}
            self.run_test_case(
                "sharded_state_dict_logic.py",
                user_defined_envs=envs,
            )


if __name__ == "__main__":
    unittest.main()
