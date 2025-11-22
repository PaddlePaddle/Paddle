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

import paddle
from paddle import nn

TEST_CONFIGS = {
    "2_card_tests": [
        {
            "world_size": 2,
            "tp": 2,
            "dp": 1,
            "sharding_degree": 1,
            "has_bias": "True",
        },
        {
            "world_size": 2,
            "tp": 2,
            "dp": 1,
            "sharding_degree": 1,
            "has_bias": "True",
        },
        {
            "world_size": 2,
            "tp": 2,
            "dp": 1,
            "sharding_degree": 1,
            "has_bias": "False",
        },
        {
            "world_size": 2,
            "tp": 2,
            "dp": 1,
            "sharding_degree": 1,
            "has_bias": "False",
        },
        {
            "world_size": 2,
            "tp": 2,
            "dp": 1,
            "sharding_degree": 1,
            "has_bias": "False",
        },
        {
            "test_type": "layer",
            "layer_type": "ColumnSequenceParallelLinear",
            "world_size": 2,
            "tp": 2,
            "dp": 1,
            "sharding_degree": 1,
            "has_bias": "True",
        },
        {
            "world_size": 2,
            "tp": 2,
            "dp": 1,
            "sharding_degree": 1,
            "has_bias": "True",
        },
        {
            "world_size": 2,
            "tp": 2,
            "sharding_degree": 1,
            "has_bias": "False",
        },
        {
            "world_size": 2,
            "tp": 1,
            "sharding_degree": 2,
            "has_bias": "False",
        },
        {
            "world_size": 2,
            "tp": 1,
            "sharding_degree": 2,
            "has_bias": "False",
        },
        {
            "world_size": 2,
            "tp": 2,
            "sharding_degree": 1,
            "has_bias": "True",
            "master_weight": "True",
        },
        {
            "world_size": 2,
            "tp": 1,
            "sharding_degree": 2,
            "has_bias": "True",
            "master_weight": "True",
        },
        {
            "world_size": 2,
            "tp": 1,
            "sharding_degree": 2,
            "has_bias": "True",
            "master_weight": "True",
        },
    ],
    "4_card_tests": [
        {
            "world_size": 4,
            "tp": 4,
            "dp": 1,
            "sharding_degree": 1,
            "has_bias": "True",
        },
        {
            "world_size": 4,
            "tp": 4,
            "dp": 1,
            "sharding_degree": 1,
            "has_bias": "True",
        },
        {
            "world_size": 4,
            "tp": 2,
            "dp": 2,
            "sharding_degree": 1,
            "has_bias": "True",
        },
        {
            "world_size": 4,
            "tp": 2,
            "dp": 2,
            "sharding_degree": 1,
            "has_bias": "True",
        },
    ],
    "4_card_hv_group_tests": [
        {
            "world_size": 4,
            "tp": 2,
            "pp": 2,
            "sharding_degree": 1,
            "has_bias": "True",
            "test_using_hv_group": 1,
        },
    ],
}


class TestFullParamWith2Devices(test_base.CommunicationTestDistBase):
    def setUp(self):
        super().setUp(num_of_devices=2, timeout=240)

    def test_full_param(self):
        for config in TEST_CONFIGS["2_card_tests"]:
            envs = {k: str(v) for k, v in config.items()}
            envs["test_using_hv_group"] = "0"
            self.run_test_case(
                "model_full_param_logic.py",
                user_defined_envs=envs,
            )


class TestFullParamWith4Devices(test_base.CommunicationTestDistBase):
    def setUp(self):
        super().setUp(num_of_devices=4, timeout=240)

    def test_full_param(self):
        for config in TEST_CONFIGS["4_card_tests"]:
            envs = {k: str(v) for k, v in config.items()}
            envs["test_using_hv_group"] = "0"
            self.run_test_case(
                "model_full_param_logic.py",
                user_defined_envs=envs,
            )


class TestFullParamWithSingleDevices(unittest.TestCase):
    class SimpleMLP(nn.Layer):
        def __init__(self, hidden_size=100, has_bias=False):
            super().__init__()
            self.embedding = nn.Embedding(24, hidden_size)
            self.linear1 = nn.Linear(
                hidden_size, hidden_size, bias_attr=has_bias
            )
            self.linear2 = nn.Linear(
                hidden_size, hidden_size, bias_attr=has_bias
            )
            self.llm_head = nn.Linear(hidden_size, 24, bias_attr=False)

        def forward(self, x):
            x = self.embedding(x)
            x = self.linear1(x)
            x = self.linear2(x)
            x = self.llm_head(x)
            return x

    def test_full_param(self):
        self.batch_size = 2
        self.hidden_size = 32
        self.has_bias = True
        model = self.SimpleMLP(
            hidden_size=self.hidden_size, has_bias=self.has_bias
        )
        model = paddle.amp.decorate(
            models=model, optimizers=None, level="O2", dtype="float16"
        )
        model.train()
        model_state_dict = model.state_dict()

        for k, v in model_state_dict.items():
            ones = paddle.ones_like(v)
            paddle.assign(ones, v)
            if k == "linear1.weight":
                zeros = paddle.zeros_like(v)
                paddle.assign(zeros, v)

        aoa_config = {
            "aoa_statements": [
                "linear1.weight, linear2.weight -> fused_weight, axis=1"
                "embedding.weight -> embedding.weight, dtype = 'float32'"
            ]
        }

        full_param_iter = model.full(aoa_config)
        full_param = dict(full_param_iter)

        param_shape = {
            # "linear1.weight" : [32,32],
            # "linear2.weight" : [32, 32],
            "embedding.weight": [24, 32],
            "linear1.bias": [32],
            "linear2.bias": [32],
            "llm_head.weight": [24, 32],
            "fused_weight": [32, 64],
        }

        for name, shape in param_shape.items():
            if name == "fused_weight":
                continue
            if not self.has_bias:
                if ".bias" in name:
                    continue
            assert name in full_param.keys()
            tensor = full_param[name]
            answer = paddle.ones_like(tensor)
            assert tensor._md5sum() == answer._md5sum()
            if name == "embedding.weight":
                assert tensor.dtype == paddle.float32
        assert "fused_weight" in full_param.keys()
        ones = paddle.ones([32, 32], 'float16')
        zeros = paddle.zeros([32, 32], 'float16')
        answer = paddle.concat([zeros, ones], axis=1)
        assert full_param["fused_weight"]._md5sum() == answer._md5sum()


class TestFullParamHVGroupWith4Devices(test_base.CommunicationTestDistBase):
    def setUp(self):
        super().setUp(num_of_devices=4, timeout=240)

    def test_full_param(self):
        for config in TEST_CONFIGS["4_card_hv_group_tests"]:
            envs = {k: str(v) for k, v in config.items()}
            envs["test_using_hv_group"] = "1"
            self.run_test_case(
                "model_full_param_logic.py",
                user_defined_envs=envs,
            )


if __name__ == "__main__":
    unittest.main()
