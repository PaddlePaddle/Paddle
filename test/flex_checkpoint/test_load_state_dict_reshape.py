# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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


class TestLoadStateDictReshape(test_base.CommunicationTestDistBase):
    def setUp(self):
        super().setUp(num_of_devices=2)

    def _run(self, case):
        self.run_test_case(
            "load_state_dict_reshape_logic.py",
            user_defined_envs={"reshape_case": case},
        )

    def test_expert_parallel_to_tensor_parallel(self):
        self._run("ep_to_tp")

    def test_reshape_permute_reshape(self):
        self._run("chain")

    def test_two_reshapes_under_aoa_config_reverse(self):
        self._run("reverse")


if __name__ == "__main__":
    unittest.main()
