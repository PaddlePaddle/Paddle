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


class TestProcessMeshPass(test_base.CommunicationTestDistBase):
    def setUp(self):
        super().setUp(
            num_of_devices=2,
            timeout=150,
        )
        self._default_envs = {
            "FLAGS_cudnn_deterministic": "1",
            "FLAGS_enable_pir_api": "1",
        }
        self._changeable_envs = {
            "backend": ["gpu"],
        }

    def test_process_mesh(self):
        envs_list = test_base.gen_product_envs_list(
            self._default_envs, self._changeable_envs
        )
        test_files = [
            "fleet_test_dp.py",
            "fleet_test_mp.py",
            "fleet_test_pp.py",
            "fleet_test_sep.py",
            "fleet_test_sharding.py",
            "process_mesh_demo_unittest.py",
        ]
        for envs in envs_list:
            for test_file in test_files:
                self.run_test_case(
                    test_file,
                    user_defined_envs=envs,
                )


if __name__ == "__main__":
    unittest.main()
