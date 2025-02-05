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

import os
import unittest

import collective.test_communication_api_base as test_base

os.environ['FLAGS_enable_pir_api'] = '0'


class TestSemiAutoParallelGlobalInput(test_base.CommunicationTestDistBase):
    def setUp(self):
        super().setUp(
            num_of_devices=8,
            timeout=120,
            nnode=1,
        )
        self._default_envs = {
            "dtype": "float32",
            "seed": "1024",
        }
        self._changeable_envs = {"backend": ["gpu"]}

    def test_dynamic(self):
        self._default_envs.update({"run_static": "0"})
        envs_list = test_base.gen_product_envs_list(
            self._default_envs, self._changeable_envs
        )
        for envs in envs_list:
            self.run_test_case(
                "semi_auto_parallel_global_input.py",
                user_defined_envs=envs,
            )

    def test_static(self):
        self._default_envs.update({"run_static": "1"})
        envs_list = test_base.gen_product_envs_list(
            self._default_envs, self._changeable_envs
        )
        for envs in envs_list:
            self.run_test_case(
                "semi_auto_parallel_global_input.py",
                user_defined_envs=envs,
            )


if __name__ == "__main__":
    unittest.main()
