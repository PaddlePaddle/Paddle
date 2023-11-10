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

import unittest

import collective.test_communication_api_base as test_base


class TestSemiAutoParallelInFramework(test_base.CommunicationTestDistBase):
    def setUp(self):
        super().setUp(
            num_of_devices=2,
            timeout=120,
        )
        self._default_envs = {
            "dtype": "float32",
            "seed": "2023",
        }
        self._changeable_envs = {"backend": ["cpu", "gpu"]}

    def test_simple_net_single_strategy_with_gradient_hook(self):
        self._changeable_envs = {"backend": ["gpu"]}
        envs_list = test_base.gen_product_envs_list(
            self._default_envs, self._changeable_envs
        )
        for envs in envs_list:
            self.run_test_case(
                "semi_auto_parallel_simple_net_gradient_hook.py",
                user_defined_envs=envs,
            )

    def test_simple_net_clear_gradient(self):
        envs_list = test_base.gen_product_envs_list(
            self._default_envs, self._changeable_envs
        )
        for envs in envs_list:
            self.run_test_case(
                "semi_auto_parallel_simple_net_clear_gradient.py",
                user_defined_envs=envs,
            )

    def test_simple_net_several_grad_api(self):
        envs_list = test_base.gen_product_envs_list(
            self._default_envs, self._changeable_envs
        )
        for envs in envs_list:
            self.run_test_case(
                "semi_auto_parallel_simple_net_grad_api.py",
                user_defined_envs=envs,
            )

    def test_simple_net_empty_grad(self):
        envs_list = test_base.gen_product_envs_list(
            self._default_envs, self._changeable_envs
        )
        for envs in envs_list:
            self.run_test_case(
                "semi_auto_parallel_simple_net_fill_zero_for_emtpy_grad.py",
                user_defined_envs=envs,
            )

    def test_simple_net_zero_grads(self):
        envs_list = test_base.gen_product_envs_list(
            self._default_envs, self._changeable_envs
        )
        for envs in envs_list:
            self.run_test_case(
                "semi_auto_parallel_simple_net_zero_grads.py",
                user_defined_envs=envs,
            )

    def test_simple_net_custom_relu(self):
        self._changeable_envs = {"backend": ["gpu"]}
        envs_list = test_base.gen_product_envs_list(
            self._default_envs, self._changeable_envs
        )
        for envs in envs_list:
            self.run_test_case(
                "semi_auto_parallel_simple_net_custom_relu.py",
                user_defined_envs=envs,
            )


if __name__ == "__main__":
    unittest.main()
