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


class TestSemiAutoParallelBasic(test_base.CommunicationTestDistBase):
    def setUp(self):
        super().setUp(
            num_of_devices=2,
            timeout=120,
        )
        self._default_envs = {"dtype": "float32", "seed": "2023"}
        self._changeable_envs = {"backend": ["cpu", "gpu"]}

    """
    def test_matmul_api(self):
        envs_list = test_base.gen_product_envs_list(
            self._default_envs, self._changeable_envs
        )
        for envs in envs_list:
            self.run_test_case(
                "semi_auto_parallel_for_matmul.py",
                user_defined_envs=envs,
            )

    def test_elementwise_api(self):
        envs_list = test_base.gen_product_envs_list(
            self._default_envs, self._changeable_envs
        )
        for envs in envs_list:
            self.run_test_case(
                "semi_auto_parallel_for_elementwise.py",
                user_defined_envs=envs,
            )

    def test_concat_api(self):
        envs_list = test_base.gen_product_envs_list(
            self._default_envs, self._changeable_envs
        )
        for envs in envs_list:
            self.run_test_case(
                "semi_auto_parallel_for_concat.py",
                user_defined_envs=envs,
            )
    """

    def test_layernorm_api(self):
        envs_list = test_base.gen_product_envs_list(
            self._default_envs, self._changeable_envs
        )
        for envs in envs_list:
            self.run_test_case(
                "semi_auto_parallel_for_layernorm.py",
                user_defined_envs=envs,
            )

    """
    def test_reduction_api(self):
        envs_list = test_base.gen_product_envs_list(
            self._default_envs, self._changeable_envs
        )
        for envs in envs_list:
            self.run_test_case(
                "semi_auto_parallel_for_reduction.py",
                user_defined_envs=envs,
            )

    def test_bitwise_api(self):
        envs_list = test_base.gen_product_envs_list(
            {"dtype": "int32", "seed": "2023"}, self._changeable_envs
        )
        for envs in envs_list:
            self.run_test_case(
                "semi_auto_parallel_for_bitwise.py",
                user_defined_envs=envs,
            )

    def test_several_replicated_spmd_api(self):
        envs_list = test_base.gen_product_envs_list(
            self._default_envs, self._changeable_envs
        )
        for envs in envs_list:
            self.run_test_case(
                "semi_auto_parallel_for_replicated_spmd.py",
                user_defined_envs=envs,
            )

    def test_add_n_api(self):
        envs_list = test_base.gen_product_envs_list(
            self._default_envs, self._changeable_envs
        )
        for envs in envs_list:
            self.run_test_case(
                "semi_auto_parallel_for_add_n.py",
                user_defined_envs=envs,
            )

    def test_custom_relu_api(self):
        envs_list = test_base.gen_product_envs_list(
            self._default_envs, self._changeable_envs
        )
        for envs in envs_list:
            self.run_test_case(
                "semi_auto_parallel_for_custom_relu.py",
                user_defined_envs=envs,
            )
    """


if __name__ == "__main__":
    unittest.main()
