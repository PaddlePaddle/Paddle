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
import tempfile
import unittest

import collective.test_communication_api_base as test_base

os.environ['FLAGS_enable_pir_api'] = '0'


class TestSemiAutoParallelDPMPStrategy(test_base.CommunicationTestDistBase):
    def setUp(self):
        super().setUp(num_of_devices=4, timeout=120, nnode=1)
        self._default_envs = {
            "dtype": "float32",
            "seed": "2023",
        }
        self._changeable_envs = {"backend": ["gpu"]}

    def test_simple_net_hybrid_strategy(self):
        envs_list = test_base.gen_product_envs_list(
            self._default_envs, self._changeable_envs
        )
        for envs in envs_list:
            ckpt_path = tempfile.TemporaryDirectory()
            envs["ckpt_path"] = ckpt_path.name
            self.run_test_case(
                "semi_auto_parallel_simple_net_dp_mp.py",
                user_defined_envs=envs,
            )
            ckpt_path.cleanup()

    def test_fused_linear_param_grad_add(self):
        envs_list = test_base.gen_product_envs_list(
            self._default_envs, self._changeable_envs
        )
        for envs in envs_list:
            self.run_test_case(
                "semi_auto_parallel_for_fused_linear_param_grad_add.py",
                user_defined_envs=envs,
            )


class TestSemiAutoParallelHybridStrategy(test_base.CommunicationTestDistBase):
    def setUp(self):
        super().setUp(
            num_of_devices=8,
            timeout=120,
            nnode=1,
        )
        self._default_envs = {
            "dtype": "float32",
            "seed": "2023",
        }
        self._changeable_envs = {"backend": ["gpu"]}

    def test_simple_net_hybrid_strategy(self):
        envs_list = test_base.gen_product_envs_list(
            self._default_envs, self._changeable_envs
        )
        for envs in envs_list:
            ckpt_path = tempfile.TemporaryDirectory()
            envs["ckpt_path"] = ckpt_path.name
            self.run_test_case(
                "semi_auto_parallel_simple_net_dp_mp_pp.py",
                user_defined_envs=envs,
            )
            ckpt_path.cleanup()


class TestSemiAutoParallelHybridStrategyWithSP(
    test_base.CommunicationTestDistBase
):
    def setUp(self):
        super().setUp(
            num_of_devices=4,
            timeout=120,
            nnode=1,
        )
        self._default_envs = {
            "dtype": "float32",
            "seed": "2023",
        }
        self._changeable_envs = {"backend": ["gpu"], "is_dp": ["false"]}

    def test_simple_net_mp_pp_sp(self):
        envs_list = test_base.gen_product_envs_list(
            self._default_envs, self._changeable_envs
        )
        for envs in envs_list:
            ckpt_path = tempfile.TemporaryDirectory()
            envs["ckpt_path"] = ckpt_path.name
            self.run_test_case(
                "semi_auto_parallel_simple_net_sp.py",
                user_defined_envs=envs,
            )
            ckpt_path.cleanup()

    def test_simple_net_dp_mp_pp_sp(self):
        super().setUp(
            num_of_devices=8,
            timeout=120,
            nnode=1,
        )
        self._changeable_envs = {"backend": ["gpu"], "is_dp": ["true"]}
        envs_list = test_base.gen_product_envs_list(
            self._default_envs, self._changeable_envs
        )
        for envs in envs_list:
            ckpt_path = tempfile.TemporaryDirectory()
            envs["ckpt_path"] = ckpt_path.name
            self.run_test_case(
                "semi_auto_parallel_simple_net_sp.py",
                user_defined_envs=envs,
            )
            ckpt_path.cleanup()


if __name__ == "__main__":
    unittest.main()
