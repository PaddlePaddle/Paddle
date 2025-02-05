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

import tempfile
import unittest

import collective.test_communication_api_base as test_base


class TestShardingParallelAPI(test_base.CommunicationTestDistBase):
    def setUp(self):
        super().setUp(num_of_devices=2, timeout=180, nnode=1)
        self._default_envs = {
            "dtype": "float32",
            "seed": "2023",
            "dp": "2",
            "mp": "1",
            "pp": "1",
            "acc_step": "2",
        }
        self._changeable_envs = {
            "backend": ["gpu"],
            "amp": ["true"],
            "amp_level": ["O2"],
            "amp_dtype": ["bfloat16"],
            "amp_master_grad": ["False"],
            "sharding_stage": ["0", "1"],
            "test_share_embedding": [
                "1",
            ],
            "test_position_embedding": [
                "1",
            ],
        }

    def test_simple_net_dp2(self):
        envs_list = test_base.gen_product_envs_list(
            self._default_envs, self._changeable_envs
        )
        for envs in envs_list:
            ckpt_path = tempfile.TemporaryDirectory()
            envs["ckpt_path"] = ckpt_path.name
            self.run_test_case(
                "parallel_api.py",
                user_defined_envs=envs,
            )
            ckpt_path.cleanup()


class TestPipelineParallelAPI(test_base.CommunicationTestDistBase):
    def setUp(self):
        super().setUp(num_of_devices=2, timeout=180, nnode=1)
        self._default_envs = {
            "dtype": "float32",
            "seed": "2023",
            "dp": "1",
            "mp": "1",
            "pp": "2",
            "acc_step": "2",
        }
        self._changeable_envs = {
            "backend": ["gpu"],
            "amp": ["true"],
            "amp_level": ["O2"],
            "amp_dtype": ["bfloat16"],
            "amp_master_grad": ["False"],
            "num_hidden_layers": ["2", "4"],
            "test_share_embedding": [
                "1",
            ],
            "test_position_embedding": [
                "1",
            ],
        }

    def test_simple_net_pp2(self):
        envs_list = test_base.gen_product_envs_list(
            self._default_envs, self._changeable_envs
        )
        for envs in envs_list:
            ckpt_path = tempfile.TemporaryDirectory()
            envs["ckpt_path"] = ckpt_path.name
            self.run_test_case(
                "parallel_api.py",
                user_defined_envs=envs,
            )
            ckpt_path.cleanup()


class TestTensorParallelAPI(test_base.CommunicationTestDistBase):
    def setUp(self):
        super().setUp(num_of_devices=2, timeout=180, nnode=1)
        self._default_envs = {
            "dtype": "float32",
            "seed": "2023",
            "dp": "1",
            "mp": "2",
            "pp": "1",
            "acc_step": "2",
        }
        self._changeable_envs = {
            "backend": ["gpu"],
            "amp": ["true"],
            "amp_level": ["O2"],
            "amp_dtype": ["bfloat16"],
            "amp_master_grad": ["true"],
            "use_lazy_init": ["true", "false"],
            "sequence_parallel": ["true", "false"],
            "prepare_input_output": ["true", "false"],
            "test_share_embedding": [
                "1",
            ],
            "test_position_embedding": [
                "1",
            ],
        }

    def test_simple_net_mp2(self):
        envs_list = test_base.gen_product_envs_list(
            self._default_envs, self._changeable_envs
        )
        has_checked_lazy_init = False
        has_checked_pure_mp = False
        for envs in envs_list:
            if envs['use_lazy_init'] == 'true':
                if has_checked_lazy_init:
                    continue
                has_checked_lazy_init = True
            if envs['sequence_parallel'] != 'true':
                if has_checked_pure_mp:
                    continue
                has_checked_pure_mp = True
            ckpt_path = tempfile.TemporaryDirectory()
            envs["ckpt_path"] = ckpt_path.name
            self.run_test_case(
                "parallel_api.py",
                user_defined_envs=envs,
            )
            ckpt_path.cleanup()


if __name__ == "__main__":
    unittest.main()
