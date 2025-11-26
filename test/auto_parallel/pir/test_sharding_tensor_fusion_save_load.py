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


class TestSimpleNetShardingTensorFusionSaveLoad(
    test_base.CommunicationTestDistBase
):
    def setUp(self):
        super().setUp(
            num_of_devices=2,
            timeout=500,
        )
        self._default_envs = {"dtype": "float32", "seed": "2024"}
        self._changeable_envs = {"backend": ["gpu"]}

    def test_mlp_save_unbalanced_param(self):
        envs_list = test_base.gen_product_envs_list(
            {"dtype": "float32", "seed": "2024", "save_unbalanced_param": "1"},
            {"backend": ["gpu"]},
        )

        for envs in envs_list:
            ckpt_path_tmp = tempfile.TemporaryDirectory()
            envs["ckpt_path"] = ckpt_path_tmp.name
            self.run_test_case(
                "sharding_tensor_fusion_save_load.py",
                user_defined_envs=envs,
            )
            ckpt_path_tmp.cleanup()

    def test_mlp_amp_save_unbalanced_param(self):
        envs_list = test_base.gen_product_envs_list(
            {"dtype": "float32", "seed": "2024", "save_unbalanced_param": "1"},
            {
                "backend": ["gpu"],
                "amp": ['1'],
                "amp_dtype": ['bfloat16'],
                'amp_level': ['O1'],
                'use_master_weight': ['1'],
                'use_master_grad': ['1'],
            },
        )

        for envs in envs_list:
            # self._log_dir.name = "./log"
            ckpt_path_tmp = tempfile.TemporaryDirectory()
            envs["ckpt_path"] = ckpt_path_tmp.name
            self.run_test_case(
                "sharding_tensor_fusion_save_load.py",
                user_defined_envs=envs,
            )
            ckpt_path_tmp.cleanup()

    def test_mlp_save_balanced_param(self):
        envs_list = test_base.gen_product_envs_list(
            {"dtype": "float32", "seed": "2024", "save_unbalanced_param": "0"},
            {"backend": ["gpu"]},
        )

        for envs in envs_list:
            ckpt_path_tmp = tempfile.TemporaryDirectory()
            envs["ckpt_path"] = ckpt_path_tmp.name
            self.run_test_case(
                "sharding_tensor_fusion_save_load.py",
                user_defined_envs=envs,
            )
            ckpt_path_tmp.cleanup()

    def test_mlp_amp_save_balanced_param(self):
        envs_list = test_base.gen_product_envs_list(
            {"dtype": "float32", "seed": "2024", "save_unbalanced_param": "0"},
            {
                "backend": ["gpu"],
                "amp": ['1'],
                "amp_dtype": ['bfloat16'],
                'amp_level': ['O1'],
                'use_master_weight': ['1'],
                'use_master_grad': ['1'],
            },
        )

        for envs in envs_list:
            # self._log_dir.name = "./log"
            ckpt_path_tmp = tempfile.TemporaryDirectory()
            envs["ckpt_path"] = ckpt_path_tmp.name
            self.run_test_case(
                "sharding_tensor_fusion_save_load.py",
                user_defined_envs=envs,
            )
            ckpt_path_tmp.cleanup()


if __name__ == "__main__":
    unittest.main()
