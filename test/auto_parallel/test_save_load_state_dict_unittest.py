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

import tempfile

import collective.test_communication_api_base as test_base


class TestSaveLoadStateDict(test_base.CommunicationTestDistBase):
    def setUp(self):
        super().setUp(
            "./log",
            num_of_devices=2,
            timeout=120,
        )
        self._default_envs = {"dtype": "float32", "seed": "2023"}
        self._changeable_envs = {"backend": ["gpu"]}

    def test_save_state_dict(self):
        envs_list = test_base.gen_product_envs_list(
            self._default_envs, self._changeable_envs
        )
        for envs in envs_list:
            ckpt_path_tmp = tempfile.TemporaryDirectory()
            ckpt_path = ckpt_path_tmp.name
            ckpt_path = "ckpt_path"
            import os

            os.system("rm -rf ./log")
            os.system(f"rm -rf {ckpt_path}")
            envs["ckpt_path"] = ckpt_path
            self.run_test_case(
                "save_state_dict_unittest.py",
                user_defined_envs=envs,
            )
            ckpt_path_tmp.cleanup()
