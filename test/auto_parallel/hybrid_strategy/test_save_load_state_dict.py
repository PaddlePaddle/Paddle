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


class TestSaveLoadStateDict(test_base.CommunicationTestDistBase):
    def setUp(self):
        self._default_envs = {}
        self._changeable_envs = {"device_num": ["1", "2", "4", "8"]}

    def test_reshard(self):
        # save with 1 device
        ckpt_path = tempfile.TemporaryDirectory()
        super().setUp(num_of_devices=1, timeout=120, nnode=1)
        self.run_test_case(
            "semi_auto_save_state_dict.py",
            user_defined_envs={"device_num": "1", "ckpt_path": ckpt_path.name},
        )

        # load with 1, 2, 4, 8 devices
        envs_list = test_base.gen_product_envs_list(
            self._default_envs, self._changeable_envs
        )
        for envs in envs_list:
            envs["ckpt_path"] = ckpt_path.name
            super().setUp(
                num_of_devices=int(envs["device_num"]),
                timeout=180,
                nnode=1,
            )
            self.run_test_case(
                "semi_auto_load_state_dict.py",
                user_defined_envs=envs,
            )
        ckpt_path.cleanup()

        # save with 4 devices
        ckpt_path = tempfile.TemporaryDirectory()
        super().setUp(num_of_devices=4, timeout=120, nnode=1)
        self.run_test_case(
            "semi_auto_save_state_dict.py",
            user_defined_envs={"device_num": "4", "ckpt_path": ckpt_path.name},
        )
        # load with 1, 2, 4, 8 devices
        envs_list = test_base.gen_product_envs_list(
            self._default_envs, self._changeable_envs
        )
        for envs in envs_list:
            envs["ckpt_path"] = ckpt_path.name
            super().setUp(
                num_of_devices=int(envs["device_num"]),
                timeout=180,
                nnode=1,
            )
            self.run_test_case(
                "semi_auto_load_state_dict.py",
                user_defined_envs=envs,
            )
        ckpt_path.cleanup()

    def test_mutual_load_between_dynamic_and_static(self):
        changeable_envs = {"device_num": ["2"]}
        envs_list = test_base.gen_product_envs_list(
            self._default_envs, changeable_envs
        )

        for envs in envs_list:
            ckpt_path = tempfile.TemporaryDirectory()
            envs["ckpt_path"] = ckpt_path.name
            super().setUp(
                num_of_devices=int(envs["device_num"]),
                timeout=180,
                nnode=1,
            )
            self.run_test_case(
                "semi_auto_parallel_mutual_load_between_dynamic_and_static.py",
                user_defined_envs=envs,
            )
            ckpt_path.cleanup()


if __name__ == '__main__':
    unittest.main()
