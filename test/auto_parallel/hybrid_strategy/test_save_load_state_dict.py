# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
from auto_parallel.hybrid_strategy.save_state_dict import ckpt_path


class TestSaveLoadStateDict(test_base.CommunicationTestDistBase):
    def setUp(self):
        self._default_envs = {}
        self._changeable_envs = {"device_num": ["1", "2", "4", "8"]}

    def test_save_load_state_dict(self):
        # save with 1 device
        os.system(f"rm -rf {ckpt_path()}")
        super().setUp(num_of_devices=1, timeout=120, nnode=1)
        self.run_test_case(
            "save_state_dict.py", user_defined_envs={"device_num": "1"}
        )

        # load with 1, 2, 4, 8 devices
        envs_list = test_base.gen_product_envs_list(
            self._default_envs, self._changeable_envs
        )
        for envs in envs_list:
            super().setUp(
                save_log_dir="./log",
                num_of_devices=int(envs["device_num"]),
                timeout=120,
                nnode=1,
            )
            self.run_test_case(
                "load_state_dict.py",
                user_defined_envs=envs,
            )
        os.system(f"rm -rf {ckpt_path()}")

        # save with 4 devices
        os.system(f"rm -rf {ckpt_path()}")
        super().setUp(num_of_devices=4, timeout=120, nnode=1)
        self.run_test_case(
            "save_state_dict.py", user_defined_envs={"device_num": "4"}
        )
        # load with 1, 2, 4, 8 devices
        envs_list = test_base.gen_product_envs_list(
            self._default_envs, self._changeable_envs
        )
        for envs in envs_list:
            super().setUp(
                save_log_dir="./log",
                num_of_devices=int(envs["device_num"]),
                timeout=120,
                nnode=1,
            )
            self.run_test_case(
                "load_state_dict.py",
                user_defined_envs=envs,
            )
        os.system(f"rm -rf {ckpt_path()}")


if __name__ == '__main__':
    unittest.main()
