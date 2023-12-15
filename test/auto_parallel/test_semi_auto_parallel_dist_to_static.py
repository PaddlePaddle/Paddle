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
import unittest

import collective.test_communication_api_base as test_base


class TestSemiAutoParallelStaticDecorate(test_base.CommunicationTestDistBase):
    def setUp(self):
        super().setUp(
            num_of_devices=2,
            timeout=300,
        )
        self._default_envs = {"dtype": "float32", "seed": "2023"}
        self._changeable_envs = {"backend": ["gpu"]}

    def test_api_function(self):
        envs_list = test_base.gen_product_envs_list(
            {"dtype": "float32", "seed": "2023"}, {"backend": ["gpu"]}
        )
        for envs in envs_list:
            self.run_test_case(
                "semi_auto_parallel_dist_to_static_api.py",
                user_defined_envs=envs,
            )

    def test_mlp(self):
        envs_list = test_base.gen_product_envs_list(
            {"dtype": "float32", "seed": "2023"}, {"backend": ["gpu"]}
        )
        for envs in envs_list:
            ckpt_path_tmp = tempfile.TemporaryDirectory()
            envs["ckpt_path"] = ckpt_path_tmp.name
            self.run_test_case(
                "semi_auto_parallel_dist_to_static_mlp.py",
                user_defined_envs=envs,
            )
            ckpt_path_tmp.cleanup()


if __name__ == "__main__":
    unittest.main()
