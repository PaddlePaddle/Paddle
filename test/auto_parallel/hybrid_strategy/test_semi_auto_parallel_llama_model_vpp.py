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

import sys
import unittest

sys.path.append("../../")
import os

import collective.test_communication_api_base as test_base

os.environ['FLAGS_enable_pir_api'] = '0'


class TestSemiAutoParallelLlama3DVPP(test_base.CommunicationTestDistBase):
    def setUp(self):
        super().setUp(num_of_devices=8, timeout=200, nnode=1)
        self._default_envs = {
            "seed": "2023",
            "dp": "2",
            "mp": "2",
            "pp": "2",
            "FLAGS_embedding_deterministic": "1",
            "FLAGS_cudnn_deterministic": "1",
            "acc_step": "4",
            "only_static": "true",
        }
        self._changeable_envs = {
            "backend": ["gpu"],
            "use_sp": ["true"],
            "use_param_group": ["true"],
            "recompute": ["true"],
            "recompute_granularity": ["full"],
            "virtual_pp_degree": ["2"],
        }

    def test_simple_net_hybrid_strategy(self):
        envs_list = test_base.gen_product_envs_list(
            self._default_envs, self._changeable_envs
        )
        self._log_dir.name = "./vpp_log"
        for envs in envs_list:
            self.run_test_case(
                "semi_auto_llama_pp_gradmerge.py",
                user_defined_envs=envs,
            )


if __name__ == "__main__":
    unittest.main()
