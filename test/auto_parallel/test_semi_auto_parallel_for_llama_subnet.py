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

import paddle


class TestSemiAutoParallelBasic(test_base.CommunicationTestDistBase):
    def setUp(self):
        super().setUp(
            num_of_devices=2,
            timeout=120,
        )
        self._default_envs = {"dtype": "float32", "seed": "2023"}
        self._changeable_envs = {"backend": ["cpu", "gpu"]}

    def test_attention_subnet(self):
        envs_list = test_base.gen_product_envs_list(
            {"dtype": "float32", "seed": "2023"}, {"backend": ["gpu"]}
        )
        cuda_version_main = int(paddle.version.cuda().split(".")[0])
        device_prop_main = paddle.device.cuda.get_device_capability()[0]
        if cuda_version_main >= 11 and device_prop_main >= 8:
            for envs in envs_list:
                self.run_test_case(
                    "semi_auto_parallel_for_llama_attention.py",
                    user_defined_envs=envs,
                )

    def test_mlp_subnet(self):
        envs_list = test_base.gen_product_envs_list(
            {"dtype": "float32", "seed": "2023"}, {"backend": ["gpu", "cpu"]}
        )
        for envs in envs_list:
            self.run_test_case(
                "semi_auto_parallel_for_llama_mlp.py",
                user_defined_envs=envs,
            )

    def test_rope_subnet(self):
        envs_list = test_base.gen_product_envs_list(
            {"dtype": "float32", "seed": "2023"}, {"backend": ["gpu"]}
        )
        for envs in envs_list:
            self.run_test_case(
                "semi_auto_parallel_for_llama_rope.py",
                user_defined_envs=envs,
            )

    def test_decoder_subnet(self):
        envs_list = test_base.gen_product_envs_list(
            {"dtype": "float32", "seed": "2023"}, {"backend": ["gpu"]}
        )
        cuda_version_main = int(paddle.version.cuda().split(".")[0])
        device_prop_main = paddle.device.cuda.get_device_capability()[0]
        if cuda_version_main >= 11 and device_prop_main >= 8:
            for envs in envs_list:
                self.run_test_case(
                    "semi_auto_parallel_for_llama_decoder.py",
                    user_defined_envs=envs,
                )

    def test_mlp_with_cross_entropy_loss_subnet(self):
        envs_list = test_base.gen_product_envs_list(
            {"dtype": "float32", "seed": "2023"}, {"backend": ["gpu"]}
        )
        for envs in envs_list:
            self.run_test_case(
                "semi_auto_parallel_for_llama_mlp_with_cross_entropy_loss.py",
                user_defined_envs=envs,
            )


if __name__ == "__main__":
    unittest.main()
