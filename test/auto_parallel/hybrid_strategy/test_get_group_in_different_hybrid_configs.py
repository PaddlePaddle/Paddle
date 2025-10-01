# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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


class TestProcessMeshDPGroupConsistency(test_base.CommunicationTestDistBase):
    def setUp(self):
        super().setUp(num_of_devices=2, timeout=200, nnode=1)

    def test_dp_parallel(self):
        """Test data parallel group creation and consistency"""
        _default_envs = {
            "dp": "2",
            "mp": "1",
            "pp": "1",
            "parallel_type": "dp",
            "FLAGS_embedding_deterministic": "1",
            "FLAGS_cudnn_deterministic": "1",
        }
        _changeable_envs = {
            "backend": ["gpu"],
        }
        envs_list = test_base.gen_product_envs_list(
            _default_envs, _changeable_envs
        )
        for envs in envs_list:
            self.run_test_case(
                "test_process_mesh_group_consistency.py",
                user_defined_envs=envs,
            )


class TestProcessMeshMPGroupConsistency(test_base.CommunicationTestDistBase):
    def setUp(self):
        super().setUp(num_of_devices=2, timeout=200, nnode=1)

    def test_mp_parallel(self):
        """Test model parallel group creation and consistency"""
        _default_envs = {
            "dp": "1",
            "mp": "2",
            "pp": "1",
            "parallel_type": "mp",
            "FLAGS_embedding_deterministic": "1",
            "FLAGS_cudnn_deterministic": "1",
        }
        _changeable_envs = {
            "backend": ["gpu"],
        }
        envs_list = test_base.gen_product_envs_list(
            _default_envs, _changeable_envs
        )
        for envs in envs_list:
            self.run_test_case(
                "test_process_mesh_group_consistency.py",
                user_defined_envs=envs,
            )


class TestProcessMeshPPGroupConsistency(test_base.CommunicationTestDistBase):
    def setUp(self):
        super().setUp(num_of_devices=2, timeout=200, nnode=1)

    def test_pp_parallel(self):
        """Test pipeline parallel group creation and consistency"""
        _default_envs = {
            "dp": "1",
            "mp": "1",
            "pp": "2",
            "parallel_type": "pp",
            "FLAGS_embedding_deterministic": "1",
            "FLAGS_cudnn_deterministic": "1",
        }
        _changeable_envs = {
            "backend": ["gpu"],
        }
        envs_list = test_base.gen_product_envs_list(
            _default_envs, _changeable_envs
        )
        for envs in envs_list:
            self.run_test_case(
                "test_process_mesh_group_consistency.py",
                user_defined_envs=envs,
            )


class TestProcessMeshSEPGroupConsistency(test_base.CommunicationTestDistBase):
    def setUp(self):
        super().setUp(num_of_devices=2, timeout=200, nnode=1)

    def test_sep_parallel(self):
        """Test sequence parallel group creation and consistency"""
        _default_envs = {
            "dp": "1",
            "mp": "1",
            "pp": "1",
            "sep": "2",
            "sharding": "1",
            "parallel_type": "sep",
            "FLAGS_embedding_deterministic": "1",
            "FLAGS_cudnn_deterministic": "1",
        }
        _changeable_envs = {
            "backend": ["gpu"],
        }
        envs_list = test_base.gen_product_envs_list(
            _default_envs, _changeable_envs
        )
        for envs in envs_list:
            self.run_test_case(
                "test_process_mesh_group_consistency.py",
                user_defined_envs=envs,
            )


class TestProcessMeshShardingGroupConsistency(
    test_base.CommunicationTestDistBase
):
    def setUp(self):
        super().setUp(num_of_devices=2, timeout=200, nnode=1)

    def test_sharding_parallel(self):
        """Test sharding parallel group creation and consistency"""
        _default_envs = {
            "dp": "1",
            "mp": "1",
            "pp": "1",
            "sep": "1",
            "sharding": "2",
            "parallel_type": "sharding",
            "FLAGS_embedding_deterministic": "1",
            "FLAGS_cudnn_deterministic": "1",
        }
        _changeable_envs = {
            "backend": ["gpu"],
        }
        envs_list = test_base.gen_product_envs_list(
            _default_envs, _changeable_envs
        )
        for envs in envs_list:
            self.run_test_case(
                "test_process_mesh_group_consistency.py",
                user_defined_envs=envs,
            )


if __name__ == "__main__":
    unittest.main()  # python run
