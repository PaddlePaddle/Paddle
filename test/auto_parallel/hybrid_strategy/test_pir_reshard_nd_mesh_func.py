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


class TestPIRNdMeshReshard(test_base.CommunicationTestDistBase):
    def setUp(self):
        super().setUp(
            num_of_devices=4,
            timeout=15,
            nnode=1,
        )
        self._default_envs = {
            "dtype": "float32",
            "seed": "2023",
            "backend": "gpu",
        }

    def test_simple_net_reshard(self):
        self.run_test_case(
            "pir_reshard_nd_mesh.py",
            user_defined_envs=self._default_envs,
        )


class TestPIRNdMeshReshardCrossMesh(test_base.CommunicationTestDistBase):
    def setUp(self):
        super().setUp(
            num_of_devices=8,
            timeout=20,
            nnode=1,
        )
        self._default_envs = {
            "dtype": "float32",
            "seed": "2023",
            "backend": "gpu",
        }

    def test_simple_net_reshard_cross_mesh(self):
        self.run_test_case(
            "pir_reshard_nd_mesh_cross_mesh.py",
            user_defined_envs=self._default_envs,
        )


if __name__ == "__main__":
    unittest.main()
