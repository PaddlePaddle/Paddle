# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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


class TestLoadTransformShardedTarget(test_base.CommunicationTestDistBase):
    def setUp(self):
        super().setUp(num_of_devices=2)

    def _run_case(self, transform_case):
        with tempfile.TemporaryDirectory() as ckpt_dir:
            self.run_test_case(
                "load_transform_dist_logic.py",
                user_defined_envs={
                    "ckpt_path": ckpt_dir,
                    "transform_case": transform_case,
                },
            )

    def test_sharded_target_global_read(self):
        self._run_case("global")

    def test_sharded_target_local_read_plan(self):
        self._run_case("local")


if __name__ == "__main__":
    unittest.main()
