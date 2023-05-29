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

import unittest


class TestConvertToProcessMeshes(unittest.TestCase):
    def test_convert_to_process_meshes(self):
        device_meshes = [[1, 8], [4, 8], [15, 8]]
        from paddle.distributed.auto_parallel.tuner.rule_based_tuner import (
            convert_to_process_meshes,
        )

        process_meshes = []
        for device_mesh in device_meshes:
            process_mesh = convert_to_process_meshes(device_mesh)
            process_meshes.append(process_mesh)


if __name__ == "__main__":
    unittest.main()
