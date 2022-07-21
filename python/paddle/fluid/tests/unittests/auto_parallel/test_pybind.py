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
# limitations under the License

import unittest
from paddle.distributed.auto_parallel.process_mesh_v2 import ProcessMesh


class TestProcessMesh(unittest.TestCase):

    def test_ctor(self):
        mesh = [[0, 1], [2, 3]]
        process_mesh = ProcessMesh(mesh)
        self.assertEqual(process_mesh.shape, [2, 2])
        self.assertEqual(process_mesh.process_ids, [0, 1, 2, 3])
        self.assertEqual(process_mesh.dim_names, ["d0", "d1"])
        self.assertEqual(process_mesh.device_type, "GPU")
        self.assertEqual(process_mesh.ndim, 2)
        process_mesh.device_type = "CPU"
        self.assertEqual(process_mesh.device_type, "CPU")
        print(process_mesh)
        # self.assertEqual(str(process_mesh), "CPU")


if __name__ == "__main__":
    unittest.main()
