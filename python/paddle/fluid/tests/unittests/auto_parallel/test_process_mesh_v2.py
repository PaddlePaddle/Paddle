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
from paddle.distributed.auto_parallel.process_mesh_v2 import (
    ProcessMesh, compute_compatible_process_mesh, merge_process_mesh)


class TestProcessMesh(unittest.TestCase):

    def test_process_mesh(self):
        mesh = [[0, 1, 2], [3, 4, 5]]
        mesh2 = [[0, 1], [2, 3]]
        process_mesh = ProcessMesh(mesh, dim_names=["x", "y"])
        process_mesh2 = ProcessMesh(mesh2)
        self.assertEqual(process_mesh.shape, [2, 3])
        self.assertEqual(process_mesh.process_ids, [0, 1, 2, 3, 4, 5])
        self.assertEqual(process_mesh.dim_names, ["x", "y"])
        self.assertEqual(process_mesh.size, 6)
        self.assertEqual(process_mesh.ndim, 2)
        self.assertEqual(process_mesh.dim_size(0), 2)
        self.assertEqual(process_mesh.dim_size(-1), 3)
        self.assertEqual(process_mesh.dim_size("x"), 2)
        self.assertEqual(process_mesh.dim_size("y"), 3)
        self.assertEqual(process_mesh.empty(), False)
        self.assertEqual(process_mesh.contains(0), True)
        self.assertEqual(process_mesh.contains(6), False)
        self.assertEqual(process_mesh, process_mesh)
        self.assertNotEqual(process_mesh, process_mesh2)
        self.assertEqual(str(process_mesh), str(process_mesh))

    def test_compute_compatible_process_mesh(self):
        process_mesh1 = ProcessMesh([[0, 1, 2], [3, 4, 5]],
                                    dim_names=["x", "y"])
        compatible_process_mesh = compute_compatible_process_mesh(
            [process_mesh1, None])
        self.assertEqual(compatible_process_mesh, process_mesh1)
        compatible_process_mesh = compute_compatible_process_mesh(
            [None, process_mesh1])
        self.assertEqual(compatible_process_mesh, process_mesh1)

        process_mesh2 = ProcessMesh([[0, 1, 2], [3, 4, 5]])
        compatible_process_mesh = compute_compatible_process_mesh(
            [process_mesh1, process_mesh2])
        self.assertEqual(compatible_process_mesh, process_mesh1)
        self.assertEqual(compatible_process_mesh, process_mesh2)

        process_mesh2 = ProcessMesh([[0, 1, 2, 3, 4, 5]])
        compatible_process_mesh = compute_compatible_process_mesh(
            [process_mesh1, process_mesh2])
        self.assertEqual(compatible_process_mesh, process_mesh1)

        process_mesh2 = ProcessMesh([[0, 1, 2]])
        compatible_process_mesh = compute_compatible_process_mesh(
            [process_mesh1, process_mesh2])
        self.assertEqual(compatible_process_mesh, process_mesh1)

    def test_merge_process_mesh(self):
        process_mesh1 = ProcessMesh([[0, 1, 2], [3, 4, 5]],
                                    dim_names=["x", "y"])
        merged_process_mesh = merge_process_mesh([process_mesh1, None])
        print(merged_process_mesh)
        self.assertEqual(merged_process_mesh, ProcessMesh([0, 1, 2, 3, 4, 5]))
        merged_process_mesh = merge_process_mesh([None, process_mesh1])
        self.assertEqual(merged_process_mesh, ProcessMesh([0, 1, 2, 3, 4, 5]))

        process_mesh2 = ProcessMesh([[0, 1, 2], [3, 4, 5]])
        merged_process_mesh = merge_process_mesh([process_mesh1, process_mesh2])
        self.assertEqual(merged_process_mesh, ProcessMesh([0, 1, 2, 3, 4, 5]))

        process_mesh2 = ProcessMesh([[0, 1, 2]])
        merged_process_mesh = merge_process_mesh([process_mesh1, process_mesh2])
        self.assertEqual(merged_process_mesh, ProcessMesh([0, 1, 2, 3, 4, 5]))

        process_mesh2 = ProcessMesh([[6, 7]])
        merged_process_mesh = merge_process_mesh([process_mesh1, process_mesh2])
        self.assertEqual(merged_process_mesh,
                         ProcessMesh([0, 1, 2, 3, 4, 5, 6, 7]))


if __name__ == "__main__":
    unittest.main()
