#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

import copy
import numpy as np
from paddle.fluid import core


class ProcessMesh(core.ProcessMesh):
    r"""
    The class `Processmesh` describes the topology of logical processes.

    Args:
        mesh (list|numpy.array): an N-dimensional array describes the toplogy
            of logical processes.
        dim_names (list, optional): the i-th element of this list gives the name of the
            i-th dimension.

    Returns:
        None

    Examples:
        .. code-block:: python

            import paddle
            import paddle.distributed as dist

            paddle.enable_static()

            mesh = dist.ProcessMesh([[2, 4, 5], [0, 1, 3]])
            assert mesh.shape == [2, 3]
            assert mesh.processe_ids == [2, 4, 5, 0, 1, 3]

    """

    def __init__(self, mesh, dim_names=None):
        if not isinstance(mesh, list) and \
           not isinstance(mesh, np.ndarray):
            raise ValueError(
                'The mesh must be an instance of list or np.ndarray.')
        if isinstance(mesh, list):
            mesh = np.array(mesh)

        self._mesh = mesh

        self._shape = list(self._mesh.shape)

        self._process_ids = self._mesh.flatten().tolist()
        assert all(isinstance(p, int) for p in self._process_ids), \
            ("All elements of the mesh must be integer")
        assert min(
            self._process_ids) >= 0, ('All elements of the mesh must be >= 0.')
        unique_process_ids = set(self._process_ids)
        assert len(unique_process_ids) == len(
            self._process_ids), ('All elements of the mesh must be unique.')

        if dim_names is not None:
            assert len(dim_names) == len(self._shape), \
                ("The length of dims_names must be same as the shape of the mesh.")
            self._dim_names = dim_names
        else:
            self._dim_names = ["d" + str(i) for i in range(len(self._shape))]

        # Follow the requirement for using pybind11
        core.ProcessMesh.__init__(self, self._shape, self._process_ids,
                                  self._dim_names)

    @property
    def mesh(self):
        return self._mesh


def compute_compatible_process_mesh(process_meshes):
    """Compute the compatible process mesh given a list of process meshes."""
    if not process_meshes:
        return None

    def _compute_compatible_of_two_process_meshes(pm1, pm2):
        if pm1 is None:
            return True, pm2
        if pm2 is None:
            return True, pm1
        if pm1 == pm2:
            return True, pm1
        if pm1.process_ids == pm2.process_ids:
            if len(pm1.shape) >= len(pm2.shape):
                return True, pm1
            else:
                return True, pm2
        process_set1 = set(pm1.process_ids)
        process_set2 = set(pm2.process_ids)
        if process_set1.issubset(process_set2):
            return True, pm2
        if process_set2.issubset(process_set1):
            return True, pm1
        return False, None

    compatible_result = None
    for process_mesh in process_meshes:
        compatible, compatible_result = _compute_compatible_of_two_process_meshes(
            compatible_result, process_mesh)
        if not compatible:
            return None
    if compatible_result.empty():
        return None
    if isinstance(compatible_result, core.ProcessMesh):
        mesh = np.array(compatible_result.process_ids).reshape(
            compatible_result.shape)
        return ProcessMesh(mesh, compatible_result.dim_names)
    elif isinstance(compatible_result, ProcessMesh):
        return ProcessMesh(compatible_result.mesh, compatible_result.dim_names)
    else:
        raise ValueError("Unrecognized ProcessMesh.")


def merge_process_mesh(process_meshes):
    """Merge a list of process meshes."""
    merged_process_mesh = None
    merged_process_ids = set()
    for process_mesh in process_meshes:
        if process_mesh is not None:
            process_ids = set(process_mesh.process_ids)
            merged_process_ids = merged_process_ids.union(process_ids)
    if len(merged_process_ids) != 0:
        merged_process_mesh = ProcessMesh(list(merged_process_ids))
    return merged_process_mesh
