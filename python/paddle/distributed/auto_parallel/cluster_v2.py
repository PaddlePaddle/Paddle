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

import numpy as np
from enum import IntEnum
from enum import unique

from paddle.fluid import core
from paddle.fluid.core import Device  # noqa: F401
from paddle.fluid.core import Link  # noqa: F401


@unique
class DeviceType(IntEnum):
    UNKNOWN = 0
    CPU = 1
    GPU = 2
    XPU = 3
    NPU = 4
    DCU = 5
    NIC = 6


@unique
class LinkType(IntEnum):
    UNKNOWN = 0
    LOC = 1
    SYS = 2
    PHB = 3
    PIX = 4
    PIB = 5
    NVL = 6
    NVB = 7
    NET = 8


class DeviceMesh(core.DeviceMesh):
    r"""
    The class `DeviceMesh` describes the topology of physical devices.

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

            mesh = dist.DeviceMesh([[2, 4, 5], [0, 1, 3]])
            assert mesh.shape == [2, 3]
            assert mesh.device_ids == [2, 4, 5, 0, 1, 3]

    """

    def __init__(self, name, mesh, dim_names=None):
        self._name = name

        if not isinstance(mesh, list) and not isinstance(mesh, np.ndarray):
            raise ValueError(
                'The mesh must be an instance of list or np.ndarray.'
            )
        if isinstance(mesh, list):
            mesh = np.array(mesh)

        self._mesh = mesh

        self._shape = list(self._mesh.shape)

        self._device_ids = self._mesh.flatten().tolist()
        assert all(
            isinstance(p, int) for p in self._device_ids
        ), "All elements of the mesh be integer"
        assert (
            min(self._device_ids) >= 0
        ), 'All elements of the mesh must be >= 0.'
        unique_device_ids = set(self._device_ids)
        assert len(unique_device_ids) == len(
            self._device_ids
        ), 'All elements of the mesh must be unique.'

        if dim_names is not None:
            assert len(dim_names) == len(
                self._shape
            ), "The length of dims_names must be same as the shape of the mesh."
            self._dim_names = dim_names
        else:
            self._dim_names = ["d" + str(i) for i in range(len(self._shape))]

        # Follow the requirement for using pybind11
        core.DeviceMesh.__init__(
            self, self._name, self._shape, self._device_ids, self._dim_names
        )

    @property
    def mesh(self):
        return self._mesh


# class Cluster:
#     """
#     The cluster represents the hardware resource.
#     """

#     def __init__(self):
#         self._device_meshes = {}

#     def device_mesh(self, device_mesh_name):
#         return self._device_meshes[device_mesh_name]

#     def add_device_mesh(self, device_mesh):
#         self._device_meshes[device_mesh.name] = device_mesh
