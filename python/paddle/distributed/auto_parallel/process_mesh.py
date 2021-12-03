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

import numpy
import copy


def _get_nested_list_shape(nested_list):
    """
    Get the shape of a nested_list.
    """
    result = []
    while isinstance(nested_list, list):
        result.append(len(nested_list))
        nested_list = nested_list[0]
    return result


def _flatten_nested_list(nested_list):
    """
    Get a list of all items in a nested_list.
    Ref: https://stackoverflow.com/questions/952914/how-to-make-a-flat-list-out-of-a-list-of-lists
    """
    result = numpy.array(nested_list).flatten().tolist()
    return result


class ProcessMesh(object):
    r"""
    The class `Processmesh` describes the topology of logical processes. 
    A mesh is an N-dimensional array. The shape of the N-dimensional
    array represents the topology of logical processes and every
    element of the N-dimensional array represent a logical process. For
    example, the 2-dimensional array [[2, 4, 5], [0, 1, 3]]
    illustrates six logical processes organized as the topology [2, 3],
    i.e., the shape of the 2-dimensional array. With the above topology,
    there are two parallel groups, where the first parallel group has a
    parallel degree of 2 and the second one has a parallel degree of 3.
    And the first logical process is the one with id=2.

    Args:
        mesh (list): an N-dimensional array (nested list) describes the toplogy
            of logical processes. The shape of the N-dimensional array
            represents the topology of logical processes and every 
            element of the N-dimensional array represents a logical process.
    
    Returns:
        None

    Raises:
        ValueError: If `mesh` is not an instance of list.

    Examples:
        .. code-block:: python

            import paddle
            import paddle.distributed as dist
            
            paddle.enable_static()
            
            mesh = dist.ProcessMesh([[2, 4, 5], [0, 1, 3]])
            assert mesh.topology == [2, 3]
            assert mesh.processes == [2, 4, 5, 0, 1, 3]

    """

    def __init__(self, mesh):
        if mesh is None or not isinstance(mesh, list):
            raise ValueError('mesh must be an instance of list.')

        processes = _flatten_nested_list(mesh)

        assert all(isinstance(p, int) for p in processes), \
            ("All elements of mesh must be integer")

        assert min(processes) >= 0, ('All elements of mesh must be >= 0.')

        unique_processes = set(processes)
        assert len(unique_processes) == len(processes), (
            'All elements of mesh must be unique.')

        self._topology = _get_nested_list_shape(mesh)
        self._processes = processes

        # Store all process meshes
        from .dist_context import get_default_distributed_context
        default_dist_cxt = get_default_distributed_context()
        default_dist_cxt.add_process_mesh(self)
        # Add new processes to process group 0 
        from .process_group import get_process_group
        pg0 = get_process_group(0)
        pg0.add_ranks(self.processes)

    @property
    def topology(self):
        r"""
        Get the topology of logical processes belonging to this ProcessMesh.
        This is the shape of `mesh` used to initialized this ProcessMesh.
        """
        return self._topology

    @property
    def processes(self):
        r"""
        Get a list of all processes belonging to this ProcessMesh.
        """
        return self._processes

    @property
    def ndim(self):
        r"""
        Get the number of dimension of ProcessMesh.
        """
        return len(self._topology)

    def __eq__(self, other):
        if not isinstance(other, ProcessMesh):
            return False
        if self.topology != other.topology or self.processes != other.processes:
            return False
        return True

    def __ne__(self, other):
        return not self.__eq__(other)

    def __str__(self):
        str = "shape {} and process group {}".format(self.topology,
                                                     self.processes)
        return str
