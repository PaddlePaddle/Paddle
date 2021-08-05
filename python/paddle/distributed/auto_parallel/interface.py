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

import numpy as np

import paddle.fluid.core as core
import paddle
from paddle.fluid.framework import Variable
from paddle.fluid.framework import in_dygraph_mode

__all__ = []

_g_process_mesh_map = dict()

# user defined map from logical process ids to physical ones
_user_defined_physical_map = None


def _append_attr_suffix(name):
    """
    Append Auto Parallel Suffix for distributed attribute.
    """
    return name + core.kAutoParallelSuffix()


def _remove_attr_suffix(name):
    """
    Remove Auto Parallel Suffix for distributed attribute.
    """
    return name.strip(core.kAutoParallelSuffix())


def _static_mode_check():
    if in_dygraph_mode():
        raise RuntimeError("Now auto-parallel only supports static mode, please"
                           " use paddle.enable_static().")


class ProcessMesh(object):
    """
    A mesh is an n-dimensional array of logical processes. The shape of the 
    n-dimensional array represents the topology of logical processes and each
    element of the n-dimensional array represent a logical process. For
    example, the following diagram is represented by the n-dimensional array
    numpy.array([[2, 4, 5], [0, 1, 3]]), and first logical process is the one
    with id=2.

    | 2 | 4 | 5 |

    | 0 | 1 | 3 |

    Args:
        mesh (numpy.ndarray): an n-dimensional array of processes. Its data type is int.
        parent (ProcessMesh): the parent ProcessMesh. None means
            no parent ProcessMesh.
    
    Returns:
        None

    Raises:
        ValueError: if ``mesh`` is not an instance of numpy.ndarray.

    Examples:
        .. code-block:: python

            import numpy as np
            import paddle
            import paddle.distributed as dist
            
            paddle.enable_static()
            
            mesh = dist.ProcessMesh(np.array([[2, 4, 5], [0, 1, 3]]))
            assert mesh.parent is None
            assert mesh.topology == [2, 3]
            assert mesh.process_group == [2, 4, 5, 0, 1, 3]
            mesh.set_placement([0, 1, 2, 3, 4, 5])
    """

    def __init__(self, mesh, parent=None):
        _static_mode_check()
        if mesh is None or not isinstance(mesh, np.ndarray):
            raise ValueError("mesh must be an instance of numpy.ndarray.")

        # Every element of mesh must be greater than or equal to 0.
        assert np.min(mesh) >= 0, ('Every element of mesh must be greater '
                                   'than or equal to 0.')

        unique_ids = set(mesh.flatten().tolist())
        assert len(unique_ids) == mesh.size, ("All logical process ids "
                                              "in mesh must be unique.")

        if parent is None:
            # For root ProcessMesh, the ids of logical processes must be range
            # from 0 to N-1, where N is the number of logical processes. 
            assert np.max(mesh) == mesh.size - 1, (
                'For root ProcessMesh, the '
                'ids of logical processes must be range from 0 to N-1, where '
                'N is the number of logical processes.')

            parent_id = core.kNoneProcessMeshIndex()
            assert len(_g_process_mesh_map.keys()) == 0, (
                'The first '
                'ProcessMesh must be the root, which has no parent.')
        else:
            assert isinstance(parent, ProcessMesh), (
                'parent must be an instance'
                ' of ProcessMesh.')
            parent_id = parent._desc.id

            # All elements in mesh must belong to its parent ProcessMesh
            parent_unique_ids = set(parent.process_group)
            assert unique_ids <= parent_unique_ids, (
                'All elements in mesh '
                'must belong to its parent ProcessMesh')

            # Get the root ProcessMesh
            root_mesh = None
            root_id = parent_id
            while root_id != core.kNoneProcessMeshIndex():
                root_mesh = _g_process_mesh_map[root_id]
                root_id = root_mesh._desc.parent
            max_id = max(root_mesh.process_group)
            assert np.max(mesh) <= max_id, (
                'Every element in mesh must be '
                'less than equal to the max one in root ProcessMesh.')

        self._topology = list(mesh.shape)
        self._processes = mesh.flatten().tolist()
        self._desc = core.ProcessMeshDesc(self._topology, self._processes,
                                          parent_id)

        self._id = self._desc.id
        self._parent_id = parent_id
        assert self._id not in _g_process_mesh_map, "%d already exists." % self._id
        _g_process_mesh_map[self._id] = self

    @property
    def topology(self):
        """
        Get the topology of logical processes belonging to this ProcessMesh.

        Args:
            None
        
        Returns:
            list: A list of `int` represents the topology of logical processes.

        Examples:
            .. code-block:: python

                import numpy as np
                import paddle
                import paddle.distributed as dist
                
                paddle.enable_static()
                
                mesh = dist.ProcessMesh(np.array([[2, 4, 5], [0, 1, 3]]))
                assert mesh.topology == [2, 3]
        """
        return self._topology

    @property
    def process_group(self):
        """
        Get all processes belonging to this ProcessMesh.

        Args:
            None
        
        Returns:
            list: A list of `int` represents all logical processes.

        Examples:
            .. code-block:: python

                import numpy as np
                import paddle
                import paddle.distributed as dist
                
                paddle.enable_static()
                
                mesh = dist.ProcessMesh(np.array([[2, 4, 5], [0, 1, 3]]))
                assert mesh.process_group == [2, 4, 5, 0, 1, 3]
        """
        return self._processes

    @property
    def parent(self):
        """
        Get the parent ProcessMesh.

        Args:
            None
        
        Returns:
            ProcessMesh: the parent ProcessMesh.

        Examples:
            .. code-block:: python

                import numpy as np
                import paddle
                import paddle.distributed as dist
                
                paddle.enable_static()
                
                mesh = dist.ProcessMesh(np.array([[2, 4, 5], [0, 1, 3]]))
                assert mesh.parent is None
        """
        if self._parent_id == core.kNoneProcessMeshIndex(): return None
        assert self._parent_id in _g_process_mesh_map, \
            "parent (%d) does not exist."%self._parent_id
        return _g_process_mesh_map[self._parent_id]

    def set_placement(self, order):
        """
        Set the order of the physical process ids.

        Args:
            order (list): order of the physical process ids.
        
        Returns:
            None

        Examples:
            .. code-block:: python

                import numpy as np
                import paddle
                import paddle.distributed as dist
                
                paddle.enable_static()
                
                mesh = dist.ProcessMesh(np.array([[2, 4, 5], [0, 1, 3]]))
                mesh.set_placement([0, 1, 2, 3, 4, 5])
        """
        assert self.parent is None, ("This function can only be called by the "
                                     "root ProcessMesh.")
        unique_ids = set(order)
        assert isinstance(order, list)

        assert len(unique_ids) == len(order), ("All physical process ids "
                                               "in must be unique.")
        assert min(order) == 0
        assert max(order) == len(order) - 1

        logical_order = self.process_group
        global _user_defined_physical_map
        if _user_defined_physical_map is None:
            _user_defined_physical_map = dict()
        assert len(logical_order) == len(order)
        for idx, l_id in enumerate(logical_order):
            _user_defined_physical_map[l_id] = order[idx]

    def __eq__(self, other):
        assert other and isinstance(other, ProcessMesh)
        if self.topology != other.topology or self.process_group != other.process_group:
            return False
        return True

    def __ne__(self, other):
        return not self.__eq__(other)


def _dims_mapping_checker(tensor, mesh, dims_mapping):
    assert len(tensor.shape) == len(dims_mapping)
    mesh_dim = len(mesh.topology)
    dim_set = set()
    for i in range(len(dims_mapping)):
        assert dims_mapping[i] == -1 or (dims_mapping[i] < mesh_dim and
                                         dims_mapping[i] >= 0)
        if dims_mapping[i] >= 0:
            assert dims_mapping[i] not in dim_set
            dim_set.add(dims_mapping[i])


def shard_tensor(x, mesh, dims_mapping):
    """
    Add distributed attributes for a tensors.

    Args:
        x (Tensor): the tensor to process.
        mesh (ProcessMesh): an n-dimensional array of logical processes.
        dims_mapping (list): a list to describe the mapping between `x` and `mesh`,
            the dimension `i` of `x` is split across the dimension represented by
            dims_mapping[i], where -1 means without parition along the corresponding
            dimension.

    Returns:
        Tensor: the tensor `x` itself.

    Examples:
        .. code-block:: python

            import numpy as np
            import paddle
            import paddle.distributed as dist
            
            paddle.enable_static()

            mesh = dist.ProcessMesh(np.array([[2, 4, 5], [0, 1, 3]]))
            x = paddle.ones([4, 6])
            dist.shard_tensor(x, mesh, [0, -1])
    """
    _static_mode_check()
    _dims_mapping_checker(x, mesh, dims_mapping)
    attr_name = _append_attr_suffix('mesh_id')
    x._set_attr(attr_name, mesh._id)
    attr_name = _append_attr_suffix('dims_mapping')
    x._set_attr(attr_name, dims_mapping)
    return x


def set_shard_mask(x, mask):
    """
    Set the mask for a tensor which mask out the tensor from some processes in its mesh.

    Args:
        x (Tensor): the tensor to process.
        mask (numpy.ndarray): the shape of `mask` must be the same as the ProcessMesh info belonging to
            the tensor `x`. The value of every element of `mask` must be one or zero, where one means 
            the tenor `x` will be put on the corresponding logical process and zero means the tensor `x`
            will not be put on the corresponding logical process.
            For example, the following diagram is represented by the n-dimensional array
            numpy.array([[2, 4, 5], [0, 1, 3]]).

                 | 2 | 4 | 5 |

                 | 0 | 1 | 3 |

            And the following diagram gives the `mask`.

                 | 1 | 0 | 1 |

                 | 0 | 1 | 0 |

            Then, the tensor `x` will be put on logical processes 2, 5 and 1.

    Returns:
        Tensor: the tensor `x` itself.

    Examples:
        .. code-block:: python

            import numpy as np
            import paddle
            import paddle.distributed as dist

            paddle.enable_static()
            
            mesh = dist.ProcessMesh(np.array([[2, 4, 5], [0, 1, 3]]))
            mask = np.array([[1, 0, 1], [0, 1, 0]])
            x = paddle.ones([4, 6])
            dist.set_shard_mask(x, mask)
    """
    _static_mode_check()
    assert isinstance(mask, np.ndarray)
    attr_name = _append_attr_suffix('mask')
    x._set_attr(attr_name, mask.flatten().tolist())
    return x


def shard_op(op_fn, mesh, dims_mapping_dict, **kwargs):
    """
    Call a functioin and add distributed attributes for ops added by the function.

    Args:
        op_fn (callable): a callable object of a API.
        mesh (ProcessMesh): an instance of ProcessMesh which specifies the logical topology of ops added by the function.
        dims_mapping_dict (dict): a mapping from tensor's name to the its dims_mapping
        kwargs (dict): a dict of parameter passed to the function `op_fn`.

    Returns:
        list: the outputs of the function `op_fn`.

    Examples:
        .. code-block:: python

            import numpy as np
            import paddle
            import paddle.distributed as dist

            paddle.enable_static()
            
            mesh = dist.ProcessMesh(np.array([[2, 4, 5], [0, 1, 3]]))
            x = paddle.ones([4, 6])
            y = paddle.zeros([4, 6])
            kwargs = {'x': x, 'y': y}
            dist.shard_op(paddle.add, mesh, None, **kwargs)
    """
    _static_mode_check()
    main_prog = paddle.fluid.default_main_program()
    main_block = main_prog.global_block()
    op_size = len(main_block.ops)
    output = op_fn(**kwargs)
    new_op_size = len(main_block.ops)
    if dims_mapping_dict is None: dims_mapping_dict = dict()
    for idx in range(op_size, new_op_size):
        op = main_block.ops[idx]
        attr_name = _append_attr_suffix('mesh_id')
        op._set_attr(attr_name, mesh._id)
        for var_name in op.input_arg_names + op.output_arg_names:
            if var_name in dims_mapping_dict:
                attr_name = _append_attr_suffix(var_name)
                op._set_attr(attr_name, dims_mapping_dict[var_name])

    if isinstance(output, Variable):
        output = [output]
    return list(output)


def set_offload_device(x, device):
    """
    Set the device that the tensor `x` will be put on.

    Args:
        x (tensor): the tensor to process.
        device (str): the device that the tensor `x` will be put on, e.g., 'gpu:0', 'cpu'.

    Returns:
        Tensor: the tensor `x` itself.

    Examples:
        .. code-block:: python

            import numpy as np
            import paddle
            import paddle.distributed as dist

            paddle.enable_static()
            
            x = paddle.ones([4, 6])
            dist.set_offload_device(x, 'cpu')
    """
    _static_mode_check()
    attr_name = _append_attr_suffix("offload_device")
    x._set_attr(attr_name, device)
    return x


def set_pipeline_stage(stage):
    """
    Set the pipeline stage of the following ops.

    Args:
        stage (int): the pipeline stage the following ops belonging to

    Returns:
        None.

    Examples:
        .. code-block:: python

            import numpy as np
            import paddle
            import paddle.distributed as dist

            paddle.enable_static()
            
            dist.set_pipeline_stage(0)
    """
    from paddle.fluid.framework import _set_pipeline_stage
    _static_mode_check()
    _set_pipeline_stage(stage)
