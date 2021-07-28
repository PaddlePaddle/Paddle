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

__all__ = []

g_process_mesh_map = dict()


class ProcessMesh(object):
    """
    A class to describe the logical topology of processes.

    Args:
        topology (list): a list to describe the process topology
        process_group (list): a list of processes belonging to this group
        parent_index (int): the index of the parent ProcessMesh. None means
            no parent ProcessMesh.

    Examples:
        .. code-block:: python

            import paddle
            import paddle.distributed as dist
            
            dp_degree = 2
            pp_degree = 2
            mp_degree = 2
            mesh = ProcessMesh([dp_degree, pp_degree, mp_degree])
    """

    def __init__(self, topology, process_group=None, parent_id=None):
        assert topology, "You must specify the topology for ProcessMesh."
        process_num = np.prod(topology)
        if process_group is None:
            process_group = list(range(process_num))
        assert len(process_group) == process_num

        if parent_id is None: parent_id = _NO_PARENT

        self.desc = core.ProcessMeshDesc(topology, process_group, parent_id)
        cur_id = self.desc.id
        self._id = cur_id
        self._parent_id = parent_id
        self._topology = topology
        self._process_group = process_group
        assert cur_id not in g_process_mesh_map, "%d already exists." % cur_id
        g_process_mesh_map[cur_id] = self

    @property
    def topology(self):
        return self._topology

    @property
    def process_group(self):
        return self._process_group

    @property
    def rank(self):
        return len(self._topology)

    @property
    def id(self):
        return self._id

    @property
    def parent(self):
        if self._parent_id == -1: return None
        assert self._parent_id in g_process_mesh_map, \
            "parent (%d) does not exist."%self._parent_id
        return g_process_mesh_map[self._parent_id]

    def __eq__(self, other):
        assert other and isinstance(other, ProcessMesh)
        if len(self._topology) != len(other._topology): return False
        if self._topology != other._topology or self._process_group != other._process_group:
            return False
        return True

    def __ne__(self, other):
        return not self.__eq__(other)


def validate_check():
    pass


def shard_tensor(tensor, mesh, dims_mapping):
    """
    Add distributed attributes for tensors.
    Inputs:
        tensor (Variable): tensor to process， it's an instance of Variable (framework.py)
        mesh (ProcessMesh): an instance of ProcessMesh
        dims_mapping (list): a list to describe the mapping between tensor shape and mesh topology
    Returns:
        The tensor itself.
    """
    validate_check()
    tensor._set_attr('MESH_ID', mesh.id)
    tensor._set_attr('DIMS_MAPPING', dims_mapping)
    return tensor


def set_shard_mask(tensor, mask_out):
    """
    Set the mask for a tensor which mask out the tensor from some processes in its mesh.
    Inputs:
        tensor (Variable): tensor to process， it's an instance of Variable (framework.py)
        mask (list): mask out tensor from some processes in mesh.
    Returns:
        The tensor itself.
    """
    validate_check()
    tensor._set_attr('MASK_OUT', mask)
    return tensor


def shard_op(fn_call, mesh, input_dims_mapping, output_dims_mapping):
    """
    Add distributed attributes for ops.
    Inputs:
        fn_call (func_call): a call of an API.
        mesh (ProcessMesh): an instance of ProcessMesh
        input_dims_mapping (dict): a mapping from input name to the input's dims_mapping
        output_dims_mapping(dict): a mapping from output name to the output's dims_mapping
    Returns:
        Output variables of the op named op_name(tuple).
    """
    validate_check()
    main_prog = paddle.fluid.default_main_program()
    main_block = main_prog.global_block()
    op_size = len(main_block.ops)
    op = main_block.ops[op_size - 1]
    op._set_distributed_attr('MESH_ID', mesh.id)
    for name in input_dims_mapping:
        op._set_attr(name, input_dims_mapping[name])
    for name in output_dims_mapping:
        op._set_attr(name, output_dims_mapping[name])


def set_offload_device(tensor, dst_device):
    """
    Set the device that the tensor on.
    Inputs:
        op (tensor): tensor to process, it's an instance of Variable (framework.py)
        dst_device (str): the device that the tensor on, e.g., 'gpu', 'cpu'.
    Returns:
        None.
    """
    tensor._set_attr('OFFLOAD_DEVICE', dst_device)


def set_pipeline_stage(stage):
    """
    Set the pipeline stage of the following ops.
    Inputs:
        stage: the pipeline stage the following ops belonging to
    Returns:
        None.
    """
    from paddle.fluid.framework import _current_pipeline_stage
    _current_pipeline_stage = stage
