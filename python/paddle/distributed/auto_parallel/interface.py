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

__all__ = []

g_process_mesh_map = dict()


class ProcessMesh(object):
    """
    A class to describe the logical topology of processes.

    Args:
        topology (list): a list to describe the process topology
        process_group (list): a list of processes belonging to this group
        parent_index (int): the index of the parent ProcessMesh. None means
            that it has no parent ProcessMesh.

    Examples:
        .. code-block:: python

            import paddle
            import paddle.distributed as dist
            
            dp_degree = 2
            pp_degree = 2
            mp_degree = 2
            mesh = ProcessMesh([dp_degree, pp_degree, mp_degree])
    """

    def __init__(self, topology, process_group=None, parent_index=None):
        assert topology, "You must specify the topology for ProcessMesh."
        process_num = np.prod(mesh)
        if process_group is None:
            process_group = list(range(process_num))
        assert len(process_group) == process_num

        if parent_index is None: parent_index = -1

        self.desc = core.ProcessMesh(topology, process_group, parent_index)
        cur_idx = self.desc.id
        assert cur_idx not in g_process_mesh_map, "%d already exists." % cur_idx
        g_process_mesh_map[cur_idx] = self

    @property
    def topology(self):
        return self.desc.topology

    @property
    def process_group(self):
        return self.desc.process_group

    @property
    def rank(self):
        return len(self.desc.topology)

    @property
    def id(self):
        return self.desc.id

    @property
    def parent(self):
        parent_idx = self.desc.parent_idx
        if parent_idx == -1: return None
        assert parent_idx in g_process_mesh_map, \
            "parent (%d) does not exist."%parent_idx
        return g_process_mesh_map[parent_idx]

    def __eq__(self, other):
        assert other
        return self.desc.equal(other.desc)

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
    tensor.desc._set_distributed_attr('mesh_id', mesh.id)
    tensor.desc._set_distributed_attr('dims_mapping', dims_mapping)
    return tensor


def set_shard_mask(tensor, mask):
    """
    Set the mask for a tensor which mask out the tensor from some processes in its mesh.
    Inputs:
        tensor (Variable): tensor to process， it's an instance of Variable (framework.py)
        mask (Variable): mask out tensor from some processes in mesh.
    Returns:
        The tensor itself.
    """
    validate_check()
    tensor.desc._set_distributed_attr('mask', mask)
    return tensor


def shard_op(op_name, mesh, input_dims_mapping, output_dims_mapping):
    """
    Add distributed attributes for ops.
    Inputs:
        op_name (string): the name of the  op to process
        mesh (ProcessMesh): an instance of ProcessMesh
        input_dims_mapping (dict): a mapping from input name to the input's dims_mapping
        output_dims_mapping(dict): a mapping from output name to the output's dims_mapping
    Returns:
        Output variables of the op named op_name(tuple).
    """
    validate_check()
    # op_mapping[op_name](parameter list from input_dims_mapping)
    op.desc._set_distributed_attr('mesh_id', mesh.id)
    op.desc._set_distributed_attr('input_dims_mapping', input_dims_mapping)
    op.desc._set_distributed_attr('output_dims_mapping', output_dims_mapping)
    # input_dims_mapping = {index: {'name': in_name, 'dims_mapping': dims_mapping}}


def set_offload_device(tensor, dst_device):
    """
    Set the device that the tensor on.
    Inputs:
        op (tensor): tensor to process, it's an instance of Variable (framework.py)
        dst_device: the device that the tensor on, e.g., 'gpu', 'cpu'.
    Returns:
        None.
    """
    tensor.desc._set_distributed_attr('offload_device', dst_device)


def set_pipeline_stage(stage):
    """
    Set the pipeline stage of the following ops.
    Inputs:
        stage: the pipeline stage the following ops belonging to
    Returns:
        None.
    """
    op.desc._set_distributed_attr('pipeline_stage', stage)
