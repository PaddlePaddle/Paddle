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
# limitations under the License

from .utils import generate_distributed_attr_uid

TENSOR_DISTRIBUTED_ATTR_MAP_FOR_PROGRAM = {}
OP_DISTRIBUTED_ATTR_MAP_FOR_PROGRAM = {}


def get_tensor_distributed_attr_program(tensor):
    distributed_attr_uid = tensor.get_distributed_attr_uid()
    global TENSOR_DISTRIBUTED_ATTR_MAP_FOR_PROGRAM
    tensor_dist_attr = TENSOR_DISTRIBUTED_ATTR_MAP_FOR_PROGRAM.get(
        distributed_attr_uid, None)
    return tensor_dist_attr


def set_tensor_distributed_attr_program(tensor, tensor_dist_attr):
    distributed_attr_uid = tensor.get_distributed_attr_uid()
    if distributed_attr_uid == -1:
        distributed_attr_uid = generate_distributed_attr_uid()
        tensor.set_distributed_attr_uid(distributed_attr_uid)
    global TENSOR_DISTRIBUTED_ATTR_MAP_FOR_PROGRAM
    TENSOR_DISTRIBUTED_ATTR_MAP_FOR_PROGRAM[
        distributed_attr_uid] = tensor_dist_attr


def get_op_distributed_attr_program(op):
    distributed_attr_uid = op.get_distributed_attr_uid()
    global TENSOR_DISTRIBUTED_ATTR_MAP_FOR_PROGRAM
    op_dist_attr = TENSOR_DISTRIBUTED_ATTR_MAP_FOR_PROGRAM.get(
        distributed_attr_uid, None)
    return op_dist_attr


def set_op_distributed_attr_program(op, op_dist_attr):
    distributed_attr_uid = op.get_distributed_attr_uid()
    if distributed_attr_uid == -1:
        distributed_attr_uid = generate_distributed_attr_uid()
        op.set_distributed_attr_uid(distributed_attr_uid)
    global TENSOR_DISTRIBUTED_ATTR_MAP_FOR_PROGRAM
    TENSOR_DISTRIBUTED_ATTR_MAP_FOR_PROGRAM[distributed_attr_uid] = op_dist_attr


class TensorDistributedAttribute:
    def __init__(self, var_desc):
        self._var_desc = var_desc
        self._process_mesh = None
        self._dims_mapping = None
        self._shard_mask = None
        self._offload_device = None
        self._is_annotated = {}

    def get_var_desc(self):
        return self._var_desc

    def get_process_mesh(self):
        return self._process_mesh

    def set_process_mesh(self, process_mesh, is_annotated=False):
        self._process_mesh = process_mesh
        self._is_annotated['process_mesh'] = is_annotated

    def get_dims_mapping(self):
        return self._dims_mapping

    def set_dims_mapping(self, dims_mapping, is_annotated=False):
        self._dims_mapping = dims_mapping
        self._is_annotated['dims_mapping'] = is_annotated

    def get_shard_mask(self):
        return self._shard_mask

    def set_shard_mask(self, shard_mask, is_annotated=False):
        self._shard_mask = shard_mask
        self._is_annotated['shard_mask'] = is_annotated

    def get_offload_device(self):
        return self._offload_device

    def set_offload_device(self, offload_device, is_annotated=False):
        self._offload_device = offload_device
        self._is_annotated['offload_device'] = is_annotated

    def is_annotated(self, dist_attr_name):
        return self._is_annotated[dist_attr_name]


class OperatorDistributedAttribute:
    def __init__(self, op_desc):
        self._op_desc = op_desc
        self._process_mesh = None
        self._inputs_dims_mappings = {}
        self._outputs_dims_mappings = {}
        self._is_annotated = {}
        self._is_annotated_inputs_dims_mapping = {}
        self._is_annotated_outputs_dims_mapping = {}
        self._impl_idx = None

    def get_op_desc(self):
        return self._op_desc

    def get_process_mesh(self):
        return self._process_mesh

    def set_process_mesh(self, process_mesh, is_annotated=False):
        self._process_mesh = process_mesh
        self._is_annotated['process_mesh'] = is_annotated

    def get_input_dims_mapping(self, name):
        return self._inputs_dims_mappings[name]

    def set_input_dims_mapping(self, name, dims_mapping, is_annotated=False):
        self._inputs_dims_mappings[name] = dims_mapping
        self._is_annotated_inputs_dims_mapping[name] = is_annotated

    def get_input_dim_mapping(self, name, dim):
        return self._inputs_dims_mappings[name][dim]

    def set_input_dim_mapping(self, name, dim, dim_mapping):
        self._inputs_dims_mappings[name][dim] = dim_mapping

    def get_output_dims_mapping(self, name):
        return self._outputs_dims_mappings[name]

    def set_output_dims_mapping(self, name, dims_mapping, is_annotated=False):
        self._outputs_dims_mappings[name] = dims_mapping
        self._is_annotated_outputs_dims_mapping[name] = is_annotated

    def get_output_dim_mapping(self, name, dim):
        return self._outputs_dims_mappings[name][dim]

    def set_output_dim_mapping(self, name, dim, dim_mapping):
        self._outputs_dims_mappings[name][dim] = dim_mapping

    def get_impl_idx(self):
        return self._impl_idx

    def set_impl_idx(self, impl_idx):
        self._impl_idx = impl_idx

    def is_annotated(self, dist_attr_name):
        return self._is_annotated[dist_attr_name]

    def is_annotated_input_dims_mapping(self, name):
        return self._is_annotated_inputs_dims_mapping[name]

    def is_annotated_output_dims_mapping(self, name):
        return self._is_annotated_outputs_dims_mapping[name]
