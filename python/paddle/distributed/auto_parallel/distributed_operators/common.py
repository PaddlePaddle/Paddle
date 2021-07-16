# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

from enum import IntEnum
from collections import defaultdict

DISTRIBUTED_OPERATORS = {}


class ShardTag(IntEnum):
    # Replicate
    Replicate = -1
    # Split
    Split = -2
    # Any
    Any = -3


class OperatorDistributedSignature:
    def __init__(self):
        self.declared_proc_mesh_ndim_set = set()
        self.declared_inputs_dims_mappings = defaultdict(dict)
        self.declared_outputs_dims_mappings = defaultdict(dict)
        self.declared_inputs_same_shard_dims_list = list()
        self.declared_outputs_same_shard_dims_list = list()
        self.declared_inputs_outputs_same_shard_dims_list = list()

    def add_valid_proc_mesh_ndim(self, ndim):
        self.declared_proc_mesh_ndim_set.add(ndim)

    def get_valid_proc_mesh_ndim_set(self):
        return self.declared_proc_mesh_ndim_set

    def set_valid_input_dim_shard(self, name, dim, tag):
        self.declared_inputs_dims_mappings[name][dim] = tag

    def get_valid_input_dims_mapping(self, name):
        return self.declared_inputs_dims_mappings[name]

    def set_valid_output_dim_shard(self, name, dim, tag):
        self.declared_outputs_dims_mappings[name][dim] = tag

    def get_valid_output_dims_mapping(self, name):
        return self.declared_outputs_dims_mappings[name]

    def add_valid_inputs_same_shard_dims(self, same_shard_dims):
        self.declared_inputs_same_shard_dims_list.append(same_shard_dims)

    def get_valid_inputs_same_shard_dims_list(self):
        return self.declared_inputs_same_shard_dims_list

    def add_valid_outputs_same_shard_dims(self, same_shard_dims):
        self.declared_outputs_same_shard_dims_list.append(same_shard_dims)

    def get_valid_outputs_same_shard_dims_list(self):
        return self.declared_outputs_same_shard_dims_list

    def add_valid_inputs_outputs_same_shard_dims(self, same_shard_dims):
        self.declared_inputs_outputs_same_shard_dims_list.append(
            same_shard_dims)

    def get_valid_inputs_outputs_same_shard_dims_list(self):
        return self.declared_inputs_outputs_same_shard_dims_list

    def is_input_compatible(self, op_dist_attr):
        # Check whether proc_mesh_ndim is valid
        proc_mesh = op_dist_attr.get_process_mesh()
        proc_mesh_ndim = len(proc_mesh)
        if proc_mesh_ndim not in self.declared_proc_mesh_ndim_set:
            return False

        # Check each input_dims_mapping
        for param_name in self.declared_inputs_dims_mappings.keys():
            op_desc = op_dist_attr.get_op_desc()
            # Each Argument must conform to its corresponding parameter
            for arg_name in op_desc.input(param_name):
                input_dims_mapping = op_dist_attr.get_input_dims_mapping(
                    arg_name, None)
                # May not need?
                if input_dims_mapping is None:
                    return False
                for dim in self.declared_inputs_dims_mappings[param_name].keys(
                ):
                    if dim < -len(input_dims_mapping) or dim >= len(
                            input_dims_mapping):
                        return False
                    if (self.declared_inputs_dims_mappings[param_name][dim] ==
                            ShardTag.Replicate and
                            input_dims_mapping[dim] != -1):
                        return False
                    if (self.declared_inputs_dims_mappings[param_name][dim] ==
                            ShardTag.Split and input_dims_mapping[dim] == -1):
                        return False

        # Check input_dims_mapping between inputs 
        for same_shard_dims in self.declared_inputs_same_shard_dims_list:
            # Save dim_mappings from first param_name 
            in_or_out, param_name, dim = same_shard_dims[0]
            assert in_or_out == "input"
            saved_dim_mapping = []
            for arg_name in op_desc.input(param_name):
                input_dims_mapping = op_dist_attr.get_input_dims_mapping(
                    arg_name)
                saved_dim_mapping.append(input_dims_mapping[dim])
            # Check other param with saved results
            for in_or_out, param_name, dim in same_shard_dims[1:]:
                assert in_or_out == "input"
                for idx, arg_name in enumerate(op_desc.input(param_name)):
                    input_dims_mapping = op_dist_attr.get_input_dims_mapping(
                        arg_name)
                    if input_dims_mapping[dim] != saved_dim_mapping[idx]:
                        return False

        return True

    def is_output_compatible(self, op_dist_attr):
        # Check whether proc_mesh_ndim is valid
        proc_mesh = op_dist_attr.get_process_mesh()
        proc_mesh_ndim = len(proc_mesh)
        if proc_mesh_ndim not in self.declared_proc_mesh_ndim_set:
            return False

        op_desc = op_dist_attr.get_op_desc()
        # Check each output_dims_mapping
        for param_name in self.declared_outputs_dims_mappings.keys():
            # Each Argument must conform to its corresponding parameter
            for arg_name in op_desc.output(param_name):
                output_dims_mapping = op_dist_attr.get_output_dims_mapping(
                    arg_name, None)
                # May not need?
                if output_dims_mapping is None:
                    return False
                for dim in self.declared_outputs_dims_mappings[param_name].keys(
                ):
                    if dim < -len(output_dims_mapping) or dim >= len(
                            output_dims_mapping):
                        return False
                    if (self.declared_outputs_dims_mappings[param_name][dim] ==
                            ShardTag.Replicate and
                            output_dims_mapping[dim] != -1):
                        return False
                    if (self.declared_outputs_dims_mappings[param_name][dim] ==
                            ShardTag.Split and output_dims_mapping[dim] == -1):
                        return False

        # Check output_dims_mapping between outputs 
        for same_shard_dims in self.declared_outputs_same_shard_dims_list:
            # Save dim_mappings from first param_name 
            in_or_out, param_name, dim = same_shard_dims[0]
            assert in_or_out == "output"
            saved_dim_mapping = []
            for arg_name in op_desc.output(param_name):
                output_dims_mapping = op_dist_attr.get_output_dims_mapping(
                    arg_name)
                saved_dim_mapping.append(output_dims_mapping[dim])
            # Check other param with saved results
            for in_or_out, param_name, dim in same_shard_dims[1:]:
                assert in_or_out == "output"
                for idx, arg_name in enumerate(op_desc.output(param_name)):
                    output_dims_mapping = op_dist_attr.get_output_dims_mapping(
                        arg_name)
                    if output_dims_mapping[dim] != saved_dim_mapping[idx]:
                        return False
        return True

    def is_input_output_compatible(self, op_dist_attr):
        # Check whether proc_mesh_ndim is valid
        proc_mesh = op_dist_attr.get_process_mesh()
        proc_mesh_ndim = len(proc_mesh)
        if proc_mesh_ndim not in self.declared_proc_mesh_ndim_set:
            return False
        op_desc = op_dist_attr.get_op_desc()
        # Check output_dims_mapping between outputs
        for same_shard_dims in self.declared_inputs_outputs_same_shard_dims_list:
            # Save dim_mappings from first param_name 
            in_or_out, param_name, dim = same_shard_dims[0]
            saved_dim_mapping = []
            if in_or_out == 'input':
                for arg_name in op_desc.input(param_name):
                    input_dims_mapping = op_dist_attr.get_input_dims_mapping(
                        arg_name)
                    saved_dim_mapping.append(input_dims_mapping[dim])
            else:
                for arg_name in op_desc.output(param_name):
                    output_dims_mapping = op_dist_attr.get_output_dims_mapping(
                        arg_name)
                    saved_dim_mapping.append(output_dims_mapping[dim])

            # Check other param with saved results
            for in_or_out, param_name, dim in same_shard_dims[1:]:
                if in_or_out == 'input':
                    for idx, arg_name in enumerate(op_desc.input(param_name)):
                        input_dims_mapping = op_dist_attr.get_input_dims_mapping(
                            arg_name)
                        if input_dims_mapping[dim] != saved_dim_mapping[idx]:
                            return False
                else:
                    for idx, arg_name in enumerate(op_desc.output(param_name)):
                        output_dims_mapping = op_dist_attr.get_output_dims_mapping(
                            arg_name)
                        if output_dims_mapping[dim] != saved_dim_mapping[idx]:
                            return False
        return True


class DistributedOperator:
    def __init__(self):
        self.impls = []
        self.name = None

    def register_impl(self, dist_impl):
        self.impls.append(dist_impl)

    def get_impl(self, impl_idx):
        return self.impls[impl_idx]

    def get_impls(self):
        return self.impls


class DistributedOperatorImpl:
    def __init__(self):
        self.dist_singnature = None
        self.name = None

    def forward(self, serial_op):
        pass

    def backward(self, serial_grad_op):
        pass

    def get_distributed_signature(self):
        return self.dist_singnature


def register_distributed_operator(name, dist_op):
    global DISTRIBUTED_OPERATORS
    DISTRIBUTED_OPERATORS[name] = dist_op


def get_distributed_operator(name):
    global DISTRIBUTED_OPERATORS
    return DISTRIBUTED_OPERATORS[name]


def register_distributed_operator_impl(name, dist_impl):
    dist_op = get_distributed_operator(name)
    dist_op.register_impl(dist_impl)


def get_distributed_operator_impl(name, impl_idx):
    global DISTRIBUTED_OPERATORS
    return DISTRIBUTED_OPERATORS[name].get_impl(impl_idx)


def find_best_compatible_distributed_operator_impl(name, op_dist_attr,
                                                   fwd=True):
    compatible_impls = []
    dist_op = get_distributed_operator(name)
    impls = dist_op.get_impls()
    if fwd:
        for idx, impl in enumerate(impls):
            if impl.is_input_compatible(op_dist_attr):
                compatible_impls.append((impl, idx))
    else:
        for idx, impl in enumerate(impls):
            if impl.is_output_compatible(op_dist_attr):
                compatible_impls.append((impl, idx))

    if compatible_impls:
        best_compatible_impl, idx = compatible_impls[0]
    else:
        best_compatible_impl, idx = None, -1

    return best_compatible_impl, idx
