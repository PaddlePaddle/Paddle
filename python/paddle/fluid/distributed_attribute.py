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

import copy
from collections import defaultdict
from . import core
from . import framework


DEFAULT_DISTRIBUTED_CONFIGURATION = None


def get_default_distributed_config():
    global DEFAULT_DISTRIBUTED_CONFIGURATION
    if DEFAULT_DISTRIBUTED_CONFIGURATION is None:
        dist_config = DistributedConfiguration()
        set_default_distributed_config(dist_config)
    return DEFAULT_DISTRIBUTED_CONFIGURATION


def set_default_distributed_config(dist_config):
    global DEFAULT_DISTRIBUTED_CONFIGURATION
    DEFAULT_DISTRIBUTED_CONFIGURATION = dist_config


# TENSOR_DISTRIBUTED_ATTR_ACC_NUM = -1
# OP_DISTRIBUTED_ATTR_ACC_NUM = -1
# TENSOR_DISTRIBUTED_ATTR_MAP_FOR_PROGRAM = {}
# OP_DISTRIBUTED_ATTR_MAP_FOR_PROGRAM = {}
# 
# 
# def generate_tensor_distributed_attr_uid():
#     global TENSOR_DISTRIBUTED_ATTR_ACC_NUM
#     TENSOR_DISTRIBUTED_ATTR_ACC_NUM = TENSOR_DISTRIBUTED_ATTR_ACC_NUM + 1
#     return TENSOR_DISTRIBUTED_ATTR_ACC_NUM
# 
# 
# def generate_op_distributed_attr_uid():
#     global OP_DISTRIBUTED_ATTR_ACC_NUM
#     OP_DISTRIBUTED_ATTR_ACC_NUM = OP_DISTRIBUTED_ATTR_ACC_NUM + 1
#     return OP_DISTRIBUTED_ATTR_ACC_NUM
# 
# 
# def get_tensor_distributed_attr_program(tensor):
#     distributed_attr_uid = tensor.get_distributed_attr_uid()
#     global TENSOR_DISTRIBUTED_ATTR_MAP_FOR_PROGRAM
#     tensor_dist_attr = TENSOR_DISTRIBUTED_ATTR_MAP_FOR_PROGRAM.get(
#         distributed_attr_uid, None)
#     return tensor_dist_attr
# 
# 
# def set_tensor_distributed_attr_program(tensor, tensor_dist_attr):
#     distributed_attr_uid = tensor.get_distributed_attr_uid()
#     if distributed_attr_uid == -1:
#         distributed_attr_uid = generate_tensor_distributed_attr_uid()
#         tensor.set_distributed_attr_uid(distributed_attr_uid)
#     global TENSOR_DISTRIBUTED_ATTR_MAP_FOR_PROGRAM
#     TENSOR_DISTRIBUTED_ATTR_MAP_FOR_PROGRAM[
#         distributed_attr_uid] = tensor_dist_attr
# 
# 
# def get_op_distributed_attr_program(op):
#     distributed_attr_uid = op.get_distributed_attr_uid()
#     global OP_DISTRIBUTED_ATTR_MAP_FOR_PROGRAM
#     op_dist_attr = OP_DISTRIBUTED_ATTR_MAP_FOR_PROGRAM.get(distributed_attr_uid,
#                                                            None)
#     return op_dist_attr
# 
# 
# def set_op_distributed_attr_program(op, op_dist_attr):
#     distributed_attr_uid = op.get_distributed_attr_uid()
#     if distributed_attr_uid == -1:
#         distributed_attr_uid = generate_op_distributed_attr_uid()
#         op.set_distributed_attr_uid(distributed_attr_uid)
#     global OP_DISTRIBUTED_ATTR_MAP_FOR_PROGRAM
#     OP_DISTRIBUTED_ATTR_MAP_FOR_PROGRAM[distributed_attr_uid] = op_dist_attr


class TensorDistributedAttribute:
    def __init__(self, desc):
        self._desc = desc
        assert self._desc is not None

        self._process_mesh = None
        self._dims_mapping = None
        self._shard_mask = None
        self._offload_device = None
        self._is_annotated = {}
        self._is_parameter = False

    def get_desc(self):
        return self._desc

    def get_process_mesh(self):
        return self._process_mesh

    def set_process_mesh(self, process_mesh, is_annotated=False):
        self._process_mesh = process_mesh
        self._is_annotated["process_mesh"] = is_annotated

    def get_dims_mapping(self):
        return self._dims_mapping

    def set_dims_mapping(self, dims_mapping, is_annotated=False):
        self._dims_mapping = dims_mapping
        self._is_annotated["dims_mapping"] = is_annotated

    def get_shard_mask(self):
        return self._shard_mask

    def set_shard_mask(self, shard_mask, is_annotated=False):
        self._shard_mask = shard_mask
        self._is_annotated["shard_mask"] = is_annotated

    def get_offload_device(self):
        return self._offload_device

    def set_offload_device(self, offload_device, is_annotated=False):
        self._offload_device = offload_device
        self._is_annotated["offload_device"] = is_annotated

    def is_annotated(self, dist_attr_name):
        return self._is_annotated.get(dist_attr_name, False)

    def mark_as_parameter(self):
        self._is_parameter = True

    def is_parameter(self):
        return self._is_parameter

    def __str__(self):
        str = "{{name: {}, distributed_attr_uid: {}".format(
            self._desc.name(), self._desc.get_distributed_attr_uid())
        if self.is_annotated("process_mesh"):
            annotated_str = "annotated"
        else:
            annotated_str = "non-annotated"
        str += ", process_mesh ({}): {}".format(annotated_str,
                                                self._process_mesh)

        str += ", is_parameter: {}".format(self._is_parameter)

        if self.is_annotated("dims_mapping"):
            annotated_str = "annotated"
        else:
            annotated_str = "non-annotated"
        str += ", dims_mapping ({}): {}".format(annotated_str,
                                                self._dims_mapping)

        if self.is_annotated("shard_mask"):
            annotated_str = "annotated"
        else:
            annotated_str = "non-annotated"
        str += ", shard_mask ({}): {}".format(annotated_str, self._shard_mask)

        if self.is_annotated("offload_device"):
            annotated_str = "annotated"
        else:
            annotated_str = "non-annotated"
        str += ", offload_device ({}): {} }}".format(annotated_str,
                                                     self._offload_device)
        return str

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            if k == "_desc":
                setattr(result, k, v)
            else:
                setattr(result, k, copy.deepcopy(v, memo))
        return result


class OperatorDistributedAttribute:
    def __init__(self, desc):
        self._desc = desc
        assert self._desc is not None

        self._process_mesh = None
        self._inputs_dims_mapping = {}
        self._outputs_dims_mapping = {}
        self._is_annotated = {}
        self._is_annotated_inputs_dims_mapping = {}
        self._is_annotated_outputs_dims_mapping = {}
        self._impl_idx = None
        self._parameters = {}

    def get_desc(self):
        return self._desc

    def get_process_mesh(self):
        return self._process_mesh

    def set_process_mesh(self, process_mesh, is_annotated=False):
        self._process_mesh = process_mesh
        self._is_annotated["process_mesh"] = is_annotated

    def get_input_dims_mapping(self, name):
        return self._inputs_dims_mapping.get(name, None)

    def set_input_dims_mapping(self, name, dims_mapping, is_annotated=False):
        self._inputs_dims_mapping[name] = dims_mapping
        self._is_annotated_inputs_dims_mapping[name] = is_annotated

    def get_input_dim_mapping(self, name, dim):
        input_dims_mapping = self._inputs_dims_mapping.get(name, None)
        assert input_dims_mapping is not None
        assert dim >= -len(input_dims_mapping) or dim < len(input_dims_mapping)
        return self._inputs_dims_mapping[name][dim]

    def set_input_dim_mapping(self, name, dim, dim_mapping):
        input_dims_mapping = self._inputs_dims_mapping[name]
        assert input_dims_mapping is not None
        assert dim >= -len(input_dims_mapping) or dim < len(input_dims_mapping)
        self._inputs_dims_mapping[name][dim] = dim_mapping

    def get_output_dims_mapping(self, name):
        return self._outputs_dims_mapping.get(name, None)

    def set_output_dims_mapping(self, name, dims_mapping, is_annotated=False):
        self._outputs_dims_mapping[name] = dims_mapping
        self._is_annotated_outputs_dims_mapping[name] = is_annotated

    def get_output_dim_mapping(self, name, dim):
        output_dims_mapping = self._outputs_dims_mapping.get(name, None)
        assert output_dims_mapping is not None
        assert dim >= -len(output_dims_mapping) or dim < len(
            output_dims_mapping)
        return self._outputs_dims_mapping[name][dim]

    def set_output_dim_mapping(self, name, dim, dim_mapping):
        output_dims_mapping = self._outputs_dims_mapping.get(name, None)
        assert output_dims_mapping is not None
        assert dim >= -len(output_dims_mapping) or dim < len(
            output_dims_mapping)
        self._outputs_dims_mapping[name][dim] = dim_mapping

    def get_impl_idx(self):
        return self._impl_idx

    def set_impl_idx(self, impl_idx):
        self._impl_idx = impl_idx

    def is_annotated(self, dist_attr_name):
        return self._is_annotated.get(dist_attr_name, False)

    def is_annotated_input_dims_mapping(self, name):
        return self._is_annotated_inputs_dims_mapping.get(name, False)

    def is_annotated_output_dims_mapping(self, name):
        return self._is_annotated_outputs_dims_mapping.get(name, False)

    def mark_as_parameter(self, arg_name):
        self._parameters[arg_name] = True

    def is_parameter(self, arg_name):
        return self._parameters.get(arg_name, False)

    def __str__(self):
        str = "{{type: {}, distributed_attr_uid: {}".format(
            self._desc.type(), self._desc.get_distributed_attr_uid())

        if self.is_annotated("process_mesh"):
            annotated_str = "annotated"
        else:
            annotated_str = "non-annotated"
        str += ", process_mesh ({}): {}".format(annotated_str,
                                                self._process_mesh)

        for arg_name in self._desc.input_arg_names():
            dims_mapping = self.get_input_dims_mapping(arg_name)
            if self.is_annotated_input_dims_mapping(arg_name):
                annotated_str = "annotated"
            else:
                annotated_str = "non-annotated"
            if self.is_parameter(arg_name):
                is_parameter_str = "parameter"
            else:
                is_parameter_str = "non-parameter"
            str += ", {}'s dims_mapping (input, {}, {}): {}".format(
                arg_name, annotated_str, is_parameter_str, dims_mapping)

        for arg_name in self._desc.output_arg_names():
            dims_mapping = self.get_output_dims_mapping(arg_name)
            if self.is_annotated_output_dims_mapping(arg_name):
                annotated_str = "annotated"
            else:
                annotated_str = "non-annotated"
            if self.is_parameter(arg_name):
                is_parameter_str = "parameter"
            else:
                is_parameter_str = "non-parameter"
            str += ", {}'s dims_mapping (output, {}, {}): {}".format(
                arg_name, annotated_str, is_parameter_str, dims_mapping)

        str += ", dist_impl idx: {} }}".format(self._impl_idx)

        return str

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            if k == "_desc":
                setattr(result, k, v)
            else:
                setattr(result, k, copy.deepcopy(v, memo))
        return result


class DistributedConfiguration:
    def __init__(self):
        self._tensor_distributed_attr_acc_num = -1
        self._op_distributed_attr_acc_num = -1
        self._tensor_distributed_attr_map_for_program = {}
        self._op_distributed_attr_map_for_program = {}
        # The following two dicts will be used for completion
        self._tensor_distributed_attr_map_for_graph = {}
        self._op_distributed_attr_map_for_graph = {}

        # The following maps are used to store comm related info
        self._comm_info_map = defaultdict(dict)
        self._comm_group_map = {}

    def generate_tensor_distributed_attr_uid(self):
        self._tensor_distributed_attr_acc_num = self._tensor_distributed_attr_acc_num + 1
        return self._tensor_distributed_attr_acc_num

    def generate_op_distributed_attr_uid(self):
        self._op_distributed_attr_acc_num = self._op_distributed_attr_acc_num + 1
        return self._op_distributed_attr_acc_num

    def get_tensor_distributed_attr_program(self, tensor):
        distributed_attr_uid = tensor.get_distributed_attr_uid()
        tensor_dist_attr = self._tensor_distributed_attr_map_for_program.get(
            distributed_attr_uid, None)
        return tensor_dist_attr

    def set_tensor_distributed_attr_program(self, tensor, tensor_dist_attr):
        distributed_attr_uid = tensor.get_distributed_attr_uid()
        if distributed_attr_uid == -1:
            distributed_attr_uid = self.generate_tensor_distributed_attr_uid()
            tensor.set_distributed_attr_uid(distributed_attr_uid)
        self._tensor_distributed_attr_map_for_program[
            distributed_attr_uid] = tensor_dist_attr

    def get_op_distributed_attr_program(self, op):
        distributed_attr_uid = op.get_distributed_attr_uid()
        op_dist_attr=self._op_distributed_attr_map_for_program.get(distributed_attr_uid,
                                                                   None)
        return op_dist_attr

    def set_op_distributed_attr_program(self, op, op_dist_attr):
        distributed_attr_uid = op.get_distributed_attr_uid()
        if distributed_attr_uid == -1:
            distributed_attr_uid = self.generate_op_distributed_attr_uid()
            op.set_distributed_attr_uid(distributed_attr_uid)
        self._op_distributed_attr_map_for_program[distributed_attr_uid] = op_dist_attr

    def get_tensor_distributed_attr_graph(self, tensor_node):
        tensor_node_id = tensor_node.id()
        tensor_node_dist_attr = self._tensor_distributed_attr_map_for_graph.get(tensor_node_id, None)
        return tensor_node_dist_attr

    def set_tensor_distributed_attr_graph(self, tensor_node, tensor_node_dist_attr):
        tensor_node_id = tensor_node.id()
        self._tensor_distributed_attr_map_for_graph[tensor_node_id] = tensor_node_dist_attr

    def get_op_distributed_attr_graph(self, op_node):
        op_node_id = op_node.id()
        op_dist_attr = self._op_distributed_attr_map_for_graph.get(op_node_id, None)
        return op_dist_attr

    def set_op_distributed_attr_graph(self, op_node, op_dist_attr):
        op_node_id = op_node.id()
        self._op_distributed_attr_map_for_graph[op_node_id] = op_dist_attr

    def initialize_distributed_attr_for_program(self, program):
        for block in program.blocks:
            for tensor in block.vars.values():
                # Need make sure var is a tensor
                tensor_dist_attr = self.get_tensor_distributed_attr_program(tensor.desc)
                if tensor_dist_attr is None:
                    distributed_attr_uid = self.generate_tensor_distributed_attr_uid()
                    tensor.desc.set_distributed_attr_uid(distributed_attr_uid)
                    tensor_dist_attr = TensorDistributedAttribute(tensor.desc)
                    self.set_tensor_distributed_attr_program(tensor.desc,
                                                             tensor_dist_attr)
                if tensor_dist_attr.get_dims_mapping() is None:
                    tensor_dims_mapping = [
                        -1 for _ in range(len(tensor.desc.shape()))
                    ]
                    tensor_dist_attr.set_dims_mapping(tensor_dims_mapping)
                if isinstance(tensor, framework.Parameter):
                    tensor_dist_attr.mark_as_parameter()
            for op in block.ops:
                op_dist_attr = self.get_op_distributed_attr_program(op.desc)
                if op_dist_attr is None:
                    distributed_attr_uid = self.generate_op_distributed_attr_uid()
                    op.desc.set_distributed_attr_uid(distributed_attr_uid)
                    op_dist_attr = OperatorDistributedAttribute(op.desc)
                    # Default distributed implementation for all operators
                    op_dist_attr.set_impl_idx(-2)
                    self.set_op_distributed_attr_program(op.desc, op_dist_attr)
                for tensor_name in op.input_arg_names:
                    # There may be a better way to find the tensor by name
                    tensor = op.block._var_recursive(tensor_name)
                    if op_dist_attr.get_input_dims_mapping(tensor_name) is None:
                        tensor_dims_mapping = [
                            -1 for _ in range(len(tensor.desc.shape()))
                        ]
                        op_dist_attr.set_input_dims_mapping(tensor_name,
                                                            tensor_dims_mapping)
                    if isinstance(tensor, framework.Parameter):
                        op_dist_attr.mark_as_parameter(tensor_name)
                for tensor_name in op.output_arg_names:
                    tensor = op.block._var_recursive(tensor_name)
                    if op_dist_attr.get_output_dims_mapping(tensor_name) is None:
                        tensor_dims_mapping = [
                            -1 for _ in range(len(tensor.desc.shape()))
                        ]
                        op_dist_attr.set_output_dims_mapping(tensor_name,
                                                            tensor_dims_mapping)
                    if isinstance(tensor, framework.Parameter):
                        op_dist_attr.mark_as_parameter(tensor_name)

    def initialize_distributed_attr_for_graph(self, graph):
        all_nodes = graph.all_nodes()
        for node in all_nodes:
            if node.is_var() and node.var() is not None:
                tensor_desc = node.var()
                # Need make sure var is a tensor
                tensor_dist_attr = self.get_tensor_distributed_attr_program(tensor_desc)
                assert tensor_dist_attr is not None, \
                    "Var must have a distributed attribute after the initialization for program."
                new_tensor_dist_attr = copy.deepcopy(tensor_dist_attr)
                self.set_tensor_distributed_attr_graph(node, new_tensor_dist_attr)

            if node.is_op() and node.op() is not None:
                op_desc = node.op()
                op_dist_attr = self.get_op_distributed_attr_program(op_desc)
                assert op_dist_attr is not None, \
                    "Op must have a distributed attribute after the initialization for program."
                new_op_dist_attr = copy.deepcopy(op_dist_attr)
                self.set_op_distributed_attr_graph(node, new_op_dist_attr)

    def clear_distributed_attr_for_program(self):
        self._tensor_distributed_attr_map_for_program.clear()
        self._op_distributed_attr_map_for_program.clear()

    def clear_distributed_attr_for_graph(self):
        self._tensor_distributed_attr_map_for_graph.clear()
        self._op_distributed_attr_map_for_graph.clear()

    def copy_distribute_attr_from_graph_to_program(self, graph, program):
        updated_tensors = {}
        all_nodes = graph.all_nodes()
        for node in all_nodes:
            if node.is_var() and node.var() is not None:
                tensor_desc = node.var()
                updated = updated_tensors.get(tensor_desc.name(), False)
                # If a var has multiples var nodes in graph, only use the first one for now
                if not updated:
                    tensor_dist_attr = self.get_tensor_distributed_attr_graph(node)
                    new_tensor_dist_attr = copy.deepcopy(tensor_dist_attr)
                    self.set_tensor_distributed_attr_program(tensor_desc,
                                                             new_tensor_dist_attr)
                    updated_tensors[tensor_desc.name()] = True
            if node.is_op() and node.op() is not None:
                op_desc = node.op()
                op_dist_attr = self.get_op_distributed_attr_graph(node)
                new_op_dist_attr = copy.deepcopy(op_dist_attr)
                self.set_op_distributed_attr_program(op_desc, new_op_dist_attr)
    
    def set_communication_info(self, rank, comm_op, process_group):
        pass
    
