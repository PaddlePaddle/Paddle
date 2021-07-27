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
# limitations under the License.

from copy import deepcopy

from paddle.fluid import core
from paddle.fluid import framework
from paddle.fluid.distributed_attribute import get_default_distributed_config 

from .distributed_operators import find_best_compatible_distributed_operator_impl
from .utils import compute_compatible_process_mesh
from .utils import compute_compatible_dim_mapping
from .utils import compute_compatible_dims_mapping

ELEMENTWISE_LIKIE_OP_LIST = ["elementwise_add", "gelu", "dropout"]
# TENSOR_DISTRIBUTED_ATTR_MAP_FOR_GRAPH = {}
# OP_DISTRIBUTED_ATTR_MAP_FOR_GRAPH = {}


def is_elementwise_like_op(op_type):
    if op_type in ELEMENTWISE_LIKIE_OP_LIST:
        return True
    else:
        return False


# def get_tensor_distributed_attr_graph(tensor_node):
#     tensor_node_id = tensor_node.id()
#     global TENSOR_DISTRIBUTED_ATTR_MAP_FOR_GRAPH
#     tensor_node_dist_attr = TENSOR_DISTRIBUTED_ATTR_MAP_FOR_GRAPH.get(
#         tensor_node_id, None)
#     return tensor_node_dist_attr


# def set_tensor_distributed_attr_graph(tensor_node, tensor_node_dist_attr):
#     tensor_node_id = tensor_node.id()
#     global TENSOR_DISTRIBUTED_ATTR_MAP_FOR_GRAPH
#     TENSOR_DISTRIBUTED_ATTR_MAP_FOR_GRAPH[
#         tensor_node_id] = tensor_node_dist_attr


# def get_op_distributed_attr_graph(op_node):
#     op_node_id = op_node.id()
#     global OP_DISTRIBUTED_ATTR_MAP_FOR_GRAPH
#     op_dist_attr = OP_DISTRIBUTED_ATTR_MAP_FOR_GRAPH.get(op_node_id, None)
#     return op_dist_attr


# def set_op_distributed_attr_graph(op_node, op_dist_attr):
#     op_node_id = op_node.id()
#     global OP_DISTRIBUTED_ATTR_MAP_FOR_GRAPH
#     OP_DISTRIBUTED_ATTR_MAP_FOR_GRAPH[op_node_id] = op_dist_attr


def update_tensor_node_process_mesh(dist_config, tensor_node, fwd=True):
    changed = False
    tensor_dist_attr = dist_config.get_tensor_distributed_attr_graph(tensor_node)
    if tensor_dist_attr.is_annotated("process_mesh"):
        return changed
    tensor_process_mesh = tensor_dist_attr.get_process_mesh()
    if fwd:
        inputs_process_meshes = []
        for pred_op_node in tensor_node.inputs:
            if pred_op_node.op() is not None:
                op_dist_attr = dist_config.get_op_distributed_attr_graph(pred_op_node)
                op_process_mesh = op_dist_attr.get_process_mesh()
                inputs_process_meshes.append(op_process_mesh)
        compatible_process_mesh = compute_compatible_process_mesh(
            inputs_process_meshes)
        if compatible_process_mesh is not None and tensor_process_mesh is None:
            tensor_dist_attr.set_process_mesh(compatible_process_mesh)
            changed = True
    else:
        outputs_process_meshes = []
        for succ_op_node in tensor_node.outputs:
            if succ_op_node.op() is not None:
                op_dist_attr = dist_config.get_op_distributed_attr_graph(succ_op_node)
                op_process_mesh = op_dist_attr.get_process_mesh()
                outputs_process_meshes.append(op_process_mesh)
        compatible_process_mesh = compute_compatible_process_mesh(
            outputs_process_meshes)
        if compatible_process_mesh is not None and tensor_process_mesh is None:
            tensor_dist_attr.set_process_mesh(compatible_process_mesh)
            changed = True
    return changed


def update_op_node_process_mesh(dist_config, op_node, fwd=True):
    changed = False
    op_dist_attr = dist_config.get_op_distributed_attr_graph(op_node)
    if op_dist_attr.is_annotated("process_mesh"):
        return changed
    op_process_mesh = op_dist_attr.get_process_mesh()
    if fwd:
        inputs_process_meshes = []
        for tensor_node in op_node.inputs:
            if tensor_node.var() is not None:
                tensor_dist_attr = dist_config.get_tensor_distributed_attr_graph(
                    tensor_node)
                tensor_process_mesh = tensor_dist_attr.get_process_mesh()
                inputs_process_meshes.append(tensor_process_mesh)
        compatible_process_mesh = compute_compatible_process_mesh(
            inputs_process_meshes)
        if compatible_process_mesh is not None and op_process_mesh is None:
            op_dist_attr.set_process_mesh(compatible_process_mesh)
            changed = True
    else:
        outputs_process_meshes = []
        for tensor_node in op_node.outputs:
            if tensor_node.var() is not None:
                tensor_dist_attr = dist_config.get_tensor_distributed_attr_graph(
                    tensor_node)
                tensor_process_mesh = tensor_dist_attr.get_process_mesh()
                outputs_process_meshes.append(tensor_process_mesh)
        compatible_process_mesh = compute_compatible_process_mesh(
            outputs_process_meshes)
        if compatible_process_mesh is not None and op_process_mesh is None:
            op_dist_attr.set_process_mesh(compatible_process_mesh)
            changed = True
    return changed


def update_op_dims_mapping_by_default_dist_impl(op_dist_attr):
    """Each operator has a default distributed operator, only allowed to be sharded in batch dimension. """
    changed = False
    op_desc = op_dist_attr.get_desc()
    batch_dim_mappings = []
    for arg_name in op_desc.input_arg_names():
        if op_dist_attr.is_parameter(arg_name):
            continue
        dims_mapping = op_dist_attr.get_input_dims_mapping(arg_name)
        # print(op_desc.type(), arg_name, dims_mapping, op_dist_attr)
        for idx, mapping in enumerate(dims_mapping[1:]):
            assert mapping == -1, \
                "{} only the batch dimension (0-dim) can be sharded, but the dimension {} is sharded by {} part."\
                    .format(op_desc.type(), idx, mapping)
        batch_dim_mappings.append(
            op_dist_attr.get_input_dim_mapping(arg_name, 0))
    for arg_name in op_desc.output_arg_names():
        if op_dist_attr.is_parameter(arg_name):
            continue
        dims_mapping = op_dist_attr.get_output_dims_mapping(arg_name)
        for idx, mapping in enumerate(dims_mapping[1:]):
            assert mapping == -1, \
                "{} only the batch dimension (0-dim) can be sharded, but the dimension {} is sharded by {} part."\
                    .format(op_desc.type(), idx, mapping)
        batch_dim_mappings.append(
            op_dist_attr.get_output_dim_mapping(arg_name, 0))
    compatible_dim_mapping = compute_compatible_dim_mapping(batch_dim_mappings)
    assert compatible_dim_mapping is not None, "There is no compatible dim mapping."
    for arg_name in op_desc.input_arg_names():
        if op_dist_attr.is_parameter(arg_name):
            continue
        if compatible_dim_mapping != op_dist_attr.get_input_dim_mapping(
                arg_name, 0):
            changed = True
        op_dist_attr.set_input_dim_mapping(arg_name, 0, compatible_dim_mapping)
    for arg_name in op_desc.output_arg_names():
        if op_dist_attr.is_parameter(arg_name):
            continue
        if compatible_dim_mapping != op_dist_attr.get_output_dim_mapping(
                arg_name, 0):
            changed = True
        op_dist_attr.set_output_dim_mapping(arg_name, 0, compatible_dim_mapping)
    return changed


def update_op_dims_mapping_by_elementwise_like_dist_impl(op_dist_attr):
    """Element-wise operator can be sharded in any way (and take care of broadcasting)."""
    changed = False
    op_desc = op_dist_attr.get_desc()

    input_arg_names = op_desc.input_arg_names()
    input_dims_mapping_dict = {}
    input_dims_mapping_lens = {}
    max_dims_mapping_len = -1
    for arg_name in input_arg_names:
        dims_mapping = op_dist_attr.get_input_dims_mapping(arg_name)
        if max_dims_mapping_len < len(dims_mapping):
            max_dims_mapping_len = len(dims_mapping)
        input_dims_mapping_dict[arg_name] = dims_mapping
        input_dims_mapping_lens[arg_name] = len(dims_mapping)

    dims_mapping_list = []
    for arg_name in input_arg_names:
        if input_dims_mapping_lens[arg_name] < max_dims_mapping_len:
            new_dims_mapping = [-1 for _ in range(max_dims_mapping_len)]
            for i in range(input_dims_mapping_lens[arg_name]):
                new_idx = (max_dims_mapping_len -
                           input_dims_mapping_lens[arg_name]) + i
                new_dims_mapping[new_idx] = input_dims_mapping_dict[arg_name][i]
            dims_mapping_list.append(new_dims_mapping)
        else:
            dims_mapping_list.append(input_dims_mapping_dict[arg_name])
    output_arg_names = op_desc.output_arg_names()
    for arg_name in output_arg_names:
        dims_mapping = op_dist_attr.get_output_dims_mapping(arg_name)
        assert len(dims_mapping) == max_dims_mapping_len
        dims_mapping_list.append(dims_mapping)

    compatible_dims_mapping = compute_compatible_dims_mapping(dims_mapping_list)
    assert compatible_dims_mapping is not None, "There is no compatible dim mapping."

    for arg_name in input_arg_names:
        if input_dims_mapping_lens[arg_name] < max_dims_mapping_len:
            new_dims_mapping = [
                -1 for _ in range(input_dims_mapping_lens[arg_name])
            ]
            for i in range(input_dims_mapping_lens[arg_name]):
                new_idx = (max_dims_mapping_len -
                           input_dims_mapping_lens[arg_name]) + i
                new_dims_mapping[i] = compatible_dims_mapping[new_idx]
            if new_dims_mapping != input_dims_mapping_dict[arg_name]:
                op_dist_attr.set_input_dims_mapping(arg_name, new_dims_mapping)
                changed = True
        else:
            if compatible_dims_mapping != input_dims_mapping_dict[arg_name]:
                op_dist_attr.set_input_dims_mapping(arg_name,
                                                    compatible_dims_mapping)
                changed = True

    for arg_name in output_arg_names:
        dims_mapping = op_dist_attr.get_output_dims_mapping(arg_name)
        if compatible_dims_mapping != dims_mapping:
            op_dist_attr.set_output_dims_mapping(arg_name,
                                                 compatible_dims_mapping)
            changed = True

    return changed


def update_tensor_node_dims_mapping(dist_config, tensor_node, fwd=True):
    changed = False
    if (not tensor_node.is_var()) or (tensor_node.var() is None):
        return False
    tensor_desc = tensor_node.var()
    tensor_dist_attr = dist_config.get_tensor_distributed_attr_graph(tensor_node)
    assert tensor_dist_attr is not None
    if tensor_dist_attr.is_annotated("dims_mapping"):
        return False
    tensor_dims_mapping = tensor_dist_attr.get_dims_mapping()
    if fwd:
        dims_mapping_list = []
        for pred_op_node in tensor_node.inputs:
            if pred_op_node.op() is not None:
                op_dist_attr = dist_config.get_op_distributed_attr_graph(pred_op_node)
                op_dims_mapping = op_dist_attr.get_output_dims_mapping(
                    tensor_desc.name())
                dims_mapping_list.append(op_dims_mapping)
        dims_mapping_list.append(tensor_dims_mapping)
        compatible_dims_mapping = compute_compatible_dims_mapping(
            dims_mapping_list)
        if (compatible_dims_mapping is not None) and \
            (compatible_dims_mapping != tensor_dims_mapping):
            tensor_dist_attr.set_dims_mapping(compatible_dims_mapping)
            changed = True
    else:
        dims_mapping_list = []
        for succ_op_node in tensor_node.outputs:
            if succ_op_node.op() is not None:
                op_dist_attr = dist_config.get_op_distributed_attr_graph(succ_op_node)
                op_dims_mapping = op_dist_attr.get_input_dims_mapping(
                    tensor_desc.name())
                dims_mapping_list.append(op_dims_mapping)
        dims_mapping_list.append(tensor_dims_mapping)
        compatible_dims_mapping = compute_compatible_dims_mapping(
            dims_mapping_list)
        if (compatible_dims_mapping is not None) and \
            (compatible_dims_mapping != tensor_dims_mapping):
            tensor_dist_attr.set_dims_mapping(compatible_dims_mapping)
            changed = True
    return changed


def update_op_node_dims_mapping(dist_config, op_node, fwd=True):
    changed = False
    if (not op_node.is_op()) or (op_node.op() is None):
        return False
    op_desc = op_node.op()
    op_dist_attr = dist_config.get_op_distributed_attr_graph(op_node)
    if fwd:
        for tensor_node in op_node.inputs:
            if tensor_node.var() is not None:
                tensor_desc = tensor_node.var()
                if op_dist_attr.is_annotated_input_dims_mapping(
                        tensor_desc.name()):
                    continue
                tensor_dist_attr = dist_config.get_tensor_distributed_attr_graph(
                    tensor_node)
                tensor_dims_mapping = tensor_dist_attr.get_dims_mapping()
                op_dims_mapping = op_dist_attr.get_input_dims_mapping(
                    tensor_desc.name())
                compatible_dims_mapping = compute_compatible_dims_mapping(
                    [op_dims_mapping, tensor_dims_mapping])
                # print("fwd0", tensor_desc.name(), op_dims_mapping, tensor_dims_mapping, compatible_dims_mapping)
                if (compatible_dims_mapping is not None) and \
                    (compatible_dims_mapping != op_dims_mapping):
                    op_dist_attr.set_input_dims_mapping(tensor_desc.name(),
                                                        compatible_dims_mapping)
                    changed = True
        # Find the most compatible implemenetations from the distributed operator
        # print("fwd1", op_desc.type(), op_dist_attr)
        op_dist_impl, op_dist_impl_idx = find_best_compatible_distributed_operator_impl(
            op_desc.type(), op_dist_attr, fwd=True)
        # print("fwd2", op_desc.type(), op_dist_impl_idx)
        if op_dist_impl is not None:
            dim_changed = op_dist_impl.update_dims_mapping(op_dist_attr)
            if dim_changed:
                changed = True
            # This statement will be replaced by a good way
            if op_dist_impl.is_compatible(op_dist_attr):
                op_dist_attr.set_impl_idx(op_dist_impl_idx)
                # print("fwd3", op_dist_impl.get_name(), op_dist_impl_idx)
            # print("fwd4", op_dist_impl.get_name(), op_dist_impl_idx, op_dist_attr)
        elif is_elementwise_like_op(op_desc.type()):
            dim_changed = update_op_dims_mapping_by_elementwise_like_dist_impl(
                op_dist_attr)
            if dim_changed:
                changed = True
            op_dist_attr.set_impl_idx(-1)
        else:
            dim_changed = update_op_dims_mapping_by_default_dist_impl(
                op_dist_attr)
            if dim_changed:
                changed = True
            op_dist_attr.set_impl_idx(-2)
    else:
        for tensor_node in op_node.outputs:
            if tensor_node.var() is not None:
                tensor_desc = tensor_node.var()
                if op_dist_attr.is_annotated_output_dims_mapping(
                        tensor_desc.name()):
                    continue
                tensor_dist_attr = dist_config.get_tensor_distributed_attr_graph(
                    tensor_node)
                tensor_dims_mapping = tensor_dist_attr.get_dims_mapping()
                op_dims_mapping = op_dist_attr.get_output_dims_mapping(
                    tensor_desc.name())
                compatible_dims_mapping = compute_compatible_dims_mapping(
                    [op_dims_mapping, tensor_dims_mapping])
                # print("bwd0", tensor_desc.name(), op_dims_mapping, tensor_dims_mapping, compatible_dims_mapping)
                if (compatible_dims_mapping is not None) and \
                    (compatible_dims_mapping != op_dims_mapping):
                    op_dist_attr.set_output_dims_mapping(
                        tensor_desc.name(), compatible_dims_mapping)
                    changed = True
        # Find the most compatible implemenetations from the distributed operator
        # print("bwd1", op_desc.type(), op_dist_attr)
        op_dist_impl, op_dist_impl_idx = find_best_compatible_distributed_operator_impl(
            op_desc.type(), op_dist_attr, fwd=False)
        # print("bwd2", op_desc.type(), op_dist_impl_idx)
        if op_dist_impl is not None:
            dim_changed = op_dist_impl.update_dims_mapping(op_dist_attr)
            if dim_changed:
                changed = True
            # This statement will be replaced by a good way
            if op_dist_impl.is_compatible(op_dist_attr):
                op_dist_attr.set_impl_idx(op_dist_impl_idx)
                # print("bwd3", op_dist_impl.get_name(), op_dist_impl_idx)
            # print("bwd4", op_dist_impl.get_name(), op_dist_impl_idx, op_dist_attr)
        elif is_elementwise_like_op(op_desc.type()):
            dim_changed = update_op_dims_mapping_by_elementwise_like_dist_impl(
                op_dist_attr)
            if dim_changed:
                changed = True
            op_dist_attr.set_impl_idx(-1)
        else:
            dim_changed = update_op_dims_mapping_by_default_dist_impl(
                op_dist_attr)
            if dim_changed:
                changed = True
            op_dist_attr.set_impl_idx(-2)
    return changed


# def initialize_distributed_attr_for_program(program):
#     for block in program.blocks:
#         for tensor in block.vars.values():
#             # Need make sure var is a tensor
#             tensor_dist_attr = get_tensor_distributed_attr_program(tensor.desc)
#             if tensor_dist_attr is None:
#                 distributed_attr_uid = generate_tensor_distributed_attr_uid()
#                 tensor.desc.set_distributed_attr_uid(distributed_attr_uid)
#                 tensor_dist_attr = TensorDistributedAttribute(tensor.desc)
#                 set_tensor_distributed_attr_program(tensor.desc,
#                                                     tensor_dist_attr)
#             if tensor_dist_attr.get_dims_mapping() is None:
#                 tensor_dims_mapping = [
#                     -1 for _ in range(len(tensor.desc.shape()))
#                 ]
#                 tensor_dist_attr.set_dims_mapping(tensor_dims_mapping)
#             if isinstance(tensor, framework.Parameter):
#                 tensor_dist_attr.mark_as_parameter()
#         for op in block.ops:
#             op_dist_attr = get_op_distributed_attr_program(op.desc)
#             if op_dist_attr is None:
#                 distributed_attr_uid = generate_op_distributed_attr_uid()
#                 op.desc.set_distributed_attr_uid(distributed_attr_uid)
#                 op_dist_attr = OperatorDistributedAttribute(op.desc)
#                 # Default distributed implementation for all operators
#                 op_dist_attr.set_impl_idx(-2)
#                 set_op_distributed_attr_program(op.desc, op_dist_attr)
#             for tensor_name in op.input_arg_names:
#                 # There may be a better way to find the tensor by name
#                 tensor = op.block._var_recursive(tensor_name)
#                 if op_dist_attr.get_input_dims_mapping(tensor_name) is None:
#                     tensor_dims_mapping = [
#                         -1 for _ in range(len(tensor.desc.shape()))
#                     ]
#                     op_dist_attr.set_input_dims_mapping(tensor_name,
#                                                         tensor_dims_mapping)
#                 if isinstance(tensor, framework.Parameter):
#                     op_dist_attr.mark_as_parameter(tensor_name)
#             for tensor_name in op.output_arg_names:
#                 tensor = op.block._var_recursive(tensor_name)
#                 if op_dist_attr.get_output_dims_mapping(tensor_name) is None:
#                     tensor_dims_mapping = [
#                         -1 for _ in range(len(tensor.desc.shape()))
#                     ]
#                     op_dist_attr.set_output_dims_mapping(tensor_name,
#                                                          tensor_dims_mapping)
#                 if isinstance(tensor, framework.Parameter):
#                     op_dist_attr.mark_as_parameter(tensor_name)


# def initialize_distributed_attr_for_graph(graph):
#     all_nodes = graph.all_nodes()
#     for node in all_nodes:
#         if node.is_var() and node.var() is not None:
#             tensor_desc = node.var()
#             # Need make sure var is a tensor
#             tensor_dist_attr = get_tensor_distributed_attr_program(tensor_desc)
#             assert tensor_dist_attr is not None, \
#                 "Var must have a distributed attribute after the initialization for program."
#             new_tensor_dist_attr = deepcopy(tensor_dist_attr)
#             set_tensor_distributed_attr_graph(node, new_tensor_dist_attr)

#         if node.is_op() and node.op() is not None:
#             op_desc = node.op()
#             op_dist_attr = get_op_distributed_attr_program(op_desc)
#             assert op_dist_attr is not None, \
#                 "Op must have a distributed attribute after the initialization for program."
#             new_op_dist_attr = deepcopy(op_dist_attr)
#             set_op_distributed_attr_graph(node, new_op_dist_attr)


# def copy_distribute_attr_from_graph_to_program(graph, program):
#     updated_tensors = {}
#     all_nodes = graph.all_nodes()
#     for node in all_nodes:
#         if node.is_var() and node.var() is not None:
#             tensor_desc = node.var()
#             updated = updated_tensors.get(tensor_desc.name(), False)
#             # If a var has multiples var nodes in graph, only use the first one for now
#             if not updated:
#                 tensor_dist_attr = get_tensor_distributed_attr_graph(node)
#                 new_tensor_dist_attr = deepcopy(tensor_dist_attr)
#                 set_tensor_distributed_attr_program(tensor_desc,
#                                                     new_tensor_dist_attr)
#                 updated_tensors[tensor_desc.name()] = True
#         if node.is_op() and node.op() is not None:
#             op_desc = node.op()
#             op_dist_attr = get_op_distributed_attr_graph(node)
#             new_op_dist_attr = deepcopy(op_dist_attr)
#             set_op_distributed_attr_program(op_desc, new_op_dist_attr)


def complete_annotation(program, dist_config = None):
    """ Complete annotation for the partial annotated program.

    Arguments:
        program: partial annotated program.

    Returns:
        program: completed annotated program.
    """

    if dist_config is None:
        dist_config = get_default_distributed_config()
    # Initialize distributed attributes for all var and op node in program 
    dist_config.initialize_distributed_attr_for_program(program)
    print(program)

    # Convert program to graph
    graph = framework.IrGraph(core.Graph(program.desc))

    # Initialize distributed attributes for all var and op node in graph
    dist_config.initialize_distributed_attr_for_graph(graph)

    # # Complete process mesh for each node
    all_nodes = list(graph.all_nodes())
    reach_fix_point = False
    while not reach_fix_point:
        changed = False
        for node in all_nodes:
            if node.is_var() and node.var() is not None:
                tensor_changed = update_tensor_node_process_mesh(dist_config, node, fwd=True)
                if tensor_changed:
                    changed = True
            if node.is_op() and node.op() is not None:
                op_changed = update_op_node_process_mesh(dist_config, node, fwd=True)
                if op_changed:
                    changed = True
        for node in reversed(all_nodes):
            if node.is_var() and node.var() is not None:
                tensor_changed = update_tensor_node_process_mesh(
                    dist_config, node, fwd=False)
                if tensor_changed:
                    changed = True
            if node.is_op() and node.op() is not None:
                op_changed = update_op_node_process_mesh(dist_config, node, fwd=False)
                if op_changed:
                    changed = True
        if changed:
            reach_fix_point = False
        else:
            reach_fix_point = True

    # Complete dims_mapping for each node
    reach_fix_point = False
    while not reach_fix_point:
        changed = False
        for node in all_nodes:
            if node.is_var() and node.var() is not None:
                tensor_changed = update_tensor_node_dims_mapping(dist_config, node, fwd=True)
                if tensor_changed:
                    changed = True
            if node.is_op() and node.op() is not None:
                op_changed = update_op_node_dims_mapping(dist_config, node, fwd=True)
                if op_changed:
                    changed = True
        for node in reversed(all_nodes):
            if node.is_var() and node.var() is not None:
                tensor_changed = update_tensor_node_dims_mapping(
                    dist_config, node, fwd=False)
                if tensor_changed:
                    changed = True
            if node.is_op() and node.op() is not None:
                op_changed = update_op_node_dims_mapping(dist_config, node, fwd=False)
                if op_changed:
                    changed = True
        if changed:
            reach_fix_point = False
        else:
            reach_fix_point = True
        
    # Copy the corresponding distributed attribute from graph to program
    dist_config.copy_distribute_attr_from_graph_to_program(graph, program)
    dist_config.clear_distributed_attr_for_graph()

    return program
