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
from paddle.fluid.distributed_attribute import TensorDistributedAttribute
from paddle.fluid.distributed_attribute import OperatorDistributedAttribute
from paddle.fluid.distributed_attribute import get_tensor_distributed_attr_program
from paddle.fluid.distributed_attribute import set_tensor_distributed_attr_program
from paddle.fluid.distributed_attribute import get_op_distributed_attr_program
from paddle.fluid.distributed_attribute import set_op_distributed_attr_program
from paddle.fluid.distributed_attribute import generate_tensor_distributed_attr_uid
from paddle.fluid.distributed_attribute import generate_op_distributed_attr_uid

from .distributed_operators import find_best_compatible_distributed_operator_impl

ELEMENT_WISE_OP_LIST = ["gelu"]
TENSOR_DISTRIBUTED_ATTR_MAP_FOR_GRAPH = {}
OP_DISTRIBUTED_ATTR_MAP_FOR_GRAPH = {}


def is_element_wise_op(op_type):
    if op_type in ELEMENT_WISE_OP_LIST:
        return True
    else:
        return False


def get_tensor_distributed_attr_graph(tensor_node):
    tensor_node_id = tensor_node.id()
    global TENSOR_DISTRIBUTED_ATTR_MAP_FOR_GRAPH
    tensor_node_dist_attr = TENSOR_DISTRIBUTED_ATTR_MAP_FOR_GRAPH.get(
        tensor_node_id, None)
    return tensor_node_dist_attr


def set_tensor_distributed_attr_graph(tensor_node, tensor_node_dist_attr):
    tensor_node_id = tensor_node.id()
    global TENSOR_DISTRIBUTED_ATTR_MAP_FOR_GRAPH
    TENSOR_DISTRIBUTED_ATTR_MAP_FOR_GRAPH[
        tensor_node_id] = tensor_node_dist_attr


def get_op_distributed_attr_graph(op_node):
    op_node_id = op_node.id()
    global OP_DISTRIBUTED_ATTR_MAP_FOR_GRAPH
    op_dist_attr = OP_DISTRIBUTED_ATTR_MAP_FOR_GRAPH.get(op_node_id, None)
    return op_dist_attr


def set_op_distributed_attr_graph(op_node, op_dist_attr):
    op_node_id = op_node.id()
    global OP_DISTRIBUTED_ATTR_MAP_FOR_GRAPH
    OP_DISTRIBUTED_ATTR_MAP_FOR_GRAPH[op_node_id] = op_dist_attr


def compute_compatible_process_mesh(process_mesh_list):
    compatible_process_mesh = None
    if not process_mesh_list:
        return compatible_process_mesh
    for process_mesh in process_mesh_list:
        if process_mesh is not None:
            if compatible_process_mesh is None:
                compatible_process_mesh = process_mesh
            else:
                assert process_mesh == compatible_process_mesh, \
                    "There is no compatible process mesh."
    return compatible_process_mesh


def update_tensor_node_process_mesh(tensor_node, fwd=True):
    changed = False
    dist_attr_in_tensor = get_tensor_distributed_attr_graph(tensor_node)
    if dist_attr_in_tensor.is_annotated("process_mesh"):
        return changed
    process_mesh_in_tensor = dist_attr_in_tensor.get_process_mesh()
    if fwd:
        inputs_process_meshes = []
        for pred_op_node in tensor_node.inputs:
            if pred_op_node.op() is not None:
                dist_attr_in_op = get_op_distributed_attr_graph(pred_op_node)
                process_mesh_in_op = dist_attr_in_op.get_process_mesh()
                inputs_process_meshes.append(process_mesh_in_op)
        compatible_process_mesh = compute_compatible_process_mesh(
            inputs_process_meshes)
        if compatible_process_mesh is not None and process_mesh_in_tensor is None:
            dist_attr_in_tensor.set_process_mesh(compatible_process_mesh)
            changed = True
    else:
        outputs_process_meshes = []
        for succ_op_node in tensor_node.outputs:
            if succ_op_node.op() is not None:
                dist_attr_in_op = get_op_distributed_attr_graph(succ_op_node)
                process_mesh_in_op = dist_attr_in_op.get_process_mesh()
                outputs_process_meshes.append(process_mesh_in_op)
        compatible_process_mesh = compute_compatible_process_mesh(
            outputs_process_meshes)
        if compatible_process_mesh is not None and process_mesh_in_tensor is None:
            dist_attr_in_tensor.set_process_mesh(compatible_process_mesh)
            changed = True
    return changed


def update_op_node_process_mesh(op_node, fwd=True):
    changed = False
    op_dist_attr = get_op_distributed_attr_graph(op_node)
    if op_dist_attr.is_annotated("process_mesh"):
        return changed
    process_mesh_in_op = op_dist_attr.get_process_mesh()
    if fwd:
        inputs_process_meshes = []
        for tensor_node in op_node.inputs:
            if tensor_node.var() is not None:
                tensor_dist_attr = get_tensor_distributed_attr_graph(
                    tensor_node)
                process_mesh_in_tensor = tensor_dist_attr.get_process_mesh()
                inputs_process_meshes.append(process_mesh_in_tensor)
        compatible_process_mesh = compute_compatible_process_mesh(
            inputs_process_meshes)
        if compatible_process_mesh is not None and process_mesh_in_op is None:
            op_dist_attr.set_process_mesh(compatible_process_mesh)
            changed = True
    else:
        outputs_process_meshes = []
        for tensor_node in op_node.outputs:
            if tensor_node.var() is not None:
                tensor_dist_attr = get_tensor_distributed_attr_graph(
                    tensor_node)
                process_mesh_in_tensor = tensor_dist_attr.get_process_mesh()
                outputs_process_meshes.append(process_mesh_in_tensor)
        compatible_process_mesh = compute_compatible_process_mesh(
            outputs_process_meshes)
        if compatible_process_mesh is not None and process_mesh_in_op is None:
            op_dist_attr.set_process_mesh(compatible_process_mesh)
            changed = True
    return changed


def compute_compatible_dim_mapping(dim_mappings):
    if not dim_mappings:
        return None
    compatible_mapping = dim_mappings[0]
    for mapping in dim_mappings:
        if compatible_mapping == -1:
            compatible_mapping = mapping
        elif mapping == -1:
            continue
        elif compatible_mapping == mapping:
            continue
        else:
            return None
    return compatible_mapping


def compute_compatible_dims_mapping(dims_mapping_list):
    if not dims_mapping_list:
        return None
    length = len(dims_mapping_list[0])
    for dims_mapping in dims_mapping_list:
        assert len(dims_mapping) == length, \
            "The length of dims_mapping in list must be same for compatible computation."
    compatible_result = []
    for dim_mappings in zip(*dims_mapping_list):
        compatible_dim_mapping = compute_compatible_dim_mapping(
            list(dim_mappings))
        if compatible_dim_mapping is None:
            return None
        compatible_result.append(compatible_dim_mapping)
    return compatible_result


def update_op_dims_mapping_by_default_dist_impl(op_dist_attr):
    """Each operator has a default distributed operator, only allowed to be sharded in batch dimension. """
    changed = False
    # print("haha", op_dist_attr)
    op_desc = op_dist_attr.get_desc()
    batch_dim_mappings = []
    for arg_name in op_desc.input_arg_names():
        dims_mapping = op_dist_attr.get_input_dims_mapping(arg_name)
        for idx, mapping in enumerate(dims_mapping[1:]):
            assert mapping == -1, \
                "{} only the batch dimension (0-dim) can be sharded, but the dimension {} is sharded by {} part."\
                    .format(op_desc.type(), idx, mapping)
        batch_dim_mappings.append(
            op_dist_attr.get_input_dim_mapping(arg_name, 0))
    for arg_name in op_desc.output_arg_names():
        dims_mapping = op_dist_attr.get_output_dims_mapping(arg_name)
        for idx, mapping in enumerate(dims_mapping[1:]):
            assert mapping == -1, \
                "{} only the batch dimension (0-dim) can be sharded, but the dimension {} is sharded by {} part."\
                    .format(op_desc.type(), idx, mapping)
        batch_dim_mappings.append(
            op_dist_attr.get_output_dim_mapping(arg_name, 0))
    compatible_dim_mapping = compute_compatible_dim_mapping(batch_dim_mappings)
    # print("compatible dim mapping", batch_dim_mappings, compatible_dim_mapping)
    assert compatible_dim_mapping is not None, "There is no compatible dim mapping."
    for arg_name in op_desc.input_arg_names():
        if compatible_dim_mapping != op_dist_attr.get_input_dim_mapping(
                arg_name, 0):
            changed = True
        op_dist_attr.set_input_dim_mapping(arg_name, 0, compatible_dim_mapping)
    for arg_name in op_desc.output_arg_names():
        if compatible_dim_mapping != op_dist_attr.get_output_dim_mapping(
                arg_name, 0):
            changed = True
        op_dist_attr.set_output_dim_mapping(arg_name, 0, compatible_dim_mapping)
    return changed


def update_op_dims_mapping_by_element_wise_dist_impl(op_dist_attr):
    """Element-wise operator can be sharded in any way."""
    changed = False
    op_desc = op_dist_attr.get_desc()

    input_arg_names = op_desc.input_arg_names()
    assert len(input_arg_names) == 1, "Element-wise op only has one input."
    output_arg_names = op_desc.output_arg_names()
    assert len(output_arg_names) == 1, "Element-wise op only has one output."

    input_dims_mapping = op_dist_attr.get_input_dims_mapping(input_arg_names[0])
    output_dims_mapping = op_dist_attr.get_output_dims_mapping(output_arg_names[
        0])
    assert len(input_dims_mapping) == len(output_dims_mapping), \
        "The input and output of element-wise op must has the same shape."

    for idx in range(len(input_dims_mapping)):
        dim_mappings = [
            op_dist_attr.get_input_dim_mapping(input_arg_names[0], idx),
            op_dist_attr.get_output_dim_mapping(output_arg_names[0], idx)
        ]
        compatible_dim_mapping = compute_compatible_dim_mapping(dim_mappings)
        assert compatible_dim_mapping is not None, "There is no compatible dim mapping."
        if compatible_dim_mapping != op_dist_attr.get_input_dim_mapping(
                input_arg_names[0], idx):
            op_dist_attr.set_input_dim_mapping(input_arg_names[0], idx,
                                               compatible_dim_mapping)
            changed = True
        if compatible_dim_mapping != op_dist_attr.get_output_dim_mapping(
                input_arg_names[0], idx):
            op_dist_attr.set_output_dim_mapping(output_arg_names[0], idx,
                                                compatible_dim_mapping)
            changed = True
    return changed


def update_op_dims_mapping_by_dist_impl(op_dist_attr, op_dist_impl):
    """Update dims mapping by the selected distributed implemention."""
    changed = False
    op_desc = op_dist_attr.get_desc()
    dist_singnature = op_dist_impl.get_distributed_signature()
    same_shard_dims_list = dist_singnature.get_valid_inputs_outputs_same_shard_dims_list(
    )
    for same_shard_dims in same_shard_dims_list:
        dim_mappings = []
        for in_or_out, para_name, dim in same_shard_dims:
            if in_or_out == 'input':
                for arg_name in op_desc.input(para_name):
                    dim_mappings.append(
                        op_dist_attr.get_input_dim_mapping(arg_name, dim))
            else:
                for arg_name in op_desc.output(para_name):
                    dim_mappings.append(
                        op_dist_attr.get_output_dim_mapping(arg_name, dim))
        compatible_dim_mapping = compute_compatible_dim_mapping(dim_mappings)
        assert compatible_dim_mapping is not None, "There is no compatible dim mapping."
        for in_or_out, para_name, dim in same_shard_dims:
            if in_or_out == 'input':
                for arg_name in op_desc.input(para_name):
                    if compatible_dim_mapping != op_dist_attr.get_input_dim_mapping(
                            arg_name, dim):
                        op_dist_attr.set_input_dim_mapping(
                            arg_name, dim, compatible_dim_mapping)
                        changed = True
            else:
                for arg_name in op_desc.output(para_name):
                    if compatible_dim_mapping != op_dist_attr.get_output_dim_mapping(
                            arg_name, dim):
                        op_dist_attr.set_output_dim_mapping(
                            arg_name, dim, compatible_dim_mapping)
                        changed = True
    return changed


def update_tensor_node_dims_mapping(tensor_node, fwd=True):
    changed = False
    if (not tensor_node.is_var()) or (tensor_node.var() is None):
        return False
    tensor_desc = tensor_node.var()
    dist_attr_in_tensor = get_tensor_distributed_attr_graph(tensor_node)
    dims_mapping_in_tensor = dist_attr_in_tensor.get_dims_mapping()
    if fwd:
        dims_mapping_list = []
        for pred_op_node in tensor_node.inputs:
            if pred_op_node.op() is not None:
                dist_attr_in_op = get_op_distributed_attr_graph(pred_op_node)
                dims_mapping_in_op = dist_attr_in_op.get_output_dims_mapping(
                    tensor_desc.name())
                assert dims_mapping_in_op is not None
                dims_mapping_list.append(dims_mapping_in_op)
        dims_mapping_list.append(dims_mapping_in_tensor)
        compatible_dims_mapping = compute_compatible_dims_mapping(
            dims_mapping_list)
        if (compatible_dims_mapping is not None) and \
            (compatible_dims_mapping != dims_mapping_in_tensor):
            dist_attr_in_tensor.set_dims_mapping(compatible_dims_mapping)
            changed = True
    else:
        dims_mapping_list = []
        for succ_op_node in tensor_node.outputs:
            if succ_op_node.op() is not None:
                dist_attr_in_op = get_op_distributed_attr_graph(succ_op_node)
                dims_mapping_in_op = dist_attr_in_op.get_input_dims_mapping(
                    tensor_desc.name())
                assert dims_mapping_in_op is not None
                dims_mapping_list.append(dims_mapping_in_op)
        dims_mapping_list.append(dims_mapping_in_tensor)
        compatible_dims_mapping = compute_compatible_dims_mapping(
            dims_mapping_list)
        if (compatible_dims_mapping is not None) and \
            (compatible_dims_mapping != dims_mapping_in_tensor):
            dist_attr_in_tensor.set_dims_mapping(compatible_dims_mapping)
            changed = True
    return changed


def update_op_node_dims_mapping(op_node, fwd=True):
    changed = False
    if (not op_node.is_op()) or (op_node.op() is None):
        return False
    op_desc = op_node.op()
    op_dist_attr = get_op_distributed_attr_graph(op_node)
    if fwd:
        for tensor_node in op_node.inputs:
            if tensor_node.var() is not None:
                tensor_desc = tensor_node.var()
                tensor_dist_attr = get_tensor_distributed_attr_graph(
                    tensor_node)
                dims_mapping_in_tensor = tensor_dist_attr.get_dims_mapping()
                assert dims_mapping_in_tensor is not None
                dims_mapping_in_op = op_dist_attr.get_input_dims_mapping(
                    tensor_desc.name())
                assert dims_mapping_in_op is not None
                compatible_dims_mapping = compute_compatible_dims_mapping(
                    [dims_mapping_in_op, dims_mapping_in_tensor])
                # print("last try", tensor_desc.name(), ",", dims_mapping_in_op, ",", dims_mapping_in_tensor, ",", compatible_dims_mapping)
                if (compatible_dims_mapping is not None) and \
                    (compatible_dims_mapping != dims_mapping_in_op):
                    op_dist_attr.set_input_dims_mapping(tensor_desc.name(),
                                                        compatible_dims_mapping)
                    changed = True
        # Find the most compatible implemenetations from the distributed operator
        op_dist_impl, op_dist_impl_idx = find_best_compatible_distributed_operator_impl(
            op_desc.type(), op_dist_attr, fwd=True)
        if op_dist_impl is not None:
            dim_changed = update_op_dims_mapping_by_dist_impl(op_dist_attr,
                                                              op_dist_impl)
            if dim_changed:
                changed = True
            op_dist_attr.set_impl_idx(op_dist_impl_idx)
        elif is_element_wise_op(op_desc.type):
            dim_changed = update_op_dims_mapping_by_element_wise_dist_impl(
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
                tensor_dist_attr = get_tensor_distributed_attr_graph(
                    tensor_node)
                dims_mapping_in_tensor = tensor_dist_attr.get_dims_mapping()
                assert dims_mapping_in_tensor is not None
                dims_mapping_in_op = op_dist_attr.get_output_dims_mapping(
                    tensor_desc.name())
                assert dims_mapping_in_op is not None
                compatible_dims_mapping = compute_compatible_dims_mapping(
                    [dims_mapping_in_op, dims_mapping_in_tensor])
                if (compatible_dims_mapping is not None) and \
                    (compatible_dims_mapping != dims_mapping_in_op):
                    op_dist_attr.set_output_dims_mapping(
                        tensor_desc.name(), compatible_dims_mapping)
                    changed = True
        # Find the most compatible implemenetations from the distributed operator
        op_dist_impl, op_dist_impl_idx = find_best_compatible_distributed_operator_impl(
            op_desc.type(), op_dist_attr, fwd=False)
        if op_dist_impl is not None:
            dim_changed = update_op_dims_mapping_by_dist_impl(op_dist_attr,
                                                              op_dist_impl)
            if dim_changed:
                changed = True
            op_dist_attr.set_impl_idx(op_dist_impl_idx)
        elif is_element_wise_op(op_desc.type):
            dim_changed = update_op_dims_mapping_by_element_wise_dist_impl(
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


def initialize_distributed_attr_for_program(program):
    for block in program.blocks:
        for tensor in block.vars.values():
            # Need make sure tensor is a tensor
            tensor_dist_attr = get_tensor_distributed_attr_program(tensor.desc)
            if tensor_dist_attr is None:
                distributed_attr_uid = generate_tensor_distributed_attr_uid()
                tensor.desc.set_distributed_attr_uid(distributed_attr_uid)
                tensor_dist_attr = TensorDistributedAttribute(tensor.desc)
                set_tensor_distributed_attr_program(tensor.desc,
                                                    tensor_dist_attr)
        for op in block.ops:
            op_dist_attr = get_op_distributed_attr_program(op.desc)
            if op_dist_attr is None:
                distributed_attr_uid = generate_op_distributed_attr_uid()
                op.desc.set_distributed_attr_uid(distributed_attr_uid)
                op_dist_attr = OperatorDistributedAttribute(op.desc)
                set_op_distributed_attr_program(op.desc, op_dist_attr)


def initialize_distributed_attr_for_graph(graph):
    all_nodes = graph.all_nodes()
    for node in all_nodes:
        if node.is_var() and node.var() is not None:
            tensor_desc = node.var()
            # Need make sure var is a tensor
            tensor_dist_attr = get_tensor_distributed_attr_program(tensor_desc)
            assert tensor_dist_attr is not None, \
                "Var must have a distributed attribute after the initialization for program."
            new_tensor_dist_attr = deepcopy(tensor_dist_attr)
            if new_tensor_dist_attr.get_dims_mapping() is None:
                tensor_dims_mapping = [
                    -1 for _ in range(len(tensor_desc.shape()))
                ]
                new_tensor_dist_attr.set_dims_mapping(tensor_dims_mapping)
            set_tensor_distributed_attr_graph(node, new_tensor_dist_attr)

        if node.is_op() and node.op() is not None:
            op_desc = node.op()
            op_dist_attr = get_op_distributed_attr_program(op_desc)
            assert op_dist_attr is not None, \
                "Op must have a distributed attribute after the initialization for program."
            new_op_dist_attr = deepcopy(op_dist_attr)
            for tensor_node in node.inputs:
                if tensor_node.var() is not None:
                    tensor_desc = tensor_node.var()
                    # Need make sure var is a tensor
                    if new_op_dist_attr.get_input_dims_mapping(tensor_desc.name(
                    )) is None:
                        tensor_dims_mapping = [
                            -1 for _ in range(len(tensor_desc.shape()))
                        ]
                        new_op_dist_attr.set_input_dims_mapping(
                            tensor_desc.name(), tensor_dims_mapping)
            for tensor_node in node.outputs:
                if tensor_node.var() is not None:
                    tensor_desc = tensor_node.var()
                    # Need make sure var is a tensor
                    if new_op_dist_attr.get_output_dims_mapping(
                            tensor_desc.name()) is None:
                        tensor_dims_mapping = [
                            -1 for _ in range(len(tensor_desc.shape()))
                        ]
                        new_op_dist_attr.set_output_dims_mapping(
                            tensor_desc.name(), tensor_dims_mapping)
            set_op_distributed_attr_graph(node, new_op_dist_attr)


def copy_distribute_attr_from_graph_to_program(graph, program):
    updated_tensors = {}
    all_nodes = graph.all_nodes()
    for node in all_nodes:
        if node.is_var() and node.var() is not None:
            tensor_desc = node.var()
            updated = updated_tensors.get(tensor_desc.name(), False)
            # If a var has multiples var nodes in graph, only use the first one for now
            if not updated:
                tensor_dist_attr = get_tensor_distributed_attr_graph(node)
                new_tensor_dist_attr = deepcopy(tensor_dist_attr)
                set_tensor_distributed_attr_program(tensor_desc,
                                                    new_tensor_dist_attr)
                updated_tensors[tensor_desc.name()] = True
        if node.is_op() and node.op() is not None:
            op_desc = node.op()
            op_dist_attr = get_op_distributed_attr_graph(node)
            new_op_dist_attr = deepcopy(op_dist_attr)
            set_op_distributed_attr_program(op_desc, new_op_dist_attr)


def complete_annotation(program):
    """ Complete annotation for the partial annotated program.

    Arguments:
        program: partial annotated program.

    Returns:
        program: completed annotated program.
    """
    # Initialize distributed attributes for all var and op node in program 
    initialize_distributed_attr_for_program(program)

    # Convert program to graph
    graph = framework.IrGraph(core.Graph(program.desc))

    # Initialize distributed attributes for all var and op node in graph
    initialize_distributed_attr_for_graph(graph)

    # # Complete process mesh for each node
    all_nodes = list(graph.all_nodes())
    reach_fix_point = False
    while not reach_fix_point:
        changed = False
        for node in all_nodes:
            if node.is_var() and node.var() is not None:
                tensor_changed = update_tensor_node_process_mesh(node, fwd=True)
                if tensor_changed:
                    changed = True
            if node.is_op() and node.op() is not None:
                op_changed = update_op_node_process_mesh(node, fwd=True)
                if op_changed:
                    changed = True
        for node in reversed(all_nodes):
            if node.is_var() and node.var() is not None:
                tensor_changed = update_tensor_node_process_mesh(
                    node, fwd=False)
                if tensor_changed:
                    changed = True
            if node.is_op() and node.op() is not None:
                op_changed = update_op_node_process_mesh(node, fwd=False)
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
                tensor_changed = update_tensor_node_dims_mapping(node, fwd=True)
                if tensor_changed:
                    changed = True
            if node.is_op() and node.op() is not None:
                op_changed = update_op_node_dims_mapping(node, fwd=True)
                if op_changed:
                    changed = True
        for node in reversed(all_nodes):
            if node.is_var() and node.var() is not None:
                tensor_changed = update_tensor_node_dims_mapping(
                    node, fwd=False)
                if tensor_changed:
                    changed = True
            if node.is_op() and node.op() is not None:
                op_changed = update_op_node_dims_mapping(node, fwd=False)
                if op_changed:
                    changed = True
        if changed:
            reach_fix_point = False
        else:
            reach_fix_point = True

    # Copy the corresponding distributed attribute from graph to program
    copy_distribute_attr_from_graph_to_program(graph, program)

    return program
