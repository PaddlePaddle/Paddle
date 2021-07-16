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

import paddle.fluid.core as core
import paddle.fluid.framework as framework
import distributed_operators as dist_ops

from .attribute import TensorDistributedAttribute
from .attribute import OperatorDistributedAttribute
from .attribute import get_tensor_distributed_attr_program
from .attribute import set_tensor_distributed_attr_program
from .attribute import get_op_distributed_attr_program
from .attribute import set_op_distributed_attr_program

TENSOR_DISTRIBUTED_ATTR_MAP_FOR_GRAPH = {}
OP_DISTRIBUTED_ATTR_MAP_FOR_GRAPH = {}


def get_tensor_distributed_attr_graph(tensor_node):
    tensor_node_id = tensor_node.id()
    global TENSOR_DISTRIBUTED_ATTR_MAP_FOR_GRAPH
    tensor_node_dist_attr = TENSOR_DISTRIBUTED_ATTR_MAP_FOR_GRAPH.get(
        tensor_node_id, None)
    return tensor_node_dist_attr


def set_tensor_distributed_attr_graph(tensor_node, tensor_node_dist_attr):
    tensor_node_id = tensor_node.id()
    global TENSOR_DISTRIBUTED_ATTR_MAP_FOR_GRAPH
    TENSOR_DISTRIBUTED_ATTR_MAP_FOR_GRAPH[tensor_node] = tensor_node_dist_attr


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
                compatible_process_mesh -= process_mesh
            else:
                assert is_same(process_mesh, compatible_process_mesh
                               ), "There is no compatible process mesh."

    return compatible_process_mesh


def update_tensor_node_process_mesh(op_node, fwd=True):
    for var_node in op_node.inputs():
        # Just skip empty vars and ctrl vars
        if var_node.var() is None:
            continue
        if fwd:
            pred_op_node = var_node.inputs()
            # Each var node have at most one predecessor operator because of SSA graph.
            # If it have no successor, just skip the following statements.
            if not pred_op_node and pred_op_node.op() is not None:
                dist_attr_in_op = get_op_distributed_attr_graph(pred_op_node)
                process_mesh_in_op = dist_attr_in_op.get_process_mesh()
                dist_attr_in_var = get_tensor_distributed_attr_graph(var_node)
                process_mesh_in_var = dist_attr_in_var.get_process_mesh()
                if process_mesh_in_op is not None and process_mesh_in_var is None:
                    dist_attr_in_var.set_process_mesh(process_mesh_in_op)
        else:
            succ_op_node = var_node.outputs()
            # Each var node have at most one successor operator because of SSA graph.
            # If it have no successor, just skip the following statements.
            if not succ_op_node and succ_op_node.op() is not None:
                dist_attr_in_op = get_op_distributed_attr_graph(succ_op_node)
                process_mesh_in_op = dist_attr_in_op.get_process_mesh()
                dist_attr_in_var = get_tensor_distributed_attr_graph(var_node)
                process_mesh_in_var = dist_attr_in_var.get_process_mesh()
                if process_mesh_in_op is not None and process_mesh_in_var is None:
                    dist_attr_in_var.set_process_mesh(process_mesh_in_op)


def update_op_node_process_mesh(op_node, fwd=True):
    op_dist_attr = get_op_distributed_attr_graph(op_node)
    if fwd:
        inputs_process_meshes = []
        for var_node in op_node.inputs():
            # Just skip empty vars and ctrl vars
            if var_node.var() is None:
                continue
            var_dist_attr = get_tensor_distributed_attr_graph(var_node)
            if var_dist_attr is not None:
                process_mesh_in_var = var_dist_attr.get_process_mesh()
                inputs_process_meshes.append(process_mesh_in_var)
        compatible_process_mesh = compute_compatible_process_mesh(
            inputs_process_meshes)
        process_mesh_in_op = op_dist_attr.get_process_mesh()
        if compatible_process_mesh is not None and process_mesh_in_op is None:
            op_dist_attr.set_process_mesh(compatible_process_mesh)
    else:
        outputs_process_meshes = []
        for var_node in op_node.outputs():
            # Just skip empty vars and ctrl vars
            if var_node.var() is None:
                continue
            var_dist_attr = get_tensor_distributed_attr_graph(var_node)
            if var_dist_attr is not None:
                process_mesh_in_var = var_dist_attr.get_process_mesh()
                outputs_process_meshes.append(process_mesh_in_var)
        compatible_process_mesh = compute_compatible_process_mesh(
            outputs_process_meshes)
        process_mesh_in_op = op_dist_attr.get_process_mesh()
        if compatible_process_mesh is not None and process_mesh_in_op is None:
            op_dist_attr.set_process_mesh(compatible_process_mesh)


def is_same_dims_mapping(left_dims_mapping, right_dims_mapping):
    if len(left_dims_mapping) != len(right_dims_mapping):
        return False
    for i in range(len(left_dims_mapping)):
        if left_dims_mapping[i] != right_dims_mapping[i]:
            return False
    return True


def compute_compatible_dim_mapping(mapping_list):
    fined_mapping = -1
    for mapping in mapping_list:
        if fined_mapping == -1:
            fined_mapping = mapping
        elif fined_mapping == mapping:
            continue
        else:
            raise ValueError("There is no compatible mapping.")
    return fined_mapping


def compute_compatible_dims_mapping(src_dims_mapping, dst_dims_mapping):
    assert len(src_dims_mapping) == len(dst_dims_mapping), \
        "The length of the source dims_mapping must be same as that of the destination dims_mapping."
    compatible_result = [-1 for _ in range(len(src_dims_mapping))]
    for i in range(src_dims_mapping):
        compatible_result[i] = compute_compatible_fined_mapping(
            [src_dims_mapping[i], dst_dims_mapping[i]])
    return compatible_result


def update_op_dims_mapping_by_impl(op_dist_attr, op_dist_impl):
    op_desc = op_dist_attr.get_op_desc()
    dist_singnature = op_dist_impl.get_distributed_signature()
    same_shard_dims_list = dist_singnature.get_valid_inputs_outputs_same_shard_dims_list(
    )
    for same_shard_dims in same_shard_dims_list:
        mapping_list = []
        for in_or_out, para_name, dim in same_shard_dims:
            if in_or_out == 'input':
                for arg_name in op_desc.input(para_name):
                    mapping_list.append(
                        op_dist_attr.get_input_dim_mapping(arg_name, dim))
            else:
                for arg_name in op_desc.output(para_name):
                    mapping_list.append(
                        op_dist_attr.get_input_dim_mapping(arg_name, dim))
        compatible_dim_mapping = compute_compatible_dim_mapping(mapping_list)
        for in_or_out, para_name, dim in same_shard_dims:
            if in_or_out == 'input':
                for arg_name in op_desc.input(para_name):
                    op_dist_attr.set_input_dim_mapping(arg_name, dim,
                                                       compatible_dim_mapping)
            else:
                for arg_name in op_desc.output(para_name):
                    op_dist_attr.set_output_dim_mapping(arg_name, dim,
                                                        compatible_dim_mapping)


def update_tensor_node_dims_mapping(op_node, fwd=True):
    changed = False
    if not op_node.is_op():
        return changed
    for var_node in op_node.inputs():
        if var_node.var() is None:
            continue
        var_desc = var_node.var()
        if fwd:
            pred_op_node = var_node.inputs()
            # Each var node have at most one predecessor operator because of SSA graph.
            # If it have no successor, just skip the following statements.
            if not pred_op_node and pred_op_node.op() is not None:
                dist_attr_in_op = get_op_distributed_attr_graph(pred_op_node)
                dims_mapping_in_op = dist_attr_in_op.get_dims_mapping(
                    var_desc.name)
                dist_attr_in_var = get_tensor_distributed_attr_graph(var_node)
                dims_mapping_in_var = dist_attr_in_var.get_dims_mapping()
                compatible_dims_mapping = compute_compatible_dims_mapping(
                    dims_mapping_in_op, dims_mapping_in_var)
                if not is_same_dims_mapping(dims_mapping_in_var,
                                            compatible_dims_mapping):
                    changed = True
                dist_attr_in_var.set_dims_mapping(compatible_dims_mapping)
        else:
            succ_op_node = var_node.outputs()
            # Each var node have at most one successor operator because of SSA graph.
            # If it have no successor, just skip the following statements.
            if not succ_op_node and succ_op_node.op() is not None:
                dist_attr_in_op = get_op_distributed_attr_graph(succ_op_node)
                dims_mapping_in_op = dist_attr_in_op.get_dims_mapping(
                    var_desc.name)
                dist_attr_in_var = get_tensor_distributed_attr_graph(var_node)
                dims_mapping_in_var = dist_attr_in_var.get_dims_mapping()
                compatible_dims_mapping = compute_compatible_dims_mapping(
                    dims_mapping_in_op, dims_mapping_in_var)
                if not is_same_dims_mapping(dims_mapping_in_var,
                                            compatible_dims_mapping):
                    changed = True
                dist_attr_in_var.set_dims_mapping(compatible_dims_mapping)
    return changed


def update_op_node_dims_mapping(op_node, fwd=True):
    changed = False
    if (not op_node.is_op()) or (op_node.op() is None):
        return changed
    op_desc = op_node.op()
    op_dist_attr = get_op_distributed_attr_graph(op_node)
    if fwd:
        for var_node in op_node.inputs():
            if var_node.var() is not None:
                var_dist_attr = get_tensor_distributed_attr_graph(var_node)
                dims_mapping_in_var = var_dist_attr.get_dims_mapping()
                dims_mapping_in_op = op_dist_attr.get_dims_mapping(var_node)
                compatible_dims_mapping = compute_compatible_dims_mapping(
                    dims_mapping_in_op, dims_mapping_in_var)
                if not is_same_dims_mapping(dims_mapping_in_op,
                                            compatible_dims_mapping):
                    changed = True
                var_desc = var_node.var()
                op_dist_attr.set_input_dims_mapping(var_desc,
                                                    compatible_dims_mapping)
        # Find compatible implemenetations from the distributed operator,
        # and return the most compatible one.
        op_dist_impl, op_dist_impl_idx = dist_ops.find_best_compatible_distributed_operator_impl(
            op_desc.type, op_dist_attr, fwd=True)
        assert op_dist_impl != None, "Cannot find a compatbile distributed operator implementations."
        update_op_dims_mapping_by_impl(op_dist_attr, op_dist_impl)
        op_dist_attr.set_impl_idx(op_dist_impl_idx)
    else:
        for var_node in op_node.outputs():
            if var_node.var() is not None:
                var_dist_attr = get_tensor_distributed_attr_graph(var_node)
                dims_mapping_in_var = var_dist_attr.get_dims_mapping()
                dims_mapping_in_op = op_dist_attr.get_dims_mapping(var_node)
                compatible_dims_mapping = compute_compatible_dims_mapping(
                    dims_mapping_in_op, dims_mapping_in_var)
                if not is_same_dims_mapping(dims_mapping_in_op,
                                            compatible_dims_mapping):
                    changed = True
                var_desc = var_node.var()
                op_dist_attr.set_output_dims_mapping(var_desc,
                                                     compatible_dims_mapping)
        # Find compatible implemenetations from the distributed operator,
        # and return the most compatible one.
        op_dist_impl, op_dist_impl_idx = dist_ops.find_best_compatible_distributed_operator_impl(
            op_desc.type, op_dist_attr, fwd=False)
        assert op_dist_impl != None, "Cannot find a compatbile distributed operator implementations."
        update_op_dims_mapping_by_impl(op_dist_attr, op_dist_impl)
        op_dist_attr.set_impl_idx(op_dist_impl_idx)

    return changed


def initialize_distributed_attr_for_graph(graph):
    all_nodes = graph.all_nodes()
    for node in all_nodes:
        if node.is_var() and node.var() is not None:
            var_desc = node.var()
            var_shape = var_desc.shape
            var_dist_attr = TensorDistributedAttribute(var_desc)
            var_dims_mapping = [-1 for _ in range(len(var_shape))]
            var_dist_attr.set_dims_mapping(var_dims_mapping)
            set_tensor_distributed_attr_graph(node, var_dist_attr)
        if node.is_op() and node.op() is not None:
            op_desc = node.op()
            op_dist_attr = OperatorDistributedAttribute(op_desc)
            for var_node in node.inputs():
                if var_node.var() is not None:
                    var_desc = var_node.var()
                    var_shape = var_desc.shape
                    var_dims_mapping = [-1 for _ in range(len(var_shape))]
                    op_dist_attr.set_input_dims_mapping(var_desc.name,
                                                        var_dims_mapping)
            for var_node in node.outputs():
                if var_node.var() is not None:
                    var_desc = var_node.var()
                    var_shape = var_desc.shape
                    var_dims_mapping = [-1 for _ in range(len(var_shape))]
                    op_dist_attr.set_output_dims_mapping(var_desc.name,
                                                         var_dims_mapping)
            set_op_distributed_attr_graph(node, op_dist_attr)


def copy_distribute_attr_to_program(graph, program):
    all_nodes = graph.all_nodes()
    for node in all_nodes:
        if node.is_var() and node.var() is not None:
            var_desc = node.var()
            # If a var_desc has multiples var nodes in graph, we only use the first one for now
            var_dist_attr = get_tensor_distributed_attr_graph(node)
            if get_tensor_distributed_attr_program(var_desc) is None:
                set_tensor_distributed_attr_program(var_desc, var_dist_attr)
        if node.is_op() and node.op() is not None:
            op_desc = node.op()
            op_dist_attr = get_op_distributed_attr_graph(node)
            if get_op_distributed_attr_program(op_desc) is None:
                set_op_distributed_attr_program(op_desc, op_dist_attr)


def complete_annotation(program):
    """ Complete annotation for the partial annotated program.

    Arguments:
        program: partial annotated program.

    Returns:
        program: completed annotated program.
    """
    graph = framework.IrGraph(core.Graph(program.desc))

    # Initialize distributed attributes for all var and op node in graph
    initialize_distributed_attr_for_graph(graph)

    ordered_nodes = graph.topology_sort()
    reverse_ordered_nodes = ordered_nodes.reverse()

    # Complete process mesh for each node
    for node in ordered_nodes:
        if node.is_op() and node.op() is not None:
            update_tensor_node_process_mesh(node, fwd=True)
            update_op_node_process_mesh(node, fwd=True)

    for node in reverse_ordered_nodes:
        if node.is_op() and node.op() is not None:
            update_tensor_node_process_mesh(node, fwd=False)
            update_op_node_process_mesh(node, fwd=False)

    # Complete dims_mapping for each node
    reach_fix_point = False
    while not reach_fix_point:
        changed = True
        for node in ordered_nodes:
            if node.is_op() and node.op() is not None:
                changed = update_tensor_node_dims_mapping(node, fwd=True)
                changed = update_op_node_dims_mapping(node, fwd=True)

        for node in reverse_ordered_nodes:
            if node.is_op() and node.op() is not None:
                changed = update_tensor_node_dims_mapping(node, fwd=False)
                changed = update_op_node_dims_mapping(node, fwd=False)

        if changed:
            reach_fix_point = False
        else:
            reach_fix_point = True

    # Copy the corresponding distributed attribute from graph's to program's
    copy_distribute_attr_to_program(graph, program)

    return program
