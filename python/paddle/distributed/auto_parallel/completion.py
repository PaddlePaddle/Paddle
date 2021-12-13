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

from .utils import compute_compatible_process_mesh
from .utils import compute_compatible_dim_mapping
from .utils import compute_compatible_dims_mapping
from .utils import print_program_with_dist_attr
from .operators import find_best_compatible_distributed_operator_impl
from .dist_context import get_default_distributed_context
from .dist_tensor import DistributedTensor
from .dist_op import DistributedOperator
from .dist_attribute import TensorDistributedAttribute
from .dist_attribute import OperatorDistributedAttribute
from paddle.distributed.fleet.meta_optimizers.common import OpRole

ELEMENTWISE_LIKE_OP_LIST = ["elementwise_add", "gelu", "dropout", "cast"]


def is_elementwise_like_op(op_type):
    if op_type in ELEMENTWISE_LIKE_OP_LIST:
        return True
    else:
        return False


def update_tensor_node_process_mesh(dist_context, tensor_node, fwd=True):
    """
    Update tensor's process mesh by using its predecessor's process mesh if in the forward direction, 
    and by using its successor's process mesh if in the backward direction. Note: only the equal 
    process meshes are compatible for now.
    """
    changed = False
    tensor_dist_attr = dist_context.get_tensor_dist_attr_for_graph(tensor_node)
    if tensor_dist_attr.is_annotated("process_mesh"):
        return changed
    tensor_process_mesh = tensor_dist_attr.process_mesh
    if fwd:
        inputs_process_meshes = []
        for pred_op_node in tensor_node.inputs:
            if pred_op_node.op() is not None:
                op_dist_attr = dist_context.get_op_dist_attr_for_graph(
                    pred_op_node)
                op_process_mesh = op_dist_attr.process_mesh
                inputs_process_meshes.append(op_process_mesh)
        compatible_process_mesh = compute_compatible_process_mesh(
            inputs_process_meshes)
        if compatible_process_mesh is not None and tensor_process_mesh is None:
            tensor_dist_attr.process_mesh = compatible_process_mesh
            changed = True
    else:
        outputs_process_meshes = []
        for succ_op_node in tensor_node.outputs:
            if succ_op_node.op() is not None:
                op_dist_attr = dist_context.get_op_dist_attr_for_graph(
                    succ_op_node)
                op_process_mesh = op_dist_attr.process_mesh
                outputs_process_meshes.append(op_process_mesh)
        compatible_process_mesh = compute_compatible_process_mesh(
            outputs_process_meshes)
        if compatible_process_mesh is not None and tensor_process_mesh is None:
            tensor_dist_attr.process_mesh = compatible_process_mesh
            changed = True
    return changed


def update_op_node_process_mesh(dist_context, op_node, fwd=True):
    """
    Update op's process mesh by using its predecessor's process mesh if in the forward direction, 
    and by using its successor's process mesh if in the backward direction. Note: only the equal 
    process meshes are compatible for now.
    """
    changed = False
    op_dist_attr = dist_context.get_op_dist_attr_for_graph(op_node)
    if op_dist_attr.is_annotated("process_mesh"):
        return changed
    op_process_mesh = op_dist_attr.process_mesh
    if fwd:
        inputs_process_meshes = []
        for tensor_node in op_node.inputs:
            if tensor_node.var() is not None:
                tensor_dist_attr = dist_context.get_tensor_dist_attr_for_graph(
                    tensor_node)
                tensor_process_mesh = tensor_dist_attr.process_mesh
                inputs_process_meshes.append(tensor_process_mesh)
        compatible_process_mesh = compute_compatible_process_mesh(
            inputs_process_meshes)
        if compatible_process_mesh is not None and op_process_mesh is None:
            op_dist_attr.process_mesh = compatible_process_mesh
            changed = True
    else:
        outputs_process_meshes = []
        for tensor_node in op_node.outputs:
            if tensor_node.var() is not None:
                tensor_dist_attr = dist_context.get_tensor_dist_attr_for_graph(
                    tensor_node)
                tensor_process_mesh = tensor_dist_attr.process_mesh
                outputs_process_meshes.append(tensor_process_mesh)
        compatible_process_mesh = compute_compatible_process_mesh(
            outputs_process_meshes)
        if compatible_process_mesh is not None and op_process_mesh is None:
            op_dist_attr.process_mesh = compatible_process_mesh
            changed = True
    return changed


def update_op_dims_mapping_by_default_dist_impl(dist_context, op_node):
    """Each operator has a default distributed operator, only allowed to be sharded in batch dimension."""
    changed = False
    if (not op_node.is_op()) or (op_node.op() is None):
        return False
    op_desc = op_node.op()
    dist_op = dist_context.get_dist_op_for_graph(op_node)
    op_dist_attr = dist_op.dist_attr
    # The following statement will be replaced by a more elegent way
    if op_desc.type() == "shape" or op_desc.type() == "slice":
        return False
    output_names = op_desc.output_names()
    xshape_arg_names = []
    if "XShape" in output_names:
        xshape_arg_names = op_desc.output("XShape")
    batch_dim_mappings = []
    for arg_name in op_desc.input_arg_names():
        serial_tensor = dist_op.get_serial_input(arg_name)
        if serial_tensor.is_parameter:
            continue
        dims_mapping = op_dist_attr.get_input_dims_mapping(arg_name)
        if len(dims_mapping) > 1:
            for idx, mapping in enumerate(dims_mapping[1:]):
                assert mapping == -1, \
                    "{} only the batch dimension (0-dim) can be sharded, but the dimension {} is sharded by {} part."\
                        .format(op_desc.type(), idx, mapping)
        batch_dim_mappings.append(dims_mapping[0])
    for arg_name in op_desc.output_arg_names():
        serial_tensor = dist_op.get_serial_output(arg_name)
        if serial_tensor.is_parameter:
            continue
        dims_mapping = op_dist_attr.get_output_dims_mapping(arg_name)
        if arg_name not in xshape_arg_names:
            if len(dims_mapping) > 1:
                for idx, mapping in enumerate(dims_mapping[1:]):
                    assert mapping == -1, \
                        "{} only the batch dimension (0-dim) can be sharded, but the dimension {} is sharded by {} part."\
                            .format(op_desc.type(), idx, mapping)
            batch_dim_mappings.append(dims_mapping[0])
        else:
            assert dims_mapping[0] == -1, \
                "{} only the batch dimension (1-dim) of XShape can be sharded, but the dimension 0 is sharded by {} part."\
                    .format(op_desc.type(), mapping)
            if len(dims_mapping) > 2:
                for idx, mapping in enumerate(dims_mapping[2:]):
                    assert mapping == -1, \
                        "{} only the batch dimension (1-dim) of XShape can be sharded, but the dimension {} is sharded by {} part."\
                            .format(op_desc.type(), idx, mapping)
            batch_dim_mappings.append(dims_mapping[1])

    compatible_dim_mapping = compute_compatible_dim_mapping(batch_dim_mappings)
    assert compatible_dim_mapping is not None, "There is no compatible dim mapping."
    for arg_name in op_desc.input_arg_names():
        serial_tensor = dist_op.get_serial_input(arg_name)
        if serial_tensor.is_parameter:
            continue
        dims_mapping = op_dist_attr.get_input_dims_mapping(arg_name)
        if compatible_dim_mapping != dims_mapping[0]:
            dims_mapping[0] = compatible_dim_mapping
            changed = True
    for arg_name in op_desc.output_arg_names():
        serial_tensor = dist_op.get_serial_output(arg_name)
        if serial_tensor.is_parameter:
            continue
        dims_mapping = op_dist_attr.get_output_dims_mapping(arg_name)
        if arg_name not in xshape_arg_names:
            if compatible_dim_mapping != dims_mapping[0]:
                dims_mapping[0] = compatible_dim_mapping
                changed = True
        else:
            if compatible_dim_mapping != dims_mapping[1]:
                dims_mapping[1] = compatible_dim_mapping
                changed = True

    return changed


def update_op_dims_mapping_by_elementwise_like_dist_impl(dist_context, op_node):
    """Element-wise operator can be sharded in any way (but should take care of broadcasting)."""
    changed = False
    if (not op_node.is_op()) or (op_node.op() is None):
        return False
    op_desc = op_node.op()
    op_dist_attr = dist_context.get_op_dist_attr_for_graph(op_node)

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


def update_tensor_node_dims_mapping(dist_context, tensor_node, fwd=True):
    changed = False
    if (not tensor_node.is_var()) or (tensor_node.var() is None):
        return False
    tensor_desc = tensor_node.var()
    # Skip reader tensor
    if tensor_desc.type() == core.VarDesc.VarType.READER:
        return False
    tensor_dist_attr = dist_context.get_tensor_dist_attr_for_graph(tensor_node)
    assert tensor_dist_attr is not None
    if tensor_dist_attr.is_annotated("dims_mapping"):
        return False
    tensor_dims_mapping = tensor_dist_attr.dims_mapping
    if fwd:
        dims_mapping_list = []
        for pred_op_node in tensor_node.inputs:
            if pred_op_node.op() is not None:
                if pred_op_node.op().type() == "create_py_reader" \
                    or pred_op_node.op().type() == "create_double_buffer_reader" \
                    or pred_op_node.op().type() == "read":
                    continue
                op_dist_attr = dist_context.get_op_dist_attr_for_graph(
                    pred_op_node)
                op_dims_mapping = op_dist_attr.get_output_dims_mapping(
                    tensor_desc.name())
                dims_mapping_list.append(op_dims_mapping)
        dims_mapping_list.append(tensor_dims_mapping)
        compatible_dims_mapping = compute_compatible_dims_mapping(
            dims_mapping_list)
        if (compatible_dims_mapping is not None) and \
            (compatible_dims_mapping != tensor_dims_mapping):
            tensor_dist_attr.dims_mapping = compatible_dims_mapping
            changed = True
    else:
        dims_mapping_list = []
        for succ_op_node in tensor_node.outputs:
            if succ_op_node.op() is not None:
                if succ_op_node.op().type() == "create_py_reader" \
                    or succ_op_node.op().type() == "create_double_buffer_reader" \
                    or succ_op_node.op().type() == "read":
                    continue
                op_dist_attr = dist_context.get_op_dist_attr_for_graph(
                    succ_op_node)
                op_dims_mapping = op_dist_attr.get_input_dims_mapping(
                    tensor_desc.name())
                dims_mapping_list.append(op_dims_mapping)
        dims_mapping_list.append(tensor_dims_mapping)
        compatible_dims_mapping = compute_compatible_dims_mapping(
            dims_mapping_list)
        if (compatible_dims_mapping is not None) and \
            (compatible_dims_mapping != tensor_dims_mapping):
            tensor_dist_attr.dims_mapping = compatible_dims_mapping
            changed = True
    return changed


def update_op_node_dims_mapping(dist_context, op_node, fwd=True):
    changed = False
    if (not op_node.is_op()) or (op_node.op() is None):
        return False
    # Skip reader op
    op_desc = op_node.op()
    if op_desc.type() == "create_py_reader" \
        or op_desc.type() == "create_double_buffer_reader" \
        or op_desc.type() == "read":
        return False
    dist_op = dist_context.get_dist_op_for_graph(op_node)
    op_dist_attr = dist_op.dist_attr
    if fwd:
        for tensor_node in op_node.inputs:
            if tensor_node.var() is not None:
                if tensor_node.var().type() == core.VarDesc.VarType.READER:
                    continue
                tensor_desc = tensor_node.var()
                if op_dist_attr.is_annotated_input_dims_mapping(
                        tensor_desc.name()):
                    continue
                tensor_dist_attr = dist_context.get_tensor_dist_attr_for_graph(
                    tensor_node)
                tensor_dims_mapping = tensor_dist_attr.dims_mapping
                op_dims_mapping = op_dist_attr.get_input_dims_mapping(
                    tensor_desc.name())
                compatible_dims_mapping = compute_compatible_dims_mapping(
                    [op_dims_mapping, tensor_dims_mapping])
                if (compatible_dims_mapping is not None) and \
                    (compatible_dims_mapping != op_dims_mapping):
                    op_dist_attr.set_input_dims_mapping(tensor_desc.name(),
                                                        compatible_dims_mapping)
                    changed = True
        # Find the most compatible implemenetations from the distributed operator
        op_dist_impl, op_dist_impl_idx = find_best_compatible_distributed_operator_impl(
            op_desc.type(), dist_op, fwd=True)
        if op_dist_impl is not None:
            dim_changed = op_dist_impl.update_dims_mapping(dist_op)
            if dim_changed:
                changed = True
            # This statement will be replaced by a good way
            if op_dist_impl.is_compatible(dist_op):
                op_dist_attr.impl_type = op_desc.type()
                op_dist_attr.impl_idx = op_dist_impl_idx
        elif is_elementwise_like_op(op_desc.type()):
            dim_changed = update_op_dims_mapping_by_elementwise_like_dist_impl(
                dist_context, op_node)
            if dim_changed:
                changed = True
            op_dist_attr.impl_type = "element-wise"
            op_dist_attr.impl_idx = -1
        else:
            dim_changed = update_op_dims_mapping_by_default_dist_impl(
                dist_context, op_node)
            if dim_changed:
                changed = True
            op_dist_attr.impl_type = "default"
            op_dist_attr.impl_idx = -2
    else:
        for tensor_node in op_node.outputs:
            if tensor_node.var() is not None:
                if tensor_node.var().type() == core.VarDesc.VarType.READER:
                    continue
                tensor_desc = tensor_node.var()
                if op_dist_attr.is_annotated_output_dims_mapping(
                        tensor_desc.name()):
                    continue
                tensor_dist_attr = dist_context.get_tensor_dist_attr_for_graph(
                    tensor_node)
                tensor_dims_mapping = tensor_dist_attr.dims_mapping
                op_dims_mapping = op_dist_attr.get_output_dims_mapping(
                    tensor_desc.name())
                compatible_dims_mapping = compute_compatible_dims_mapping(
                    [op_dims_mapping, tensor_dims_mapping])
                if (compatible_dims_mapping is not None) and \
                    (compatible_dims_mapping != op_dims_mapping):
                    op_dist_attr.set_output_dims_mapping(
                        tensor_desc.name(), compatible_dims_mapping)
                    changed = True
        # Find the most compatible implemenetations from the distributed operator
        op_dist_impl, op_dist_impl_idx = find_best_compatible_distributed_operator_impl(
            op_desc.type(), dist_op, fwd=False)
        if op_dist_impl is not None:
            dim_changed = op_dist_impl.update_dims_mapping(dist_op)
            if dim_changed:
                changed = True
            # This statement will be replaced by a good way
            if op_dist_impl.is_compatible(dist_op):
                op_dist_attr.impl_type = op_desc.type()
                op_dist_attr.impl_idx = op_dist_impl_idx
        elif is_elementwise_like_op(op_desc.type()):
            dim_changed = update_op_dims_mapping_by_elementwise_like_dist_impl(
                dist_context, op_node)
            if dim_changed:
                changed = True
            op_dist_attr.impl_type = "element-wise"
            op_dist_attr.impl_idx = -1
        else:
            dim_changed = update_op_dims_mapping_by_default_dist_impl(
                dist_context, op_node)
            if dim_changed:
                changed = True
            op_dist_attr.impl_type = "default"
            op_dist_attr.impl_idx = -2
    return changed


def complete_annotation(program, dist_context=None):
    """ Complete annotation for the partial annotated program.

    Arguments:
        program: partial annotated program.
        dist_context: the distributed context is used to store distributed attributes for program.
            If not provided, the default one will be used.
    Returns:
        program: completed annotated program.
    """

    # Use the default distribted context for completeion if there is no one
    if dist_context is None:
        dist_context = get_default_distributed_context()
        dist_context.serial_program = program
    else:
        dist_context.serial_program = program

    # print_program_with_dist_attr(program, dist_context)

    # Initialize distributed attributes for all var and op node in program
    dist_context.init_dist_attr_for_program()

    # Initialize distributed attributes for all var and op node in graph
    dist_context.init_dist_attr_for_graph()

    # Complete process mesh for each node
    all_nodes = list(dist_context.serial_graph.all_nodes())

    def sort_key_fun(node):
        first = -1
        if node.is_op():
            first = 0
        else:
            first = 1
        second = -1
        if node.is_op() and node.op() is not None:
            second = node.op().id()
        if node.is_var() and node.var() is not None:
            second = node.var().id()
        return (first, second)

    all_nodes.sort(key=sort_key_fun)

    reach_fix_point = False
    while not reach_fix_point:
        total_changed = False
        reach_fwd_fix_point = False
        reach_bwd_fix_point = False
        while not reach_fwd_fix_point:
            changed = False
            for node in all_nodes:
                if node.is_var() and node.var() is not None:
                    tensor_changed = update_tensor_node_process_mesh(
                        dist_context, node, fwd=True)
                    if tensor_changed:
                        changed = True
                if node.is_op() and node.op() is not None:
                    op_changed = update_op_node_process_mesh(
                        dist_context, node, fwd=True)
                    if op_changed:
                        changed = True
            if changed:
                reach_fwd_fix_point = False
                total_changed = True
            else:
                reach_fwd_fix_point = True
        while not reach_bwd_fix_point:
            changed = False
            for node in all_nodes:
                if node.is_var() and node.var() is not None:
                    tensor_changed = update_tensor_node_process_mesh(
                        dist_context, node, fwd=False)
                    if tensor_changed:
                        changed = True
                if node.is_op() and node.op() is not None:
                    op_changed = update_op_node_process_mesh(
                        dist_context, node, fwd=False)
                    if op_changed:
                        changed = True
            if changed:
                reach_bwd_fix_point = False
                total_changed = True
            else:
                reach_bwd_fix_point = True
        if total_changed:
            reach_fix_point = False
        else:
            reach_fix_point = True
            # Validation the completion of process meshes and should be moved to a proper location
            is_wrong = False
            for node in all_nodes:
                if node.is_var() and node.var() is not None:
                    tensor_dist_attr = dist_context.get_tensor_dist_attr_for_graph(
                        node)
                    if tensor_dist_attr.process_mesh is None:
                        msg_str = ""
                        for op_node in node.inputs:
                            if op_node.op() is not None:
                                op_dist_attr = dist_context.get_op_dist_attr_for_graph(
                                    op_node)
                                msg_str += "{} [{}], ".format(
                                    op_node.op().type(),
                                    op_dist_attr.process_mesh)
                            else:
                                msg_str += "{} [{}], ".format(op_node.name(),
                                                              None)
                        for op_node in node.outputs:
                            if op_node.op() is not None:
                                op_dist_attr = dist_context.get_op_dist_attr_for_graph(
                                    op_node)
                                msg_str += "{} [{}], ".format(
                                    op_node.op().type(),
                                    op_dist_attr.process_mesh)
                            else:
                                msg_str += "{} [{}], ".format(op_node.name(),
                                                              None)
                        msg_str = "Cannot decide ProcessMesh of {} among {}. Please use shard_tensor api explicitly to annotate it".format(
                            node.var().name(), msg_str[:-2])
                        is_wrong = True
                        print(msg_str)
                if node.is_op() and node.op() is not None:
                    op_dist_attr = dist_context.get_op_dist_attr_for_graph(node)
                    if op_dist_attr.process_mesh is None:
                        msg_str = ""
                        for tensor_node in node.inputs:
                            if tensor_node.var() is not None:
                                tensor_dist_attr = dist_context.get_tensor_dist_attr_for_graph(
                                    tensor_node)
                                msg_str += "{} [{}], ".format(
                                    tensor_node.var().name(),
                                    tensor_dist_attr.process_mesh)
                            else:
                                msg_str += "{} [{}], ".format(
                                    tensor_node.name(), None)
                        for tensor_node in node.outputs:
                            if tensor_node.var() is not None:
                                tensor_dist_attr = dist_context.get_tensor_dist_attr_for_graph(
                                    tensor_node)
                                msg_str += "{} [{}], ".format(
                                    tensor_node.var().name(),
                                    tensor_dist_attr.process_mesh)
                            else:
                                msg_str += "{} [{}], ".format(
                                    tensor_node.name(), None)
                        msg_str = "Cannot decide ProcessMesh of {} among {}. Please use shard_op api explicitly to annotate it".format(
                            node.op().type(), msg_str[:-2])
                        is_wrong = True
                        print(msg_str)
                if node.is_op() and node.op() is None:
                    print("op op is None", node.name())
            if is_wrong:
                assert False, "Cannot complete process_meshes of the program."

    # Complete dims_mapping for each node
    reach_fix_point = False
    while not reach_fix_point:
        changed = False
        for node in all_nodes:
            if node.is_var() and node.var() is not None:
                tensor_changed = update_tensor_node_dims_mapping(
                    dist_context, node, fwd=True)
                if tensor_changed:
                    changed = True
            if node.is_op() and node.op() is not None:
                op_changed = update_op_node_dims_mapping(
                    dist_context, node, fwd=True)
                if op_changed:
                    changed = True
        for node in reversed(all_nodes):
            if node.is_var() and node.var() is not None:
                tensor_changed = update_tensor_node_dims_mapping(
                    dist_context, node, fwd=False)
                if tensor_changed:
                    changed = True
            if node.is_op() and node.op() is not None:
                op_changed = update_op_node_dims_mapping(
                    dist_context, node, fwd=False)
                if op_changed:
                    changed = True
        if changed:
            reach_fix_point = False
        else:
            reach_fix_point = True

    # Copy the corresponding distributed attribute from graph to program
    dist_context.copy_dist_attr_from_graph_to_program()
    dist_context.clear_dist_info_for_graph()

    # Do the validation check and amend some completion
    dist_context.amend_dist_attr_for_program()

    # print_program_with_dist_attr(program, dist_context)
    dist_context.validate_dist_attr_for_program()

    return program


def complete_backward_annotation(auto_parallel_main_prog, dist_context=None):
    """Complete the annotation of vars and ops in the backward phase for parallel program."""

    def _is_grad_var_name(name):
        if "@GRAD" in name:
            return True
        return False

    def _get_forward_varname_from_grad_varname(grad_var_name):
        assert _is_grad_var_name(
            grad_var_name), "[{}] is not a grad varnme.".format(grad_var_name)
        return grad_var_name[:grad_var_name.find("@GRAD")]

    def _get_op_by_id(ops, id):
        for op in ops:
            if op.desc.id() == id:
                return op
        return None

    if dist_context is None:
        dist_context = get_default_distributed_context()

    first_backward_op_idx = -1
    for idx, op in enumerate(auto_parallel_main_prog.global_block().ops):
        if int(op.attr('op_role')) == int(
                int(core.op_proto_and_checker_maker.OpRole.Backward) | int(
                    core.op_proto_and_checker_maker.OpRole.Loss)):
            assert op.type == "fill_constant"
            first_backward_op_idx = idx
            break

    assert first_backward_op_idx >= 0, "No backward procedure found in this program."

    ops = list(auto_parallel_main_prog.global_block().ops)
    vars = auto_parallel_main_prog.global_block().vars
    dist_op_context = dist_context.dist_op_context

    for idx in range(first_backward_op_idx, len(ops)):

        # complete the initial grad loss op
        if idx == first_backward_op_idx:
            assert ops[idx].type == "fill_constant"
            assert len(
                ops[idx].input_arg_names
            ) == 0, "first backward op should has only ONE output, but got [{}]".format(
                len(ops[idx].input_arg_names))
            assert len(
                ops[idx].output_arg_names
            ) == 1, "first backward op should has only ONE output, but got [{}]".format(
                len(ops[idx].output_arg_names))

            grad_var = vars[ops[idx].output_arg_names[0]]
            forward_var_name = _get_forward_varname_from_grad_varname(
                grad_var.name)
            forward_var = vars[forward_var_name]

            # TODO complete other attribte for grad var
            tensor_dist_attr = TensorDistributedAttribute()
            process_mesh = dist_context.get_tensor_dist_attr_for_program(
                forward_var).process_mesh
            dims_mapping = dist_context.get_tensor_dist_attr_for_program(
                forward_var).dims_mapping
            tensor_dist_attr.dims_mapping = dims_mapping
            tensor_dist_attr.process_mesh = process_mesh
            dist_context.set_tensor_dist_attr_for_program(grad_var,
                                                          tensor_dist_attr)

            op_dist_attr = OperatorDistributedAttribute()
            op_dist_attr.process_mesh = process_mesh
            op_dist_attr.set_output_dims_mapping(grad_var.name, dims_mapping)
            dist_context.set_op_dist_attr_for_program(ops[idx], op_dist_attr)
            continue

        # complete the annotation of grad op (xxx_grad op or sum op)
        # xxx_grad op will have a corresponding forward op in gradopidx2opidx
        grad_op = ops[idx]
        if grad_op.desc.id() in dist_op_context.gradopidx2opidx:
            # TODO support the case where one forward op corresponding to multiple xxx_grad op
            forward_op = _get_op_by_id(
                ops[:first_backward_op_idx],
                dist_op_context.gradopidx2opidx[grad_op.desc.id()])
            assert forward_op is not None

            # op dist attr
            forward_op_dist_attr = dist_context.get_op_dist_attr_for_program(
                forward_op)
            forward_op_process_mesh = forward_op_dist_attr.process_mesh
            grad_op_dist_attr = OperatorDistributedAttribute()
            grad_op_dist_attr.process_mesh = forward_op_process_mesh

            # var 
            for input_name in grad_op.input_arg_names:
                input_var = vars[input_name]
                ref_dims_mapping = None
                if "@GRAD" in input_name:
                    forward_name = _get_forward_varname_from_grad_varname(
                        input_name)
                    ref_dims_mapping = forward_op_dist_attr.get_output_dims_mapping(
                        forward_name)
                else:
                    if forward_op_dist_attr.get_input_dims_mapping(input_name):
                        ref_dims_mapping = forward_op_dist_attr.get_input_dims_mapping(
                            input_name)
                    else:
                        ref_dims_mapping = forward_op_dist_attr.get_output_dims_mapping(
                            input_name)

                assert ref_dims_mapping is not None, "[{}] 's dims mapping is NONE".format(
                    input_var.name)
                grad_op_dist_attr.set_input_dims_mapping(input_name,
                                                         ref_dims_mapping)

            for output_name in grad_op.desc.output_names():
                assert len(grad_op.desc.output(output_name)) in [0, 1]
                if _is_grad_var_name(output_name):
                    input_name = _get_forward_varname_from_grad_varname(
                        output_name)
                else:
                    assert grad_op.type in [
                        "cast", "c_identity", "c_allreduce_sum"
                    ]
                    input_name = "X"
                assert input_name in forward_op.desc.input_names(
                ), "var [{}] in op [{}]'s output but could not find [{}] in its forward op".format(
                    output_name, grad_op.type, input_name)
                if len(grad_op.desc.output(output_name)) == 1:
                    # tensor dist attr
                    output_var = vars[grad_op.desc.output(output_name)[0]]
                    forward_name = _get_forward_varname_from_grad_varname(
                        output_var.name)
                    ref_dims_mapping = forward_op_dist_attr.get_input_dims_mapping(
                        forward_name)

                    output_var_dist_attr = TensorDistributedAttribute()
                    output_var_dist_attr.dims_mapping = ref_dims_mapping
                    output_var_dist_attr.process_mesh = forward_op_process_mesh
                    dist_context.set_tensor_dist_attr_for_program(
                        output_var, output_var_dist_attr)

                    grad_op_dist_attr.set_output_dims_mapping(output_var.name,
                                                              ref_dims_mapping)

            dist_context.set_op_dist_attr_for_program(grad_op,
                                                      grad_op_dist_attr)

        # only sum op for merge mutiple version grad has no a corresponding mapping in gradopidx2opidx
        else:
            assert grad_op.type == "sum", "got unexpect op [{}]".format(
                str(grad_op.type))
            assert all(map(_is_grad_var_name, grad_op.input_arg_names))
            assert len(grad_op.output_arg_names) == 1

            ref_forward_var_name = _get_forward_varname_from_grad_varname(
                grad_op.output_arg_names[0])
            forward_var = vars[ref_forward_var_name]
            ref_forward_var_dims_mapping = dist_context.get_tensor_dist_attr_for_program(
                forward_var).dims_mapping
            ref_forward_var_process_mesh = dist_context.get_tensor_dist_attr_for_program(
                forward_var).process_mesh

            # output
            tensor_dist_attr = TensorDistributedAttribute()
            tensor_dist_attr.dims_mapping = ref_forward_var_dims_mapping
            tensor_dist_attr.process_mesh = ref_forward_var_process_mesh
            dist_context.set_tensor_dist_attr_for_program(
                vars[grad_op.output_arg_names[0]], tensor_dist_attr)

            # op
            grad_op_dist_attr = OperatorDistributedAttribute()
            grad_op_dist_attr.process_mesh = ref_forward_var_process_mesh
            for var_name in grad_op.input_arg_names:
                assert _get_forward_varname_from_grad_varname(
                    var_name) == ref_forward_var_name
                grad_op_dist_attr.set_input_dims_mapping(
                    var_name, ref_forward_var_dims_mapping)

            grad_op_dist_attr.set_output_dims_mapping(
                grad_op.output_arg_names[0], ref_forward_var_dims_mapping)
            dist_context.set_op_dist_attr_for_program(grad_op,
                                                      grad_op_dist_attr)


def complete_update_annotation(auto_parallel_main_prog, dist_context):
    """Complete the annotation of vars and ops in the update phase for parallel program."""

    if dist_context is None:
        dist_context = get_default_distributed_context()

    ops = list(auto_parallel_main_prog.global_block().ops)
    vars = auto_parallel_main_prog.global_block().vars
    learning_rate_completed = False

    for idx in range(len(ops)):

        # complete the annotation of the optimizer op.
        # TODO to add attribute for moment var
        op = ops[idx]
        if int(op.attr('op_role')) == int(OpRole.Optimize):

            if "Grad" in op.input_names and "Param" in ops[idx].input_names:
                assert len(op.input(
                    "Param")) == 1, "Only support one-to-one now."
                assert len(op.input(
                    "Grad")) == 1, "Only support one-to-one now."
                param = vars[op.input("Param")[0]]
                grad_var = vars[op.input("Grad")[0]]

                param_dist_attr = dist_context.get_tensor_dist_attr_for_program(
                    param)
                assert param_dist_attr is not None
                ref_process_mesh = dist_context.get_tensor_dist_attr_for_program(
                    param).process_mesh
                assert ref_process_mesh is not None
                ref_dims_mapping = dist_context.get_tensor_dist_attr_for_program(
                    param).dims_mapping
                assert ref_dims_mapping is not None
                op_dist_attr = OperatorDistributedAttribute()
                op_dist_attr.process_mesh = ref_process_mesh
                op_dist_attr.set_input_dims_mapping(grad_var.name,
                                                    ref_dims_mapping)
                op_dist_attr.set_input_dims_mapping(param.name,
                                                    ref_dims_mapping)
                op_dist_attr.set_output_dims_mapping(param.name,
                                                     ref_dims_mapping)
                learning_var = vars[op.input("LearningRate")[0]]
                op_dist_attr.set_input_dims_mapping(learning_var.name, [-1])
                op_dist_attr.set_output_dims_mapping(learning_var.name, [-1])

                if not learning_rate_completed:
                    learning_rate_completed = True
                    var_dist_attr = TensorDistributedAttribute()
                    var_dist_attr.process_mesh = ref_process_mesh
                    var_dist_attr.dims_mapping = [-1]
                    dist_context.set_tensor_dist_attr_for_program(learning_var,
                                                                  var_dist_attr)

                for input_name in op.desc.input_names():

                    if input_name in [
                            'Param', 'Grad', 'LearningRate', "SkipUpdate",
                            "Beta1Tensor", "Beta2Tensor", "EpsilonTensor",
                            "MasterParam"
                    ]:
                        continue

                    assert len(op.desc.input(input_name)) == 1
                    input_var = vars[op.desc.input(input_name)[0]]
                    input_var_attr = TensorDistributedAttribute()

                    if "Beta1Pow" in input_name or "Beta2Pow" in input_name:
                        input_var_attr.dims_mapping = [-1]
                        op_dist_attr.set_input_dims_mapping(input_var.name,
                                                            [-1])
                        op_dist_attr.set_output_dims_mapping(input_var.name,
                                                             [-1])
                    else:
                        assert "Moment" in input_name
                        input_var_attr.dims_mapping = ref_dims_mapping
                        op_dist_attr.set_input_dims_mapping(input_var.name,
                                                            ref_dims_mapping)
                        op_dist_attr.set_output_dims_mapping(input_var.name,
                                                             ref_dims_mapping)

                    input_var_attr.process_mesh = ref_process_mesh
                    dist_context.set_tensor_dist_attr_for_program(
                        input_var, input_var_attr)

                dist_context.set_op_dist_attr_for_program(op, op_dist_attr)
                continue
