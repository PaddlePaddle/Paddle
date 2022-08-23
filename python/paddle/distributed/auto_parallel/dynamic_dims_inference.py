# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

import copy
from paddle.fluid import core
from .dist_context import _node_id
from .utils import is_valid_list_index


def compute_compatible_dynamic_dim(dynamic_dims):
    if not dynamic_dims:
        return None
    compatible_dim = 0
    for dim in dynamic_dims:
        if dim == 1:
            compatible_dim = 1
    return compatible_dim


def compute_compatible_dynamic_dims(dynamic_dims_list):
    if not dynamic_dims_list:
        return None
    length = len(dynamic_dims_list[0])
    for dynamic_dims in dynamic_dims_list:
        assert dynamic_dims is not None, \
            "Dims dim must not be None for compatible computation"
        assert len(dynamic_dims) == length, \
            "The length of dynamic_dims in list must be same for compatible computation."
    compatible_result = []
    for dynamic_dims in zip(*dynamic_dims_list):
        compatible_dynamic_dim = compute_compatible_dynamic_dim(
            list(dynamic_dims))
        compatible_result.append(compatible_dynamic_dim)
    if not compatible_result:
        return None
    return compatible_result


def compute_compatible_and_update_dynamic_dim(dynamic_dims_list, index_list):
    assert len(dynamic_dims_list) == len(index_list)
    changed = False
    dynamic_dims = []
    for i in range(len(dynamic_dims_list)):
        assert is_valid_list_index(dynamic_dims_list[i], index_list[i])
        dynamic_dims.append(dynamic_dims_list[i][index_list[i]])
    compatible_dynamic_dim = compute_compatible_dynamic_dim(dynamic_dims)
    if compatible_dynamic_dim is None:
        return False
    for i in range(len(dynamic_dims_list)):
        if compatible_dynamic_dim != dynamic_dims_list[i][index_list[i]]:
            dynamic_dims_list[i][index_list[i]] = compatible_dynamic_dim
            changed = True
    return changed


# This dict is used to store rules for infering dynamic dims
dynamic_dims_rules = {}


def dynamic_dims_fill_constant_batch_size_like(dist_op):
    changed = False
    op_desc = dist_op.serial_op.desc
    assert op_desc.type() == "fill_constant_batch_size_like"
    op_dist_attr = dist_op.dist_attr
    dynamic_dims_list = []

    dynamic_dims = []
    shape = op_desc.attr("shape")
    for i in shape:
        if i == 0:
            dynamic_dims.append(1)
        else:
            dynamic_dims.append(0)
    dynamic_dims_list.append(dynamic_dims)

    output_arg_names = op_desc.output_arg_names()
    for arg_name in output_arg_names:
        dynamic_dims = op_dist_attr.get_output_dynamic_dims(arg_name)
        dynamic_dims_list.append(dynamic_dims)

    compatible_dynamic_dims = compute_compatible_dynamic_dims(dynamic_dims_list)
    if compatible_dynamic_dims is None:
        return False

    for arg_name in output_arg_names:
        if compatible_dynamic_dims != op_dist_attr.get_output_dynamic_dims(
                arg_name):
            op_dist_attr.set_output_dynamic_dims(arg_name,
                                                 compatible_dynamic_dims)
            changed = True

    return changed


dynamic_dims_rules[
    "fill_constant_batch_size_like"] = dynamic_dims_fill_constant_batch_size_like


def dynamic_dims_gather(dist_op):
    changed = False
    op_desc = dist_op.serial_op.desc
    assert op_desc.type() == "gather"
    op_dist_attr = dist_op.dist_attr
    dynamic_dims_list = []

    input_arg_name = op_desc.input("X")[0]
    dynamic_dims = op_dist_attr.get_input_dynamic_dims(input_arg_name)
    dynamic_dims_list.append(dynamic_dims)

    output_arg_names = op_desc.output_arg_names()
    for arg_name in output_arg_names:
        dynamic_dims = op_dist_attr.get_output_dynamic_dims(arg_name)
        dynamic_dims_list.append(dynamic_dims)

    compatible_dynamic_dims = compute_compatible_dynamic_dims(dynamic_dims_list)
    if compatible_dynamic_dims is None:
        return False

    if compatible_dynamic_dims != op_dist_attr.get_input_dynamic_dims(
            input_arg_name):
        op_dist_attr.set_input_dynamic_dims(input_arg_name,
                                            compatible_dynamic_dims)
        changed = True

    for arg_name in output_arg_names:
        if compatible_dynamic_dims != op_dist_attr.get_output_dynamic_dims(
                arg_name):
            op_dist_attr.set_output_dynamic_dims(arg_name,
                                                 compatible_dynamic_dims)
            changed = True

    return changed


dynamic_dims_rules["gather"] = dynamic_dims_gather

# def dynamic_dims_elementwise_like(dist_op):
#     pass


def dynamic_dims_concat(dist_op):
    changed = False
    op_desc = dist_op.serial_op.desc
    assert op_desc.type() == "concat"
    op_dist_attr = dist_op.dist_attr
    dynamic_dims_list = []

    input_arg_names = op_desc.input_arg_names()
    for arg_name in input_arg_names:
        dynamic_dims = op_dist_attr.get_input_dynamic_dims(arg_name)
        dynamic_dims_list.append(dynamic_dims)

    output_arg_names = op_desc.output_arg_names()
    for arg_name in output_arg_names:
        dynamic_dims = op_dist_attr.get_output_dynamic_dims(arg_name)
        dynamic_dims_list.append(dynamic_dims)

    compatible_dynamic_dims = compute_compatible_dynamic_dims(dynamic_dims_list)
    if compatible_dynamic_dims is None:
        return False

    for arg_name in input_arg_names:
        if compatible_dynamic_dims != op_dist_attr.get_input_dynamic_dims(
                arg_name):
            op_dist_attr.set_input_dynamic_dims(arg_name,
                                                compatible_dynamic_dims)
            changed = True

    for arg_name in output_arg_names:
        if compatible_dynamic_dims != op_dist_attr.get_output_dynamic_dims(
                arg_name):
            op_dist_attr.set_output_dynamic_dims(arg_name,
                                                 compatible_dynamic_dims)
            changed = True

    return changed


dynamic_dims_rules["concat"] = dynamic_dims_concat


def dynamic_dims_assign(dist_op):
    changed = False
    op_desc = dist_op.serial_op.desc
    assert op_desc.type() == "assign"
    op_dist_attr = dist_op.dist_attr
    dynamic_dims_list = []

    input_arg_names = op_desc.input_arg_names()
    for arg_name in input_arg_names:
        dynamic_dims = op_dist_attr.get_input_dynamic_dims(arg_name)
        dynamic_dims_list.append(dynamic_dims)

    output_arg_names = op_desc.output_arg_names()
    for arg_name in output_arg_names:
        dynamic_dims = op_dist_attr.get_output_dynamic_dims(arg_name)
        dynamic_dims_list.append(dynamic_dims)

    compatible_dynamic_dims = compute_compatible_dynamic_dims(dynamic_dims_list)
    if compatible_dynamic_dims is None:
        return False

    for arg_name in input_arg_names:
        if compatible_dynamic_dims != op_dist_attr.get_input_dynamic_dims(
                arg_name):
            op_dist_attr.set_input_dynamic_dims(arg_name,
                                                compatible_dynamic_dims)
            changed = True

    for arg_name in output_arg_names:
        if compatible_dynamic_dims != op_dist_attr.get_output_dynamic_dims(
                arg_name):
            op_dist_attr.set_output_dynamic_dims(arg_name,
                                                 compatible_dynamic_dims)
            changed = True

    return changed


dynamic_dims_rules["assign"] = dynamic_dims_assign

# def dynamic_dims_read_from_array(dist_op):
#     pass
# dynamic_dims_rules["read_from_array"] = dynamic_dims_read_from_array

# def dynamic_dims_write_to_array(dist_op):
#     pass
# dynamic_dims_rules["write_from_array"] = dynamic_dims_write_to_array


def dynamic_dims_scale(dist_op):
    changed = False
    op_desc = dist_op.serial_op.desc
    assert op_desc.type() == "scale"
    op_dist_attr = dist_op.dist_attr
    dynamic_dims_list = []

    input_arg_name = op_desc.input("X")[0]
    dynamic_dims = op_dist_attr.get_input_dynamic_dims(input_arg_name)
    dynamic_dims_list.append(dynamic_dims)

    output_arg_name = op_desc.output("Out")[0]
    dynamic_dims = op_dist_attr.get_output_dynamic_dims(output_arg_name)
    dynamic_dims_list.append(dynamic_dims)

    compatible_dynamic_dims = compute_compatible_dynamic_dims(dynamic_dims_list)
    if compatible_dynamic_dims is None:
        return False

    if compatible_dynamic_dims != op_dist_attr.get_input_dynamic_dims(
            input_arg_name):
        op_dist_attr.set_input_dynamic_dims(input_arg_name,
                                            compatible_dynamic_dims)
        changed = True

    if compatible_dynamic_dims != op_dist_attr.get_output_dynamic_dims(
            output_arg_name):
        op_dist_attr.set_output_dynamic_dims(output_arg_name,
                                             compatible_dynamic_dims)
        changed = True

    return changed


dynamic_dims_rules["scale"] = dynamic_dims_scale


def dynamic_dims_unsqueeze2(dist_op):
    changed = False
    op_desc = dist_op.serial_op.desc
    assert op_desc.type() == "unsqueeze2"
    op_dist_attr = dist_op.dist_attr
    dynamic_dims_list = []

    axes = op_desc.attr("axes")
    input_arg_name = op_desc.input("X")[0]
    input_dynamic_dims = op_dist_attr.get_input_dynamic_dims(input_arg_name)
    dynamic_dims_list.append(input_dynamic_dims)

    output_arg_name = op_desc.output("Out")[0]
    output_dynamic_dims = op_dist_attr.get_output_dynamic_dims(output_arg_name)
    for axis in axes:
        # Remove -1 at the axis to make sure the length of the list same as the input
        output_dynamic_dims.pop(axis)
    dynamic_dims_list.append(output_dynamic_dims)

    compatible_dynamic_dims = compute_compatible_dynamic_dims(dynamic_dims_list)
    if compatible_dynamic_dims is None:
        return False

    if compatible_dynamic_dims != op_dist_attr.get_input_dynamic_dims(
            input_arg_name):
        op_dist_attr.set_input_dynamic_dims(input_arg_name,
                                            compatible_dynamic_dims)
        changed = True

    for axis in axes:
        # Add -1 at the axis to make sure the length of the list same as the output
        output_dynamic_dims.insert(axis, 0)
        compatible_dynamic_dims.insert(axis, 0)
    if compatible_dynamic_dims != op_dist_attr.get_output_dynamic_dims(
            output_arg_name):
        op_dist_attr.set_output_dynamic_dims(output_arg_name,
                                             compatible_dynamic_dims)
        changed = True

    return changed


dynamic_dims_rules["unsqueeze2"] = dynamic_dims_unsqueeze2


def dynamic_dims_matmul_v2(dist_op):
    changed = False
    op_desc = dist_op.serial_op.desc
    op_dist_attr = dist_op.dist_attr
    x_name = op_desc.input('X')[0]
    y_name = op_desc.input('Y')[0]
    out_name = op_desc.output('Out')[0]
    trans_x = op_desc.attr('trans_x')
    trans_y = op_desc.attr('trans_y')
    x_dynamic_dims = op_dist_attr.get_input_dynamic_dims(x_name)
    y_dynamic_dims = op_dist_attr.get_input_dynamic_dims(y_name)
    out_dynamic_dims = op_dist_attr.get_output_dynamic_dims(out_name)
    x_dynamic_dims_len = len(x_dynamic_dims)
    y_dynamic_dims_len = len(y_dynamic_dims)
    out_dynamic_dims_len = len(out_dynamic_dims)

    # Add dim mapping to Make sure the length dynamic_dims be at least 2
    if x_dynamic_dims_len == 1:
        assert trans_x is False
        x_dynamic_dims.insert(0, 0)
        out_dynamic_dims.insert(out_dynamic_dims_len - 1, 0)
    if y_dynamic_dims_len == 1:
        assert trans_y is False
        y_dynamic_dims.insert(1, 0)
        out_dynamic_dims.insert(out_dynamic_dims_len, 0)

    new_x_dynamic_dims_len = len(x_dynamic_dims)
    new_y_dynamic_dims_len = len(y_dynamic_dims)
    new_out_dynamic_dims_len = len(out_dynamic_dims)
    # Deal with dim > 2 and take care of broadcasting
    if new_out_dynamic_dims_len > 2:
        broadcast_x_dynamic_dims = []
        broadcast_y_dynamic_dims = []
        broadcast_out_dynamic_dims = []

        for i in range(new_out_dynamic_dims_len - new_x_dynamic_dims_len):
            broadcast_x_dynamic_dims.append(out_dynamic_dims[i])
        for i in range(new_x_dynamic_dims_len - 2):
            broadcast_x_dynamic_dims.append(x_dynamic_dims[i])

        for i in range(new_out_dynamic_dims_len - new_y_dynamic_dims_len):
            broadcast_y_dynamic_dims.append(out_dynamic_dims[i])
        for i in range(new_y_dynamic_dims_len - 2):
            broadcast_y_dynamic_dims.append(y_dynamic_dims[i])

        for i in range(new_out_dynamic_dims_len - 2):
            broadcast_out_dynamic_dims.append(out_dynamic_dims[i])

        compatible_dynamic_dims = compute_compatible_dynamic_dims([
            broadcast_x_dynamic_dims, broadcast_y_dynamic_dims,
            broadcast_out_dynamic_dims
        ])
        if compatible_dynamic_dims is None:
            return False

        for i in range(new_x_dynamic_dims_len - 2):
            new_idx = i + (new_out_dynamic_dims_len - new_x_dynamic_dims_len)
            if x_dynamic_dims[i] != compatible_dynamic_dims[new_idx]:
                x_dynamic_dims[i] = compatible_dynamic_dims[new_idx]
                changed = True

        for i in range(new_y_dynamic_dims_len - 2):
            new_idx = i + (new_out_dynamic_dims_len - new_y_dynamic_dims_len)
            if y_dynamic_dims[i] != compatible_dynamic_dims[new_idx]:
                y_dynamic_dims[i] = compatible_dynamic_dims[new_idx]
                changed = True

        for i in range(new_out_dynamic_dims_len - 2):
            if out_dynamic_dims[i] != compatible_dynamic_dims[i]:
                out_dynamic_dims[i] = compatible_dynamic_dims[i]
                changed = True

    # The following which uses negative index can be work
    # when len(out_dynamic_dims) > 2 and len(out_dynamic_dims) <=2

    if trans_x:
        x_dynamic_dims[-1], x_dynamic_dims[-2] = x_dynamic_dims[
            -2], x_dynamic_dims[-1]
    if trans_y:
        y_dynamic_dims[-1], y_dynamic_dims[-2] = y_dynamic_dims[
            -2], y_dynamic_dims[-1]

    dim_changed = compute_compatible_and_update_dynamic_dim(
        [x_dynamic_dims, y_dynamic_dims], [-1, -2])
    if dim_changed:
        changed = True

    dim_changed = compute_compatible_and_update_dynamic_dim(
        [x_dynamic_dims, out_dynamic_dims], [-2, -2])
    if dim_changed:
        changed = True

    dim_changed = compute_compatible_and_update_dynamic_dim(
        [y_dynamic_dims, out_dynamic_dims], [-1, -1])
    if dim_changed:
        changed = True

    if trans_x:
        x_dynamic_dims[-1], x_dynamic_dims[-2] = x_dynamic_dims[
            -2], x_dynamic_dims[-1]
    if trans_y:
        y_dynamic_dims[-1], y_dynamic_dims[-2] = y_dynamic_dims[
            -2], y_dynamic_dims[-1]

    # Remove unnecessary dim mapping to make sure the length of dynamic_dims is same as its tensor
    if x_dynamic_dims_len == 1:
        x_dynamic_dims.pop(0)
        out_dynamic_dims.pop(out_dynamic_dims_len - 1)
    if y_dynamic_dims_len == 1:
        y_dynamic_dims.pop(1)
        out_dynamic_dims.pop(out_dynamic_dims_len)

    assert len(x_dynamic_dims) == x_dynamic_dims_len
    assert len(y_dynamic_dims) == y_dynamic_dims_len
    assert len(out_dynamic_dims) == out_dynamic_dims_len

    return changed


dynamic_dims_rules["matmul_v2"] = dynamic_dims_matmul_v2


def dynamic_dims_elementwise_add(dist_op):
    changed = False
    op_desc = dist_op.serial_op.desc
    assert op_desc.type() == "elementwise_add"
    op_dist_attr = dist_op.dist_attr
    dynamic_dims_list = []

    input_arg_names = op_desc.input_arg_names()
    input_dynamic_dims_dict = {}
    input_dynamic_dims_lens = {}
    input_max_dynamic_dims_len = -1
    for arg_name in input_arg_names:
        dynamic_dims = op_dist_attr.get_input_dynamic_dims(arg_name)
        if input_max_dynamic_dims_len < len(dynamic_dims):
            input_max_dynamic_dims_len = len(dynamic_dims)
        input_dynamic_dims_dict[arg_name] = dynamic_dims
        input_dynamic_dims_lens[arg_name] = len(dynamic_dims)
    for arg_name in input_arg_names:
        if input_dynamic_dims_lens[arg_name] < input_max_dynamic_dims_len:
            new_dynamic_dims = [0 for _ in range(input_max_dynamic_dims_len)]
            for i in range(input_dynamic_dims_lens[arg_name]):
                new_idx = (input_max_dynamic_dims_len -
                           input_dynamic_dims_lens[arg_name]) + i
                new_dynamic_dims[new_idx] = input_dynamic_dims_dict[arg_name][i]
            dynamic_dims_list.append(new_dynamic_dims)
        else:
            dynamic_dims_list.append(input_dynamic_dims_dict[arg_name])

    output_arg_names = op_desc.output_arg_names()
    output_dynamic_dims_dict = {}
    output_dynamic_dims_lens = {}
    output_max_dynamic_dims_len = -1
    for arg_name in output_arg_names:
        dynamic_dims = op_dist_attr.get_output_dynamic_dims(arg_name)
        if output_max_dynamic_dims_len < len(dynamic_dims):
            output_max_dynamic_dims_len = len(dynamic_dims)
        output_dynamic_dims_dict[arg_name] = dynamic_dims
        output_dynamic_dims_lens[arg_name] = len(dynamic_dims)
    for arg_name in output_arg_names:
        if output_dynamic_dims_lens[arg_name] < output_max_dynamic_dims_len:
            new_dynamic_dims = [0 for _ in range(output_max_dynamic_dims_len)]
            for i in range(output_dynamic_dims_lens[arg_name]):
                new_idx = (output_max_dynamic_dims_len -
                           output_dynamic_dims_lens[arg_name]) + i
                new_dynamic_dims[new_idx] = output_dynamic_dims_dict[arg_name][
                    i]
            dynamic_dims_list.append(new_dynamic_dims)
        else:
            dynamic_dims_list.append(output_dynamic_dims_dict[arg_name])

    assert input_max_dynamic_dims_len == output_max_dynamic_dims_len
    max_dynamic_dims_len = input_max_dynamic_dims_len
    compatible_dynamic_dims = compute_compatible_dynamic_dims(dynamic_dims_list)
    if compatible_dynamic_dims is None:
        return False

    for arg_name in input_arg_names:
        if input_dynamic_dims_lens[arg_name] < max_dynamic_dims_len:
            new_dynamic_dims = [
                0 for _ in range(input_dynamic_dims_lens[arg_name])
            ]
            for i in range(input_dynamic_dims_lens[arg_name]):
                new_idx = (max_dynamic_dims_len -
                           input_dynamic_dims_lens[arg_name]) + i
                new_dynamic_dims[i] = compatible_dynamic_dims[new_idx]
            if new_dynamic_dims != input_dynamic_dims_dict[arg_name]:
                op_dist_attr.set_input_dynamic_dims(arg_name, new_dynamic_dims)
                changed = True
        else:
            if compatible_dynamic_dims != input_dynamic_dims_dict[arg_name]:
                op_dist_attr.set_input_dynamic_dims(arg_name,
                                                    compatible_dynamic_dims)
                changed = True

    for arg_name in output_arg_names:
        if output_dynamic_dims_lens[arg_name] < max_dynamic_dims_len:
            new_dynamic_dims = [
                0 for _ in range(output_dynamic_dims_lens[arg_name])
            ]
            for i in range(output_dynamic_dims_lens[arg_name]):
                new_idx = (max_dynamic_dims_len -
                           output_dynamic_dims_lens[arg_name]) + i
                new_dynamic_dims[i] = compatible_dynamic_dims[new_idx]
            if new_dynamic_dims != output_dynamic_dims_dict[arg_name]:
                op_dist_attr.set_output_dynamic_dims(arg_name, new_dynamic_dims)
                changed = True
        else:
            if compatible_dynamic_dims != output_dynamic_dims_dict[arg_name]:
                op_dist_attr.set_output_dynamic_dims(arg_name,
                                                     compatible_dynamic_dims)
                changed = True

    return changed


dynamic_dims_rules["elementwise_add"] = dynamic_dims_elementwise_add


def dynamic_dims_softmax(dist_op):
    changed = False
    op_desc = dist_op.serial_op.desc
    assert op_desc.type() == "softmax"
    op_dist_attr = dist_op.dist_attr
    dynamic_dims_list = []

    input_arg_names = op_desc.input_arg_names()
    for arg_name in input_arg_names:
        dynamic_dims = op_dist_attr.get_input_dynamic_dims(arg_name)
        dynamic_dims_list.append(dynamic_dims)

    output_arg_names = op_desc.output_arg_names()
    for arg_name in output_arg_names:
        dynamic_dims = op_dist_attr.get_output_dynamic_dims(arg_name)
        dynamic_dims_list.append(dynamic_dims)

    compatible_dynamic_dims = compute_compatible_dynamic_dims(dynamic_dims_list)
    if compatible_dynamic_dims is None:
        return False

    for arg_name in input_arg_names:
        if compatible_dynamic_dims != op_dist_attr.get_input_dynamic_dims(
                arg_name):
            op_dist_attr.set_input_dynamic_dims(arg_name,
                                                compatible_dynamic_dims)
            changed = True

    for arg_name in output_arg_names:
        if compatible_dynamic_dims != op_dist_attr.get_output_dynamic_dims(
                arg_name):
            op_dist_attr.set_output_dynamic_dims(arg_name,
                                                 compatible_dynamic_dims)
            changed = True

    return changed


dynamic_dims_rules["dynamic_dim"] = dynamic_dims_softmax


class DynamicDimensionsInference:

    def __init__(self, dist_context):
        self._dist_context = dist_context
        self._node_pairs_between_graphs = []
        all_nodes = self._dist_context.serial_ordered_nodes
        for idx, node in enumerate(all_nodes):
            if node.is_var() and node.var() is not None:
                if node.node.graph_id() != 0:
                    for before_node in reversed(all_nodes[:idx]):
                        if before_node.is_var() and before_node.var() is not None \
                            and before_node.node.graph_id() == node.node.graph_id() - 1 \
                                and before_node.var().name() == node.var().name():
                            self._node_pairs_between_graphs.append(
                                (before_node, node))
                    for after_node in all_nodes[idx + 1:]:
                        if after_node.is_var() and after_node.var() is not None \
                            and after_node.node.graph_id() == node.node.graph_id() - 1 \
                                and after_node.var().name() == node.var().name():
                            self._node_pairs_between_graphs.append(
                                (after_node, node))

    def _update_dynamic_dims_between_graphs(self):
        changed = False
        for parent_node, child_node in self._node_pairs_between_graphs:
            parent_node_dist_attr = self._dist_context.get_dist_attr_for_graph(
                parent_node)
            child_node_dist_attr = self._dist_context.get_dist_attr_for_graph(
                child_node)
            parent_node_dynamic_dims = parent_node_dist_attr.dynamic_dims
            child_node_dynamic_dims = child_node_dist_attr.dynamic_dims
            compatible_dynamic_dims = compute_compatible_dynamic_dims(
                [parent_node_dynamic_dims, child_node_dynamic_dims])
            if (compatible_dynamic_dims is not None) \
                and (compatible_dynamic_dims != parent_node_dynamic_dims):
                parent_node_dist_attr.dynamic_dims = compatible_dynamic_dims
                changed = True
            if (compatible_dynamic_dims is not None) \
                and (compatible_dynamic_dims != child_node_dynamic_dims):
                child_node_dist_attr.dynamic_dims = compatible_dynamic_dims
                changed = True
        return changed

    def _update_tensor_node_dynamic_dims(self, tensor_node, fwd=True):
        changed = False
        if (not tensor_node.is_var()) or (tensor_node.var() is None):
            return False
        tensor_desc = tensor_node.var()
        # Skip reader tensor
        if tensor_desc.type() == core.VarDesc.VarType.READER \
            or tensor_desc.type == core.VarDesc.VarType.LOD_TENSOR_ARRAY \
            or tensor_desc.type == core.VarDesc.VarType.STEP_SCOPES:
            return False
        tensor_dist_attr = self._dist_context.get_tensor_dist_attr_for_graph(
            tensor_node)
        assert tensor_dist_attr is not None
        if tensor_dist_attr.is_annotated("dynamic_dims"):
            return False
        tensor_dynamic_dims = tensor_dist_attr.dynamic_dims
        if fwd:
            # print("tensor fwd 1", tensor_desc.name(), tensor_desc.original_id(), tensor_dist_attr, changed, flush=True)
            dynamic_dims_list = []
            for pred_op_node in tensor_node.inputs:
                if pred_op_node.op() is not None:
                    if pred_op_node.op().type() == "create_py_reader" \
                        or pred_op_node.op().type() == "create_double_buffer_reader" \
                        or pred_op_node.op().type() == "read":
                        continue
                    op_dist_attr = self._dist_context.get_op_dist_attr_for_graph(
                        pred_op_node)
                    op_dynamic_dims = op_dist_attr.get_output_dynamic_dims(
                        tensor_desc.name())
                    dynamic_dims_list.append(op_dynamic_dims)
            dynamic_dims_list.append(tensor_dynamic_dims)
            compatible_dynamic_dims = compute_compatible_dynamic_dims(
                dynamic_dims_list)
            if (compatible_dynamic_dims is not None) and \
                (compatible_dynamic_dims != tensor_dynamic_dims):
                tensor_dist_attr.dynamic_dims = compatible_dynamic_dims
                changed = True
            # print("tensor fwd 2", tensor_desc.name(), tensor_desc.original_id(), tensor_dist_attr, changed, flush=True)
        else:
            # print("tensor bwd 1", tensor_desc.name(), tensor_desc.original_id(), tensor_dist_attr, changed, flush=True)
            dynamic_dims_list = []
            for succ_op_node in tensor_node.outputs:
                if succ_op_node.op() is not None:
                    if succ_op_node.op().type() == "create_py_reader" \
                        or succ_op_node.op().type() == "create_double_buffer_reader" \
                        or succ_op_node.op().type() == "read":
                        continue
                    op_dist_attr = self._dist_context.get_op_dist_attr_for_graph(
                        succ_op_node)
                    op_dynamic_dims = op_dist_attr.get_input_dynamic_dims(
                        tensor_desc.name())
                    dynamic_dims_list.append(op_dynamic_dims)
            dynamic_dims_list.append(tensor_dynamic_dims)
            compatible_dynamic_dims = compute_compatible_dynamic_dims(
                dynamic_dims_list)
            if (compatible_dynamic_dims is not None) and \
                (compatible_dynamic_dims != tensor_dynamic_dims):
                tensor_dist_attr.dynamic_dims = compatible_dynamic_dims
                changed = True
            # print("tensor bwd 2", tensor_desc.name(), tensor_desc.original_id(), tensor_dist_attr, changed, flush=True)
        return changed

    def _update_op_node_dynamic_dims(self, op_node, fwd=True):
        changed = False
        if (not op_node.is_op()) or (op_node.op() is None):
            return False
        # Skip reader op
        op_desc = op_node.op()
        if op_desc.type() == "create_py_reader" \
            or op_desc.type() == "create_double_buffer_reader" \
            or op_desc.type() == "while" \
            or op_desc.type() == "read":
            return False
        dist_op = self._dist_context.get_dist_op_for_graph(op_node)
        op_dist_attr = dist_op.dist_attr
        if fwd:
            # print("op fwd 1", op_desc.type(), op_desc.original_id(), op_dist_attr, changed, flush=True)
            for tensor_node in op_node.inputs:
                if tensor_node.is_var() and tensor_node.var() is not None:
                    if tensor_node.var().type() == core.VarDesc.VarType.READER:
                        continue
                    tensor_desc = tensor_node.var()
                    if op_dist_attr.is_annotated_input_dynamic_dims(
                            tensor_desc.name()):
                        continue
                    tensor_dist_attr = self._dist_context.get_tensor_dist_attr_for_graph(
                        tensor_node)
                    tensor_dynamic_dims = tensor_dist_attr.dynamic_dims
                    op_dynamic_dims = op_dist_attr.get_input_dynamic_dims(
                        tensor_desc.name())
                    compatible_dynamic_dims = compute_compatible_dynamic_dims(
                        [op_dynamic_dims, tensor_dynamic_dims])
                    if (compatible_dynamic_dims is not None) and \
                        (compatible_dynamic_dims != op_dynamic_dims):
                        op_dist_attr.set_input_dynamic_dims(
                            tensor_desc.name(), compatible_dynamic_dims)
                        changed = True
            dynamic_dims_rule = dynamic_dims_rules.get(op_desc.type(), None)
            if dynamic_dims_rule:
                changed = dynamic_dims_rule(dist_op)
            # print("op fwd 1", op_desc.type(), op_desc.original_id(), op_dist_attr, changed, flush=True)
        else:
            # print("op bwd 1", op_desc.type(), op_desc.original_id(), op_dist_attr, changed, flush=True)
            for tensor_node in op_node.outputs:
                if tensor_node.is_var() and tensor_node.var() is not None:
                    if tensor_node.var().type() == core.VarDesc.VarType.READER:
                        continue
                    tensor_desc = tensor_node.var()
                    if op_dist_attr.is_annotated_output_dynamic_dims(
                            tensor_desc.name()):
                        continue
                    tensor_dist_attr = self._dist_context.get_tensor_dist_attr_for_graph(
                        tensor_node)
                    tensor_dynamic_dims = tensor_dist_attr.dynamic_dims
                    op_dynamic_dims = op_dist_attr.get_output_dynamic_dims(
                        tensor_desc.name())
                    compatible_dynamic_dims = compute_compatible_dynamic_dims(
                        [op_dynamic_dims, tensor_dynamic_dims])
                    if (compatible_dynamic_dims is not None) and \
                        (compatible_dynamic_dims != op_dynamic_dims):
                        op_dist_attr.set_output_dynamic_dims(
                            tensor_desc.name(), compatible_dynamic_dims)
                        changed = True
            dynamic_dims_rule = dynamic_dims_rules.get(op_desc.type(), None)
            if dynamic_dims_rule:
                changed = dynamic_dims_rule(dist_op)
            # print("op bwd 2", op_desc.type(), op_desc.original_id(), op_dist_attr, changed, flush=True)
        return changed

    def infer_dynamic_dims(self):
        reach_fix_point = False
        while not reach_fix_point:
            changed = False
            for is_fwd in [True, False]:
                all_nodes = self._dist_context.serial_ordered_nodes \
                    if is_fwd else reversed(self._dist_context.serial_ordered_nodes)
                for node in all_nodes:
                    if node.is_var() and node.var() is not None:
                        tensor_changed = self._update_tensor_node_dynamic_dims(
                            node, fwd=is_fwd)
                        if tensor_changed:
                            changed = True
                    if node.is_op() and node.op() is not None:
                        op_changed = self._update_op_node_dynamic_dims(
                            node, fwd=is_fwd)
                        if op_changed:
                            changed = True
                graph_changed = self._update_dynamic_dims_between_graphs()
                if graph_changed:
                    changed = True
            if changed:
                reach_fix_point = False
            else:
                reach_fix_point = True
