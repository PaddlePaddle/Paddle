# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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
import math

import paddle
from paddle import pir
from paddle.autograd.backward_utils import ValueDict, ValueSet
from paddle.base import core

_PADDLE_DTYPE_2_NBYTES = {
    core.DataType.BOOL: 1,
    core.DataType.FLOAT16: 2,
    core.DataType.BFLOAT16: 2,
    core.DataType.FLOAT32: 4,
    core.DataType.FLOAT64: 8,
    core.DataType.INT8: 1,
    core.DataType.INT16: 2,
    core.DataType.INT32: 4,
    core.DataType.INT64: 8,
    core.DataType.UINT8: 1,
    core.DataType.COMPLEX64: 8,
    core.DataType.COMPLEX128: 16,
}


def analyze_mid_hold_values(program, saved_values, outputs, fwd_op_end_idx):
    forward_ops = set(program.global_block().ops[0 : fwd_op_end_idx + 1])
    backward_ops = set(program.global_block().ops[fwd_op_end_idx + 1 :])
    mid_hold_values = ValueSet()
    for op in forward_ops:
        for result in op.results():
            all_used_ops = result.all_used_ops()
            if (
                any(op in backward_ops for op in all_used_ops)
                and result not in saved_values
                and result not in outputs
            ):
                mid_hold_values.add(result)
    return mid_hold_values


def clone_graph(program, origin_ops, graph_inputs, clone_insertion_op):
    pir.set_insertion_point(clone_insertion_op)
    all_ops = program.global_block().ops
    value_map = paddle.pir.IrMapping()
    origin_ops = set(origin_ops)
    cloned_ops = []
    for input_value in graph_inputs:
        value_map.add(input_value, input_value)
    for op in all_ops:
        if op in origin_ops:
            cloned_ops.append(
                op.clone(value_map, paddle.pir.CloneOptions(False, True))
            )
    pir.set_insertion_point_to_block_end(program.global_block())
    return cloned_ops, value_map


def find_parent_ops(value):
    parent_ops = set()
    parent_op = value.get_defining_op()
    parent_ops.add(parent_op)
    op_inputs = parent_op.operands_source()
    for op_input in op_inputs:
        parent_ops = parent_ops | find_parent_ops(op_input)
    return parent_ops


def find_child_ops(value):
    child_ops = set()
    used_ops = value.all_used_ops()
    child_ops |= set(used_ops)
    op_results = ValueSet()
    for used_op in used_ops:
        op_results = op_results | ValueSet(used_op.results())
    for op_result in op_results:
        child_ops = child_ops | find_child_ops(op_result)
    return child_ops


def replace_mid_values_with_forward_subgraph(
    program, saved_values, mid_values, fwd_op_end_idx
):
    def _extract_forward_recompute_subgraph_for_backward(
        saved_values, mid_values
    ):
        def _find_recompute_ops(
            recompute_value,
            saved_values,
            marked_recompute_ops,
            needed_saved_values,
        ):
            define_op = recompute_value.get_defining_op()
            if define_op in marked_recompute_ops:
                return
            op_inputs = define_op.operands_source()
            if len(op_inputs) == 0 and define_op.name() not in [
                "pd_op.full",
                "pd_op.full_int_array",
            ]:
                raise Exception(
                    "Every path to recompute value {} must have saved value or starting point of the path is one of op in [pd_op.full, pd_op.full_int_array], but find {} op".format(
                        recompute_value, define_op.name()
                    )
                )
            for op_input in op_inputs:
                if op_input in saved_values:
                    if op_input not in needed_saved_values:
                        needed_saved_values.add(op_input)
                    continue
                _find_recompute_ops(
                    op_input,
                    saved_values,
                    marked_recompute_ops,
                    needed_saved_values,
                )
            marked_recompute_ops.add(define_op)
            return

        # {inputs:[...], ops: [...], needed_outputs: [...]}
        recompute_subgraph_ops = set()
        recompute_subgraph_inputs = ValueSet()
        recompute_subgraph_outputs_backward_needed = mid_values
        for recompute_value in mid_values:
            _find_recompute_ops(
                recompute_value,
                saved_values,
                recompute_subgraph_ops,
                recompute_subgraph_inputs,
            )
        recompute_subgraph = {
            "inputs": recompute_subgraph_inputs,
            "recompute_ops": recompute_subgraph_ops,
            "outputs": recompute_subgraph_outputs_backward_needed,
        }
        return recompute_subgraph

    forward_ops = set(program.global_block().ops[0 : fwd_op_end_idx + 1])
    backward_ops = set(program.global_block().ops[fwd_op_end_idx + 1 :])
    first_backward_op = program.global_block().ops[fwd_op_end_idx + 1]

    # 1. find forward subgraph to recompute mid values that backward need to hold.
    recompute_forward_subgraph = (
        _extract_forward_recompute_subgraph_for_backward(
            saved_values, mid_values
        )
    )

    # 2. clone subgraph which need to be recomputed
    origin_ops = recompute_forward_subgraph["recompute_ops"]
    origin_subgraph_inputs = recompute_forward_subgraph["inputs"]
    origin_subgraph_outputs = recompute_forward_subgraph["outputs"]
    cloned_ops, value_map = clone_graph(
        program, origin_ops, origin_subgraph_inputs, first_backward_op
    )

    # 3. replace mid values that backward need to hold with recompute subgraph's outputs
    cloned_subgraph_outputs = ValueSet()
    for origin_value in origin_subgraph_outputs:
        cloned_value = value_map.look_up(origin_value)
        origin_value.replace_grad_users_with(cloned_value, backward_ops)
        cloned_subgraph_outputs.add(cloned_value)

    # 4. reset recomputed ops location in program
    reseted_ops = set()
    backward_ops_list = program.global_block().ops[fwd_op_end_idx + 1 :]
    for op in backward_ops_list:
        op_inputs = op.operands_source()
        for op_input in op_inputs:
            if op_input in cloned_subgraph_outputs:
                parent_ops = find_parent_ops(op_input)
                for cloned_op in cloned_ops:
                    if cloned_op in parent_ops and cloned_op not in reseted_ops:
                        cloned_op.move_before(op)
                        reseted_ops.add(cloned_op)
    return program, fwd_op_end_idx


def recompute(program, saved_values, outputs, fwd_op_end_idx):
    """
    recompute intermediate values to save memory.
    Args:
        program(Program): The program to be recomputed.
        saved_values(list[valueiable]): A list of saved valueiables.
        outputs(list[valueiable]): A list of forward outputs.
    Returns:
        recomputed_program(Program): The recomputed program.
    """
    saved_values = ValueSet(saved_values)
    outputs = ValueSet(outputs)

    # 1. Analyze the program, get all forward porgram mid hold values
    mid_hold_values = analyze_mid_hold_values(
        program, saved_values, outputs, fwd_op_end_idx
    )

    # 2. Extract the recompute subgraph and replace forward mid hold values with recompute subgraph's outputs
    program, fwd_op_end_idx = replace_mid_values_with_forward_subgraph(
        program, saved_values, mid_hold_values, fwd_op_end_idx
    )

    return program, fwd_op_end_idx


def min_cut_auto_recompute(
    program,
    inputs,
    outputs,
    grad_outputs,
    fwd_op_end_idx,
    recomputable_ops=None,
):
    # model value as graph's node, op as graph's edge
    def _classify_value_node(program, grad_outputs, fwd_op_end_idx):
        all_ops = program.global_block().ops
        required_fw_value_nodes = ValueSet()
        required_fw_ops = set(all_ops[0 : fwd_op_end_idx + 1])
        for required_fw_op in required_fw_ops:
            fw_op_outputs = required_fw_op.results()
            required_fw_value_nodes = required_fw_value_nodes | ValueSet(
                fw_op_outputs
            )
        required_bw_value_nodes = ValueSet()
        required_bw_ops = set()
        for grad_output in grad_outputs:
            required_bw_ops = (
                required_bw_ops
                | find_child_ops(grad_output)
                | find_parent_ops(grad_output)
            )
        for required_bw_op in required_bw_ops:
            bw_op_outputs = required_bw_op.results()
            required_bw_value_nodes = required_bw_value_nodes | ValueSet(
                bw_op_outputs
            )
        unclaimed_value_nodes = ValueSet()
        unclaimed_ops = {
            op
            for op in all_ops
            if op not in required_fw_ops and op not in required_bw_ops
        }
        for unclaimed_op in unclaimed_ops:
            unclaimed_op_outputs = unclaimed_op.results()
            unclaimed_value_nodes = unclaimed_value_nodes | ValueSet(
                unclaimed_op_outputs
            )
        return (
            required_fw_value_nodes,
            required_bw_value_nodes,
            unclaimed_value_nodes,
        )

    try:
        import networkx as nx
    except ImportError as e:
        raise RuntimeError(
            "Need networkx installed to perform smart recomputation "
            "heuristics"
        ) from e

    (
        required_fw_value_nodes,
        required_bw_value_nodes,
        unclaimed_value_nodes,
    ) = _classify_value_node(program, grad_outputs, fwd_op_end_idx)

    if len(required_bw_value_nodes) == 0:
        return program, fwd_op_end_idx

    def _find_value_node_users(value_node):
        '''
        Find all the value nodes which use the same value node to be computed.
        '''
        users = ValueSet()
        for op in value_node.all_used_ops():
            if op.name() == "builtin.combine":
                combine_result = op.results()[0]
                for combine_res_used_op in combine_result.all_used_ops():
                    results = combine_res_used_op.results()
                    for result in results:
                        if (
                            len(result.all_used_ops()) == 1
                            and result.all_used_ops()[0] == "builtin.split"
                        ):
                            split_results = result.all_used_ops()[0].results()
                            users |= ValueSet(split_results)
                        else:
                            users.add(result)
            else:
                results = op.results()
                for result in results:
                    if (
                        len(result.all_used_ops()) == 1
                        and result.all_used_ops()[0] == "builtin.split"
                    ):
                        split_results = result.all_used_ops()[0].results()
                        users |= ValueSet(split_results)
                    else:
                        users.add(result)
        return users

    all_ops = program.global_block().ops

    dist_from_bw = ValueDict()
    # caculate value node the shortest dist to backward graph
    for op in reversed(all_ops):
        if op.name() == "builtin.combine":
            continue
        op_results = op.results()
        for op_result in op_results:
            used_ops = op_result.all_used_ops()
            if len(used_ops) == 1 and used_ops[0].name() == "builtin.split":
                continue
            real_users = _find_value_node_users(op_result)
            if op_result not in required_fw_value_nodes:
                dist_from_bw[op_result] = 0
            else:
                dist_from_bw[op_result] = int(1e9)
                for user in real_users:
                    dist_from_bw[op_result] = min(
                        dist_from_bw[op_result], dist_from_bw[user] + 1
                    )

    default_recomputable_ops = [
        "pd_op.full_int_array",
        "pd_op.full",
        "pd_op.sum",
        "pd_op.divide",
        "pd_op.subtract",
        "pd_op.add",
        "pd_op.multiply",
        "pd_op.elementwise_pow",
        "pd_op.reshape",
        "pd_op.full_like",
        "pd_op.assign",
        "pd_op.expand",
        "pd_op.scale",
        "pd_op.exp",
        "pd_op.equal",
        "pd_op.where",
    ]
    view_ops = []

    default_recomputable_ops += view_ops

    recomputable_ops = (
        set(recomputable_ops)
        if recomputable_ops is not None
        else set(default_recomputable_ops)
    )

    random_ops = ["pd_op.randint", "pd_op.uniform", "pd_op.dropout"]
    compute_intensive_ops = [
        "pd_op.matmul",
        "pd_op.conv2d",
        "pd_op.layer_norm",
        "pd_op.batchnorm",
        "pd_op.softmax",
        "pd_op.add_n",
    ]

    unrecomputable_ops = random_ops + compute_intensive_ops

    fusible_ops = recomputable_ops | set(random_ops)

    AGGRESSIVE_RECOMPUTATION = False
    # Restricts the amount of computation recompute can do.
    MAX_DIST_FROM_BW = 3

    def _is_fusible(value_node1, value_node2):
        return (
            value_node1.get_defining_op().name() in fusible_ops
            and value_node2.get_defining_op().name() in fusible_ops
        )

    def _get_real_input_nodes(output_value_node):
        real_input_nodes = ValueSet()
        define_op = output_value_node.get_defining_op()
        if define_op.name() == "builtin.split":
            op_input = define_op.operands_source()[0]
            real_define_op = op_input.get_defining_op()
            input_value_nodes = real_define_op.operands_source()
        else:
            input_value_nodes = define_op.operands_source()
        for input_value_node in input_value_nodes:
            if input_value_node.get_defining_op().name() == "builtin.combine":
                real_input_nodes |= ValueSet(
                    input_value_node.get_defining_op().operands_source()
                )
            else:
                real_input_nodes.add(input_value_node)
        return real_input_nodes

    def _get_real_define_op_name(value_node):
        define_op = value_node.get_defining_op()
        if define_op.name() == "builtin.split":
            op_input = define_op.operands_source()[0]
            return op_input.get_defining_op().name()
        else:
            return define_op.name()

    def _is_dynamic_value_node(value_node):
        return -1 in value_node.shape

    def _cal_value_node_size(value_node):
        # todo(wanghao107) hack for dynamic shape
        if _is_dynamic_value_node(value_node):
            return 1
        return value_node.numel() * _PADDLE_DTYPE_2_NBYTES[value_node.dtype]

    def _is_materialized_backwards(value_node):
        cur_value_nodes = ValueSet()
        cur_value_nodes.add(value_node)
        while len(cur_value_nodes) > 0:
            cur_value_node = cur_value_nodes.pop()
            users = _find_value_node_users(cur_value_node)
            for user in users:
                if user not in required_fw_value_nodes and not _is_fusible(
                    cur_value_node, user
                ):
                    return True
                if (
                    user not in required_fw_value_nodes
                    and _get_real_define_op_name(user) in view_ops
                ):
                    cur_value_nodes.add(user)
        return False

    def _is_materialized(value_node, placeholder_value_nodes):
        if value_node in placeholder_value_nodes:
            return True
        users = _find_value_node_users(value_node)
        return not all(_is_fusible(value_node, user) for user in users)

    def _get_node_weight(value_node, placeholder_value_nodes):
        mem_sz = _cal_value_node_size(value_node)

        # Heuristic to bias towards nodes closer to the backwards pass
        mem_sz = int(
            mem_sz * (1.1 ** max(min(dist_from_bw[value_node], 100), 1))
        )
        if _is_materialized(value_node, placeholder_value_nodes):
            return mem_sz
        else:
            return mem_sz * 2

    def _ban_recomputation(value_node):
        if AGGRESSIVE_RECOMPUTATION:
            return value_node.get_defining_op().name() in unrecomputable_ops
        else:
            if value_node.get_defining_op().name() not in recomputable_ops:
                return True

            # If a node *must* be materialized in the backwards pass, then we
            # should never recompute it. This is a pretty subtle point.  In
            # general, the assumption we make is that recomputing a node in the
            # backwards pass is "free". However, if a node must be materialized
            # in the backwards pass, then recomputing it is never free.
            if _is_materialized_backwards(value_node):
                return True

            if dist_from_bw[value_node] > MAX_DIST_FROM_BW:
                return True
            # If the output of an op is 4x smaller (arbitrary choice),
            # then we don't allow recomputation.
            output_size = _cal_value_node_size(value_node)
            inputs = _get_real_input_nodes(value_node)
            inputs_size = sum(_cal_value_node_size(i) for i in inputs)
            return output_size * 4 < inputs_size

    outputs = ValueSet(outputs)
    inputs = ValueSet(inputs)
    value_id_dict = {}
    nx_graph = nx.DiGraph()
    for value_node in (
        required_fw_value_nodes
        | required_bw_value_nodes
        | unclaimed_value_nodes
    ):
        if value_node in outputs or not value_node.initialized():
            continue

        if value_node.get_defining_op().name() == "builtin.combine":
            continue

        if (
            len(value_node.all_used_ops()) == 1
            and value_node.all_used_ops()[0] == "builtin.split"
        ):
            continue

        if value_node in required_bw_value_nodes:
            nx_graph.add_edge(value_node.id + "_in", "sink", capacity=math.inf)
            value_id_dict[value_node.id] = value_node
            continue

        if value_node in inputs:
            nx_graph.add_edge(
                "source", value_node.id + "_in", capacity=math.inf
            )
            value_id_dict[value_node.id] = value_node

        # If a node can't be recomputed (too expensive or involves randomness),
        # we prevent it from being recomputed by adding an inf edge to the source
        # We only need to ban nodes in the fw pass, as those are the only ones that would be recomputed.
        if (
            _ban_recomputation(value_node)
            and value_node in required_fw_value_nodes
        ):
            nx_graph.add_edge(
                "source", value_node.id + "_in", capacity=math.inf
            )
            value_id_dict[value_node.id] = value_node

        # todo(wanghao107) hack for dynamic shape
        if _is_dynamic_value_node(value_node):
            weight = 1
        else:
            weight = _get_node_weight(
                value_node, placeholder_value_nodes=inputs | outputs
            )

        # Creates the weights on the "node" edge
        nx_graph.add_edge(
            value_node.id + "_in", value_node.id + "_out", capacity=weight
        )
        value_id_dict[value_node.id] = value_node

        users = _find_value_node_users(value_node)
        for user in users:
            nx_graph.add_edge(
                value_node.id + "_out", user.id + "_in", capacity=math.inf
            )
    _, partition = nx.minimum_cut(nx_graph, "source", "sink")
    reachable, non_reachable = partition
    cutset = set()
    for u, nbrs in ((n, nx_graph[n]) for n in reachable):
        cutset.update((u, v) for v in nbrs if v in non_reachable)

    cut_value_nodes = ValueSet()
    for value_node_in, value_node_out in cutset:
        assert value_node_in[:-3] == value_node_out[:-4]
        value_node = value_id_dict[value_node_in[:-3]]
        cut_value_nodes.add(value_node)

    saved_values = cut_value_nodes
    program_after_recompute, fwd_op_end_idx_after_recompute = recompute(
        program, saved_values, outputs, fwd_op_end_idx
    )
    return program_after_recompute, fwd_op_end_idx_after_recompute
