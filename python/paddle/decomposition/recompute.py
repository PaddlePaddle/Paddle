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
from __future__ import annotations

import functools
import math
import os
from collections import deque
from typing import TYPE_CHECKING

import paddle
from paddle import pir
from paddle.autograd import backward_utils
from paddle.base import core

if TYPE_CHECKING:
    from collections.abc import Sequence

_PADDLE_DTYPE_2_NBYTES = {
    core.DataType.BOOL: 1,
    core.DataType.FLOAT16: 2,
    core.DataType.BFLOAT16: 2,
    core.DataType.FLOAT32: 4,
    core.DataType.FLOAT64: 8,
    core.DataType.FLOAT8_E4M3FN: 1,
    core.DataType.FLOAT8_E5M2: 1,
    core.DataType.INT8: 1,
    core.DataType.INT16: 2,
    core.DataType.INT32: 4,
    core.DataType.INT64: 8,
    core.DataType.UINT8: 1,
    core.DataType.COMPLEX64: 8,
    core.DataType.COMPLEX128: 16,
}

# define the default recompute ops that can be fused between pairs
DEFAULT_RECOMPUTABLE_OPS: list[str] = [
    "pd_op.full_int_array",
    "pd_op.full",
    "pd_op.sum",
    "pd_op.divide",
    "pd_op.subtract",
    "pd_op.add",
    "pd_op.multiply",
    "pd_op.elementwise_pow",
    "pd_op.rsqrt",
    "pd_op.reshape",
    "pd_op.full_like",
    "pd_op.assign",
    "pd_op.expand",
    "pd_op.scale",
    "pd_op.exp",
    "pd_op.equal",
    "pd_op.where",
    "pd_op.sin",
    "pd_op.cos",
    "pd_op.add_n",
    "pd_op.any",
    "pd_op.bitwise_and",
    "pd_op.cast",
    "pd_op.concat",
    "pd_op.full_with_tensor",
    "pd_op.gather_nd",
    "pd_op.greater_than",
    "pd_op.less_than",
    "pd_op.logical_and",
    "pd_op.logical_not",
    "pd_op.not_equal",
    "pd_op.pow",
    "pd_op.shape",
    "pd_op.slice",
    "pd_op.squeeze",
    "pd_op.unsqueeze",
    "pd_op.transpose",
    "pd_op.where",
    "pd_op.prod",
    "pd_op.log",
    "pd_op.log1p",
    "pd_op.logit",
    "pd_op.max",
    "pd_op.expand_as",
    "pd_op.split",
    "pd_op.arange",
    "pd_op.put_along_axis",
    "pd_op.tanh",
    "pd_op.atan",
    "pd_op.atanh",
    "pd_op.sinh",
    "pd_op.asin",
    "pd_op.asinh",
    "pd_op.cosh",
    "pd_op.acos",
    "pd_op.acosh",
    "pd_op.abs",
    "pd_op.sign",
    "pd_op.expm1",
    "pd_op.erf",
    "pd_op.erfinv",
    "pd_op.ceil",
    "pd_op.floor",
    "pd_op.frac",
    "pd_op.round",
    "pd_op.trunc",
    "pd_op.equal",
    "pd_op.angle",
    "pd_op.as_complex",
    "pd_op.as_real",
    "pd_op.complex",
    "pd_op.real",
    "pd_op.imag",
    "pd_op.conj",
    "pd_op.not_equal",
    "pd_op.greater_equal",
    "pd_op.greater_than",
    "pd_op.less_equal",
    "pd_op.less_than",
    "pd_op.bitwise_and",
    "pd_op.bitwise_not",
    "pd_op.bitwise_or",
    "pd_op.bitwise_xor",
    "pd_op.isinf",
    "pd_op.isnan",
    # "pd_op.gather",
    "pd_op.sigmoid",
]

VIEW_OPS: list[str] = []

RANDOM_OPS: list[str] = ["pd_op.randint", "pd_op.uniform", "pd_op.dropout"]

COMPUTE_INTENSIVE_OPS: list[str] = [
    "pd_op.matmul",
    "pd_op.conv2d",
    "pd_op.layer_norm",
    "pd_op.batchnorm",
    "pd_op.softmax",
    "pd_op.all_reduce_",
    "pd_op.c_broadcast_",
    "pd_op.reduce_",
]

AGGRESSIVE_RECOMPUTATION = False
# Restricts the amount of computation recompute can do.
MAX_DIST_FROM_BW = 3


def DebugPrint(*args):
    flag = os.getenv("FLAGS_print_auto_recompute_debug")
    if flag and str(flag).lower() in ("1", "true"):
        print(*args)


class JudgeFusionLoop:
    def __init__(self, program, unrecomputable_ops):
        self.ops = program.global_block().ops
        self.operand_value_set = set()
        self.result_value_set = set()
        self.unrecomputable_ops = unrecomputable_ops
        self.has_unfusible_on_path_map = self._set_has_unfusible_on_path_map()

    def _set_has_unfusible_on_path_map(self):
        def _get_used_external_value(op):
            defined_values = set()
            used_values = []
            _get_used_external_value_impl(defined_values, used_values, op)
            return used_values

        def _get_used_external_value_impl(defined_values, used_values, op):
            for operand in op.operands_source():
                if operand not in defined_values:
                    used_values.append(operand)
                    defined_values.add(operand)
            for block in op.blocks():
                for value in block.args():
                    defined_values.add(value)
                for _, value in block.kwargs():
                    defined_values.add(value)
            for block in op.blocks():
                for inner_op in block.ops():
                    _get_used_external_value_impl(
                        defined_values, used_values, inner_op
                    )
            for result_value in op.results():
                defined_values.add(result_value)

        def _get_producer_ops(op):
            producers = set()
            for operand in _get_used_external_value(op):
                if operand.get_defining_op() is None:
                    continue
                source_op = operand.get_defining_op()
                if source_op.get_parent_block() == op.get_parent_block():
                    producers.add(source_op)
                    self.operand_value_set.add(operand)
            return producers

        def _get_consumer_ops(op):
            consumers = set()
            for result in op.results():
                for parent_op in result.all_used_ops():
                    while parent_op is not None:
                        if (
                            parent_op.get_parent_block()
                            == op.get_parent_block()
                        ):
                            consumers.add(parent_op)
                            self.result_value_set.add(result)
                            break
                        parent_op = (
                            parent_op.get_parent_block().parent_op
                            if parent_op.get_parent_block() is not None
                            else None
                        )
            return consumers

        def _get_producer_ops_recursivly(root):
            visited = set()
            queue = deque()
            result = set()
            queue.append(root)
            visited.add(root)
            while queue:
                cur = queue.popleft()
                result.add(cur)
                for new_op in _get_producer_ops(cur):
                    if new_op in visited:
                        continue
                    visited.add(new_op)
                    queue.append(new_op)
            return result

        def _get_consumer_ops_recursivly(root):
            visited = set()
            queue = deque()
            result = set()
            queue.append(root)
            visited.add(root)
            while queue:
                cur = queue.popleft()
                result.add(cur)
                for new_op in _get_consumer_ops(cur):
                    if new_op in visited:
                        continue
                    visited.add(new_op)
                    queue.append(new_op)
            return result

        has_unfusible_on_path_map = {
            op1: {op2: False for op2 in self.ops} for op1 in self.ops
        }
        for op in self.ops:
            if op.name() in self.unrecomputable_ops:
                upstream_set = _get_producer_ops_recursivly(op)
                downstream_set = _get_consumer_ops_recursivly(op)

                for upstream_op in upstream_set:
                    for downstream_op in downstream_set:
                        has_unfusible_on_path_map[upstream_op][
                            downstream_op
                        ] = True
                        has_unfusible_on_path_map[downstream_op][
                            upstream_op
                        ] = True
        return has_unfusible_on_path_map

    def _has_unfusible_op_on_any_path(self, op1, op2):
        return (
            self.has_unfusible_on_path_map[op1][op2]
            if op1 is not None and op2 is not None
            else False
        )

    def _get_operand_value_set(self):
        return backward_utils.ValueSet(self.operand_value_set)

    def _get_result_value_set(self):
        return backward_utils.ValueSet(self.result_value_set)


def auto_recompute(
    program: paddle.static.Program,
    inputs: Sequence[pir.Value],
    outputs: Sequence[pir.Value],
    grad_outputs: Sequence[pir.Value],
    fwd_op_end_idx: int,
    backward_op_start_idx: int,
    recomputable_ops: Sequence[str] | None = None,
) -> tuple[paddle.static.Program, int]:
    '''
    Considering the compiler fuse strategy, we model the pir graph.
    Convert the pir calculation graph into a networkx calculation
    graph. Find the cut point through the min-cut algorithm,
    which is the value to be saved in pir forward calculation graph.

    Recompute the forward computation graph to replace intermediate
    variables in the forward graph held by the backward graph.

    .. warning::
        This API is experimental and likely to change.

    Args:
        program (Program): The program to be recomputed.
        inputs:(list[Value]|tuple(Value)): The input Values
            of the forward graph.
        outputs:(list[Value]|tuple(Value)): The out Values
            of the forward graph.
        grad_outputs:(list[Value]|tuple(Value)): initial gradient values
            of `outputs` .
        forward_op_end_idx(int): The index of the last forward op.
        backward_op_start_idx(int): The index of the start backward op.
        recomputable_ops(list[str]|tuple(str)|None): The op names that can
            be recomputed. If 'recompute_ops' is None, we will use the
            default recomputable_ops. Default None.
    Returns:
        recomputed_program(Program): The recomputed program.
        fwd_op_end_idx(int): The index of the last forward op in recomputed program.

    Examples:
        .. code-block:: python

        >>> import numpy as np
        >>> import paddle
        >>> from paddle.autograd.ir_backward import grad as ir_grad
        >>> from paddle.base import core
        >>> from paddle.decomposition import decompose
        >>> def forward(x):
        ...     y = paddle.sin(x)
        ...     z = paddle.cos(y)
        ...     return z

        >>> np_x = np.random.random(size=[4096, 4096]).astype("float32")
        >>> paddle.enable_static()
        >>> core._set_prim_all_enabled(True)
        >>> main_program = paddle.static.Program()
        >>> with paddle.static.program_guard(main_program):
        >>>     x = paddle.static.data(
        >>>         name="x", shape=[4096, 4096], dtype="float32"
        >>>     )
        >>>     x.stop_gradient = False
        >>>     out = forward(x)
        >>>     out_grad = paddle.full(
        >>>         shape=out.shape, fill_value=3, dtype="float32"
        >>>     )
        >>>     [out] = decompose(main_program, [out])
        >>>     [dx] = ir_grad(out, [x], out_grad)
        >>>     main_program, _ = paddle.decomposition.auto_recompute(
        >>>         main_program,
        >>>         [x],
        >>>         [out],
        >>>         grad_outputs=[out_grad],
        >>>         fwd_op_end_idx=2,
        >>>         backward_op_start_idx=4
        >>>     )
        >>>     exe = paddle.static.Executor(paddle.CUDAPlace(0))
        >>>     res = exe.run(
        >>>         feed={'x': np_x},
        >>>         fetch_list=[dx],
        >>>     )
        >>>     print(main_program)
        {
            (%0) = "pd_op.data" () {dtype:(pd_op.DataType)float32,name:"x",place:(pd_op.Place)Place(undefined:0),shape:(pd_op.IntArray)[4096,4096],stop_gradient:[false]} : () -> pd_op.tensor<4096x4096xf32>
            (%1) = "pd_op.sin" (%0) {stop_gradient:[false]} : (pd_op.tensor<4096x4096xf32>) -> pd_op.tensor<4096x4096xf32>
            (%2) = "pd_op.cos" (%1) {stop_gradient:[false]} : (pd_op.tensor<4096x4096xf32>) -> pd_op.tensor<4096x4096xf32>
            (%3) = "pd_op.full" () {dtype:(pd_op.DataType)float32,place:(pd_op.Place)Place(undefined:0),shape:(pd_op.IntArray)[4096,4096],stop_gradient:[true],value:(Float)3} : () -> pd_op.tensor<4096x4096xf32>
            (%4) = "pd_op.sin" (%0) {stop_gradient:[false]} : (pd_op.tensor<4096x4096xf32>) -> pd_op.tensor<4096x4096xf32>
            (%5) = "pd_op.sin" (%4) {stop_gradient:[false]} : (pd_op.tensor<4096x4096xf32>) -> pd_op.tensor<4096x4096xf32>
            (%6) = "pd_op.full" () {dtype:(pd_op.DataType)float32,place:(pd_op.Place)Place(cpu),shape:(pd_op.IntArray)[1],stop_gradient:[true],value:(Float)-1} : () -> pd_op.tensor<1xf32>
            (%7) = "pd_op.scale" (%5, %6) {bias:(Float)0,bias_after_scale:true,stop_gradient:[false]} : (pd_op.tensor<4096x4096xf32>, pd_op.tensor<1xf32>) -> pd_op.tensor<4096x4096xf32>
            (%8) = "pd_op.multiply" (%7, %3) {stop_gradient:[false]} : (pd_op.tensor<4096x4096xf32>, pd_op.tensor<4096x4096xf32>) -> pd_op.tensor<4096x4096xf32>
            (%9) = "pd_op.cos" (%0) {stop_gradient:[false]} : (pd_op.tensor<4096x4096xf32>) -> pd_op.tensor<4096x4096xf32>
            (%10) = "pd_op.multiply" (%9, %8) {stop_gradient:[false]} : (pd_op.tensor<4096x4096xf32>, pd_op.tensor<4096x4096xf32>) -> pd_op.tensor<4096x4096xf32>
            (%11) = "pd_op.fetch" (%10) {col:(Int32)0,is_persistable:[true],name:"fetch0",stop_gradient:[false]} : (pd_op.tensor<4096x4096xf32>) -> pd_op.tensor<4096x4096xf32>
        }
    '''
    # 1. find smart recompute needed saved values by min-cut algorithm
    # 1.1 classify value nodes
    import networkx as nx

    # model value as graph's node, op as graph's edge
    (
        required_fw_value_nodes,
        required_bw_value_nodes,
        unclaimed_value_nodes,
    ) = classify_value_node(program, grad_outputs, fwd_op_end_idx)

    if len(required_bw_value_nodes) == 0:
        return program, fwd_op_end_idx

    all_ops = program.global_block().ops
    # 1.2 cal value nodes dist to backward
    dist_from_bw = cal_value_nodes_dist_to_backward(
        all_ops, required_fw_value_nodes
    )

    # 1.3 classify ops
    default_recomputable_ops = DEFAULT_RECOMPUTABLE_OPS
    view_ops = VIEW_OPS

    default_recomputable_ops += view_ops

    recomputable_ops = (
        set(recomputable_ops)
        if recomputable_ops is not None
        else set(default_recomputable_ops)
    )

    random_ops = RANDOM_OPS
    compute_intensive_ops = COMPUTE_INTENSIVE_OPS

    unrecomputable_ops = random_ops + compute_intensive_ops

    fusible_ops = recomputable_ops | set(random_ops)

    def _is_fusible(value_node1, value_node2):
        return (
            value_node1.get_defining_op().name() in fusible_ops
            and value_node2.get_defining_op().name() in fusible_ops
        )

    def _is_materialized_backwards(value_node):
        cur_value_nodes = backward_utils.ValueSet()
        cur_value_nodes.add(value_node)
        while len(cur_value_nodes) > 0:
            cur_value_node = cur_value_nodes.pop()
            users = find_value_node_users(cur_value_node)
            for user in users:
                if user not in required_fw_value_nodes and not _is_fusible(
                    cur_value_node, user
                ):
                    return True
                if (
                    user not in required_fw_value_nodes
                    and get_real_define_op_name(user) in view_ops
                ):
                    cur_value_nodes.add(user)
        return False

    def _is_materialized(value_node, placeholder_value_nodes):
        if value_node in placeholder_value_nodes:
            return True
        users = find_value_node_users(value_node)
        return not all(_is_fusible(value_node, user) for user in users)

    def _get_node_weight(value_node, placeholder_value_nodes):
        mem_sz = cal_value_node_size(value_node)

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
            output_size = cal_value_node_size(value_node)
            inputs = get_real_input_nodes(value_node)
            inputs_size = sum(cal_value_node_size(i) for i in inputs)
            return output_size * 4 < inputs_size

    # 1.4  Model pir graph. Convert the pir calculation graph into a networkx calculation graph.
    outputs = backward_utils.ValueSet(outputs)
    inputs = backward_utils.ValueSet(inputs)
    value_id_dict = {}
    nx_graph = nx.DiGraph()

    judge_fusion_loop = JudgeFusionLoop(program, unrecomputable_ops)
    forward_ops = set(program.global_block().ops[: fwd_op_end_idx + 1])

    for value_node in (
        required_fw_value_nodes
        | required_bw_value_nodes
        | unclaimed_value_nodes
    ):
        if value_node in outputs or not value_node.initialized():
            continue

        if value_node.get_defining_op().name() == "builtin.combine":
            continue

        if len(value_node.all_used_ops()) == 1 and value_node.all_used_ops()[
            0
        ].name() in ["builtin.split", "builtin.slice"]:
            continue

        if value_node in required_bw_value_nodes:
            DebugPrint(
                "add edge link from: ", value_node.id, " -> ", "sink", " (inf) "
            )
            nx_graph.add_edge(value_node.id + "_in", "sink", capacity=math.inf)
            value_id_dict[value_node.id] = value_node
            continue

        if value_node in inputs:
            DebugPrint(
                "add edge link from: ",
                " source ",
                " -> ",
                value_node.id,
                " (inf)",
            )
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
            DebugPrint(
                "add edge link from: ",
                " source ",
                " -> ",
                value_node.id,
                "(inf)",
            )
            nx_graph.add_edge(
                "source", value_node.id + "_in", capacity=math.inf
            )
            value_id_dict[value_node.id] = value_node

        # todo(wanghao107) hack for dynamic shape
        if is_dynamic_value_node(value_node):
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

        users = find_value_node_users(value_node)
        for user in users:
            DebugPrint(
                "add edge link from: ",
                value_node.id,
                " -> ",
                user.id,
                " (inf) ",
            )
            nx_graph.add_edge(
                value_node.id + "_out", user.id + "_in", capacity=math.inf
            )
        for user in value_node.all_used_ops():
            if user in forward_ops:
                if judge_fusion_loop._has_unfusible_op_on_any_path(
                    value_node.get_defining_op(), user
                ):
                    DebugPrint(
                        "add edge link from: ",
                        " source ",
                        " -> ",
                        value_node.id,
                        "(inf)",
                    )
                    nx_graph.add_edge(
                        "source", value_node.id + "_in", capacity=math.inf
                    )

                    DebugPrint(
                        "add edge link from: ",
                        value_node.id,
                        " -> ",
                        " sink ",
                        "(inf)",
                    )

                    nx_graph.add_edge(
                        value_node.id + "_out", "sink", capacity=math.inf
                    )

    # 1.5  find saved values by minimum cut.
    cut_value, partition = nx.minimum_cut(nx_graph, "source", "sink")
    DebugPrint("Cut Value:", cut_value)
    reachable, non_reachable = partition
    cutset = set()
    for u, nbrs in ((n, nx_graph[n]) for n in reachable):
        cutset.update((u, v) for v in nbrs if v in non_reachable)

    cut_value_nodes = backward_utils.ValueSet()
    for value_node_in, value_node_out in cutset:
        assert value_node_in[:-3] == value_node_out[:-4]
        value_node = value_id_dict[value_node_in[:-3]]
        cut_value_nodes.add(value_node)

    saved_values = cut_value_nodes
    # (TODO: wanghao107): remove it and fix model
    # saved_values = cut_value_nodes | inputs
    saved_values = cut_value_nodes
    DebugPrint("program before recompute:", program)
    # 2.patition the joint graph by saved values.
    (
        program_after_recompute,
        fwd_op_end_idx_after_recompute,
    ) = partition_joint_graph(
        program,
        saved_values,
        inputs,
        outputs,
        fwd_op_end_idx,
        backward_op_start_idx,
    )
    DebugPrint("program after recompute:", program_after_recompute)
    return program_after_recompute, fwd_op_end_idx_after_recompute


def partition_joint_graph(
    program: paddle.static.Program,
    saved_values: list[pir.Value],
    inputs: list[pir.Value],
    outputs: list[pir.Value],
    fwd_op_end_idx: int,
    backward_op_start_idx: int,
) -> tuple[paddle.static.Program, int]:
    """
    Partition the joint graph, recompute the intermediate values
    by saved values to save memory.
    Args:
        program(Program): The program to be recomputed.
        saved_values(list[valueiable]): The saved values
            of forward graph which used by backward graph.
        inputs:(list[Value]|tuple(Value)): The input Values
            of the forward graph.
        outputs(list[valueiable]): The out values
            of the forward graph.
        forward_op_end_idx(int): The index of the last forward op.
        backward_op_start_idx(int): The index of the start backward op.
    Returns:
        recomputed_program(Program): The recomputed program.
        fwd_op_end_idx(int): The index of the last forward op in
            recomputed program.
    """
    saved_values = backward_utils.ValueSet(saved_values)
    outputs = backward_utils.ValueSet(outputs)

    # 1. Analyze the program, get all forward program mid hold values
    mid_hold_values = analyze_mid_hold_values(
        program,
        saved_values,
        inputs,
        outputs,
        fwd_op_end_idx,
        backward_op_start_idx,
    )
    mem = 0
    for mid in mid_hold_values:
        mem += cal_value_node_size(mid)
    DebugPrint("Saved Memory is: ", mem / 1024 / 1024 / 1024, "GB")

    def getIdx(program, op):
        for idx, op_iter in enumerate(program.global_block().ops):
            if op == op_iter:
                return idx
        raise RuntimeError("op not found in program")

    # 2. Extract the recompute subgraph and replace forward mid hold values with recompute subgraph's outputs
    program, fwd_op_end_idx = replace_mid_values_with_forward_subgraph(
        program,
        saved_values,
        mid_hold_values,
        fwd_op_end_idx,
        backward_op_start_idx,
    )

    return program, fwd_op_end_idx


def replace_mid_values_with_forward_subgraph(
    program, saved_values, mid_values, fwd_op_end_idx, backward_op_start_idx
):
    def _extract_forward_recompute_subgraph_for_backward(
        saved_values, mid_values
    ):
        def _find_recompute_ops(
            recompute_value,
            saved_values,
            marked_recompute_ops,
            needed_saved_values,
            chain,
        ):
            def getIdx(program, op):
                for idx, op_iter in enumerate(program.global_block().ops):
                    if op == op_iter:
                        return idx
                raise RuntimeError("op not found in program")

            new_chain = list(chain)
            new_chain.append(recompute_value)
            define_op = recompute_value.get_defining_op()
            if define_op in marked_recompute_ops:
                return
            op_inputs = define_op.operands_source()
            if len(op_inputs) == 0 and define_op.name() not in [
                "pd_op.full",
                "pd_op.full_int_array",
            ]:

                def getIdx(program, op):
                    for idx, op_iter in enumerate(program.global_block().ops):
                        if op == op_iter:
                            return idx
                    raise RuntimeError("op not found in program")

                raise Exception(
                    f"Every path to recompute value {recompute_value} must have saved value or starting point of the path is one of op in [pd_op.full, pd_op.full_int_array], but find {define_op.name()} op"
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
                    new_chain,
                )
            marked_recompute_ops.add(define_op)

            return

        recompute_subgraph_ops = set()
        recompute_subgraph_inputs = backward_utils.ValueSet()
        recompute_subgraph_outputs_backward_needed = mid_values

        def getIdx(program, op):
            for idx, op_iter in enumerate(program.global_block().ops):
                if op == op_iter:
                    return idx
            raise RuntimeError("op not found in program")

        for recompute_value in mid_values:
            _find_recompute_ops(
                recompute_value,
                saved_values,
                recompute_subgraph_ops,
                recompute_subgraph_inputs,
                [],
            )

        DebugPrint("Recompute Ops: ", len(recompute_subgraph_ops))
        DebugPrint("Recompute Ops: ", recompute_subgraph_ops)
        recompute_subgraph = {
            "inputs": recompute_subgraph_inputs,
            "recompute_ops": recompute_subgraph_ops,
            "outputs": recompute_subgraph_outputs_backward_needed,
        }
        return recompute_subgraph

    forward_ops = set(program.global_block().ops[: fwd_op_end_idx + 1])
    backward_ops = set(program.global_block().ops[backward_op_start_idx:])
    first_backward_op = program.global_block().ops[backward_op_start_idx]

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

    for origin_op in origin_ops:
        origin_op.set_bool_attr("is_recompute_op", True)
    for cloned_op in cloned_ops:
        cloned_op.set_bool_attr("is_recompute_bw_op", True)

    # 3. replace mid values that backward need to hold with recompute subgraph's outputs
    cloned_subgraph_outputs = backward_utils.ValueSet()
    for origin_value in origin_subgraph_outputs:
        cloned_value = value_map.look_up(origin_value)
        origin_value.replace_grad_users_with(cloned_value, backward_ops)
        cloned_subgraph_outputs.add(cloned_value)

    # 4. reset recomputed ops location in program
    reseted_ops = set()
    backward_ops_list = program.global_block().ops[backward_op_start_idx:]
    for op in backward_ops_list:
        op_inputs = op.operands_source()
        for op_input in op_inputs:
            if op_input in cloned_subgraph_outputs:
                parent_ops = find_parent_ops(op_input)
                for cloned_op in cloned_ops:
                    if cloned_op in parent_ops and cloned_op not in reseted_ops:
                        cloned_op.move_before(op)
                        reseted_ops.add(cloned_op)
    DebugPrint("recompute program", program)
    return program, fwd_op_end_idx


def classify_value_node(program, grad_outputs, fwd_op_end_idx):
    all_ops = program.global_block().ops
    required_fw_value_nodes = backward_utils.ValueSet()
    required_fw_ops = set(all_ops[: fwd_op_end_idx + 1])
    for required_fw_op in required_fw_ops:
        fw_op_outputs = required_fw_op.results()
        required_fw_value_nodes = (
            required_fw_value_nodes | backward_utils.ValueSet(fw_op_outputs)
        )
    required_bw_value_nodes = backward_utils.ValueSet()
    required_bw_ops = set()
    for grad_output in grad_outputs:
        required_bw_ops = required_bw_ops | find_child_ops(grad_output)
        required_bw_ops.add(grad_output.get_defining_op())
    for required_bw_op in required_bw_ops:
        bw_op_outputs = (
            required_bw_op.results() if required_bw_op is not None else []
        )
        required_bw_value_nodes = (
            required_bw_value_nodes | backward_utils.ValueSet(bw_op_outputs)
        )
    unclaimed_value_nodes = backward_utils.ValueSet()
    unclaimed_ops = {
        op
        for op in all_ops
        if op not in required_fw_ops and op not in required_bw_ops
    }
    for unclaimed_op in unclaimed_ops:
        unclaimed_op_outputs = unclaimed_op.results()
        unclaimed_value_nodes = unclaimed_value_nodes | backward_utils.ValueSet(
            unclaimed_op_outputs
        )
    return (
        required_fw_value_nodes,
        required_bw_value_nodes | unclaimed_value_nodes,
        backward_utils.ValueSet(),
    )


def find_value_node_users(value_node):
    '''
    Find all the value nodes which use the same value node to be computed.
    '''
    users = backward_utils.ValueSet()
    for op in value_node.all_used_ops():
        if op.name() == "builtin.combine":
            combine_result = op.results()[0]
            for combine_res_used_op in combine_result.all_used_ops():
                results = combine_res_used_op.results()
                for result in results:
                    if len(
                        result.all_used_ops()
                    ) == 1 and result.all_used_ops()[0].name() in [
                        "builtin.split",
                        "builtin.slice",
                    ]:
                        split_results = result.all_used_ops()[0].results()
                        users |= backward_utils.ValueSet(split_results)
                    else:
                        users.add(result)
        else:
            results = op.results()
            for result in results:
                if len(result.all_used_ops()) == 1 and result.all_used_ops()[
                    0
                ].name() in ["builtin.split", "builtin.slice"]:
                    split_results = result.all_used_ops()[0].results()
                    users |= backward_utils.ValueSet(split_results)
                else:
                    users.add(result)
    return users


def get_real_input_nodes(output_value_node):
    real_input_nodes = backward_utils.ValueSet()
    define_op = output_value_node.get_defining_op()
    if define_op.name() in ["builtin.split", "builtin.slice"]:
        op_input = define_op.operands_source()[0]
        real_define_op = op_input.get_defining_op()
        input_value_nodes = real_define_op.operands_source()
    else:
        input_value_nodes = define_op.operands_source()
    for input_value_node in input_value_nodes:
        if input_value_node.get_defining_op().name() == "builtin.combine":
            real_input_nodes |= backward_utils.ValueSet(
                input_value_node.get_defining_op().operands_source()
            )
        else:
            real_input_nodes.add(input_value_node)
    return real_input_nodes


def get_real_define_op_name(value_node):
    define_op = value_node.get_defining_op()
    if define_op.name() in ["builtin.split", "builtin.slice"]:
        op_input = define_op.operands_source()[0]
        return op_input.get_defining_op().name()
    else:
        return define_op.name()


def is_dynamic_value_node(value_node):
    try:
        return -1 in value_node.shape
    except:
        raise ValueError(f"value node not found in program: {value_node} ")


def cal_value_node_size(value_node):
    # todo(wanghao107) hack for dynamic shape
    if is_dynamic_value_node(value_node):
        return 1
    return (
        functools.reduce(lambda x, y: x * y, value_node.shape, 1)
        * _PADDLE_DTYPE_2_NBYTES[value_node.dtype]
    )


def cal_value_nodes_dist_to_backward(all_ops, required_fw_value_nodes):
    dist_from_bw = backward_utils.ValueDict()
    # calculate value node the shortest dist to backward graph
    for op in reversed(all_ops):
        if op.name() == "builtin.combine":
            continue
        op_results = op.results()
        for op_result in op_results:
            used_ops = op_result.all_used_ops()
            if len(used_ops) == 1 and used_ops[0].name() in [
                "builtin.split",
                "builtin.slice",
            ]:
                continue
            real_users = find_value_node_users(op_result)
            if op_result not in required_fw_value_nodes:
                dist_from_bw[op_result] = 0
            else:
                dist_from_bw[op_result] = int(1e9)
                for user in real_users:
                    dist_from_bw[op_result] = min(
                        dist_from_bw[op_result], dist_from_bw[user] + 1
                    )
    return dist_from_bw


def all_used_op_consider_combine(program, value):
    def filter_unused_combine(op):
        if (
            op.name() == "builtin.combine"
            and len(op.result(0).all_used_ops()) == 0
        ):
            return False
        return True

    return list(filter(filter_unused_combine, value.all_used_ops()))


def analyze_mid_hold_values(
    program,
    saved_values,
    inputs,
    outputs,
    fwd_op_end_idx,
    backward_op_start_idx,
):
    forward_ops = set(program.global_block().ops[: fwd_op_end_idx + 1])
    backward_ops = set(program.global_block().ops[backward_op_start_idx:])
    mid_hold_values = backward_utils.ValueSet()
    for op in forward_ops:
        for result in op.results():
            all_used_ops = all_used_op_consider_combine(program, result)
            if (
                any(op in backward_ops for op in all_used_ops)
                and result not in saved_values
                and result not in outputs
                and result not in inputs
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
                op.clone(value_map, paddle.pir.CloneOptions(False, True, True))
            )
    pir.set_insertion_point_to_block_end(program.global_block())
    return cloned_ops, value_map


def find_parent_ops(value):
    visited = backward_utils.ValueSet()

    def _find_parent_ops(value):
        parent_ops = set()
        if value in visited:
            return parent_ops
        visited.add(value)
        parent_op = value.get_defining_op()
        if parent_op is not None:
            parent_ops.add(parent_op)
        else:
            return parent_ops
        op_inputs = parent_op.operands_source()
        for op_input in op_inputs:
            if not value.initialized():
                continue
            parent_ops = parent_ops | _find_parent_ops(op_input)
        return parent_ops

    return _find_parent_ops(value)


def find_child_ops(value):
    visited = backward_utils.ValueSet()

    def _find_child_ops(value):
        child_ops = set()
        if value in visited:
            return child_ops
        visited.add(value)
        used_ops = value.all_used_ops()
        child_ops |= set(used_ops)
        op_results = backward_utils.ValueSet()
        for used_op in used_ops:
            op_results = op_results | backward_utils.ValueSet(used_op.results())
        for op_result in op_results:
            if not op_result.initialized():
                continue
            child_ops = child_ops | _find_child_ops(op_result)
        return child_ops

    return _find_child_ops(value)
