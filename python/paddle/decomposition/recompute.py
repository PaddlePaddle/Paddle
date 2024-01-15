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
import paddle
from paddle import pir
from paddle.autograd.backward_utils import (
    ValueSet,
)


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


def analyze_forward_inputs(program, fwd_op_end_idx):
    forward_inputs = ValueSet()
    for i in range(fwd_op_end_idx + 1):
        op = program.global_block().ops[i]
        if op.num_operands() == 0:
            outputs = op.results()
            forward_inputs += ValueSet(outputs)
    return forward_inputs


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
            # if len(op_inputs) == 0 and define_op.name() not in ["pd_op.full", "pd_op.full_int_array"]:
            # raise Exception("Every path to recompute value {} must have saved value or  starting point of the path is one of op in [pd_op.full, pd_op.full_int_array], but find {} op".format(recompute_value, define_op.name()))
            if len(op_inputs) == 0:
                raise Exception(
                    "Every path to recompute value {} must have saved value ".format(
                        recompute_value
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
    # TODO(wanghao107)优化子图替换功能，优化显存，多子图有序插入
    origin_ops = recompute_forward_subgraph["recompute_ops"]
    origin_subgraph_inputs = recompute_forward_subgraph["inputs"]
    origin_subgraph_outputs = recompute_forward_subgraph["outputs"]
    cloned_ops, value_map = clone_graph(
        program, origin_ops, origin_subgraph_inputs, first_backward_op
    )

    # 3. replace mid values that backward need to hold with recompute subgraph's outputs
    for origin_value in origin_subgraph_outputs:
        origin_value.replace_grad_users_with(
            value_map.look_up(origin_value), backward_ops
        )

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
