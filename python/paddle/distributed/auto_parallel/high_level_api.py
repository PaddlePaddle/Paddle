# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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
import warnings

import paddle
import paddle.distributed as dist
from paddle.base import (
    default_main_program,
)
from paddle.base.framework import (
    in_dygraph_mode,
)
from paddle.distributed.auto_parallel.static.tuner.pir_rule_based_tuner import (
    _PIR_PATTERNS,
    match_all_patterns,
)


class ToDistributedConfig:
    def __init__(self):
        self.input_spec = None
        self.num_hidden_layers = None


def record_program_ops_pre_hook(layer, inputs):
    """
    A pre-hook to mark op numbers before enter layer.forward.
    """
    if not in_dygraph_mode():
        if layer._op_recorder.start < 0:
            layer._op_recorder.start = len(
                default_main_program().global_block().ops
            )
            # print(f"start program is : {default_main_program()}")
            layer._op_recorder.is_valid = True
        else:
            layer._op_recorder.is_valid = False
            warnings.warn(
                f"{layer._full_name} has recorded the op information before. Please check whether you call this layer twice."
            )


def record_program_ops_post_hook(layer, inputs, outputs):
    """
    A post-hook to mark op numbers after enter layer.forward, and record corresponding ops of the layer.
    """
    if not in_dygraph_mode():
        assert (
            layer._op_recorder.start >= 0
            and layer._op_recorder.is_valid is True
        ), f"{layer._full_name} has not recorded the start of the corresponding ops before"
        end = len(default_main_program().global_block().ops)
        # some layers, such as llama_rotary_embedding, will not add new ops to program
        # assert end > layer._op_recorder.start, f"{layer._full_name} has not added new ops to the program"
        ops = []
        if end > layer._op_recorder.start:
            layer._op_recorder.end = end
            ops = (
                default_main_program()
                .global_block()
                .ops[layer._op_recorder.start : layer._op_recorder.end]
            )
        layer._op_recorder.ops = ops
        # print(f"layer: {layer._full_name}, start: {layer._op_recorder.start}, end: {end}, corresponding ops: {ops}")


def get_layer_pp_info(mesh, num_hidden_layers, layer_index):
    if "pp" in mesh.dim_names:
        pp_degree = mesh.get_dim_size("pp")
        layer_per_stage = math.ceil(num_hidden_layers / pp_degree)
        # input_need_reshard = layer_index % layer_per_stage == 0
        # return layer_index // layer_per_stage, input_need_reshard
        return layer_index // layer_per_stage
    else:
        # return None, False
        return None


# mesh, config: input_spec
def to_distributed(model, mesh, config):

    # # # step1: register pre-hooks and post-hooks, thus recording corresponding static ops in following paddle.jit.to_static
    for layer in model.sublayers():
        pre_hook_helper = layer.register_forward_pre_hook(
            record_program_ops_pre_hook
        )
        post_hook_helper = layer.register_forward_post_hook(
            record_program_ops_post_hook
        )
        layer._op_recorder.hooks.append(pre_hook_helper)
        layer._op_recorder.hooks.append(post_hook_helper)

    # # # step2: call @to_static, get program, and corresponding static ops of each layer
    #               (1) with FLAGS_enable_pir_api=False, get program based on var and op, default to False
    #               (2) with FLAGS_enable_pir_api=True, get pir program

    static_func = paddle.jit.to_static(
        model.forward, input_spec=config.input_spec, full_graph=True
    )
    pir_program = static_func.concrete_program.main_program
    print(f"convert to pir program: {pir_program}")
    # record pir_program ops_to_ids
    op_to_id = {}
    for idx, op in enumerate(pir_program.global_block().ops):
        op_to_id[op] = idx

    # # # step3: get the mapping [dynamic-layers : static ops]
    layer_to_ops = {}
    ops_id_to_layer = {}
    for layer in model.sublayers():
        layer_ops = layer._op_recorder.ops
        layer_to_ops[layer._full_name] = layer_ops
        ops_id = []
        for op in layer_ops:
            assert op in op_to_id.keys(), f"{op.name} is not in pir program"
            ops_id.append(op_to_id[op])
        ops_id_to_layer[tuple(ops_id)] = layer

    # # # # step4: pattern recogincation
    results = match_all_patterns(pir_program)
    print(f"match patterns based on pir program is: {results}")

    # # # # step5: mark pir programs ops dist infos
    matched_programs = {}
    for pattern_name, matched_patterns in results.items():
        # process one pattern
        pattern_ops_dist_infos = _PIR_PATTERNS[pattern_name].ops_dist_infos
        assert (
            pattern_ops_dist_infos is not None
        ), f"{pattern_name} does not contain ops_dist_infos, cannot reshard, please check"
        # print(f"{pattern_name} op dist infos are {pattern_ops_dist_infos}")
        print(
            f"matched patterns are {matched_patterns}"
        )  # [dict{pattern_node_id : graph_node_id, ..., ...}, dict, dict]
        processed_patterns = []
        for matched_pattern in matched_patterns:
            # convert pattern_ops_dist_infos to program_ops_dist_infos
            program_ops_dist_infos = {}
            for pattern_ops_id, op_dist_info in pattern_ops_dist_infos.items():
                program_ops_id = []
                for pattern_op_id in pattern_ops_id:
                    assert (
                        pattern_op_id in matched_pattern.keys()
                    ), "pattern not matched"
                    program_op_id = matched_pattern[pattern_op_id]
                    program_ops_id.append(program_op_id)
                program_ops_dist_infos[tuple(program_ops_id)] = op_dist_info
            processed_patterns.append(program_ops_dist_infos)

        matched_programs[pattern_name] = processed_patterns
        print(f"matched program and ops dist infos are {matched_programs}")

    DECODER_LAYER_NAME = 'decoder_layer'
    num_hidden_layers = len(matched_programs[DECODER_LAYER_NAME])
    print(f"num_hidden_layers by pattern matching is {num_hidden_layers}")
    assert (
        num_hidden_layers == config.num_hidden_layers
    ), "num_hidden_layers by pattern matching is not compatible with configuration"

    # # # # step6-0: SHARD PATRAMETERS, get dynamic layer dist infos
    for pattern_name, processed_patterns in matched_programs.items():
        assert (
            len(processed_patterns) == num_hidden_layers
        ), "transformer patterns matched are incomplete"
        for idx, processed_pattern in enumerate(processed_patterns):
            pp_stage_id = get_layer_pp_info(mesh, num_hidden_layers, idx)
            local_mesh = mesh
            if pp_stage_id is None:
                print("pp_stage_id is None")
                local_mesh = mesh
            else:
                print(f"pp_stage_id is {pp_stage_id}")
                local_mesh = mesh.get_mesh_with_dim("pp", pp_stage_id)
            print(f"local_mesh is {local_mesh}")
            for program_ops_id, dist_infos in processed_pattern.items():
                if program_ops_id not in ops_id_to_layer.keys():
                    print(
                        f"program_ops: {program_ops_id} is not corresponding to a dynamic layer"
                    )
                else:
                    dynamic_layer = ops_id_to_layer[program_ops_id]
                    # shard layers
                    # print(f"sharding info is {dist_infos.print_dist_infos()}")
                    mesh_num_dims = len(local_mesh.shape)
                    print(
                        f"local mesh shape is {local_mesh.shape}, num_dims is {mesh_num_dims}"
                    )
                    # breakpoint()
                    sharding_info = dist_infos.get_dist_info(mesh_num_dims)
                    print(
                        f"shard layer {dynamic_layer._full_name}, sharding info is {sharding_info}"
                    )
                    dynamic_layer.weight = dist.shard_tensor(
                        dynamic_layer.weight, local_mesh, sharding_info[0]
                    )
                    if dynamic_layer.bias is not None:
                        dynamic_layer.bias = dist.shard_tensor(
                            dynamic_layer.bias, local_mesh, sharding_info[1]
                        )
    # # # # step6-0: SHARD INPUTS

    # # # # step7: clean layer_op recorder hooks
    for layer in model.sublayers():
        for hook_helper in layer._op_recorder.hooks:
            hook_helper.remove()

    return model
