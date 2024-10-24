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
    clear_used_patterns,
    get_pattern,
    match_all_patterns,
    register_used_patterns,
)


class ToDistributedConfig:
    def __init__(self):
        self.input_spec = None
        self.sequence_parallel = False


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


def transpose_reshard_embedding_layer_output(layer, inputs, outputs):
    if hasattr(layer, "current_mesh"):
        current_mesh = layer.__getattr__("current_mesh")
        new_output = paddle.transpose(outputs, [1, 0, 2])
        new_output = dist.reshard(
            new_output, current_mesh, [dist.Shard(1), dist.Shard(0)]
        )
        return new_output


def reshard_transpose_attention_layer_input(layer, inputs):
    # print(f"inputs are {list(inputs)}")
    # breakpoint()
    new_inputs = list(inputs)
    x = new_inputs[0]
    if hasattr(layer, "current_mesh"):
        current_mesh = layer.__getattr__("current_mesh")
        new_x = dist.reshard(x, current_mesh, [dist.Shard(1), dist.Replicate()])
        new_x = paddle.transpose(new_x, [1, 0, 2])
        new_inputs[0] = new_x
        # print(f"new_inputs are: {new_inputs}")
        # breakpoint()
        return tuple(new_inputs)


def transpose_reshard_attention_layer_output(layer, inputs, outputs):
    # print(f"outputs are: {outputs}")
    # breakpoint()
    attn_out = outputs
    if hasattr(layer, "current_mesh"):
        current_mesh = layer.__getattr__("current_mesh")
        new_attn_out = paddle.transpose(attn_out, [1, 0, 2])
        new_attn_out = dist.reshard(
            new_attn_out, current_mesh, [dist.Shard(1), dist.Shard(0)]
        )
        # print(f"new_outputs are: {new_attn_out}")
        # breakpoint()
        return new_attn_out


def reshard_transpose_mlp_layer_input(layer, inputs):
    new_inputs = list(inputs)
    mlp_input = new_inputs[0]
    if hasattr(layer, "current_mesh"):
        current_mesh = layer.__getattr__("current_mesh")
        new_mlp_input = dist.reshard(
            mlp_input, current_mesh, [dist.Shard(1), dist.Replicate()]
        )
        # print(f"new_mlp_input is {new_mlp_input}")
        # breakpoint()
        new_inputs[0] = new_mlp_input
        return tuple(new_inputs)


def transpose_reshard_mlp_layer_output(layer, inputs, outputs):
    # print(f"outputs are: {outputs}")
    # breakpoint()
    mlp_out = outputs
    if hasattr(layer, "current_mesh"):
        current_mesh = layer.__getattr__("current_mesh")
        new_mlp_out = dist.reshard(
            mlp_out, current_mesh, [dist.Shard(1), dist.Shard(0)]
        )
        # print(f"new_outputs are: {new_mlp_out}")
        # breakpoint()
        return new_mlp_out


def reshard_transpose_rms_norm_layer_output(layer, inputs, outputs):
    # print(f"inputs are: {inputs}")
    # print(f"outputs are: {outputs}")
    # breakpoint()
    if hasattr(layer, "current_mesh"):
        current_mesh = layer.__getattr__("current_mesh")
        new_output = dist.reshard(
            outputs, current_mesh, [dist.Shard(1), dist.Replicate()]
        )
        new_output = paddle.transpose(new_output, [1, 0, 2])
        return new_output


def reshard_all_inputs(layer, inputs):
    # print(f"inputs are {inputs}")
    if hasattr(layer, "current_mesh"):
        current_mesh = layer.__getattr__("current_mesh")
        if type(inputs) is tuple:
            new_inputs = []
            for input in inputs:
                if paddle.is_tensor(input):
                    if input.is_dist():
                        new_input = dist.reshard(
                            input,
                            current_mesh,
                            input.placements,
                        )
                    else:
                        new_input = dist.shard_tensor(
                            input,
                            current_mesh,
                            [dist.Shard(0), dist.Replicate()],
                        )
                    new_inputs.append(new_input)
                else:
                    new_inputs.append(input)
            # print(f"new_inputs are: {tuple(new_inputs)}")
            # breakpoint()
            return tuple(new_inputs)
        else:
            if input.is_dist():
                new_input = dist.reshard(
                    input, current_mesh, [dist.Shard(0), dist.Replicate()]
                )
            else:
                new_input = dist.shard_tensor(
                    input, current_mesh, [dist.Shard(0), dist.Replicate()]
                )
            return new_input


def reshard_all_outputs(layer, inputs, outputs):
    if hasattr(layer, "next_mesh"):
        next_mesh = layer.__getattr__("next_mesh")
        # print(f"outputs are {outputs}")
        if type(outputs) is tuple:
            new_outputs = []
            for output in outputs:
                if paddle.is_tensor(output):
                    new_output = dist.reshard(
                        output, next_mesh, [dist.Shard(0), dist.Replicate()]
                    )
                    new_outputs.append(new_output)
                else:
                    new_outputs.append(output)
            return new_outputs
        else:
            new_output = dist.reshard(
                outputs, next_mesh, [dist.Shard(0), dist.Replicate()]
            )
            return new_output


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
        return layer_index // layer_per_stage
    else:
        # return None, False
        return None


# mesh, config: input_spec
def to_distributed(model, dataloader, mesh, config):
    paddle.distributed.init_parallel_env()

    with_pp = True if "pp" in mesh.dim_names else False
    with_sp = True if config.sequence_parallel else False
    # print(f"pp: {with_pp}, sp: {with_sp}")
    # breakpoint()

    # # shard dataloader
    if with_pp:
        first_stage_mesh = mesh.get_mesh_with_dim("pp", 0)
        last_stage_mesh = mesh.get_mesh_with_dim("pp", 1)
        loader = dist.shard_dataloader(
            dataloader,
            meshes=[first_stage_mesh, last_stage_mesh],
            shard_dims="dp",
        )
    else:
        loader = dist.shard_dataloader(
            dataloader, meshes=[mesh], shard_dims="dp"
        )

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
    # print(f"convert to pir program: {pir_program}")
    # breakpoint()

    # record pir_program ops_to_ids
    op_to_id = {}
    for idx, op in enumerate(pir_program.global_block().ops):
        op_to_id[op] = idx

    # # # step3: get the mapping [dynamic-layers : static ops]
    ops_id_to_layer = {}
    op_id_to_layer = {}
    for layer in model.sublayers():
        layer_ops = layer._op_recorder.ops
        ops_id = []
        for op in layer_ops:
            assert op in op_to_id.keys(), f"{op.name} is not in pir program"
            op_id = op_to_id[op]
            op_id_to_layer[op_id] = layer
            ops_id.append(op_id)
        ops_id_to_layer[tuple(ops_id)] = layer
    # print(f"ops id to layer is {ops_id_to_layer}")
    # breakpoint()

    # # # # step4: pattern recogincation
    DECODER_LAYER_NAME = 'decoder_layer'
    register_used_patterns(DECODER_LAYER_NAME)
    results = match_all_patterns(pir_program)
    # print(f"match patterns based on pir program is: {results}")
    # breakpoint()

    # # # # step5: mark pir programs ops dist infos
    matched_programs = {}
    for pattern_name, matched_patterns in results.items():
        # process one pattern
        pattern_ops_dist_infos = get_pattern(pattern_name).ops_dist_infos
        assert (
            pattern_ops_dist_infos is not None
        ), f"{pattern_name} does not contain ops_dist_infos, cannot reshard, please check"
        # print(f"{pattern_name} op dist infos are {pattern_ops_dist_infos}")
        # print(
        #     f"matched patterns are {matched_patterns}"
        # )  # [dict{pattern_node_id : graph_node_id, ..., ...}, dict, dict]
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
        # print(f"matched program and ops dist infos are {matched_programs}")

    # # # #: shard model
    num_hidden_layers = len(matched_programs[DECODER_LAYER_NAME])
    # print(f"num_hidden_layers by pattern matching is {num_hidden_layers}")

    # # # # step6: tensor parallel
    for pattern_name, processed_patterns in matched_programs.items():
        assert (
            len(processed_patterns) == num_hidden_layers
        ), "transformer patterns matched are incomplete"
        for idx, processed_pattern in enumerate(processed_patterns):
            local_mesh = mesh
            if with_pp:
                pp_stage_id = get_layer_pp_info(mesh, num_hidden_layers, idx)
                local_mesh = mesh.get_mesh_with_dim("pp", pp_stage_id)

            for program_ops_id, dist_infos in processed_pattern.items():
                assert (
                    program_ops_id in ops_id_to_layer.keys()
                ), f"program_ops: {program_ops_id} is not corresponding to a dynamic layer"
                dynamic_layer = ops_id_to_layer[program_ops_id]
                # shard layers
                # print(f"sharding info is {dist_infos.print_dist_infos()}")
                mesh_num_dims = len(local_mesh.shape)
                # breakpoint()
                sharding_info = dist_infos.get_dist_info(mesh_num_dims)
                dynamic_layer.weight = dist.shard_tensor(
                    dynamic_layer.weight, local_mesh, sharding_info[0]
                )
                if dynamic_layer.bias is not None:
                    dynamic_layer.bias = dist.shard_tensor(
                        dynamic_layer.bias, local_mesh, sharding_info[1]
                    )

    # # # # step6: pipeline parallel
    if with_pp:
        decoder_layers = []
        for pattern_name, matched_all_patterns in results.items():
            if pattern_name == DECODER_LAYER_NAME:
                for matched_pattern in matched_all_patterns:
                    program_ops_id = []
                    for a, b in matched_pattern.items():
                        program_ops_id.append(b)
                    if tuple(sorted(program_ops_id)) in ops_id_to_layer.keys():
                        decoder_layers.append(
                            ops_id_to_layer[tuple(sorted(program_ops_id))]
                        )
        # print(f"matched decoder layers are: {decoder_layers}")

        if decoder_layers is not None:
            num_decoder_blocks = len(decoder_layers)
            assert (
                num_decoder_blocks == num_hidden_layers
            ), "decoder pattern layers matched are incomplete"

            pp_degree = mesh.get_dim_size("pp")
            num_blocks_per_stage = num_decoder_blocks // pp_degree
            for i in range(num_decoder_blocks):
                pp_stage_id = get_layer_pp_info(mesh, num_decoder_blocks, i)
                current_mesh = mesh.get_mesh_with_dim("pp", pp_stage_id)
                decoder_layer = decoder_layers[i]
                decoder_layer.__setattr__("current_mesh", current_mesh)
                pre_hook_helper = decoder_layer.register_forward_pre_hook(
                    reshard_all_inputs
                )

    # # # # step7: sequence parallel
    if with_sp:
        clear_used_patterns()
        EMBEDDING_LAYER_NAME = "embedding"
        ATTENTION_LAYER_NAME = "attention"
        MLP_LAYER_NAME = "mlp"
        RMS_NORM_LAYER_NAME = "rmsnorm"
        used_patterns = [
            EMBEDDING_LAYER_NAME,
            ATTENTION_LAYER_NAME,
            MLP_LAYER_NAME,
            RMS_NORM_LAYER_NAME,
        ]
        register_used_patterns(used_patterns)
        results = match_all_patterns(pir_program)
        # print(f"match patterns based on pir program is: {results}")

        matched_layers = {}
        for pattern_name, matched_all_patterns in results.items():
            if pattern_name in used_patterns:
                for matched_pattern in matched_all_patterns:
                    program_ops_id = []
                    for a, b in matched_pattern.items():
                        program_ops_id.append(b)
                    if tuple(sorted(program_ops_id)) in ops_id_to_layer.keys():
                        if pattern_name in matched_layers.keys():
                            matched_layers[pattern_name].append(
                                ops_id_to_layer[tuple(sorted(program_ops_id))]
                            )
                        else:
                            matched_layers[pattern_name] = [
                                ops_id_to_layer[tuple(sorted(program_ops_id))]
                            ]
        # print(f"matched layers are: {matched_layers}")

        # init mesh
        GLOBAL_MESH = []
        if with_pp:
            pp_degree = mesh.get_dim_size("pp")
            for i in range(pp_degree):
                local_mesh = mesh.get_mesh_with_dim("pp", i)
                GLOBAL_MESH.append(local_mesh)
        else:
            GLOBAL_MESH.append(mesh)

        # embedding: from [b/dp_degree, s, h] reshard+transpose to [s/mp_degree, b/dp_degree, h]
        embedding_layer = matched_layers[EMBEDDING_LAYER_NAME][0]
        embedding_layer_mesh = GLOBAL_MESH[0]
        embedding_layer.__setattr__("current_mesh", embedding_layer_mesh)
        post_hook_helper = embedding_layer.register_forward_post_hook(
            transpose_reshard_embedding_layer_output
        )

        # attention: input from [s/mp_degree, b/dp_degree, h] to [b/dp_degree, s, h], output from [b/dp_degree, s, h] to [s/mp_degree, b/dp_degree, h]
        attention_layers = matched_layers[ATTENTION_LAYER_NAME]
        num_attention_layers = len(attention_layers)
        if attention_layers is not None:
            for i in range(num_attention_layers):
                current_mesh = GLOBAL_MESH[0]
                if with_pp:
                    pp_stage_id = get_layer_pp_info(
                        mesh, num_attention_layers, i
                    )
                    current_mesh = GLOBAL_MESH[pp_stage_id]
                attention_layer = attention_layers[i]
                attention_layer.__setattr__("current_mesh", current_mesh)
                pre_hook_helper = attention_layer.register_forward_pre_hook(
                    reshard_transpose_attention_layer_input
                )
                post_hook_helper = attention_layer.register_forward_post_hook(
                    transpose_reshard_attention_layer_output
                )

        # mlp: input from [s/mp_degree, b/dp_degree, h] to [s, b/dp_degree, h], output from [s, b/dp_degree, h] to [s/mp_degree, b/dp_degree, h]
        mlp_layers = matched_layers[MLP_LAYER_NAME]
        num_mlp_layers = len(mlp_layers)
        if mlp_layers is not None:
            for i in range(num_mlp_layers):
                current_mesh = GLOBAL_MESH[0]
                if with_pp:
                    pp_stage_id = get_layer_pp_info(
                        mesh, num_attention_layers, i
                    )
                    current_mesh = GLOBAL_MESH[pp_stage_id]
                mlp_layer = mlp_layers[i]
                mlp_layer.__setattr__("current_mesh", current_mesh)
                pre_hook_helper = mlp_layer.register_forward_pre_hook(
                    reshard_transpose_mlp_layer_input
                )
                post_hook_helper = mlp_layer.register_forward_post_hook(
                    transpose_reshard_mlp_layer_output
                )

        # rms norm: for the last rms norm (after decoder blocks), input from [s/mp_degree, b/dp_degree, h] to [b, s, h]
        rms_norm_layers = matched_layers[RMS_NORM_LAYER_NAME]
        if rms_norm_layers is not None:
            last_rms_norm_layer = rms_norm_layers[-1]
            current_mesh = GLOBAL_MESH[-1]
            last_rms_norm_layer.__setattr__("current_mesh", current_mesh)
            post_hook_helper = last_rms_norm_layer.register_forward_post_hook(
                reshard_transpose_rms_norm_layer_output
            )

    # # # # step8: clean layer_op recorder hooks
    for layer in model.sublayers():
        for hook_helper in layer._op_recorder.hooks:
            hook_helper.remove()

    return model, loader
