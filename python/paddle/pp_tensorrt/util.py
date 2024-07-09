# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved
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

import os
import paddle
import numpy as np

try:
    import tensorrt as trt
except Exception as e:
    pass
from paddle import base
from paddle import pir

import paddle.static as static
import paddle.nn.functional as F
import paddle.nn as nn
from paddle.nn import TransformerEncoderLayer, TransformerEncoder


def map_dtype(pd_dtype):
    if pd_dtype == "FLOAT32":
        return trt.float32
    elif pd_dtype == "FLOAT16":
        return trt.float16
    elif pd_dtype == "INT32":
        return trt.int32
    elif pd_dtype == "INT8":
        return trt.int8
    # Add other dtype mappings as needed
    else:
        raise TypeError(f"Unsupported dtype: {pd_dtype}")


def run_pir_pass(program, partition_mode=False):
    pm = pir.PassManager(opt_level=4)
    pm.enable_print_statistics()
    pm.enable_ir_printing()
    passes = [
        # {'dead_code_elimination_pass': {}},
        {'multihead_matmul_fuse_pass': {}},
        {'transpose_flatten_concat_fuse_pass': {}},
        # {'fused_gemm_epilogue_pass': {}},
        {'fused_dropout_add_pass': {}},
        # {'fused_weight_only_linear_pass': {}},
        {'fused_linear_param_grad_add_pass': {}},
        {'fuse_allreduce_split_to_reducescatter_pass': {}},
        {'inplace_pass': {}},
        # {'replace_fetch_with_shadow_output_pass': {}},
        {'identity_op_clean_pass': {}},
        {'map_op_to_another_pass': {}},
        {'matmul_scale_fuse_pass': {}},
        {'matmul_transpose_fuse_pass': {}},
        # {'matmul_add_act_fuse_pass': {}},
        {'silu_fuse_pass': {}},
        # {'fc_elementwise_layernorm_fuse_pass': {}},
        # {'conv2d_bn_fuse_pass': {}},
        # {'conv2d_add_fuse_pass': {}},
        # {'conv2d_add_act_fuse_pass': {}},
        # {'embedding_eltwise_layernorm_fuse_pass': {}},
        # {'add_norm_fuse_pass': {}},
        {'group_norm_silu_fuse_pass': {}},
        {'fused_dot_product_attention_pass': {}},
        {'fused_flash_attn_pass': {}},
        {'remove_redundant_transpose_pass': {}},
        # {'delete_weight_dequant_linear_op_pass': {}},
        # {'delete_quant_dequant_linear_op_pass': {}},
        # {'transfer_layout_pass': {}},
        {'fused_rotary_position_embedding_pass': {}},
        {'trt_op_marker_pass': {}},
        # {'trt_sub_graph_extract_pass': {}}
    ]
    if partition_mode:
        passes = [{'trt_sub_graph_extract_pass': {}}]

    pm = pir.PassManager(opt_level=4)
    paddle.base.libpaddle.pir.infer_symbolic_shape_pass(pm, program)

    pm.enable_print_statistics()
    pm.enable_ir_printing()
    for pass_item in passes:
        for pass_name, pass_attr in pass_item.items():
            pm.add_pass(pass_name, pass_attr)
    pm.run(program)
    return program


def forbid_op_lower_trt(program, op_name):
    for op in program.global_block().ops:
        if op.name() == op_name:
            op.set_bool_attr("__l_trt__", False)


def enforce_op_lower_trt(program, op_name):
    for op in program.global_block().ops:
        if op.name() == op_name:
            op.set_bool_attr("__l_trt__", True)


def predict_program(program, feed_data, fetch_var_list):
    with paddle.pir_utils.IrGuard():
        with paddle.static.program_guard(program):
            executor = paddle.static.Executor()
            output = executor.run(
                program, feed=feed_data, fetch_list=fetch_var_list
            )
            return output


def warmup_shape_infer(program, min_shape_feed, max_shape_feed):
    with paddle.pir_utils.IrGuard():
        with paddle.static.program_guard(program):
            executor = paddle.static.Executor()
            output_var = program.list_vars()[-1]
            # Run the program with input_data
            for _ in range(1):
                output_original = executor.run(
                    program, feed=min_shape_feed, fetch_list=[output_var]
                )

            # Run the program with input_data_max_shape (fake max_shape input)
            for _ in range(1):
                executor.run(
                    program, feed=max_shape_feed, fetch_list=[output_var]
                )


def get_r50_program():
    paddle.enable_static()
    # static_resnet50 = StaticResNet50()
    from paddle.vision.models import wide_resnet50_2

    with paddle.pir_utils.IrGuard():
        infer_program = paddle.static.Program()
        startup_program = paddle.static.Program()
        with static.program_guard(infer_program, startup_program):
            scope = paddle.static.global_scope()
            input_data = paddle.static.data(
                shape=[1, 3, 224, 224], dtype='float32', name='input'
            )
            model = wide_resnet50_2()
            model.eval()
            output = model(input_data)
        place = paddle.CUDAPlace(0)
        exe = static.Executor(place)
        exe.run(startup_program)

    # paddle.static.io.save(infer_program, "./resnet")

    params = infer_program.global_block().all_parameters()
    param_dict = {}
    for v in params:
        name = v.get_defining_op().attrs()["parameter_name"]
        param_dict.update({name: np.array(scope.var(name).get_tensor())})

    return infer_program, scope, param_dict


def get_dummy_program():
    paddle.enable_static()
    with paddle.pir_utils.IrGuard():
        main_program = paddle.static.Program()
        default_startup_program = paddle.static.Program()
        with paddle.static.program_guard(main_program, default_startup_program):
            scope = paddle.static.global_scope()
            input = paddle.static.data(
                shape=[-1, 64], dtype='float32', name='input'
            )
            weight_numpy = np.random.rand(64, 64).astype('float32')
            weight = paddle.create_parameter(
                name="w",
                shape=[64, 64],
                dtype='float32',
                attr=paddle.ParamAttr(
                    initializer=paddle.nn.initializer.Assign(weight_numpy)
                ),
            )
            bias_numpy = np.random.rand(64).astype('float32')
            bias = paddle.create_parameter(
                name="b",
                shape=[64],
                dtype='float32',
                attr=paddle.ParamAttr(
                    initializer=paddle.nn.initializer.Assign(bias_numpy)
                ),
            )
            # shape = paddle._C_ops.full_int_array(value=0)
            x = paddle.matmul(input, weight)
            x_1 = paddle.add(x, bias)
            y = paddle.nn.functional.relu(x_1)
            y = paddle.nn.functional.gelu(y)
            y_2 = paddle.nn.functional.gelu(x_1)
        main_program = run_pir_pass(main_program)
        exe = paddle.static.Executor(paddle.CUDAPlace(0))
        exe.run(default_startup_program)

        params = main_program.global_block().all_parameters()
        param_dict = {}
        # save parameters
        for v in params:
            name = v.get_defining_op().attrs()["parameter_name"]
            param_dict.update({name: np.array(scope.var(name).get_tensor())})
    return main_program, scope, param_dict


class BertModel(nn.Layer):
    def __init__(
        self,
        vocab_size,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
    ):
        super(BertModel, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, hidden_size)
        encoder_layer = TransformerEncoderLayer(
            hidden_size, num_attention_heads, hidden_size * 4
        )
        self.encoder = TransformerEncoder(encoder_layer, num_hidden_layers)

    def forward(self, input_ids):
        embeddings = self.embeddings(input_ids)
        encoded_output = self.encoder(embeddings)
        return encoded_output


def get_bert_program():
    paddle.enable_static()

    vocab_size = 30522  # BERT base vocab size
    hidden_size = 768
    num_hidden_layers = 2
    num_attention_heads = 12
    seq_length = 128

    with paddle.pir_utils.IrGuard():
        main_program = static.default_main_program()
        startup_program = static.default_startup_program()
        with static.program_guard(main_program, startup_program):
            scope = paddle.static.global_scope()
            input_ids = static.data(
                name='input_ids', shape=[-1, -1], dtype='int64'
            )

            bert_model = BertModel(
                vocab_size, hidden_size, num_hidden_layers, num_attention_heads
            )
            bert_model.eval()
            logits = bert_model(input_ids)

    place = (
        paddle.CUDAPlace(0)
        if paddle.is_compiled_with_cuda()
        else paddle.CPUPlace()
    )
    pir_program = main_program

    with paddle.pir_utils.IrGuard():
        with paddle.static.program_guard(pir_program, startup_program):
            x = np.ones([1, seq_length]).astype('int64')
            executor = paddle.static.Executor(place)
            executor.run(startup_program)
            fetches = executor.run(
                pir_program,
                feed={"input_ids": x},
                fetch_list=pir_program.list_vars()[-1],
            )
    params = main_program.global_block().all_parameters()
    param_dict = {}
    # save parameters
    for v in params:
        name = v.get_defining_op().attrs()["parameter_name"]
        param_dict.update({name: np.array(scope.var(name).get_tensor())})
    return pir_program, scope, param_dict


if __name__ == "__main__":
    pir_program, scope, param_dict = get_bert_program()
    pir_program = run_pir_pass(pir_program)
    x = np.ones([1, 768]).astype('int64')
    place = (
        paddle.CUDAPlace(0)
        if paddle.is_compiled_with_cuda()
        else paddle.CPUPlace()
    )
    executor = paddle.static.Executor(place)
    with paddle.pir_utils.IrGuard():
        with paddle.static.program_guard(pir_program):
            fetches = executor.run(
                pir_program,
                feed={"input_ids": x},
                fetch_list=pir_program.list_vars()[-1],
            )
            print(fetches)
