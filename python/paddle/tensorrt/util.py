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


import paddle

try:
    import tensorrt as trt
except Exception as e:
    pass
from paddle import pir


def map_dtype(pd_dtype):
    if pd_dtype == "FLOAT32":
        return trt.float32
    elif pd_dtype == "FLOAT16":
        return trt.float16
    elif pd_dtype == "INT32":
        return trt.int32
    elif pd_dtype == "INT8":
        return trt.int8
    elif pd_dtype == "BOOL":
        return trt.bool
    # Add other dtype mappings as needed
    else:
        raise TypeError(f"Unsupported dtype: {pd_dtype}")


def run_pir_pass(program, partition_mode=False):
    pm = pir.PassManager(opt_level=4)
    pm.enable_print_statistics()
    paddle.base.libpaddle.pir.infer_symbolic_shape_pass(pm, program)
    passes = [
        {'multihead_matmul_fuse_pass': {}},
        {'transpose_flatten_concat_fuse_pass': {}},
        {'fused_dropout_add_pass': {}},
        {'fused_linear_param_grad_add_pass': {}},
        {'fuse_allreduce_split_to_reducescatter_pass': {}},
        {'inplace_pass': {}},
        {'identity_op_clean_pass': {}},
        {'map_op_to_another_pass': {}},
        {'matmul_scale_fuse_pass': {}},
        {'matmul_transpose_fuse_pass': {}},
        {'silu_fuse_pass': {}},
        {'group_norm_silu_fuse_pass': {}},
        {'fused_dot_product_attention_pass': {}},
        {'fused_flash_attn_pass': {}},
        {'remove_redundant_transpose_pass': {}},
        {'fused_rotary_position_embedding_pass': {}},
        {'trt_op_marker_pass': {}},
    ]
    if partition_mode:
        passes = [{'trt_sub_graph_extract_pass': {}}]

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
            place = paddle.CUDAPlace(0)
            executor = paddle.static.Executor(place)
            output = executor.run(
                program, feed=feed_data, fetch_list=fetch_var_list
            )
            return output


def warmup_shape_infer(program, min_shape_feed, max_shape_feed):
    paddle.framework.set_flags({"FLAGS_enable_collect_shape": True})
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
    paddle.framework.set_flags({"FLAGS_enable_collect_shape": False})
