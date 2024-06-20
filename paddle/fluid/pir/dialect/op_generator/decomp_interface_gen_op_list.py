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

# =====================================
# DecompInterface gen op list
# =====================================

# come into effect in generated file pd_op.h
# manual decomp interface declare are located in manual_op.h
decomp_interface_declare_gen_op_list = [
    "add_n",
    "any",
    "batch_norm",
    "batch_norm_",
    "bce_loss",
    "bmm",
    "clip",
    "dropout",
    "elu",
    "embedding",
    "flatten",
    "floor_divide",
    "full_like",
    "gelu",
    "hardswish",
    "hardsigmoid",
    "group_norm",
    "index_sample",
    "index_select",
    "instance_norm",
    "layer_norm",
    "leaky_relu",
    "log_softmax",
    "mean",
    "mean_all",
    "meshgrid",
    "one_hot",
    "p_norm",
    "pow",
    "reciprocal",
    "relu",
    "relu6",
    "sigmoid_cross_entropy_with_logits",
    "silu",
    "swiglu",
    "softmax",
    "square",
    "squared_l2_norm",
    "squeeze",
    "stack",
    "unsqueeze",
    "unbind",
    "huber_loss",
]

# come into effect in generated file op_decomp.cc
# manual decomp interface implementation are located in manual_op_decomp.cc
decomp_interface_implementation_gen_op_list = [
    "any",
    "add_n",
    "bce_loss",
    "bmm",
    "dropout",
    "elu",
    "embedding",
    "flatten",
    "floor_divide",
    "full_like",
    "gelu",
    "hardswish",
    "hardsigmoid",
    "group_norm",
    "index_sample",
    "index_select",
    "instance_norm",
    "layer_norm",
    "leaky_relu",
    "log_softmax",
    "mean",
    "mean_all",
    "meshgrid",
    "p_norm",
    "pow",
    "reciprocal",
    "relu",
    "relu6",
    "sigmoid_cross_entropy_with_logits",
    "silu",
    "swiglu",
    "softmax",
    "square",
    "squared_l2_norm",
    "squeeze",
    "stack",
    "unsqueeze",
    "unbind",
    "huber_loss",
]

# xshape output will no longer used after decomp, but return none to keep output num the same as origin op
decomp_ops_contain_unused_output = ["squeeze", "unsqueeze"]

# prim op with one input and one output, with no attribute
UNARY_PRIM_VJP_OPS = [
    'abs_grad',
    'erf_grad',
    'exp_grad',
    'floor_grad',
    'log_grad',
    'rsqrt_grad',
    'sin_grad',
    'cos_grad',
    'tanh_grad',
]

# prim op with two inputs and one output, with no attribute
BINARY_PRIM_VJP_OPS = [
    'matmul_grad',
    'add_grad',
    'divide_grad',
    'subtract_grad',
    'multiply_grad',
    'elementwise_pow_grad',
    'maximum_grad',
    'reduce_as_grad',
]

OTHER_PRIM_VJP_OPS = [
    'sum_grad',
    'reshape_grad',
    'roll_grad',
    'transpose_grad',
    'max_grad',
    'squeeze_grad',
    'unsqueeze_grad',
]


CUSTOM_VJP = [
    'gelu_grad',
    'hardswish_grad',
    'leaky_relu_grad',
    'mean_grad',
    'minimum_grad',
    'pow_grad',
    'relu_grad',
    'sigmoid_grad',
    'silu_grad',
    'softmax_grad',
    'sqrt_grad',
    'swiglu_grad',
    'layer_norm_grad',
    'group_norm_grad',
]  # custom vjp list of composite op

# declare belongs to codegen, but implementation not
OTHER_VJP = ["concat_grad", "stack_grad", 'slice_grad']

vjp_list = (
    UNARY_PRIM_VJP_OPS + BINARY_PRIM_VJP_OPS + CUSTOM_VJP + OTHER_PRIM_VJP_OPS
)

decomp_vjp_interface_declare_gen_op_list = vjp_list + OTHER_VJP

decomp_vjp_interface_implementation_gen_op_list = vjp_list
