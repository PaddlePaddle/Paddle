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
    "batch_norm",
    "dropout",
    "full_like",
    "gelu",
    "layer_norm",
    "mean",
    "pow",
    "relu",
    "rsqrt",
    "silu",
    "softmax",
    "sqrt",
    "squeeze",
    "stack",
    "unsqueeze",
]

# come into effect in generated file op_decomp.cc
# manual decomp interface implementation are located in manual_op_decomp.cc
decomp_interface_implementation_gen_op_list = [
    "add_n",
    "dropout",
    "full_like",
    "gelu",
    "layer_norm",
    "mean",
    "pow",
    "relu",
    "rsqrt",
    "silu",
    "softmax",
    "sqrt",
    "squeeze",
    "stack",
    "unsqueeze",
]


# xshape output will no longer used after decomp, but return none to keep output num the same as origin op
decomp_ops_contain_unused_output = ["squeeze", "unsqueeze"]
