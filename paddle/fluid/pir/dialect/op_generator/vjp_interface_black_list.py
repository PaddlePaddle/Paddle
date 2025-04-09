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
# VjpInterface gen op list
# =====================================
# we don't support vjp function code
# gen now, so we use a whitelist to
# control the generation of Vjp methods.
# TODO(wanghao107)
# remove this file and support Vjp methods
# code gen.

# Operators which only has composite implementation should be added below.
# For example
# * `silu_double_grad` only has composite implementation, so `silu_grad` was added below.
# * `log_double_grad` has both composite and kernel implementation, so `log_grad` should not be added below.

vjp_interface_black_list = [
    'silu_grad',
    'exp_grad',
    'abs_double_grad',
    'where_grad',
    'bmm_grad',
    'index_put_grad',
    'gather_nd_grad',
    'take_along_axis_grad',
    'index_add_grad',
    'acos_grad',
]
