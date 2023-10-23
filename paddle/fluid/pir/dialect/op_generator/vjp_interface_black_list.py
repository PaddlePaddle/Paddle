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


vjp_interface_black_list = [
    'write_to_array',
    'fused_attention',
    'fused_feedforward',
    'set_value',
    'set_value_with_tensor',
    'silu_grad',
    'fused_dropout_add',
    'fused_rotary_position_embedding',
]
