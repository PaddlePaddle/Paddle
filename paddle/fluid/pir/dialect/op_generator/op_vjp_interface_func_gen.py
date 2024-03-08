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

from vjp_interface_black_list import vjp_interface_black_list


def gen_op_vjp_interface_func(args, op_info, op_info_items):
    if (
        op_info.backward_name
        and op_info.op_phi_name[0] not in vjp_interface_black_list
        and args.dialect_name != "onednn_op"
    ):
        return ["paddle::dialect::VjpInterface"], None, None
    else:
        return [], None, None
