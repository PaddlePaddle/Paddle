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

from op_infermeta_func_gen import gen_op_infermeta_func
from op_member_access_func_gen import gen_op_member_access_func
from op_vjp_interface_func_gen import gen_op_vjp_interface_func

all_gen_op_func_list = [
    gen_op_infermeta_func,
    gen_op_member_access_func,
    gen_op_vjp_interface_func,
]


def gen_op_all_func(args, op_info, op_info_items):
    interface_list = []
    declare_list = []
    impl_list = []
    for func in all_gen_op_func_list:
        interface, declare, impl = func(args, op_info, op_info_items)
        interface_list += interface
        if declare is not None:
            declare_list.append(declare)
        if impl is not None:
            impl_list.append(impl)
    return interface_list, declare_list, impl_list
