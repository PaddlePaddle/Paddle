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


class ReshardFunction:
    def is_suitable(self, dist_tensor, dist_attr):
        return "ReshardFunction is_suitable not implemented"

    def eval(self, program, op, src_tensor, dst_dist_attr):
        return "ReshardFunction eval not implemented"

    def is_partial(self, dist_attr):
        if len(dist_attr.partial_status) > 0:
            return True
        return False

    def is_replicated(self, dist_attr):
        dims_mapping_set = set(dist_attr.dims_mapping)
        if (
            len(dist_attr.partial_status) == 0
            and len(dims_mapping_set) == 1
            and -1 in dims_mapping_set
        ):
            return True
        return False


def choose_reshard_func(src_dist_attr, dst_dist_attr):
    global _g_reshard_func_list
    print(f'debug _g_reshard_func_list: {_g_reshard_func_list}')
    for reshard_func in _g_reshard_func_list:
        if reshard_func.is_suitable(src_dist_attr, dst_dist_attr):
            print(f'debug find reshard_func: {reshard_func}')
            return reshard_func
    return None


def register_reshard_func(reshard_func):
    global _g_reshard_func_list
    _g_reshard_func_list.append(reshard_func)


def clean_reshard_funcs():
    global _g_reshard_func_list
    _g_reshard_func_list.clear()


_g_reshard_func_list = []
