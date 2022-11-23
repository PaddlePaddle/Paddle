# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

from paddle.fluid import unique_name
from paddle.fluid.dygraph.base import switch_to_static_graph


@switch_to_static_graph
def _generate_unique_var_name_sync_with_main_program(prefix):
    return unique_name.generate(prefix)


def rename_var_with_generator(names_old):
    dict_rename_var_old_new = dict()
    names_old = list(names_old)
    for var_idx, name_old in enumerate(names_old):
        while True:
            temp_name = name_old.split('_')
            if len(temp_name) > 1 and temp_name[-1].isnumeric():
                temp_name = "_".join(temp_name[:-1])
            else:
                temp_name = "_".join(temp_name)
            name_new = _generate_unique_var_name_sync_with_main_program(
                temp_name)
            if name_new not in names_old[:var_idx] + names_old[var_idx + 1:]:
                break
        dict_rename_var_old_new[name_old] = name_new
    return dict_rename_var_old_new
