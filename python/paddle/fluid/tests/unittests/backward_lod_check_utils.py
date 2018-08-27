# Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

import paddle.fluid as fluid
import paddle.fluid.core as core
import numpy as np


def _set_persistable(input_vars):
    for var in input_vars:
        var.persistable = True


def get_all_lod_tensors_with_grad(main_prog=None, startup_prog=None):
    grad_suffix = '@GRAD'
    if main_prog is None:
        main_prog = fluid.default_main_program()

    if startup_prog is None:
        startup_prog = fluid.default_startup_program()

    all_vars = dict()

    check_func = lambda var: var.type == core.VarDesc.VarType.LOD_TENSOR  #and var.lod_level > 0

    for var in startup_prog.list_vars():
        if check_func(var):
            all_vars[var.name] = var

    for var in main_prog.list_vars():
        if check_func(var):
            all_vars[var.name] = var

    new_vars = []
    grad_vars = []
    for var_name, var in all_vars.items():
        grad_var_name = var_name + grad_suffix
        if not var_name.endswith(grad_suffix) and grad_var_name in all_vars:
            new_vars.append(var)
            grad_vars.append(all_vars[grad_var_name])

    _set_persistable(new_vars)
    _set_persistable(grad_vars)
    return new_vars, grad_vars
