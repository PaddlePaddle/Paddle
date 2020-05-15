#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

# TODO: import framework api under this directory 
__all__ = [
    'append_backward', 'gradients', 'Executor', 'global_scope', 'scope_guard',
    'BuildStrategy', 'CompiledProgram', 'default_main_program',
    'default_startup_program', 'create_global_var', 'create_parameter', 'Print',
    'py_func', 'ExecutionStrategy', 'name_scope', 'ParallelExecutor',
    'ParamAttr', 'Program', 'program_guard', 'Variable', 'WeightNormParamAttr',
    'CPUPlace', 'CUDAPlace', 'CUDAPinnedPlace'
]

from . import random
from .random import manual_seed
from ..fluid.executor import Executor, global_scope, scope_guard
from ..fluid.backward import append_backward, gradients
from ..fluid.compiler import BuildStrategy, CompiledProgram, ExecutionStrategy
from ..fluid.framework import default_main_program, default_startup_program, name_scope, Program, program_guard, Variable
from ..fluid.layers.control_flow import Print
from ..fluid.layers.nn import py_func
from ..fluid.parallel_executor import ParallelExecutor
from ..fluid.param_attr import ParamAttr, WeightNormParamAttr
from ..fluid.layers.tensor import create_global_var, create_parameter
from ..fluid.core import CPUPlace, CUDAPlace, CUDAPinnedPlace
