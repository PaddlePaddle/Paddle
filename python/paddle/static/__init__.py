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
    'BuildStrategy', 'CompiledProgram', 'Print', 'py_func', 'ExecutionStrategy',
    'name_scope', 'ParallelExecutor', 'program_guard', 'WeightNormParamAttr',
    'default_main_program', 'default_startup_program', 'Program', 'save',
    'load', 'data', 'InputSpec'
]

from . import nn
from .input import data  #DEFINE_ALIAS
from .input import InputSpec  #DEFINE_ALIAS
from ..fluid.executor import Executor  #DEFINE_ALIAS
from ..fluid.executor import global_scope  #DEFINE_ALIAS
from ..fluid.executor import scope_guard  #DEFINE_ALIAS
from ..fluid.backward import append_backward  #DEFINE_ALIAS
from ..fluid.backward import gradients  #DEFINE_ALIAS
from ..fluid.compiler import BuildStrategy  #DEFINE_ALIAS
from ..fluid.compiler import CompiledProgram  #DEFINE_ALIAS
from ..fluid.compiler import ExecutionStrategy  #DEFINE_ALIAS
from ..fluid.framework import default_main_program  #DEFINE_ALIAS
from ..fluid.framework import default_startup_program  #DEFINE_ALIAS
from ..fluid.framework import Program  #DEFINE_ALIAS
from ..fluid.framework import name_scope  #DEFINE_ALIAS
from ..fluid.framework import program_guard  #DEFINE_ALIAS
from ..fluid.layers.control_flow import Print  #DEFINE_ALIAS
from ..fluid.layers.nn import py_func  #DEFINE_ALIAS
from ..fluid.parallel_executor import ParallelExecutor  #DEFINE_ALIAS
from ..fluid.param_attr import WeightNormParamAttr  #DEFINE_ALIAS
from ..tensor.io import save  #DEFINE_ALIAS
from ..tensor.io import load  #DEFINE_ALIAS
