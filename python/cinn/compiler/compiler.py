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

import cinn

from ..runtime import CINNLowerLevelIRJIT
from .compute_code_generator import ComputeCodeGenerator
from .schedule_code_generator import ScheduleCodeGenerator


def ast_to_llir(fn, inputs_signature):
    function_name = fn.__name__
    # 1. Parse CINN Compute
    llir_compute_generator = ComputeCodeGenerator(
        function_name, inputs_signature
    )
    llir_compute_generator.visit(fn.parse())
    # 2. Parse CINN Schedule
    llir_schedule_generator = ScheduleCodeGenerator(
        llir_compute_generator.cinn_llir_func
    )
    llir_schedule_generator.visit(fn.parse())
    return llir_schedule_generator.cinn_llir_func


def llir_to_runtime_module(llir_func, target, function_name, arg_names):
    cinn_builder = cinn.lang.Module.Builder(function_name, target)
    cinn_builder.add_function(llir_func)
    llir_module = cinn_builder.build()
    return cinn.runtime.Module(llir_module, target, function_name, arg_names)


def compile(fn, **kwargs):
    if isinstance(fn, CINNLowerLevelIRJIT):
        llir_func = ast_to_llir(fn, kwargs["jit_inputs_signature"])
    else:
        raise Exception("Current Only support compile from CINNLowerLevelIRJIT")

    rt_module = llir_to_runtime_module(
        llir_func, kwargs["target"], fn.__name__, kwargs["arg_names"]
    )

    return rt_module
