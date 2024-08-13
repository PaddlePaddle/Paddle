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

from paddle import cinn
from paddle.cinn import lang

from ..runtime import CinnLowerLevelIrJit
from .compute_code_generator import ComputeCodeGenerator
from .schedule_code_generator import ScheduleCodeGenerator


def ast_to_llir(fn, inputs_signature):
    function_name = fn.__name__
    # 1. Parse CINN Compute
    llir_compute_generator = ComputeCodeGenerator(
        fn, function_name, inputs_signature
    )
    cinn_llir_func = llir_compute_generator.parse()

    # 2. Parse CINN Schedule
    llir_schedule_generator = ScheduleCodeGenerator(fn, cinn_llir_func)
    return llir_schedule_generator.parse()


def llir_to_runtime_module(llir_func, target, function_name, arg_names):
    cinn_builder = lang.Module.Builder(function_name, target)
    cinn_builder.add_function(llir_func)
    llir_module = cinn_builder.build()
    return cinn.runtime.Module(llir_module, target, function_name, arg_names)


def compile(fn, just_convert=False, jit_inputs_signature=[], **kwargs):
    if isinstance(fn, CinnLowerLevelIrJit):
        llir_func = ast_to_llir(fn, jit_inputs_signature)
    else:
        raise Exception("Current Only support compile from CinnLowerLevelIrJit")

    if just_convert:
        return llir_func

    rt_module = llir_to_runtime_module(
        llir_func, kwargs["target"], fn.__name__, kwargs["arg_names"]
    )

    return rt_module
