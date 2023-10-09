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


from ..runtime import CinnLowerLevelIrJit
from .compute_code_generator import ComputeCodeGenerator


def ast_to_llir(fn, inputs_signature):
    function_name = fn.__name__
    # 1. Parse CINN Compute
    llir_compute_generator = ComputeCodeGenerator(
        fn, function_name, inputs_signature
    )
    cinn_llir_func = llir_compute_generator.parse()
    return cinn_llir_func


def compile(fn, just_convert=False, jit_inputs_signature=[], **kwargs):
    if isinstance(fn, CinnLowerLevelIrJit):
        llir_func = ast_to_llir(fn, jit_inputs_signature)
    else:
        raise Exception("Current Only support compile from CinnLowerLevelIrJit")

    if just_convert:
        return llir_func
    return llir_func
