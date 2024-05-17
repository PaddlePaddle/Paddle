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
# Copyright (c) 2021 CINN Authors. All Rights Reserved.
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

import os

from .runtime.cinn_jit import to_cinn_llir  # noqa: F401

cinndir = os.path.dirname(os.path.abspath(__file__))
runtime_include_dir = os.path.join(cinndir, "libs")
cuhfile = os.path.join(runtime_include_dir, "cinn_cuda_runtime_source.cuh")

if os.path.exists(cuhfile):
    os.environ.setdefault('runtime_include_dir', runtime_include_dir)

from .backends import (  # noqa: F401
    Compiler,
    ExecutionEngine,
    ExecutionOptions,
)
from .common import (  # noqa: F401
    BFloat16,
    Bool,
    CINNValue,
    CINNValuePack,
    DefaultHostTarget,
    DefaultNVGPUTarget,
    DefaultTarget,
    Float,
    Float16,
    Int,
    RefCount,
    Shared_CINNValuePack_,
    String,
    Target,
    Type,
    UInt,
    Void,
    _CINNValuePack_,
    get_target,
    is_compiled_with_cuda,
    is_compiled_with_cudnn,
    make_const,
    reset_name_id,
    set_target,
    type_of,
)
