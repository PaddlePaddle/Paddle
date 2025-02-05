#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#   Copyright (c) 2021 NVIDIA Corporation. All rights reserved.
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

from ..base import Scope  # noqa: F401
from ..base.backward import append_backward, gradients
from ..base.compiler import (
    BuildStrategy,
    CompiledProgram,
    IpuCompiledProgram,
    IpuStrategy,
)
from ..base.executor import Executor, global_scope, scope_guard
from ..base.framework import (
    Operator,  # noqa: F401
    Parameter,  # noqa: F401
    Program,
    Variable,
    cpu_places,
    cuda_places,
    default_main_program,
    default_startup_program,
    device_guard,
    ipu_shard_guard,
    name_scope,
    program_guard,
    set_ipu_shard,
    xpu_places,
)
from ..base.param_attr import WeightNormParamAttr
from ..tensor.creation import create_global_var, create_parameter
from . import amp, nn  # noqa: F401
from .input import (
    InputSpec,
    data,
    setitem,  # noqa: F401
)
from .io import (
    deserialize_persistables,
    deserialize_program,
    is_persistable,  # noqa: F401
    load,
    load_from_file,
    load_inference_model,
    load_program_state,
    load_vars,  # noqa: F401
    normalize_program,
    save,
    save_inference_model,
    save_to_file,
    save_vars,  # noqa: F401
    serialize_persistables,
    serialize_program,
    set_program_state,
)
from .nn.common import ExponentialMovingAverage, py_func
from .nn.control_flow import Print
from .nn.metric import accuracy, auc, ctr_metric_bundle

__all__ = [
    'append_backward',
    'gradients',
    'Executor',
    'global_scope',
    'scope_guard',
    'BuildStrategy',
    'CompiledProgram',
    'ipu_shard_guard',
    'IpuCompiledProgram',
    'IpuStrategy',
    'Print',
    'py_func',
    'name_scope',
    'program_guard',
    'WeightNormParamAttr',
    'ExponentialMovingAverage',
    'default_main_program',
    'default_startup_program',
    'Program',
    'data',
    'InputSpec',
    'save',
    'load',
    'save_inference_model',
    'load_inference_model',
    'serialize_program',
    'serialize_persistables',
    'save_to_file',
    'deserialize_program',
    'deserialize_persistables',
    'load_from_file',
    'normalize_program',
    'load_program_state',
    'set_program_state',
    'cpu_places',
    'cuda_places',
    'xpu_places',
    'Variable',
    'create_global_var',
    'accuracy',
    'auc',
    'device_guard',
    'create_parameter',
    'set_ipu_shard',
    'ctr_metric_bundle',
]
