#  Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserve.
#
#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#Unless required by applicable law or agreed to in writing, software
#distributed under the License is distributed on an "AS IS" BASIS,
#WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#See the License for the specific language governing permissions and
#limitations under the License.
# import all class inside framework into fluid module
import framework
from framework import *
# import all class inside executor into fluid module
import executor
from executor import *

import io
import evaluator
import initializer
import layers
import nets
import optimizer
import backward
import regularizer
from param_attr import ParamAttr
from data_feeder import DataFeeder
from core import LoDTensor, CPUPlace, CUDAPlace
from distribute_transpiler import DistributeTranspiler
import clip

Tensor = LoDTensor
__all__ = framework.__all__ + executor.__all__ + [
    'io', 'initializer', 'layers', 'nets', 'optimizer', 'backward',
    'regularizer', 'LoDTensor', 'CPUPlace', 'CUDAPlace', 'Tensor', 'ParamAttr'
    'DataFeeder', 'clip', 'DistributeTranspiler'
]


def __read_gflags_from_env__():
    """
    Enable reading gflags from environment variables.

    Returns:
        None
    """
    import sys
    import core
    read_env_flags = ['use_pinned_memory']
    if core.is_compile_gpu():
        read_env_flags.append('fraction_of_gpu_memory_to_use')
    core.init_gflags([sys.argv[0]] +
                     ["--tryfromenv=" + ",".join(read_env_flags)])

    if core.is_compile_gpu():
        core.init_devices(["CPU", "GPU:0"])
    else:
        core.init_devices(["CPU"])


__read_gflags_from_env__()
