#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserve.
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

from __future__ import print_function
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
import learning_rate_decay
import backward
import regularizer
from param_attr import ParamAttr
from data_feeder import DataFeeder
from core import LoDTensor, CPUPlace, CUDAPlace
from distribute_transpiler import DistributeTranspiler
from distribute_transpiler_simple import SimpleDistributeTranspiler
import clip
from memory_optimization_transpiler import memory_optimize
import profiler

Tensor = LoDTensor

__all__ = framework.__all__ + executor.__all__ + [
    'io', 'initializer', 'layers', 'nets', 'optimizer', 'learning_rate_decay',
    'backward', 'regularizer', 'LoDTensor', 'CPUPlace', 'CUDAPlace', 'Tensor',
    'ParamAttr'
    'DataFeeder', 'clip', 'SimpleDistributeTranspiler', 'DistributeTranspiler',
    'memory_optimize', 'profiler'
]


def __bootstrap__():
    """
    Enable reading gflags from environment variables.

    Returns:
        None
    """
    import sys
    import core
    import os

    try:
        num_threads = int(os.getenv('OMP_NUM_THREADS', '1'))
    except ValueError:
        num_threads = 1

    if num_threads > 1:
        print(
            'WARNING: OMP_NUM_THREADS set to {0}, not 1. The computation '
            'speed will not be optimized if you use data parallel. It will '
            'fail if this PaddlePaddle binary is compiled with OpenBlas since'
            ' OpenBlas does not support multi-threads.'.format(num_threads),
            file=sys.stderr)
        print('PLEASE USE OMP_NUM_THREADS WISELY.', file=sys.stderr)

    os.environ['OMP_NUM_THREADS'] = str(num_threads)

    read_env_flags = [
        'use_pinned_memory', 'check_nan_inf', 'benchmark', 'warpctc_dir'
    ]
    if core.is_compiled_with_cuda():
        read_env_flags += ['fraction_of_gpu_memory_to_use']
    core.init_gflags([sys.argv[0]] +
                     ["--tryfromenv=" + ",".join(read_env_flags)])
    core.init_glog(sys.argv[0])
    core.init_devices()


layers.monkey_patch_variable()
__bootstrap__()
