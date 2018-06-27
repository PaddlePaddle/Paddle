#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

import trainer
from trainer import Trainer
from trainer import BeginEpochEvent
from trainer import EndEpochEvent
from trainer import BeginStepEvent
from trainer import EndStepEvent
from trainer import CheckpointConfig

import inferencer
from inferencer import Inferencer

import io
import evaluator
import initializer
import layers
import nets
import optimizer
import backward
import regularizer
import average
import metrics
import transpiler
from param_attr import ParamAttr, WeightNormParamAttr
from data_feeder import DataFeeder
from core import LoDTensor, CPUPlace, CUDAPlace, CUDAPinnedPlace, Scope
from transpiler import DistributeTranspiler, InferenceTranspiler, \
    memory_optimize, release_memory
from concurrency import (Go, make_channel, channel_send, channel_recv,
                         channel_close, Select)
from lod_tensor import create_lod_tensor, create_random_int_lodtensor
import clip
import profiler
import unique_name
import recordio_writer
import parallel_executor
from parallel_executor import *

Tensor = LoDTensor

__all__ = framework.__all__ + executor.__all__ + concurrency.__all__ + \
          trainer.__all__ + inferencer.__all__ + transpiler.__all__ + \
          parallel_executor.__all__ + lod_tensor.__all__ + [
              'io',
              'initializer',
              'layers',
              'transpiler'
              'nets',
              'optimizer',
              'learning_rate_decay',
              'backward',
              'regularizer',
              'LoDTensor',
              'CPUPlace',
              'CUDAPlace',
              'CUDAPinnedPlace',
              'Tensor',
              'ParamAttr',
              'WeightNormParamAttr',
              'DataFeeder',
              'clip',
              'profiler',
              'unique_name',
              'recordio_writer',
              'Scope',
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

    in_test = 'unittest' in sys.modules

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
        'use_pinned_memory', 'check_nan_inf', 'benchmark', 'warpctc_dir',
        'eager_delete_scope', 'use_mkldnn', 'initial_cpu_memory_in_mb'
    ]
    if core.is_compiled_with_cuda():
        read_env_flags += [
            'fraction_of_gpu_memory_to_use', 'cudnn_deterministic'
        ]
    core.init_gflags([sys.argv[0]] +
                     ["--tryfromenv=" + ",".join(read_env_flags)])
    core.init_glog(sys.argv[0])
    # don't init_p2p when in unittest to save time.
    core.init_devices(not in_test)


# TODO(panyx0718): Avoid doing complex initialization logic in __init__.py.
# Consider paddle.init(args) or paddle.main(args)
layers.monkey_patch_variable()
__bootstrap__()
