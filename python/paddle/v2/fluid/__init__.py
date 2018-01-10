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
import backward
import regularizer
from param_attr import ParamAttr
from data_feeder import DataFeeder
from core import LoDTensor, CPUPlace, CUDAPlace
from distribute_transpiler import DistributeTranspiler
import clip
from memory_optimization_transpiler import memory_optimize

Tensor = LoDTensor
__all__ = framework.__all__ + executor.__all__ + [
    'io', 'initializer', 'layers', 'nets', 'optimizer', 'backward',
    'regularizer', 'LoDTensor', 'CPUPlace', 'CUDAPlace', 'Tensor', 'ParamAttr'
    'DataFeeder', 'clip', 'DistributeTranspiler', 'memory_optimize'
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

    read_env_flags = ['use_pinned_memory', 'check_nan_inf']
    if core.is_compile_gpu():
        read_env_flags.append('fraction_of_gpu_memory_to_use')
    core.init_gflags([sys.argv[0]] +
                     ["--tryfromenv=" + ",".join(read_env_flags)])
    core.init_glog(sys.argv[0])
    core.init_devices()


__bootstrap__()
