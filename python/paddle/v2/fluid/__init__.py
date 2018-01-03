# import all class inside framework into fluid module
from core import LoDTensor

import backward
import clip
import evaluator
# import all class inside executor into fluid module
import executor
import framework
import initializer
import io
import layers
import nets
import optimizer
import regularizer
from data_feeder import DataFeeder
from distribute_transpiler import DistributeTranspiler
from executor import *
from framework import *
from param_attr import ParamAttr

Tensor = LoDTensor
__all__ = framework.__all__ + executor.__all__ + [
    'io', 'initializer', 'layers', 'nets', 'optimizer', 'backward',
    'regularizer', 'LoDTensor', 'CPUPlace', 'CUDAPlace', 'Tensor', 'ParamAttr'
    'DataFeeder', 'clip', 'DistributeTranspiler'
]


def __bootstrap__():
    """
    Enable reading gflags from environment variables.

    Returns:
        None
    """
    import sys
    import core
    read_env_flags = ['use_pinned_memory', 'check_nan_inf']
    if core.is_compile_gpu():
        read_env_flags.append('fraction_of_gpu_memory_to_use')
    core.init_gflags([sys.argv[0]] +
                     ["--tryfromenv=" + ",".join(read_env_flags)])
    core.init_glog(sys.argv[0])

    if core.is_compile_gpu():
        core.init_devices(["CPU", "GPU:0"])
    else:
        core.init_devices(["CPU"])


__bootstrap__()
