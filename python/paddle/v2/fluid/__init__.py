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
from core import LoDTensor, CPUPlace, GPUPlace

Tensor = LoDTensor
__all__ = framework.__all__ + executor.__all__ + [
    'io', 'initializer', 'layers', 'nets', 'optimizer', 'backward',
    'regularizer', 'LoDTensor', 'CPUPlace', 'GPUPlace', 'Tensor', 'ParamAttr'
    'DataFeeder'
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


__read_gflags_from_env__()
