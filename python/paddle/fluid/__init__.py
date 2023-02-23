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

import os
import sys
import atexit

# The legacy core need to be removed before "import core",
# in case of users installing paddlepadde without -U option
core_suffix = 'so'
if os.name == 'nt':
    core_suffix = 'pyd'

legacy_core = (
    os.path.abspath(os.path.dirname(__file__)) + os.sep + 'core.' + core_suffix
)
if os.path.exists(legacy_core):
    sys.stderr.write('Deleting legacy file ' + legacy_core + '\n')
    try:
        os.remove(legacy_core)
    except Exception as e:
        raise e

# import all class inside framework into fluid module
from . import framework
from .framework import *

# import all class inside executor into fluid module
from . import executor
from .executor import *

from . import data_feed_desc
from .data_feed_desc import *

from . import dataset
from .dataset import *

from .data import *

from . import trainer_desc

from . import io
from . import evaluator
from . import initializer
from .initializer import set_global_initializer
from . import layers
from . import dygraph
from . import contrib
from . import nets
from . import optimizer
from . import backward
from .backward import gradients
from . import regularizer
from . import average
from . import metrics
from . import transpiler
from . import incubate
from .param_attr import ParamAttr, WeightNormParamAttr
from .data_feeder import DataFeeder

from .core import LoDTensor, LoDTensorArray, Scope, _Scope
from .core import (
    CPUPlace,
    XPUPlace,
    CUDAPlace,
    CUDAPinnedPlace,
    NPUPlace,
    IPUPlace,
    MLUPlace,
    CustomPlace,
)
from .incubate import fleet
from .transpiler import (
    DistributeTranspiler,
    memory_optimize,
    release_memory,
    DistributeTranspilerConfig,
)
from .lod_tensor import create_lod_tensor, create_random_int_lodtensor
from . import profiler
from . import unique_name
from . import parallel_executor
from .parallel_executor import *
from . import compiler
from .compiler import *
from paddle.fluid.layers.math_op_patch import monkey_patch_variable
from .dygraph.layers import *
from .dygraph.base import enable_dygraph, disable_dygraph
from .io import save, load, load_program_state, set_program_state
from .dygraph.varbase_patch_methods import monkey_patch_varbase
from .core import _cuda_synchronize
from .trainer_desc import (
    TrainerDesc,
    DistMultiTrainer,
    PipelineTrainer,
    HeterPipelineTrainer,
    MultiTrainer,
    HeterXpuTrainer,
)
from .transpiler import HashName, RoundRobin
from .backward import append_backward

Tensor = LoDTensor
enable_imperative = enable_dygraph
disable_imperative = disable_dygraph

__all__ = (
    framework.__all__
    + executor.__all__
    + trainer_desc.__all__
    + transpiler.__all__
    + parallel_executor.__all__
    + lod_tensor.__all__
    + data_feed_desc.__all__
    + compiler.__all__
    + backward.__all__
    + [
        'io',
        'initializer',
        'layers',
        'contrib',
        'data',
        'dygraph',
        'enable_dygraph',
        'disable_dygraph',
        'enable_imperative',
        'disable_imperative',
        'transpiler',
        'nets',
        'optimizer',
        'backward',
        'regularizer',
        'LoDTensor',
        'LoDTensorArray',
        'CPUPlace',
        'XPUPlace',
        'CUDAPlace',
        'CUDAPinnedPlace',
        'NPUPlace',
        'IPUPlace',
        'MLUPlace',
        'Tensor',
        'ParamAttr',
        'WeightNormParamAttr',
        'DataFeeder',
        'profiler',
        'unique_name',
        'Scope',
        'save',
        'load',
        '_cuda_synchronize',
    ]
)


def __bootstrap__():
    """
    Enable reading gflags from environment variables.

    Returns:
        None
    """
    import sys
    import os
    import platform
    from . import core

    # NOTE(zhiqiu): When (1)numpy < 1.19; (2) python < 3.7,
    # unittest is always imported in numpy (maybe some versions not).
    # so is_test is True and p2p is not inited.
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
            file=sys.stderr,
        )
        print('PLEASE USE OMP_NUM_THREADS WISELY.', file=sys.stderr)

    os.environ['OMP_NUM_THREADS'] = str(num_threads)

    flag_prefix = "FLAGS_"
    read_env_flags = [
        key[len(flag_prefix) :]
        for key in core.globals().keys()
        if key.startswith(flag_prefix)
    ]

    def remove_flag_if_exists(name):
        if name in read_env_flags:
            read_env_flags.remove(name)

    sysstr = platform.system()
    if 'Darwin' in sysstr:
        remove_flag_if_exists('use_pinned_memory')

    if os.name == 'nt':
        remove_flag_if_exists('cpu_deterministic')

    if core.is_compiled_with_ipu():
        # Currently we request all ipu available for training and testing
        #   finer control of pod of IPUs will be added later
        read_env_flags += []

    core.init_gflags(["--tryfromenv=" + ",".join(read_env_flags)])
    # Note(zhouwei25): sys may not have argv in some cases,
    # Such as: use Python/C API to call Python from C++
    try:
        core.init_glog(sys.argv[0])
    except Exception:
        sys.argv = [""]
        core.init_glog(sys.argv[0])
    # don't init_p2p when in unittest to save time.
    core.init_devices()
    core.eager._init_eager_and_static_tensor_operants()
    core.init_default_kernel_signatures()
    core.init_memory_method()


# TODO(panyx0718): Avoid doing complex initialization logic in __init__.py.
# Consider paddle.init(args) or paddle.main(args)
monkey_patch_variable()
__bootstrap__()
monkey_patch_varbase()

# NOTE(zhiqiu): register npu_finalize on the exit of Python,
# do some clean up manually.
if core.is_compiled_with_npu():
    atexit.register(core.npu_finalize)
# NOTE(Aurelius84): clean up ExecutorCacheInfo in advance manually.
atexit.register(core.clear_executor_cache)

# NOTE(Aganlengzi): clean up KernelFactory in advance manually.
# NOTE(wangran16): clean up DeviceManger in advance manually.
# Keep clear_kernel_factory running before clear_device_manager
atexit.register(core.clear_device_manager)
atexit.register(core.clear_kernel_factory)
