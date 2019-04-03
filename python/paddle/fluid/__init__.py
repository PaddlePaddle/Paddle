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
import os
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

from . import async_executor
from .async_executor import *

from . import trainer_desc
from . import inferencer

from . import io
from . import evaluator
from . import initializer
from . import layers
from . import dygraph
from . import contrib
from . import nets
from . import optimizer
from . import backward
from . import regularizer
from . import average
from . import metrics
from . import transpiler
from . import incubate
from . import distribute_lookup_table
from .param_attr import ParamAttr, WeightNormParamAttr
from .data_feeder import DataFeeder
from .core import LoDTensor, LoDTensorArray, CPUPlace, CUDAPlace, CUDAPinnedPlace, Scope, _Scope
from .incubate import fleet
from .incubate import data_generator
from .transpiler import DistributeTranspiler, \
    memory_optimize, release_memory, DistributeTranspilerConfig
from .lod_tensor import create_lod_tensor, create_random_int_lodtensor
from . import clip
from . import profiler
from . import unique_name
from . import recordio_writer
from . import parallel_executor
from .parallel_executor import *
from . import compiler
from .compiler import *
from paddle.fluid.layers.math_op_patch import monkey_patch_variable
from . import install_check

Tensor = LoDTensor

__all__ = framework.__all__ + executor.__all__ + \
    trainer_desc.__all__ + inferencer.__all__ + transpiler.__all__ + \
    parallel_executor.__all__ + lod_tensor.__all__ + \
    data_feed_desc.__all__ + async_executor.__all__ + compiler.__all__ + [
        'io',
        'initializer',
        'layers',
        'contrib',
        'dygraph',
        'transpiler',
        'nets',
        'optimizer',
        'learning_rate_decay',
        'backward',
        'regularizer',
        'LoDTensor',
        'LoDTensorArray',
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
        'install_check',
    ]


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
    sysstr = platform.system()
    read_env_flags = [
        'check_nan_inf', 'benchmark', 'eager_delete_scope',
        'initial_cpu_memory_in_mb', 'init_allocated_mem', 'free_idle_memory',
        'paddle_num_threads', "dist_threadpool_size", 'eager_delete_tensor_gb',
        'fast_eager_deletion_mode', 'memory_fraction_of_eager_deletion',
        'allocator_strategy', 'reader_queue_speed_test_mode',
        'print_sub_graph_dir', 'pe_profile_fname', 'warpctc_dir',
        'inner_op_parallelism', 'enable_parallel_graph',
        'fuse_parameter_groups_size', 'multiple_of_cupti_buffer_size',
        'enable_subgraph_optimize', 'fuse_parameter_memory_size',
        'tracer_profile_fname'
    ]
    if 'Darwin' not in sysstr:
        read_env_flags.append('use_pinned_memory')

    if os.name != 'nt':
        read_env_flags.append('cpu_deterministic')

    if core.is_compiled_with_mkldnn():
        read_env_flags.append('use_mkldnn')

    if core.is_compiled_with_ngraph():
        read_env_flags.append('use_ngraph')

    if core.is_compiled_with_dist():
        #env for rpc
        read_env_flags.append('rpc_deadline')
        read_env_flags.append('rpc_server_profile_path')
        read_env_flags.append('enable_rpc_profiler')
        read_env_flags.append('rpc_send_thread_num')
        read_env_flags.append('rpc_get_thread_num')
        read_env_flags.append('rpc_prefetch_thread_num')
        read_env_flags.append('rpc_disable_reuse_port')

        # env for communicator
        read_env_flags.append('communicator_independent_recv_thread')
        read_env_flags.append('communicator_send_queue_size')
        read_env_flags.append('communicator_max_send_grad_num_before_recv')
        read_env_flags.append('communicator_thread_pool_size')
        read_env_flags.append('communicator_max_merge_var_num')
        read_env_flags.append('communicator_fake_rpc')
        if core.is_compiled_with_brpc():
            read_env_flags.append('max_body_size')
            #set brpc max body size
            os.environ['FLAGS_max_body_size'] = "2147483647"

    if core.is_compiled_with_cuda():
        read_env_flags += [
            'fraction_of_gpu_memory_to_use', 'initial_gpu_memory_in_mb',
            'reallocate_gpu_memory_in_mb', 'cudnn_deterministic',
            'enable_cublas_tensor_op_math', 'conv_workspace_size_limit',
            'cudnn_exhaustive_search', 'memory_optimize_debug', 'selected_gpus',
            'sync_nccl_allreduce', 'limit_of_tmp_allocation',
            'times_excess_than_required_tmp_allocation',
            'enable_inplace_whitelist', 'cudnn_batchnorm_spatial_persistent'
        ]
    core.init_gflags([sys.argv[0]] +
                     ["--tryfromenv=" + ",".join(read_env_flags)])
    core.init_glog(sys.argv[0])
    # don't init_p2p when in unittest to save time.
    core.init_devices(not in_test)


# TODO(panyx0718): Avoid doing complex initialization logic in __init__.py.
# Consider paddle.init(args) or paddle.main(args)
monkey_patch_variable()
__bootstrap__()
