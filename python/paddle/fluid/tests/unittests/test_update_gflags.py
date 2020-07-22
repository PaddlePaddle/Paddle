#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

import unittest
import numpy as np
from op_test import OpTest
import paddle
import paddle.fluid as fluid
import os
from paddle.fluid import core
import platform


class TEST_Update_Gflags(unittest.TestCase):
    def test_out(self):
        sysstr = platform.system()
        read_env_flags = [
            'check_nan_inf', 'fast_check_nan_inf', 'benchmark',
            'eager_delete_scope', 'fraction_of_cpu_memory_to_use',
            'initial_cpu_memory_in_mb', 'init_allocated_mem',
            'paddle_num_threads', 'dist_threadpool_size',
            'eager_delete_tensor_gb', 'fast_eager_deletion_mode',
            'memory_fraction_of_eager_deletion', 'allocator_strategy',
            'reader_queue_speed_test_mode', 'print_sub_graph_dir',
            'pe_profile_fname', 'inner_op_parallelism', 'enable_parallel_graph',
            'fuse_parameter_groups_size', 'multiple_of_cupti_buffer_size',
            'fuse_parameter_memory_size', 'tracer_profile_fname',
            'dygraph_debug', 'use_system_allocator', 'enable_unused_var_check',
            'free_idle_chunk', 'free_when_no_cache_hit'
        ]
        if 'Darwin' not in sysstr:
            read_env_flags.append('use_pinned_memory')
        if os.name != 'nt':
            read_env_flags.append('cpu_deterministic')

        if core.is_compiled_with_mkldnn():
            read_env_flags.append('use_mkldnn')

        if core.is_compiled_with_dist():
            #env for rpc
            read_env_flags.append('rpc_deadline')
            read_env_flags.append('rpc_retry_times')
            read_env_flags.append('rpc_server_profile_path')
            read_env_flags.append('enable_rpc_profiler')
            read_env_flags.append('rpc_send_thread_num')
            read_env_flags.append('rpc_get_thread_num')
            read_env_flags.append('rpc_prefetch_thread_num')
            read_env_flags.append('rpc_disable_reuse_port')
            read_env_flags.append('rpc_retry_bind_port')

            read_env_flags.append('worker_update_interval_secs')

            if core.is_compiled_with_brpc():
                read_env_flags.append('max_body_size')
                #set brpc max body size
                os.environ['FLAGS_max_body_size'] = "2147483647"

        if core.is_compiled_with_cuda():
            read_env_flags += [
                'fraction_of_gpu_memory_to_use', 'initial_gpu_memory_in_mb',
                'reallocate_gpu_memory_in_mb', 'cudnn_deterministic',
                'enable_cublas_tensor_op_math', 'conv_workspace_size_limit',
                'cudnn_exhaustive_search', 'selected_gpus',
                'sync_nccl_allreduce', 'cudnn_batchnorm_spatial_persistent',
                'gpu_allocator_retry_time', 'local_exe_sub_scope_limit',
                'gpu_memory_limit_mb'
            ]
        origin_value = os.getenv("FLAGS_fraction_of_gpu_memory_to_use")
        for i in range(100):
            os.environ['FLAGS_fraction_of_gpu_memory_to_use'] = str(i / 100.0)
            test_result = core.init_gflags(
                ["--tryfromenv=" + ",".join(read_env_flags)])
            self.assertTrue(test_result)
        if origin_value is not None:
            os.environ['FLAGS_fraction_of_gpu_memory_to_use'] = origin_value
            test_result = core.init_gflags(
                ["--tryfromenv=" + ",".join(read_env_flags)])
            self.assertTrue(test_result)


if __name__ == "__main__":
    unittest.main()
