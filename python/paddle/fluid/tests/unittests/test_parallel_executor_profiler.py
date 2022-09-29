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

import unittest
import numpy as np
import paddle.fluid as fluid
import paddle.fluid.core as core
from paddle.fluid.tests.unittests.test_profiler import TestProfiler
import os

# NCCL 2.7 decides to use shared memory while NCCL 2.6 didn't, hence causing the error.
# include/shm.h:28 NCCL WARN Call to posix_fallocate failed: No space left on device
#
# Set environment variables NCCL_SHM_DISABLE=1 to disables the Shared Memory (SHM) transports
# and force to use P2P which is the default transports way of NCCL2.6.
os.environ['NCCL_SHM_DISABLE'] = str(1)


class TestPEProfiler(TestProfiler):

    def test_cpu_profiler(self):
        exe = fluid.Executor(fluid.CPUPlace())
        self.net_profiler(exe, 'CPU', "Default", use_parallel_executor=True)

    @unittest.skipIf(not core.is_compiled_with_cuda(),
                     "profiler is enabled only with GPU")
    def test_cuda_profiler(self):
        exe = fluid.Executor(fluid.CUDAPlace(0))
        self.net_profiler(exe, 'GPU', "OpDetail", use_parallel_executor=True)

    @unittest.skipIf(not core.is_compiled_with_cuda(),
                     "profiler is enabled only with GPU")
    def test_all_profiler(self):
        exe = fluid.Executor(fluid.CUDAPlace(0))
        self.net_profiler(exe, 'All', "AllOpDetail", use_parallel_executor=True)


if __name__ == '__main__':
    unittest.main()
