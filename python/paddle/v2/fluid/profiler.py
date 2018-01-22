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

import paddle.v2.fluid.core as core
from contextlib import contextmanager
import os

__all__ = ['CudaProfiler']

NVPROF_CONFIG = [
    "gpustarttimestamp",
    "gpuendtimestamp",
    "gridsize3d",
    "threadblocksize",
    "streamid",
    "enableonstart 0",
    "conckerneltrace",
]


@contextmanager
def cuda_profiler(output_file, output_mode=None, config=None):
    """The CUDA profiler.
    This fuctions is used to profile CUDA program by CUDA runtime application
    programming interface. The profiling result will be written into
    `output_file` with Key-Value pair format or Comma separated values format.
    The user can set the output mode by `output_mode` argument and set the
    counters/options for profiling by `config` argument. The default config
    is ['gpustarttimestamp', 'gpustarttimestamp', 'gridsize3d',
    'threadblocksize', 'streamid', 'enableonstart 0', 'conckerneltrace'].

    Args:
        output_file (string) : The output file name, the result will be
            written into this file.
        output_mode (string) : The output mode has Key-Value pair format and
            Comma separated values format. It should be 'kvp' or 'csv'.
        config (list of string) : The profiler options and counters can refer
            to "Compute Command Line Profiler User Guide".
    """
    if output_mode is None:
        output_mode = 'csv'
    if output_mode not in ['kvp', 'csv']:
        raise ValueError("The output mode must be 'kvp' or 'csv'.")
    config = NVPROF_CONFIG if config is None else config
    config_file = 'nvprof_config_file'
    with open(config_file, 'wb') as fp:
        fp.writelines(["%s\n" % item for item in config])
    core.nvprof_init(output_file, output_mode, config_file)
    # Enables profiler collection by the active CUDA profiling tool.
    core.nvprof_start()
    yield
    # Disables profiler collection.
    core.nvprof_stop()
    os.remove(config_file)
