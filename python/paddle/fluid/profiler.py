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

import core
from contextlib import contextmanager
import os

__all__ = [
    'cuda_profiler', 'reset_profiler', 'profiler', 'start_profiler',
    'stop_profiler'
]

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


def reset_profiler():
    """The profiler clear interface.
    reset_profiler will clear the previous time record.
    """
    core.reset_profiler()


def start_profiler(state):
    """Enable the profiler.

    Args:
        state (string) : The profiling state, which should be 'CPU', 'GPU'
            or 'All'. 'CPU' means only profile CPU. 'GPU' means profiling
            GPU as well. 'All' also generates timeline.
    """
    if core.is_profiler_enabled():
        return
    if state not in ['CPU', 'GPU', "All"]:
        raise ValueError("The state must be 'CPU' or 'GPU' or 'All'.")
    if state == "GPU":
        prof_state = core.ProfilerState.kCUDA
    elif state == "CPU":
        prof_state = core.ProfilerState.kCPU
    else:
        prof_state = core.ProfilerState.kAll
    core.enable_profiler(prof_state)


def stop_profiler(sorted_key=None, profile_path='/tmp/profile'):
    """Stop the profiler.

    Args:
        sorted_key (string) : If None, the profiling results will be printed
            in the order of first end time of events. Otherwise, the profiling
            results will be sorted by the this flag. This flag should be one
            of 'calls', 'total', 'max', 'min' or 'ave'.
            The `calls` means sorting by the number of calls.
            The `total` means sorting by the total execution time.
            The `max` means sorting by the maximum execution time.
            The `min` means sorting by the minimum execution time.
            The `ave` means sorting by the average execution time.
        profile_path (string) : If state == 'All', it will write a profile
            proto output file.
    """
    if not core.is_profiler_enabled():
        return
    sorted_key = 'default' if sorted_key is None else sorted_key
    if sorted_key not in ['default', 'calls', 'total', 'max', 'min', 'ave']:
        raise ValueError("The sorted_key must be None or in 'calls', 'total', "
                         "'max', 'min' and 'ave'")
    key_map = {
        'default': core.EventSortingKey.kDefault,
        'calls': core.EventSortingKey.kCalls,
        'total': core.EventSortingKey.kTotal,
        'max': core.EventSortingKey.kMax,
        'min': core.EventSortingKey.kMin,
        'ave': core.EventSortingKey.kAve,
    }
    # TODO(qingqing) : redirect C++ ostream to Python stream.
    # with core.ostream_redirect(stdout=True, stderr=True):
    core.disable_profiler(key_map[sorted_key], profile_path)


@contextmanager
def profiler(state, sorted_key=None, profile_path='/tmp/profile'):
    """The profiler interface.
    Different from cuda_profiler, this profiler can be used to profile both CPU
    and GPU program. By defalut, it records the CPU and GPU operator kernels,
    if you want to profile other program, you can refer the profiling tutorial
    to add more records.

    Args:
        state (string) : The profiling state, which should be 'CPU' or 'GPU',
            telling the profiler to use CPU timer or GPU timer for profiling.
            Although users may have already specified the execution place
            (CPUPlace/CUDAPlace) in the begining, for flexibility the profiler
            would not inherit this place.
        sorted_key (string) : If None, the profiling results will be printed
            in the order of first end time of events. Otherwise, the profiling
            results will be sorted by the this flag. This flag should be one
            of 'calls', 'total', 'max', 'min' or 'ave'.
            The `calls` means sorting by the number of calls.
            The `total` means sorting by the total execution time.
            The `max` means sorting by the maximum execution time.
            The `min` means sorting by the minimum execution time.
            The `ave` means sorting by the average execution time.
        profile_path (string) : If state == 'All', it will write a profile
            proto output file.
    """
    start_profiler(state)
    yield
    stop_profiler(sorted_key, profile_path)
