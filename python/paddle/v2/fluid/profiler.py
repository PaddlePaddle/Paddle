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


def reset_profiler():
    core.reset_profiler()


@contextmanager
def profiler(state, sorted_key=None):
    """The profiler interface.
    Different from cuda_profiler, this fuction can be used to profile both CPU
    and GPU program.

    Args:
        state (string) : The profiler state, It should be 'CPU' or 'GPU'.
        sorted_key (string) : If None, the profiler results will be printed
            without sorting. Otherwise, the profiler results will be sorted
            by the this flag. This flag should be one of 'calls', 'total',
            'max', 'min' or 'ave'.
            The `calls` means sorting by the calling counter.
            The `total` means sorting by the total execution time.
            The `max` means sorting by the maximum execution time.
            The `min` means sorting by the minimum execution time.
            The `ave` means sorting by the average execution time.
    """

    if state not in ['CPU', 'GPU']:
        raise ValueError("The state must be 'CPU' or 'GPU'.")
    prof_state = core.ProfilerState.kCUDA if state == "GPU" else core.ProfilerState.kCPU
    core.enable_profiler(prof_state)
    yield

    if sorted_key not in ['calls', 'total', 'max', 'min', 'ave']:
        raise ValueError("The state must be in 'calls', 'total', "
                         "'max', 'min', 'ave'")
    sorted_key = 'default' if sorted_key is None else sorted_key
    key_map = {
        'default': core.EventSortingKey.kDefault,
        'calls': core.EventSortingKey.kCalls,
        'total': core.EventSortingKey.kTotal,
        'max': core.EventSortingKey.kMax,
        'min': core.EventSortingKey.kMin,
        'ave': core.EventSortingKey.kAve,
    }
    with core.ostream_redirect(stdout=True, stderr=True):
        core.disable_profiler(key_map[sorted_key])
