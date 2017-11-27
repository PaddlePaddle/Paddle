import paddle.v2.fluid.core as core


def nvporf_init(output_file, output_mode=None):
    """
    Initialize the CUDA profiler.
    This methods must be called before nvprof_start.

    :param output_file: The output file name.
    :type output_file: string
    :param output_mode: The output mode has Key-Value pair format and
                        Comma separated values format.
                        It should be 'key-value' or 'csv'.
    :type output_mode: string
    """
    if output_mode is None:
        output_mode = 'csv'
    if output_mode != 'key-value' or output_mode != 'csv':
        raise ValueError("The output mode must be 'key-value' or 'csv'.")
    core.nvprof_init(output_file, output_mode)


def nvporf_start():
    """
    Enables profiler collection by the active CUDA profiling tool.
    """
    core.nvprof_start()


def nvporf_stop():
    """
    Disables profiler collection.
    """
    core.nvprof_stop()


class profiler(object):
    def __init__(self, output_file, output_mode=None, enabled=True):
        self.enabled = enabled
        if not self.enabled:
            return
        self.entered = False
        nvporf_init(output_file, output_mode)

    def __enter__(self):
        if not self.enabled:
            return
        if self.entered:
            raise RuntimeError("The profiler traces are not reentrant")
        self.entered = True
        nvporf_start()
        return self

    def __exit__(self, exc_type, exc_value, tb):
        if exc_value is not None:
            raise exc_value
        if not self.enabled:
            return
        nvporf_stop()
