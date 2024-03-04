# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

import functools
import sys
from contextlib import ContextDecorator, contextmanager
from typing import Any
from warnings import warn

from paddle.base import core
from paddle.base.core import TracerEventType, _RecordEvent

_is_profiler_used = False
_has_optimizer_wrapped = False

_AllowedEventTypeList = [
    TracerEventType.Dataloader,
    TracerEventType.ProfileStep,
    TracerEventType.Forward,
    TracerEventType.Backward,
    TracerEventType.Optimization,
    TracerEventType.PythonOp,
    TracerEventType.PythonUserDefined,
]


class RecordEvent(ContextDecorator):
    r"""
    Interface for recording a time range by user defined.

    Args:
        name (str): Name of the record event.
        event_type (TracerEventType, optional): Optional, default value is
            `TracerEventType.PythonUserDefined`. It is reserved for internal
            purpose, and it is better not to specify this parameter.

    Examples:
        .. code-block:: python
            :name: code-example1

            >>> import paddle
            >>> import paddle.profiler as profiler
            >>> # method1: using context manager
            >>> paddle.seed(2023)
            >>> with profiler.RecordEvent("record_add"):
            ...     data1 = paddle.randn(shape=[3])
            ...     data2 = paddle.randn(shape=[3])
            ...     result = data1 + data2
            >>> # method2: call begin() and end()
            >>> record_event = profiler.RecordEvent("record_add")
            >>> record_event.begin()
            >>> data1 = paddle.randn(shape=[3])
            >>> data2 = paddle.randn(shape=[3])
            >>> result = data1 + data2
            >>> record_event.end()

    Note:
        RecordEvent will take effect only when :ref:`Profiler <api_paddle_profiler_Profiler>` is on and at the state of `RECORD`.
    """

    def __init__(
        self,
        name: str,
        event_type: TracerEventType = TracerEventType.PythonUserDefined,
    ):
        self.name = name
        self.event_type = event_type
        self.event = None

    def __enter__(self):
        self.begin()
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any):
        self.end()

    def begin(self):
        r"""
        Record the time of beginning.

        Examples:

            .. code-block:: python
                :name: code-example2

                >>> import paddle
                >>> import paddle.profiler as profiler
                >>> record_event = profiler.RecordEvent("record_sub")
                >>> record_event.begin()
                >>> paddle.seed(2023)
                >>> data1 = paddle.randn(shape=[3])
                >>> data2 = paddle.randn(shape=[3])
                >>> result = data1 - data2
                >>> record_event.end()
        """
        if not _is_profiler_used:
            return
        if self.event_type not in _AllowedEventTypeList:
            warn(
                "Only TracerEvent Type in [{}, {}, {}, {}, {}, {},{}]\
                  can be recorded.".format(
                    *_AllowedEventTypeList
                )
            )
            self.event = None
        else:
            self.event = _RecordEvent(self.name, self.event_type)

    def end(self):
        r"""
        Record the time of ending.

        Examples:

            .. code-block:: python
                :name: code-example3

                >>> import paddle
                >>> import paddle.profiler as profiler
                >>> record_event = profiler.RecordEvent("record_mul")
                >>> record_event.begin()
                >>> paddle.seed(2023)
                >>> data1 = paddle.randn(shape=[3])
                >>> data2 = paddle.randn(shape=[3])
                >>> result = data1 * data2
                >>> record_event.end()
        """
        if self.event:
            self.event.end()


def load_profiler_result(filename: str):
    r"""
    Load dumped profiler data back to memory.

    Args:
        filename(str): Name of the exported protobuf file of profiler data.

    Returns:
        ``ProfilerResult`` object, which stores profiling data.

    Examples:
        .. code-block:: python

            >>> # doctest: +REQUIRES(env:GPU)
            >>> import paddle.profiler as profiler
            >>> import paddle
            >>> paddle.device.set_device('gpu')
            >>> with profiler.Profiler(
            ...         targets=[profiler.ProfilerTarget.CPU, profiler.ProfilerTarget.GPU],
            ...         scheduler = (3, 10)) as p:
            ...     for iter in range(10):
            ...         #train()
            ...         p.step()
            >>> p.export('test_export_protobuf.pb', format='pb')
            >>> profiler_result = profiler.load_profiler_result('test_export_protobuf.pb')
    """
    return core.load_profiler_result(filename)


def in_profiler_mode():
    return _is_profiler_used


def wrap_optimizers():
    def optimizer_wrapper(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if in_profiler_mode():
                with RecordEvent(
                    'Optimization Step', event_type=TracerEventType.Optimization
                ):
                    return func(*args, **kwargs)
            else:
                return func(*args, **kwargs)

        return wrapper

    global _has_optimizer_wrapped
    if _has_optimizer_wrapped:
        return
    from paddle import optimizer

    for classname in optimizer.__all__:
        if classname != 'Optimizer':
            classobject = getattr(optimizer, classname)
            if getattr(classobject, 'step', None) is not None:
                classobject.step = optimizer_wrapper(classobject.step)
    _has_optimizer_wrapped = True


@contextmanager
def _nvprof_range(iter_id, start, end, exit_after_prof=True):
    """
    A range profiler interface (not public yet).
    Examples:
        .. code-block:: python

            >>> import paddle
            >>> model = Model()
            >>> for i in range(max_iter):
            ...     with paddle.profiler.utils._nvprof_range(i, 10, 20):
            ...         out = model(in)
    """
    if start >= end:
        yield
        return

    try:
        if iter_id == start:
            core.nvprof_start()
            core.nvprof_enable_record_event()
        if iter_id >= start:
            core.nvprof_nvtx_push(str(iter_id))
        yield
    finally:
        if iter_id < end:
            core.nvprof_nvtx_pop()
        if iter_id == end - 1:
            core.nvprof_stop()
            if exit_after_prof:
                sys.exit()


@contextmanager
def job_schedule_profiler_range(iter_id, start, end, exit_after_prof=True):
    if start >= end:
        yield False
        return

    try:
        if iter_id >= start and iter_id < end:
            yield True
        else:
            yield False
    finally:
        if iter_id == end - 1:
            if exit_after_prof:
                sys.exit()
