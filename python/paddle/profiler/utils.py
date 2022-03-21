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

from typing import Any
from warnings import warn
import functools
from contextlib import ContextDecorator

from paddle.fluid.core import (_RecordEvent, TracerEventType)
import paddle.fluid.core as core

_AllowedEventTypeList = [
    TracerEventType.Dataloader, TracerEventType.ProfileStep,
    TracerEventType.UserDefined, TracerEventType.Forward,
    TracerEventType.Backward, TracerEventType.Optimization,
    TracerEventType.PythonOp, TracerEventType.PythonUserDefined
]


class RecordEvent(ContextDecorator):
    r"""
    Interface for recording a time range.

    Parameters:
        name(str): Name of the record event

    Examples:
        .. code-block:: python

            import paddle
            import paddle.profiler as profiler
            # method1: using context manager
            with profiler.RecordEvent("record_add"):
                data1 = paddle.randn(shape=[3])
                data2 = paddle.randn(shape=[3])
                result = data1 + data2
            # method2: call begin() and end()
            record_event = profiler.RecordEvent("record_add")
            record_event.begin()
            data1 = paddle.randn(shape=[3])
            data2 = paddle.randn(shape=[3])
            result = data1 + data2
            record_event.end()

    Note:
        RecordEvent will take effect only when profiler is on and at the state of RECORD.
    """

    def __init__(self,
                 name: str,
                 event_type: TracerEventType=TracerEventType.UserDefined):
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
        Record the time of begining.

        .. code-block:: python

            import paddle
            import paddle.profiler as profiler
            record_event = profiler.RecordEvent("record_sub")
            record_event.begin()
            data1 = paddle.randn(shape=[3])
            data2 = paddle.randn(shape=[3])
            result = data1 - data2
            record_event.end()
        """
        if self.event_type not in _AllowedEventTypeList:
            warn("Only TracerEvent Type in [{}, {}, {}, {}, {}, {},{}]\
                  can be recorded.".format(*_AllowedEventTypeList))
            self.event = None
        else:
            if self.event_type == TracerEventType.UserDefined:
                self.event_type == TracerEventType.PythonUserDefined
            self.event = _RecordEvent(self.name, self.event_type)

    def end(self):
        r'''
        Record the time of ending.

        .. code-block:: python

            import paddle
            import paddle.profiler as profiler
            record_event = profiler.RecordEvent("record_mul")
            record_event.begin()
            data1 = paddle.randn(shape=[3])
            data2 = paddle.randn(shape=[3])
            result = data1 * data2
            record_event.end()
        '''
        if self.event:
            self.event.end()


def load_profiler_result(filename: str):
    r"""
    Load dumped profiler data back to memory.

    Parameters:
        filename(str): Name of the exported protobuf file of profiler data.

    Returns:
        ProfilerResult object.

    Examples:
        .. code-block:: python

            # required: gpu
            import paddle.profiler as profiler
            with profiler.Profiler(
                    targets=[profiler.ProfilerTarget.CPU, profiler.ProfilerTarget.GPU],
                    scheduler = (3, 10)) as p:
                for iter in range(10):
                    #train()
                    p.step()
            p.export('test_export_protobuf.pb', format='pb')
            profiler_result = profiler.load_profiler_result('test_export_protobuf.pb')
    """
    return core.load_profiler_result(filename)


def wrap_optimizers():
    def optimizer_warpper(func):
        @functools.wraps(func)
        def warpper(*args, **kwargs):
            with RecordEvent(
                    'Optimization Step',
                    event_type=TracerEventType.Optimization):
                return func(*args, **kwargs)

        return warpper

    import paddle.optimizer as optimizer
    for classname in optimizer.__all__:
        if classname != 'Optimizer':
            classobject = getattr(optimizer, classname)
            if getattr(classobject, 'step', None) != None:
                classobject.step = optimizer_warpper(classobject.step)
