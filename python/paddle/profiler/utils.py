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

from paddle.fluid.core import (_RecordEvent, TracerEventType,
                               load_profiler_result)
from typing import Any
from warnings import warn
import functools
from contextlib import ContextDecorator

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
    event_type(TracerEventType): Type of the record event, can be used for statistics.

    Examples:
        .. code-block:: python
        import paddle.profiler as profiler
        with profiler.RecordEvent(name='op1', event_type=TracerEventType=TracerEventType.UserDefined):
            op1()
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
        if self.event_type not in _AllowedEventTypeList:
            warn("Only TracerEvent Type in [{}, {}, {}, {}, {}, {},{}]\
                  can be recorded.".format(*_AllowedEventTypeList))
            self.event = None
        else:
            if self.event_type == TracerEventType.UserDefined:
                self.event_type == TracerEventType.PythonUserDefined
            self.event = _RecordEvent(self.name, self.event_type)

    def end(self):
        if self.event:
            self.event.end()


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
