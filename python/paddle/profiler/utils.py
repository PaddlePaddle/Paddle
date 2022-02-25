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

from paddle.fluid.core import (_RecordEvent, TracerEventType)
from typing import Any
from warnings import warn
import functools
try:
    # Available in Python >= 3.2
    from contextlib import ContextDecorator
except ImportError:

    class ContextDecorator(object):
        def __enter__(self):
            raise NotImplementedError

        def __exit__(self, exc_type, exc_val, exc_tb):
            raise NotImplementedError

        def __call__(self, func):
            @functools.wraps(func)
            def wrapped(*args, **kwargs):
                with self:
                    return func(*args, **kwargs)

            return wrapped


_AllowedEventTypeList = [
    TracerEventType.Dataloader, TracerEventType.ProfileStep,
    TracerEventType.UserDefined, TracerEventType.Forward,
    TracerEventType.Backward, TracerEventType.Optimization,
    TracerEventType.PythonOp
]


class RecordEvent(ContextDecorator):
    '''
  Interface for recording a time range.
  Examples:
    .. code-block:: python
    import paddle.profiler as profiler
    with profiler.RecordEvent(name='op1'):
      op1()
  '''

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


def wrap_functional():
    def functional_warpper(func):
        @functools.wraps(func)
        def warpper(*args, **kwargs):
            with RecordEvent(
                    func.__name__, event_type=TracerEventType.PythonOp):
                return func(*args, **kwargs)

        return warpper

    import paddle.nn.functional as functional
    for funcname in functional.__all__:
        funcobject = getattr(functional, funcname)
        setattr(functional, funcname, functional_warpper(funcobject))
