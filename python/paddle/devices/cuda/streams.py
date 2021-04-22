# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

from paddle.fluid import core_avx, core
from .. import cuda


class Stream(core.CUDAStream):
    def __init__(self, device, priority=None):

        if priority is None:
            priority = 2
        assert priority == 1 or priority == 2, "priority must be 1(high) or 2(normal)"
        super(Stream, self).__init__(device, core_avx.Priority(priority))

    def wait_event(self, event):
        super(Stream, self).wait_event(event)

    def wait_stream(self, stream):
        self.wait_event(stream.record_event())

    def record_event(self, event=None):
        if event is None:
            event = Event()
        event.record(self)
        return event

    def query(self):
        return super(Stream, self).query()

    def synchronize(self):
        super(Stream, self).synchronize()


class Event(core.CUDAEvent):
    def __init__(self, enable_timing=False, blocking=False, interprocess=False):
        flags = core_avx._get_cuda_flags(enable_timing, blocking, interprocess)
        super(Event, self).__init__(flags)
        pass

    def record(self, stream=None):
        if stream is None:
            stream = cuda.current_stream()
        super(Event, self).record(stream)

    def query(self):
        super(Event, self).query()

    def synchronize(self):
        super(Event, self).Synchronize()
