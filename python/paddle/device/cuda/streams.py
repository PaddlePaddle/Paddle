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
from __future__ import annotations

from paddle.base.core import CUDAEvent as Event, CUDAPlace, CUDAStream as Stream


def create_stream(
    device_id: CUDAPlace | int | None = None,
    priority: int = 2,
    device_type: str | None = None,  # Ignored for compatibility
    blocking: bool = False,  # Ignored for compatibility
):
    """
    Factory Function, used to create CUDA Stream
    """
    return Stream(device_id, priority)


def create_event(
    enable_timing: bool = False,
    blocking: bool = False,
    interprocess: bool = False,
    device_type: str | None = None,
    device_id: int = 0,
):
    """
    Factory Function, used to create CUDA Event
    """
    return Event(enable_timing, blocking, interprocess)


# from paddle.base.core import CUDAStream, CUDAEvent
# from typing import Optional, Union
# import ctypes
# import paddle

# class Event:
#     """Unified Event interface for CUDA devices."""

#     def __init__(
#         self,
#         enable_timing: bool = False,
#         blocking: bool = False,
#         interprocess: bool = False,
#         device_type: str = None,
#         device_id: int = 0,
#     ):
#         self._event = CUDAEvent(enable_timing, blocking, interprocess)

#     def record(self, stream: Stream | None = None) -> None:
#         self._event.record(stream._stream)


#     def query(self) -> bool:
#         return self._event.query()

#     def elapsed_time(self, end_event:Event) -> int:
#         """Calculate elapsed time between two events."""
#         return self._event.elapsed_time(end_event._event)

#     def synchronize(self) -> None:
#         """Wait for the event to occur."""
#         self._event.synchronize()


#     def __repr__(self):
#         return self._event

# class Stream:
#     """Unified Stream interface for CUDA devices."""

#     def __init__(
#         self,
#         device_id: Optional[Union[paddle.CUDAPlace, int]] = None,
#         priority: int = 2,
#         device_type: str = None, # Ignored for compatibility
#         blocking: bool = False,  # Ignored for compatibility
#     ):
#         self._stream = CUDAStream(device_id, priority)
#         self.device_id = device_id

#     def wait_event(self, event: Event) -> None:
#         self._stream.wait_event(event._event)

#     def wait_stream(self, stream: Stream) -> None:
#         self._stream.wait_stream(stream._stream)

#     def record_event(self, event: Event | None = None)  -> Event:
#         event.record(self)
#         return event

#     def query(self) -> bool:
#         return self._stream.query()

#     def synchronize(self) -> None:
#         self._stream.synchronize()

#     @property
#     def _as_parameter_(self):
#         return ctypes.c_void_p(self._stream.cuda_stream)

#     def __eq__(self, o: Stream | None) -> bool:
#         if isinstance(o, Stream):
#             return super().__eq__(o)
#         return False

#     def __hash__(self) -> int:
#         return hash((self._stream, self.device_id))

#     def __repr__(self) -> str:
#         return f'<paddle.device.cuda.Stream device={self.device} stream={self._as_parameter_.value:#x}>'
