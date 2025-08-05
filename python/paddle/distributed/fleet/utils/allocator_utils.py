# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

import contextlib

import paddle
from paddle.base import core


class MonotonicAllocatorManager:
    @staticmethod
    def enable():
        core._MonotonicAllocatorManager._enable()

    @staticmethod
    def disable():
        core._MonotonicAllocatorManager._disable()

    @staticmethod
    def allocate_buffer(size):
        place = paddle.framework._current_expected_place()
        # The framework's device_context is lazily inited. When manually calling allocate,
        # the device_context may not have finished initialization yet. Therefore, calling
        # paddle.empty([1]) triggers the initialization of device_context.
        paddle.empty([1])
        core._MonotonicAllocatorManager._allocate_buffer(place, size)

    @staticmethod
    def deallocate_buffer():
        place = paddle.framework._current_expected_place()
        core._MonotonicAllocatorManager._deallocate_buffer(place)

    @staticmethod
    def reset_buffer():
        place = paddle.framework._current_expected_place()
        core._MonotonicAllocatorManager._reset_buffer(place)

    @staticmethod
    @contextlib.contextmanager
    def switch_to_monotonic_allocator():
        MonotonicAllocatorManager.enable()
        try:
            yield
        finally:
            MonotonicAllocatorManager.disable()
