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
from collections.abc import Generator
from typing import Any

import paddle
from paddle.base import core


class ZeroFragmentationAllocatorManager:
    """Manager for zero-fragmentation memory allocator operations.

    This class provides static methods to control the behavior of the
    zero-fragmentation memory allocator in PaddlePaddle.
    """

    _core_manager = core._ZeroFragmentationAllocatorManager

    @staticmethod
    def allocate_buffer(size: int) -> None:
        """Allocate a buffer using the monotonic allocator with preallocation.

        Args:
            size: Size of the buffer to allocate in bytes.
        """
        with ZeroFragmentationAllocatorManager._prealloc_context():
            size = min(
                paddle.core.allocator_get_max_free_block_size(
                    paddle.framework._current_expected_place()
                ),
                size,
            )
            ZeroFragmentationAllocatorManager._allocate(size)

    @staticmethod
    @contextlib.contextmanager
    def zero_fragmentation_allocator_context() -> Generator[None, Any, None]:
        """Context manager for temporarily enabling the monotonic allocator."""
        ZeroFragmentationAllocatorManager._core_manager._enable()
        try:
            yield
        finally:
            ZeroFragmentationAllocatorManager._core_manager._disable()

    @staticmethod
    def _allocate(size: int) -> paddle.Tensor:
        """Internal method to allocate memory buffer.

        Args:
            size: Size of the buffer to allocate in bytes.

        Returns:
            A paddle.Tensor of uint8 type with the requested size.
        """
        paddle.empty(shape=[size], dtype='uint8')

    @staticmethod
    @contextlib.contextmanager
    def _prealloc_context() -> Generator[None, Any, None]:
        """Context manager for preallocation mode."""
        ZeroFragmentationAllocatorManager._core_manager._enable()
        ZeroFragmentationAllocatorManager._core_manager._enable_prealloc()
        try:
            yield
        finally:
            ZeroFragmentationAllocatorManager._core_manager._disable_prealloc()
            ZeroFragmentationAllocatorManager._core_manager._disable()
