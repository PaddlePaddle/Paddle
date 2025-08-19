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

_core_manager = core._ZeroFragmentationAllocatorManager


class ZeroFragmentationAllocatorManager:
    """Manager for zero-fragmentation memory allocator operations.

    This class provides methods to control the behavior of the zero-fragmentation
    memory allocator in PaddlePaddle, which helps reduce memory fragmentation
    during tensor allocation.

    Example:
        >>> # Enable zero-fragmentation allocator
        >>> ZeroFragmentationAllocatorManager.enable()
        >>>
        >>> # Allocate a buffer with zero fragmentation
        >>> ZeroFragmentationAllocatorManager.allocate_buffer(1024)
        >>>
        >>> # Use context manager for temporary zero-fragmentation mode
        >>> with ZeroFragmentationAllocatorManager.enter_zero_fragmentation_mode():
        ...     # Your memory intensive operations here
        ...     pass
    """

    @staticmethod
    def disable() -> None:
        """Disable the zero-fragmentation memory allocator."""
        _core_manager._disable()

    @staticmethod
    def enable() -> None:
        """Enable the zero-fragmentation memory allocator."""
        _core_manager._enable()

    @staticmethod
    def allocate_buffer(size: int) -> None:
        """Allocate a buffer using the monotonic allocator with preallocation.

        Args:
            size: Size of the buffer to allocate in bytes.
        """
        if size < 0:
            raise ValueError(f"Buffer size must be positive, got {size}")

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
    def enter_zero_fragmentation_mode() -> Generator[None, Any, None]:
        """Context manager for temporarily enabling the monotonic allocator."""
        _core_manager._enter_zero_fragmentation_mode()
        try:
            yield
        finally:
            _core_manager._exit_zero_fragmentation_mode()

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
        """Context manager for preallocation mode.

        This handles enabling/disabling the allocator and entering/exiting
        preallocation mode with proper cleanup.
        """
        was_enabled = _core_manager._is_enabled()
        if not was_enabled:
            _core_manager._enable()
        _core_manager._enter_zero_fragmentation_mode()
        _core_manager._begin_preallocation()
        try:
            yield
        finally:
            _core_manager._end_preallocation()
            _core_manager._exit_zero_fragmentation_mode()
            if not was_enabled:
                _core_manager._disable()
