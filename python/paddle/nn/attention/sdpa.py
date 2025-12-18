#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

from enum import IntEnum
from typing import TYPE_CHECKING

import paddle
from paddle.base.wrapped_decorator import signature_safe_contextmanager

if TYPE_CHECKING:
    from collections.abc import Iterable


class SDPBackend(IntEnum):
    """
    An enum-like class that contains the different backends for scaled dot product attention.
    This backend class is designed to be used with the sdpa_kernel context manager.

    The following Enums are available:
        - ERROR: An error occurred when trying to determine the backend.
        - MATH: The math backend for scaled dot product attention.
        - FLASH_ATTENTION: The flash attention backend for scaled dot product attention.
        - EFFICIENT_ATTENTION: The efficient attention backend for scaled dot product attention.

    See :func:`paddle.nn.attention.sdpa_kernel` for more details.

    .. warning:: This class is in beta and subject to change.
    """

    ERROR = -1
    MATH = 0
    FLASH_ATTENTION = 1
    EFFICIENT_ATTENTION = 2


_backend_enabled = {
    SDPBackend.MATH: True,
    SDPBackend.FLASH_ATTENTION: paddle.framework._global_flags().get(
        "FLAGS_flash_attn_available", False
    ),
    SDPBackend.EFFICIENT_ATTENTION: paddle.framework._global_flags().get(
        "FLAGS_mem_efficient_attn_available", False
    ),
}
_current_priority = [
    SDPBackend.FLASH_ATTENTION,
    SDPBackend.EFFICIENT_ATTENTION,
    SDPBackend.MATH,
]


def _get_enabled_backends():
    global _backend_enabled
    return [backend for backend, enabled in _backend_enabled.items() if enabled]


def _set_enabled_backends(backends: list[SDPBackend]):
    global _backend_enabled
    for backend in _backend_enabled:
        _backend_enabled[backend] = False
    for backend in backends:
        if backend in _backend_enabled:
            _backend_enabled[backend] = True


def _get_backend_priority():
    global _current_priority
    return _current_priority.copy()


def _set_backend_priority(priority: list[SDPBackend]):
    global _current_priority
    _current_priority = priority.copy()


def _validate_backends(backends):
    if isinstance(backends, SDPBackend):
        backends = [backends]

    if not isinstance(backends, (list, tuple)):
        raise TypeError(
            "backends must be an instance of SDPBackend or a list of SDPBackend instances"
        )

    for backend in backends:
        if not isinstance(backend, SDPBackend):
            raise TypeError(
                f"All backends must be SDPBackend instances, got {type(backend)}"
            )

    return list(dict.fromkeys(backends))


def _cur_sdpa_kernel_backends(with_priority: bool = False):
    backends = _get_enabled_backends()

    if with_priority:
        curr_priority = _get_backend_priority()
        backends = sorted(
            backends,
            key=lambda backend: curr_priority.index(backend)
            if backend in curr_priority
            else float('inf'),
        )

    return backends


def _sdpa_kernel(backends: Iterable[SDPBackend], set_priority: bool = False):
    _set_enabled_backends(list(backends))

    if set_priority:
        user_priority = list(backends)
        previous_priority = _get_backend_priority()

        for backend in previous_priority:
            if backend not in user_priority:
                user_priority.append(backend)

        _set_backend_priority(user_priority)


@signature_safe_contextmanager
def sdpa_kernel(
    backends: list[SDPBackend] | SDPBackend, set_priority: bool = False
):
    """
    Context manager to select which backend to use for scaled dot product attention.

    .. warning:: This function is beta and subject to change.

    Args:
        backends (Union[list[SDPBackend], SDPBackend]): A backend or list of backends
            for scaled dot product attention.
        set_priority (bool, optional): Whether the ordering of the backends is
            interpreted as their priority order. Default: False.

    Example:

        >>> import paddle
        >>> from paddle.nn.functional import scaled_dot_product_attention
        >>> from paddle.nn.attention import SDPBackend, sdpa_kernel

        >>> # Create dummy tensors
        >>> query = paddle.rand(shape=[2, 4, 8, 16])
        >>> key = paddle.rand(shape=[2, 4, 8, 16])
        >>> value = paddle.rand(shape=[2, 4, 8, 16])
        >>> # Example 1: Only enable math backend
        >>> with sdpa_kernel(SDPBackend.MATH):
        ...     out = scaled_dot_product_attention(query, key, value)
        >>> print(out.shape)
        [2, 4, 8, 16]
        >>> # Example 2: Enable multiple backends
        >>> with sdpa_kernel([SDPBackend.MATH, SDPBackend.EFFICIENT_ATTENTION]):
        ...     out = scaled_dot_product_attention(query, key, value)
        >>> print(out.shape)
        [2, 4, 8, 16]
        >>> # Example 3: Set priority order for multiple backends
        >>> with sdpa_kernel(
        ...     [SDPBackend.MATH, SDPBackend.EFFICIENT_ATTENTION],
        ...     set_priority=True,
        ... ):
        ...     out = scaled_dot_product_attention(query, key, value)
        >>> print(out.shape)
        [2, 4, 8, 16]
        >>> # doctest: +SKIP('FlashAttention may not be available in all environments')
        >>> # Example 4: Flash attention (skipped due to environment requirements)
        >>> with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        ...     out = scaled_dot_product_attention(query, key, value)
        >>> # doctest: -SKIP

    This context manager can be used to select which backend to use for scaled dot product attention.
    Upon exiting the context manager, the previous state of the flags will be restored.
    """
    assert isinstance(backends, (list, SDPBackend)), (
        "Backend must be an instance of SDPBackend or a list of SDPBackend instances"
    )
    backends = _validate_backends(backends)

    if not backends:
        raise ValueError("At least one backend must be specified")

    previous_backends = _cur_sdpa_kernel_backends(with_priority=set_priority)
    try:
        _sdpa_kernel(backends, set_priority)

        yield {}

    finally:
        _sdpa_kernel(previous_backends, set_priority)
