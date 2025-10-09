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

from __future__ import annotations

from typing import TYPE_CHECKING, Union

from typing_extensions import TypeAlias

from paddle.base import core

from .custom_streams import (  # noqa: F401
    Event,
    Stream,
    create_event,
    create_stream,
)

if TYPE_CHECKING:
    from paddle import CPUPlace

    _CPUPlaceLike: TypeAlias = Union[
        CPUPlace,
        str,  # some string like "iluvatar_gpu" "metax_gpu:0", etc.
        int,  # some int like 0, 1, etc.
    ]


def device_count() -> int:
    '''
    Return the number of GPUs available.

    Returns:
        int: the number of GPUs available.

    Note:
        This function returns 0 when compiled without CUDA support.

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> paddle.device.device_count()

    '''
    return 0


def get_rng_state(
    device: _CPUPlaceLike | None = None,
) -> core.GeneratorState:
    r'''
    Get the random state for the default generator.

    Returns:
        Tensor: The random state tensor.

    Examples:

        .. code-block:: python

            >>> # doctest: +REQUIRES(env:CUSTOM_DEVICE)
            >>> import paddle
            >>> paddle.device.get_rng_state()

    '''
    return core.default_cpu_generator().get_state()


def set_rng_state(
    new_state: core.GeneratorState, device: _CPUPlaceLike | None = None
) -> None:
    """
    Set the random number generator state of the specified device.

    Args:
        new_state (core.GeneratorState): The desired RNG state to set.
            This should be a state object previously obtained from ``get_rng_state()``.
        device (DeviceLike, optional): The device to set the RNG state for.
            If not specified, uses the current default device (as returned by ``paddle.framework._current_expected_place_()``).
            Can be a device object, integer device ID, or device string.

    Returns:
        None

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> # Save RNG state
            >>> state = paddle.device.get_rng_state()
            >>> # Do some random operations
            >>> x = paddle.randn([2, 3])
            >>> # Restore RNG state
            >>> paddle.device.set_rng_state(state)
    """
    core.default_cpu_generator().set_state(new_state)


def manual_seed(seed: int) -> None:
    r"""Set the seed for generating random numbers for the current Device.

    .. warning::
        If you are working with a multi-Device model, this function is insufficient
        to get determinism.  To seed all Devices, use :func:`manual_seed_all`.

    Sets the seed for global default generator, which manages the random number generation.

    Args:
        seed(int): The random seed to set.

    Returns:
        None

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> paddle.device.manual_seed(102)
            >>> # paddle.cuda.manual_seed(102) is equivalent to paddle.device.manual_seed(102)
            >>> paddle.cuda.manual_seed(102)

    """
    seed = int(seed)
    core.default_cpu_generator().manual_seed(seed)


def max_memory_allocated(device: _CPUPlaceLike | None = None) -> int:
    r"""
    The API max_memory_allocated is not supported in CPU PaddlePaddle.
    Please reinstall PaddlePaddle with GPU or XPU support to call this API.
    """
    raise ValueError(
        "The API paddle.device.max_memory_allocated is not supported in CPU PaddlePaddle. "
        "Please reinstall PaddlePaddle with GPU or XPU support to call this API."
    )


def max_memory_reserved(device: _CPUPlaceLike | None = None) -> int:
    r"""
    The API max_memory_reserved is not supported in CPU PaddlePaddle.
    Please reinstall PaddlePaddle with GPU or XPU support to call this API.
    """
    raise ValueError(
        "The API paddle.device.max_memory_reserved is not supported in CPU PaddlePaddle. "
        "Please reinstall PaddlePaddle with GPU or XPU support to call this API."
    )


def reset_max_memory_allocated(device: _CPUPlaceLike | None = None) -> None:
    r"""
    The API reset_max_memory_allocated is not supported in CPU PaddlePaddle.
    Please reinstall PaddlePaddle with GPU or XPU support to call this API.
    """
    raise ValueError(
        "The API paddle.device.reset_max_memory_allocated is not supported in CPU PaddlePaddle. "
        "Please reinstall PaddlePaddle with GPU or XPU support to call this API."
    )


def reset_max_memory_reserved(device: _CPUPlaceLike | None = None) -> None:
    r"""
    The API reset_max_memory_reserved is not supported in CPU PaddlePaddle.
    Please reinstall PaddlePaddle with GPU or XPU support to call this API.
    """
    raise ValueError(
        "The API paddle.device.reset_max_memory_reserved is not supported in CPU PaddlePaddle. "
        "Please reinstall PaddlePaddle with GPU or XPU support to call this API."
    )
