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

import paddle
from paddle.base import core

__all__ = ["initial_seed"]


def initial_seed() -> int:
    """
    Returns the initial seed for generating random numbers as a Python `int`.

    Returns:
        int: The 64-bit initial seed of the default generator on CPU place only.

    Examples:
        .. code-block:: pycon

            >>> import paddle
            >>> s = paddle.random.initial_seed()
    """
    return core.default_cpu_generator().initial_seed()


def get_rng_state(
    device: str | None = None,
) -> list[core.GeneratorState]:
    """
    Get all random states of random generators of specified device.

    Args:
        device(str): This parameter determines the specific running device.
            It can be ``cpu``, ``gpu``, ``xpu``, Default is None.
            If None, return the generators of current device (specified by ``set_device``).

    Returns:
        list[GeneratorState], object.

    Examples:
        .. code-block:: pycon

            >>> import paddle
            >>> sts = paddle.get_rng_state()
    """
    state_list = []
    if device is None:
        place = paddle.framework._current_expected_place_()
    else:
        place = paddle.device._convert_to_place(device)

    if isinstance(place, paddle.CPUPlace):
        state_list.append(core.default_cpu_generator().get_state())
    elif isinstance(place, paddle.CUDAPlace):
        for i in range(core.get_cuda_device_count()):
            state_list.append(core.default_cuda_generator(i).get_state())
    elif isinstance(place, paddle.XPUPlace):
        for i in range(core.get_xpu_device_count()):
            state_list.append(core.default_xpu_generator(i).get_state())
    elif isinstance(place, paddle.CustomPlace):
        dev_cnt = sum(
            [
                place.get_device_type() == s.split(':')[0]
                for s in core.get_available_custom_device()
            ]
        )
        for i in range(dev_cnt):
            state_list.append(
                core.default_custom_device_generator(
                    core.CustomPlace(place.get_device_type(), i)
                ).get_state()
            )
    else:
        raise ValueError(
            f"get_rng_state is not implemented for current device: {place}"
        )

    return state_list
