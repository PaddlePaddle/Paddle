#   Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

# TODO: define random api
import paddle
from paddle import base
from paddle.base import core

__all__ = []


def get_rng_state(device=None, use_index=False):
    """
    Get all random states of random generators of specified device.
    Args:
        device(str): This parameter determines the specific running device.
            It can be ``cpu``, ``gpu``, ``xpu``, Default is None.
            If None, return the generators of current device (specified by ``set_device``).
        use_index(bool): If use index is True, return the index that saved in the generator
    Returns:
        GeneratorState:  object.
    Examples:
        .. code-block:: python
            >>> import paddle
            >>> sts = paddle.incubate.get_rng_state()
    """

    def get_state(generator):
        if use_index:
            return generator.get_state_index()
        else:
            return generator.get_state()

    state_list = []
    if device is None:
        place = base.framework._current_expected_place()
    else:
        place = paddle.device._convert_to_place(device)

    if isinstance(place, core.CPUPlace):
        state_list.append(get_state(core.default_cpu_generator()))
    elif isinstance(place, core.CUDAPlace):
        for i in range(core.get_cuda_device_count()):
            state_list.append(get_state(core.default_cuda_generator(i)))
    elif isinstance(place, core.XPUPlace):
        for i in range(core.get_xpu_device_count()):
            state_list.append(get_state(core.default_xpu_generator(i)))
    elif isinstance(place, core.CustomPlace):
        dev_cnt = sum(
            [
                place.get_device_type() == s.split(':')[0]
                for s in core.get_available_custom_device()
            ]
        )
        for i in range(dev_cnt):
            state_list.append(
                get_state(
                    core.default_custom_device_generator(
                        core.CustomPlace(place.get_device_type(), i)
                    )
                )
            )
    else:
        raise ValueError(
            f"get_rng_state is not implemented for current device: {place}"
        )

    return state_list


def set_rng_state(state_list, device=None, use_index=False):
    """

    Sets generator state for all device generators.

    Args:
        state_list(list|tuple): The device states to set back to device generators. state_list is obtained from get_rng_state().
        device(str): This parameter determines the specific running device.
            It can be ``cpu``, ``gpu``, ``xpu``, Default is None.
            If None, return the generators of current device (specified by ``set_device``).
        use_index(bool): If use index is True, state_list should be the indices of the states

    Returns:
        None.

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> sts = paddle.incubate.get_rng_state()
            >>> paddle.incubate.set_rng_state(sts)

    """

    def set_state(generator, state):
        if use_index:
            generator.set_state_index(state)
        else:
            generator.set_state(state)

    if device is None:
        place = base.framework._current_expected_place()
    else:
        place = device._convert_to_place(device)

    if isinstance(place, core.CUDAPlace):
        if not len(state_list) == core.get_cuda_device_count():
            raise ValueError(
                "Length of gpu state list should be equal to the gpu device count"
            )
        for i in range(core.get_cuda_device_count()):
            set_state(core.default_cuda_generator(i), state_list[i])
    elif isinstance(place, core.XPUPlace):
        if not len(state_list) == core.get_xpu_device_count():
            raise ValueError(
                "Length of xpu state list should be equal to the xpu device count"
            )
        for i in range(core.get_xpu_device_count()):
            set_state(core.default_xpu_generator(i), state_list[i])
    elif isinstance(place, core.CustomPlace):
        dev_cnt = sum(
            [
                place.get_device_type() == s.split(':')[0]
                for s in core.get_available_custom_device()
            ]
        )
        if not len(state_list) == dev_cnt:
            raise ValueError(
                f"Length of custom device state list should be equal to the {place.get_dtype_type()} device count"
            )
        for i in range(dev_cnt):
            set_state(
                core.default_custom_device_generator(
                    core.CustomPlace(place.get_device_type(), i)
                ),
                state_list[i],
            )
    elif isinstance(place, core.CPUPlace):
        if not len(state_list) == 1:
            raise ValueError("Length of cpu state list should be equal to 1")
        set_state(core.default_cpu_generator(), state_list[0])
    else:
        raise ValueError(
            f"set_rng_state is not implemented for current device: {place}"
        )


def register_rng_state_as_index(state_list=None, device=None):
    """

    The register_rng_state_as_index function creates and registers a new generator state within the generator.
    It enables users to manage multiple generator states via indices,
    offering a convenient way to switch between these states without directly manipulating the generator's state.

    Args:
        state_list(list|tuple): A list or tuple representing the RNG states for devices.
            If not provided, the function will register the current state.
        device(str): This parameter determines the specific running device.
            It can be ``cpu``, ``gpu``, ``xpu``, Default is None.
            If None, return the generators of current device (specified by ``set_device``).

    Returns:
        A list of indices representing the positions at which the new states were saved within the generator.
        These indices can be used to switch between states using set_rng_state(use_index=True)


    Examples:
        .. code-block:: python

            >>> import paddle
            >>> old_index = paddle.incubate.get_rng_state(use_index=True)
            >>> print(old_index)
            [0]
            >>> new_index = paddle.incubate.register_rng_state_as_index()
            >>> print(new_index)
            [1]
            >>> paddle.incubate.set_rng_state(old_index, use_index=True)
            >>> paddle.incubate.set_rng_state(new_index, use_index=True)

    """
    new_state_index_list = []

    if device is None:
        place = base.framework._current_expected_place()
    else:
        place = device._convert_to_place(device)

    if state_list is None:
        state_list = get_rng_state(device)

    if isinstance(place, core.CUDAPlace):
        if not len(state_list) == core.get_cuda_device_count():
            raise ValueError(
                "Length of gpu state list should be equal to the gpu device count"
            )
        for i in range(core.get_cuda_device_count()):
            new_state_index_list.append(
                core.default_cuda_generator(i).register_state_index(
                    state_list[i]
                )
            )
    elif isinstance(place, core.XPUPlace):
        if not len(state_list) == core.get_xpu_device_count():
            raise ValueError(
                "Length of xpu state list should be equal to the xpu device count"
            )
        for i in range(core.get_xpu_device_count()):
            new_state_index_list.append(
                core.default_xpu_generator(i).register_state_index(
                    state_list[i]
                )
            )
    elif isinstance(place, core.CustomPlace):
        dev_cnt = sum(
            [
                place.get_device_type() == s.split(':')[0]
                for s in core.get_available_custom_device()
            ]
        )
        if not len(state_list) == dev_cnt:
            raise ValueError(
                f"Length of custom device state list should be equal to the {place.get_dtype_type()} device count"
            )
        for i in range(dev_cnt):
            new_state_index_list.append(
                core.default_custom_device_generator(
                    core.CustomPlace(place.get_device_type(), i)
                ).register_state_index(state_list[i])
            )
    elif isinstance(place, core.CPUPlace):
        if not len(state_list) == 1:
            raise ValueError("Length of cpu state list should be equal to 1")
        new_state_index_list.append(
            core.default_cpu_generator().register_state_index(state_list[0])
        )
    else:
        raise ValueError(
            f"register_rng_state_index is not implemented for current device: {place}"
        )
    return new_state_index_list
