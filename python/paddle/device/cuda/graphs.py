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

import os

from paddle.base.core import (
    CUDAPlace,
    CustomPlace,
    get_all_custom_device_type,
    is_compiled_with_cuda,
    is_compiled_with_custom_device,
    is_compiled_with_rocm,
)


def check_compiled_with_custom_device():
    custom_device_flag = False
    custom_devices_types = get_all_custom_device_type()
    for device_type in custom_devices_types:
        if is_compiled_with_custom_device(device_type):
            custom_device_flag = True
            break
    return custom_device_flag


if (
    is_compiled_with_cuda()
    or is_compiled_with_rocm()
    or check_compiled_with_custom_device()
):
    from paddle.base.core import CUDAGraph as CoreCUDAGraph

    def is_cuda_graph_supported():
        return True

else:
    CoreCUDAGraph = None

    def is_cuda_graph_supported():
        return False


def current_expected_place():
    for device in get_all_custom_device_type():
        selected_devices = os.getenv(f"FLAGS_selected_{device}s", "0").split(
            ","
        )
        device_id = int(selected_devices[0])
        return CustomPlace(device, device_id)
    return None


ALL_MODES = ["global", "thread_local", "relaxed"]
cuda_graph_id = 0


class CUDAGraph:
    def __init__(self, place=None, mode="thread_local", pool_id=None):
        assert CoreCUDAGraph is not None, (
            "CUDA Graph is only supported on PaddlePaddle compiled with NVIDIA GPU."
        )

        self._graph = None
        if place is None and check_compiled_with_custom_device():
            place = current_expected_place()
        elif place is None:
            device_id = int(os.environ.get('FLAGS_selected_gpus', 0))
            place = CUDAPlace(device_id)

        self._place = place
        assert mode in ALL_MODES
        self._mode = ALL_MODES.index(mode)
        self._pool_id = pool_id

    def capture_begin(self):
        CoreCUDAGraph.begin_capture_with_pool_id(
            self._place, self._mode, self._pool_id
        )

    def capture_end(self):
        self._graph = CoreCUDAGraph.end_capture()

    def replay(self):
        self._graph.replay()

    def reset(self):
        self._graph.reset()

    def print_to_dot_files(self, dirname, flags=None):
        if not isinstance(dirname, (str, bytes)):
            dirname = dirname.name
        os.makedirs(name=dirname, exist_ok=True)
        assert os.path.isdir(dirname), (
            f"The dirname {dirname} should be a directory"
        )
        if flags is None:
            flags = 2047  # only all information. It can be any integer inside [1, 2048)
        self._graph.print_to_dot_files(dirname, flags)
