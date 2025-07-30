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

import time
import unittest

import numpy as np

import paddle

# Disable static mode so that .numpy() can be called.
paddle.disable_static()

from paddle.incubate.tensor.manipulation import (
    async_offload,
    async_reload,
    create_xpu_async_load,
)


def print_debug_info(tensor, name):
    """Prints debug information for a tensor."""
    # print(f"{name} is on device: {tensor.place}")
    # print(f"{name} shape: {tensor.shape}, dtype: {tensor.dtype}")
    try:
        arr = tensor.numpy()
        flat = arr.flatten()
        # print(f"{name} first 5 elements: {flat[:5]}")
    except Exception as e:
        # print(f"{name} cannot be converted to numpy array: {e}")
        raise


class TestLargeTensorOffloadAndReloadRepeated(unittest.TestCase):
    def test_large_data_performance_repeated(self):
        # Repeat the offload and reload process 100 times.
        for i in range(1):
            # print(f"\n--- Iteration {i+1} ---")
            # Create a large tensor on XPU.
            large_arr = np.empty((512, 512, 1000), dtype="float32")
            large_tensor = paddle.to_tensor(large_arr, place=paddle.XPUPlace(0))
            print_debug_info(large_tensor, "large_tensor (original)")
            loader = create_xpu_async_load()

            # Offload the tensor.
            t0 = time.time()
            cpu_large, task_offload = async_offload(large_tensor, loader)
            task_offload.cpu_wait()  # Wait for offload completion.
            t1 = time.time()
            offload_time = t1 - t0
            # print(f"Offload time: {offload_time:.4f} seconds")

            # Reload the tensor.
            t2 = time.time()
            xpu_large, task_reload = async_reload(cpu_large, loader)
            task_reload.cpu_wait()  # Wait for reload completion.
            t3 = time.time()
            reload_time = t3 - t2
            # print(f"Reload time: {reload_time:.4f} seconds")


if __name__ == '__main__':
    # print("Default Paddle device:", paddle.get_device())
    unittest.main()
