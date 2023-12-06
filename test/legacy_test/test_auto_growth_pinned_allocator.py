# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

import unittest

import numpy as np

import paddle


class TestPinnedAllocator(unittest.TestCase):
    def test_main(self):
        if not paddle.is_compiled_with_cuda():
            return

        paddle.set_flags({'FLAGS_use_auto_growth_pinned_allocator': True})
        x_np = np.random.random([1024, 1024, 4]).astype(np.float32)
        x_pd_gpu = paddle.to_tensor(x_np)

        x_pd_pin = x_pd_gpu.pin_memory(False)
        paddle.device.cuda.synchronize()
        np.testing.assert_equal(x_np, x_pd_pin.numpy())

        x_pd_pin = x_pd_gpu.pin_memory()
        np.testing.assert_equal(x_np, x_pd_pin.numpy())


if __name__ == "__main__":
    unittest.main()
