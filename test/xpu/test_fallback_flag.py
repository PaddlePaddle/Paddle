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

import unittest

import paddle


class TestFLAGSEnableApiKernelFallback(unittest.TestCase):
    def test_FLAGS_enable_api_kernel_fallback(self):
        FLAGS_enable_api_kernel_fallback_prev: bool = paddle.get_flags(
            ["FLAGS_enable_api_kernel_fallback"]
        )["FLAGS_enable_api_kernel_fallback"]
        paddle.set_flags({"FLAGS_enable_api_kernel_fallback": False})
        x = paddle.to_tensor(1.0, dtype="float64")
        with self.assertRaisesRegex(RuntimeError, "not registered"):
            z = paddle.sqrt(x)
        paddle.set_flags(
            {
                'FLAGS_enable_api_kernel_fallback': FLAGS_enable_api_kernel_fallback_prev
            }
        )


if __name__ == "__main__":
    unittest.main()
