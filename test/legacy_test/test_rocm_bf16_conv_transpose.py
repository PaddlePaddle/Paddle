# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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
import paddle.nn.functional as F
from paddle.base import core


class TestRocmBf16ConvTranspose(unittest.TestCase):
    def setUp(self):
        if not paddle.is_compiled_with_rocm():
            self.skipTest("ROCm only")

    def test_registry_has_gpu_bf16(self):
        kernels = core._get_all_register_op_kernels("phi")
        trans = kernels.get("conv2d_transpose", [])
        trans_grad = kernels.get("conv2d_transpose_grad", [])

        def has_gpu_bf16(items):
            return any(
                "bfloat16" in x and "place[Place(gpu" in x for x in items
            )

        self.assertTrue(
            has_gpu_bf16(trans), msg=f"conv2d_transpose kernels: {trans}"
        )
        self.assertTrue(
            has_gpu_bf16(trans_grad),
            msg=f"conv2d_transpose_grad kernels: {trans_grad}",
        )

    def test_conv2d_transpose_bf16_smoke(self):
        paddle.set_device("gpu:0")
        x = paddle.randn([1, 8, 32, 32], dtype="float32").astype("bfloat16")
        w = paddle.randn([8, 4, 3, 3], dtype="float32").astype("bfloat16")
        y = F.conv2d_transpose(x, w, padding=1)
        self.assertEqual(y.dtype, paddle.bfloat16)


if __name__ == "__main__":
    unittest.main()
