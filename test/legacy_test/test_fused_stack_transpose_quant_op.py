#  Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import unittest

import numpy as np

import paddle
import paddle.incubate.nn.functional as F
from paddle import core


@unittest.skipIf(
    not core.is_compiled_with_cuda(), "core is not compiled with CUDA "
)
class TestFusedStackTransposeQuantOp(unittest.TestCase):
    def setUp(self):
        np.random.seed(2025)
        self.dtype = 'bfloat16'
        self.transpose = True

    def restore_stack_quant(self, out, scale):
        # Expand scale to [M, K] shape assuming block size 128 x 128
        scale = paddle.repeat_interleave(scale, repeats=128, axis=0)
        scale = paddle.repeat_interleave(scale, repeats=128, axis=1)
        x = out.astype('float32') * scale
        return x

    def check_main(self, N, M, K):
        paddle.disable_static()
        x_tensor_list = [
            paddle.randn([M, K], dtype=self.dtype).clip(min=-50, max=50)
            for _ in range(N)
        ]
        x_fp32 = paddle.stack(x_tensor_list).reshape([-1, K]).astype('float32')
        out, scale = F.fused_stack_transpose_quant(
            x_tensor_list, transpose=self.transpose
        )
        x_restored = self.restore_stack_quant(out, scale)
        if self.transpose:
            x_restored = (
                x_restored.reshape([N, K, M])
                .transpose([0, 2, 1])
                .reshape([-1, K])
            )
        paddle.enable_static()

        if not paddle.is_compiled_with_cuda():
            return
        np.testing.assert_allclose(
            x_fp32.numpy(),
            x_restored.numpy(),
            rtol=0.01,
            atol=0.2,
        )

    def test_fused_stack_transpose_quant(self):
        self.check_main(1, 2048, 128)

    def test_fused_stack_transpose_quant2(self):
        self.check_main(4, 2048, 128)


class TestFusedStackTransposeQuantOp1(TestFusedStackTransposeQuantOp):
    def setUp(self):
        super().setUp()
        self.transpose = False


if __name__ == "__main__":
    unittest.main()
