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

import numpy as np

import paddle


class TestCrossEntropyWithSoftmaxBwdOperator(unittest.TestCase):
    def setUp(self):
        """Initialize test environment"""
        paddle.seed(1024)
        self.place = (
            paddle.CUDAPlace(0)
            if paddle.is_compiled_with_cuda()
            else paddle.CPUPlace()
        )

    def test_custom_backward_operator(self):
        # 1. Prepare test data
        batch_size = 1
        seq_length = 4096
        vocab_size = 129280
        labels = paddle.randint(
            low=0, high=12900, shape=[batch_size, seq_length, 1]
        ).cast(paddle.int64)
        logits = paddle.uniform(
            shape=[batch_size, seq_length, vocab_size], dtype=paddle.float32
        )
        logits.stop_gradient = False

        # 2. Native cross-entropy calculation
        loss_func = paddle.nn.CrossEntropyLoss(
            reduction="none", ignore_index=-100
        )
        masked_lm_loss = loss_func(logits, labels)

        # 3. Separate operator forward pass
        softmax_val, separate_loss = paddle._C_ops.cross_entropy_with_softmax(
            logits, labels, False, True, True, -100, -1
        )

        # 4. Verify forward pass consistency
        np.testing.assert_allclose(
            masked_lm_loss.numpy(),
            separate_loss.numpy(),
            rtol=1e-5,
            atol=1e-8,
            err_msg="Forward result mismatch between composite and separate ops",
        )

        # 5. Backward pass (native)
        loss = masked_lm_loss.sum()
        loss.backward()
        original_grad = logits.grad

        # 6. Custom backward pass (simulating bfloat16 downcasting)
        broadcasted_loss = loss.expand_as(separate_loss)
        custom_grad = paddle.incubate.nn.functional.cross_entropy_with_softmax_bwd_w_downcast(
            labels, softmax_val, broadcasted_loss, axis=-1
        )

        # 7. Verify gradient consistency
        np.testing.assert_allclose(
            custom_grad.numpy(),
            original_grad.numpy(),
            rtol=1e-3,  # Relaxed tolerance for downcast precision
            atol=1e-5,
            err_msg="Backward gradient mismatch between custom and original ops",
        )


if __name__ == '__main__':
    unittest.main()
