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
import paddle.incubate.nn.functional as F
from paddle import _C_ops


def create_test_data(
    batch_size=1, seq_len=4096, vocab_size=129280, num_labels=12900
):
    labels = paddle.uniform(
        [batch_size, seq_len, 1], min=0, max=num_labels
    ).cast(paddle.int64)

    preds = paddle.uniform(
        [batch_size, seq_len, vocab_size], dtype=paddle.float32
    )
    preds.stop_gradient = False

    return labels, preds


class TestCustomCrossEntropyBwd(unittest.TestCase):

    def compute_losses(self, preds, labels):
        loss_func = paddle.nn.CrossEntropyLoss(
            reduction="none", ignore_index=-100
        )
        masked_lm_loss = loss_func(preds, labels)

        softmax_val, separate_loss = _C_ops.cross_entropy_with_softmax(
            preds, labels, False, True, False, -100, -1
        )

        np.testing.assert_allclose(
            masked_lm_loss.numpy(), separate_loss.numpy(), atol=1e-6
        )

        return masked_lm_loss, softmax_val, separate_loss

    def compute_gradients(self, preds, labels, masked_lm_loss, softmax_val):
        masked_lm_loss.retain_grads()
        loss = masked_lm_loss.sum()
        loss.backward(retain_graph=True)

        custom_grad = F.cross_entropy_with_softmax_bwd_w_downcast(
            labels, softmax_val, masked_lm_loss.grad
        )

        separate_grad = _C_ops.cross_entropy_with_softmax_grad(
            labels,
            softmax_val,
            masked_lm_loss.grad,
            False,
            True,
            False,
            -100,
            -1,
        )

        return separate_grad, custom_grad

    def verify_results(
        self, separate_loss, masked_lm_loss, separate_grad, custom_grad, preds
    ):
        # float32 compare with float32, not exactly the same because non-deterministic
        np.testing.assert_allclose(
            separate_grad.numpy(), preds.grad.numpy(), atol=1e-7, rtol=1e-5
        )

        # float32 compare with float16, not exactly the same because non-deterministic, and dtype cast
        np.testing.assert_allclose(
            separate_grad.numpy(),
            custom_grad.astype("float32").numpy(),
            atol=1e-2,
            rtol=1e-2,
        )

        # float32 compare with float16, not exactly the same because non-deterministic, and dtype cast
        np.testing.assert_allclose(
            custom_grad.astype("float32").numpy(),
            preds.grad.numpy(),
            atol=1e-2,
            rtol=1e-2,
        )

    def test_custom_bwd(self):
        labels, preds = create_test_data()

        masked_lm_loss, softmax_val, separate_loss = self.compute_losses(
            preds, labels
        )

        separate_grad, custom_grad = self.compute_gradients(
            preds, labels, masked_lm_loss, softmax_val
        )

        self.verify_results(
            separate_loss, masked_lm_loss, separate_grad, custom_grad, preds
        )


if __name__ == "__main__":
    unittest.main()
