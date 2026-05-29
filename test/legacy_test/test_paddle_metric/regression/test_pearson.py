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

from __future__ import annotations

import numpy as np

import paddle
from paddle.metric.regression import PearsonCorrCoef


class TestPearsonCorrCoef:
    """Regression tests for PearsonCorrCoef — catches multi-output bugs."""

    def test_single_output(self):
        """Basic single-output Pearson."""
        preds = paddle.randn([20])
        target = preds + paddle.randn([20]) * 0.1
        metric = PearsonCorrCoef(num_outputs=1)
        metric.update(preds, target)
        result = metric.compute()
        assert result.item() > 0.5

    def test_multi_output(self):
        """Multi-output must return per-output correlation."""
        num_outputs = 3
        preds = paddle.randn([20, num_outputs])
        target = preds + paddle.randn([20, num_outputs]) * 0.1
        metric = PearsonCorrCoef(num_outputs=num_outputs)
        metric.update(preds, target)
        result = metric.compute()
        assert result.shape == [num_outputs], (
            f"Expected shape [{num_outputs}], got {result.shape}"
        )
        # Each output should have high correlation
        for i in range(num_outputs):
            assert result[i].item() > 0.3, (
                f"Output {i} correlation too low: {result[i].item()}"
            )

    def test_multi_output_independent(self):
        """Different outputs should have different correlations."""
        np.random.seed(42)
        n = 100
        num_outputs = 3
        preds_np = np.random.randn(n, num_outputs).astype("float32")
        # Output 0: high correlation
        # Output 1: low correlation (noise)
        # Output 2: negative correlation
        target_np = np.stack(
            [
                preds_np[:, 0] * 2 + np.random.randn(n) * 0.1,
                np.random.randn(n).astype("float32"),
                -preds_np[:, 2] * 2 + np.random.randn(n) * 0.1,
            ],
            axis=1,
        ).astype("float32")

        preds = paddle.to_tensor(preds_np)
        target = paddle.to_tensor(target_np)
        metric = PearsonCorrCoef(num_outputs=num_outputs)
        metric.update(preds, target)
        result = metric.compute()
        assert result[0].item() > 0.9, (
            f"Expected high corr, got {result[0].item()}"
        )
        assert result[2].item() < -0.5, (
            f"Expected negative corr, got {result[2].item()}"
        )

    def test_multi_batch_accumulation(self):
        """Multiple updates should accumulate correctly."""
        num_outputs = 2
        metric = PearsonCorrCoef(num_outputs=num_outputs)
        for _ in range(5):
            preds = paddle.randn([10, num_outputs])
            target = preds + paddle.randn([10, num_outputs]) * 0.2
            metric.update(preds, target)
        result = metric.compute()
        assert result.shape == [num_outputs]
        for i in range(num_outputs):
            assert result[i].item() > 0.3

    def test_reset(self):
        """Reset should clear accumulated state."""
        preds = paddle.randn([20, 2])
        target = preds + paddle.randn([20, 2]) * 0.1
        metric = PearsonCorrCoef(num_outputs=2)
        metric.update(preds, target)
        metric.reset()
        metric.update(preds, target)
        result = metric.compute()
        assert result.shape == [2]
