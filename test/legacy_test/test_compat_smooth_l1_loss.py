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
import warnings

import numpy as np

import paddle
import paddle.compat.nn.functional as F_compat


def _smooth_l1_elt(val, beta):
    abs_val = abs(val)
    if beta == 0:
        return abs_val
    if abs_val < beta:
        return 0.5 * val * val / beta
    return abs_val - 0.5 * beta


def smooth_l1_ref(input, label, reduction='mean', beta=1.0):
    diff = input - label
    out = np.vectorize(_smooth_l1_elt)(diff, beta)
    if reduction == 'sum':
        return np.sum(out)
    elif reduction == 'mean':
        return np.mean(out)
    return out


class TestCompatSmoothL1LossFunctional(unittest.TestCase):
    def setUp(self):
        np.random.seed(2025)
        self.input_np = np.random.random([20, 30]).astype(np.float32)
        self.label_np = np.random.random([20, 30]).astype(np.float32)
        paddle.disable_static()

    def test_match_torch_formula(self):
        x = paddle.to_tensor(self.input_np)
        y = paddle.to_tensor(self.label_np)
        for reduction in ['none', 'mean', 'sum']:
            for beta in [0.5, 1.0, 2.0]:
                out = F_compat.smooth_l1_loss(
                    x, y, reduction=reduction, beta=beta
                )
                ref = smooth_l1_ref(
                    self.input_np, self.label_np, reduction=reduction, beta=beta
                )
                np.testing.assert_allclose(out.numpy(), ref, rtol=1e-5)

    def test_match_base_api_is_huber_false(self):
        x = paddle.to_tensor(self.input_np)
        y = paddle.to_tensor(self.label_np)
        out = F_compat.smooth_l1_loss(x, y, reduction='mean', beta=2.0)
        base = paddle.nn.functional.smooth_l1_loss(
            x, y, reduction='mean', delta=2.0, is_huber=False
        )
        np.testing.assert_allclose(out.numpy(), base.numpy(), rtol=1e-6)

    def test_target_alias_keyword(self):
        x = paddle.to_tensor(self.input_np)
        y = paddle.to_tensor(self.label_np)
        out = F_compat.smooth_l1_loss(x, target=y, reduction='mean')
        ref = smooth_l1_ref(self.input_np, self.label_np, reduction='mean')
        np.testing.assert_allclose(out.numpy(), ref, rtol=1e-5)

    def test_beta_zero_is_l1(self):
        x = paddle.to_tensor(self.input_np)
        y = paddle.to_tensor(self.label_np)
        out = F_compat.smooth_l1_loss(x, y, reduction='mean', beta=0.0)
        l1 = paddle.nn.functional.l1_loss(x, y, reduction='mean')
        np.testing.assert_allclose(out.numpy(), l1.numpy(), rtol=1e-6)
        self.assertFalse(np.isnan(out.numpy()).any())

    def test_negative_beta_raises(self):
        x = paddle.to_tensor(self.input_np)
        y = paddle.to_tensor(self.label_np)
        with self.assertRaises(ValueError):
            F_compat.smooth_l1_loss(x, y, beta=-1.0)

    def test_size_average_reduce_deprecation(self):
        x = paddle.to_tensor(self.input_np)
        y = paddle.to_tensor(self.label_np)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            out = F_compat.smooth_l1_loss(x, y, size_average=False)
        # size_average=False -> reduction='sum'
        ref = smooth_l1_ref(self.input_np, self.label_np, reduction='sum')
        np.testing.assert_allclose(out.numpy(), ref, rtol=1e-5)
        self.assertTrue(
            any(issubclass(item.category, DeprecationWarning) for item in w)
        )

    def test_forbidden_keywords(self):
        x = paddle.to_tensor(self.input_np)
        y = paddle.to_tensor(self.label_np)
        for bad_kwargs in [{"delta": 1.0}, {"is_huber": True}, {"label": y}]:
            with self.assertRaises(TypeError) as cm:
                F_compat.smooth_l1_loss(x, **bad_kwargs)
            self.assertIn(
                "paddle.nn.functional.smooth_l1_loss", str(cm.exception)
            )

    def test_size_average_reduce_mapping_variants(self):
        # Complement test_size_average_reduce_deprecation (size_average=False ->
        # 'sum') by exercising the remaining reduce/size_average -> reduction
        # branches: reduce=False -> 'none' (wins over size_average), and any
        # other non-None combination -> 'mean'.
        x = paddle.to_tensor(self.input_np)
        y = paddle.to_tensor(self.label_np)
        cases = [
            ({'reduce': False}, 'none'),
            ({'reduce': True}, 'mean'),
            ({'size_average': True}, 'mean'),
        ]
        for kwargs, expected_reduction in cases:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                out = F_compat.smooth_l1_loss(x, y, **kwargs)
            ref = smooth_l1_ref(
                self.input_np, self.label_np, reduction=expected_reduction
            )
            np.testing.assert_allclose(out.numpy(), ref, rtol=1e-5)
            self.assertTrue(
                any(
                    issubclass(item.category, DeprecationWarning) for item in w
                ),
                f"expected a DeprecationWarning for kwargs={kwargs}",
            )


class TestCompatSmoothL1LossLayer(unittest.TestCase):
    def setUp(self):
        np.random.seed(2025)
        self.input_np = np.random.random([20, 30]).astype(np.float32)
        self.label_np = np.random.random([20, 30]).astype(np.float32)
        paddle.disable_static()

    def test_layer_match(self):
        x = paddle.to_tensor(self.input_np)
        y = paddle.to_tensor(self.label_np)
        for reduction in ['none', 'mean', 'sum']:
            for beta in [0.0, 1.0, 2.0]:
                loss = paddle.compat.nn.SmoothL1Loss(
                    reduction=reduction, beta=beta
                )
                out = loss(x, y)
                ref = smooth_l1_ref(
                    self.input_np, self.label_np, reduction=reduction, beta=beta
                )
                np.testing.assert_allclose(out.numpy(), ref, rtol=1e-5)

    def test_layer_target_alias(self):
        x = paddle.to_tensor(self.input_np)
        y = paddle.to_tensor(self.label_np)
        loss = paddle.compat.nn.SmoothL1Loss()
        out = loss(x, target=y)
        ref = smooth_l1_ref(self.input_np, self.label_np, reduction='mean')
        np.testing.assert_allclose(out.numpy(), ref, rtol=1e-5)

    def test_layer_forbidden_keywords(self):
        with self.assertRaises(TypeError) as cm:
            paddle.compat.nn.SmoothL1Loss(delta=1.0)
        self.assertIn("paddle.nn.SmoothL1Loss", str(cm.exception))

    def test_layer_size_average_reduce_mapping(self):
        # Legacy size_average / reduce must be translated into reduction with a
        # DeprecationWarning in __init__, mirroring torch.nn.SmoothL1Loss.
        # reduce=False wins over size_average; otherwise size_average=False ->
        # 'sum', and any other non-None combination -> 'mean'.
        x = paddle.to_tensor(self.input_np)
        y = paddle.to_tensor(self.label_np)
        cases = [
            ({'reduce': False}, 'none'),
            ({'reduce': True}, 'mean'),
            ({'size_average': False}, 'sum'),
            ({'size_average': True}, 'mean'),
            ({'reduce': False, 'size_average': False}, 'none'),
        ]
        for kwargs, expected_reduction in cases:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                loss = paddle.compat.nn.SmoothL1Loss(**kwargs)
            self.assertEqual(loss.reduction, expected_reduction)
            self.assertTrue(
                any(
                    issubclass(item.category, DeprecationWarning) for item in w
                ),
                f"expected a DeprecationWarning for kwargs={kwargs}",
            )
            out = loss(x, y)
            ref = smooth_l1_ref(
                self.input_np, self.label_np, reduction=expected_reduction
            )
            np.testing.assert_allclose(out.numpy(), ref, rtol=1e-5)

    def test_layer_no_deprecation_warning_default(self):
        # Without size_average / reduce, no DeprecationWarning should fire and
        # reduction must be kept as-is.
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            loss = paddle.compat.nn.SmoothL1Loss(reduction='sum', beta=2.0)
        self.assertEqual(loss.reduction, 'sum')
        self.assertEqual(loss.beta, 2.0)
        self.assertFalse(
            any(issubclass(item.category, DeprecationWarning) for item in w)
        )

    def test_layer_extra_repr(self):
        loss = paddle.compat.nn.SmoothL1Loss(reduction='sum', beta=2.0)
        self.assertEqual(loss.extra_repr(), "reduction=sum, beta=2.0")
        # extra_repr should also be surfaced through the Layer's repr.
        self.assertIn("reduction=sum, beta=2.0", repr(loss))


if __name__ == "__main__":
    unittest.main()
