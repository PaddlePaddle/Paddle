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
from paddle.nn import (
    BCELoss,
    BCEWithLogitsLoss,
    CosineEmbeddingLoss,
    CrossEntropyLoss,
    HingeEmbeddingLoss,
    KLDivLoss,
    L1Loss,
    MarginRankingLoss,
    MSELoss,
    MultiLabelMarginLoss,
    MultiLabelSoftMarginLoss,
    MultiMarginLoss,
    NLLLoss,
    PoissonNLLLoss,
    SmoothL1Loss,
    SoftMarginLoss,
    TripletMarginLoss,
)


class TestLegacyLossArgs(unittest.TestCase):
    def assertSuggests(self, loss_ctor, expected_reduction, **legacy_kwargs):
        with self.assertRaises(ValueError) as cm:
            loss_ctor(**legacy_kwargs)
        msg = str(cm.exception)
        self.assertIn(f"reduction='{expected_reduction}'", msg)

    def test_no_legacy_all_constructible(self):
        # Ensure all 17 losses still construct with reduction only
        ctors = [
            L1Loss,
            NLLLoss,
            PoissonNLLLoss,
            KLDivLoss,
            MSELoss,
            BCELoss,
            BCEWithLogitsLoss,
            HingeEmbeddingLoss,
            MultiLabelMarginLoss,
            SmoothL1Loss,
            SoftMarginLoss,
            CrossEntropyLoss,
            MultiLabelSoftMarginLoss,
            CosineEmbeddingLoss,
            MarginRankingLoss,
            MultiMarginLoss,
            TripletMarginLoss,
        ]
        for ctor in ctors:
            _ = ctor(reduction='mean')

    def test_no_args_all_constructible_with_defaults(self):
        # Ensure all 17 losses construct with default args (no legacy, no explicit reduction)
        ctors = [
            L1Loss,
            NLLLoss,
            PoissonNLLLoss,
            KLDivLoss,
            MSELoss,
            BCELoss,
            BCEWithLogitsLoss,
            HingeEmbeddingLoss,
            MultiLabelMarginLoss,
            SmoothL1Loss,
            SoftMarginLoss,
            CrossEntropyLoss,
            MultiLabelSoftMarginLoss,
            CosineEmbeddingLoss,
            MarginRankingLoss,
            MultiMarginLoss,
            TripletMarginLoss,
        ]
        for ctor in ctors:
            _ = ctor()

    # Cover legacy combos across the family (not each loss needs all combos)

    def test_cross_entropy_reduce_false(self):
        self.assertSuggests(CrossEntropyLoss, 'none', reduce=False)

    def test_mse_reduce_true_size_average_false(self):
        self.assertSuggests(MSELoss, 'sum', reduce=True, size_average=False)

    def test_bcewithlogits_reduce_true_size_average_true(self):
        self.assertSuggests(
            BCEWithLogitsLoss, 'mean', reduce=True, size_average=True
        )

    def test_l1_size_average_false_only(self):
        self.assertSuggests(L1Loss, 'sum', size_average=False)

    def test_kldiv_reduce_true_size_average_none(self):
        self.assertSuggests(KLDivLoss, 'mean', reduce=True, size_average=None)

    def test_multimargin_reduce_false(self):
        self.assertSuggests(MultiMarginLoss, 'none', reduce=False)

    def test_multilabel_margin_size_average_false(self):
        self.assertSuggests(MultiLabelMarginLoss, 'sum', size_average=False)

    def test_cosine_embedding_reduce_true_size_average_true(self):
        self.assertSuggests(
            CosineEmbeddingLoss, 'mean', reduce=True, size_average=True
        )

    def test_margin_ranking_reduce_true_size_average_false(self):
        self.assertSuggests(
            MarginRankingLoss, 'sum', reduce=True, size_average=False
        )

    def test_soft_margin_reduce_false(self):
        self.assertSuggests(SoftMarginLoss, 'none', reduce=False)

    def test_smooth_l1_size_average_false(self):
        self.assertSuggests(SmoothL1Loss, 'sum', size_average=False)

    def test_bce_reduce_true_size_average_true(self):
        self.assertSuggests(BCELoss, 'mean', reduce=True, size_average=True)

    def test_nll_reduce_true_size_average_none(self):
        self.assertSuggests(NLLLoss, 'mean', reduce=True, size_average=None)

    def test_poisson_nll_reduce_false(self):
        self.assertSuggests(PoissonNLLLoss, 'none', reduce=False)

    def test_multilabel_soft_margin_size_average_false(self):
        self.assertSuggests(MultiLabelSoftMarginLoss, 'sum', size_average=False)

    def test_triplet_margin_reduce_true_size_average_false(self):
        self.assertSuggests(
            TripletMarginLoss, 'sum', reduce=True, size_average=False
        )

    def test_ce_positional_soft_label_guard_by_ignore_index(self):
        # CrossEntropyLoss(weight=None, ignore_index=int, reduction='mean', soft_label=True)
        w = paddle.ones([3], dtype='float32')
        _ = CrossEntropyLoss(w, -100, 'mean', True)

    def test_ce_positional_legacy_reduce_trigger(self):
        # CrossEntropyLoss(weight=None, size_average=True, ignore_index, reduce=True)
        with self.assertRaises(ValueError) as cm:
            w = paddle.ones([3], dtype='float32')
            CrossEntropyLoss(w, True, -100, True)
        self.assertIn("reduction='mean'", str(cm.exception))

    def test_kldiv_positional_log_target_guard(self):
        # KLDivLoss(reduction='mean', log_target=True)
        _ = KLDivLoss('mean', True)

    def test_kldiv_positional_legacy_reduce_trigger(self):
        # KLDivLoss(log_target=True)(Not provide reduction string, treat as legacy reduce)
        with self.assertRaises(ValueError) as cm:
            KLDivLoss(True)
        self.assertIn("reduction='mean'", str(cm.exception))

    def test_poisson_positional_eps_float_guard(self):
        # PoissonNLLLoss(log_input, full, eps)
        _ = PoissonNLLLoss(True, False, 1e-8)

    def test_poisson_positional_legacy_reduce_trigger(self):
        # PoissonNLLLoss(log_input, full, size_average=True, epsilon, reduce=True)
        with self.assertRaises(ValueError) as cm:
            PoissonNLLLoss(True, False, True, 1e-8, True)
        self.assertIn("reduction='mean'", str(cm.exception))


if __name__ == '__main__':
    unittest.main()
