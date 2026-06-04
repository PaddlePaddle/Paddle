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

"""Paddle metric functional API."""

from paddle.metric.functional.classification import (
    accuracy,
    auroc,
    average_precision,
    calibration_error,
    cohen_kappa,
    confusion_matrix,
    eer,
    exact_match,
    f1_score,
    fbeta_score,
    hamming_distance,
    hinge_loss,
    jaccard_index,
    logauc,
    matthews_corrcoef,
    negative_predictive_value,
    precision,
    precision_at_fixed_recall,
    precision_recall_curve,
    recall,
    recall_at_fixed_precision,
    roc,
    sensitivity_at_specificity,
    specificity,
    specificity_at_sensitivity,
    stat_scores,
)

__all__ = [
    "accuracy",
    "auroc",
    "average_precision",
    "calibration_error",
    "cohen_kappa",
    "confusion_matrix",
    "eer",
    "exact_match",
    "f1_score",
    "fbeta_score",
    "hamming_distance",
    "hinge_loss",
    "jaccard_index",
    "logauc",
    "matthews_corrcoef",
    "negative_predictive_value",
    "precision",
    "precision_at_fixed_recall",
    "precision_recall_curve",
    "recall",
    "recall_at_fixed_precision",
    "roc",
    "sensitivity_at_specificity",
    "specificity",
    "specificity_at_sensitivity",
    "stat_scores",
]
