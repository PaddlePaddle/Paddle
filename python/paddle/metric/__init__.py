#   Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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

"""Paddle metric module: enhanced Metric base class with state management,
distributed sync, serialization, and metric composition."""

from paddle.metric.aggregation import (
    CatMetric,
    MaxMetric,
    MeanMetric,
    MinMetric,
    RunningMean,
    RunningSum,
    SumMetric,
)
from paddle.metric.collections import MetricCollection
from paddle.metric.metric import CompositionalMetric, Metric

__all__ = [
    "Metric",
    "CompositionalMetric",
    "MetricCollection",
    "CatMetric",
    "MaxMetric",
    "MeanMetric",
    "MinMetric",
    "RunningMean",
    "RunningSum",
    "SumMetric",
]


_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    # Backward-compatible functional API (delegates to C++ ops)
    "accuracy": ("paddle.metric.compat", "accuracy"),
    # Classification
    "Accuracy": ("paddle.metric.classification", "Accuracy"),
    "AUROC": ("paddle.metric.classification", "AUROC"),
    "ROC": ("paddle.metric.classification", "ROC"),
    "Precision": ("paddle.metric.classification", "Precision"),
    "Recall": ("paddle.metric.classification", "Recall"),
    "F1Score": ("paddle.metric.classification", "F1Score"),
    "FBetaScore": ("paddle.metric.classification", "FBetaScore"),
    "ConfusionMatrix": ("paddle.metric.classification", "ConfusionMatrix"),
    "AveragePrecision": ("paddle.metric.classification", "AveragePrecision"),
    "CalibrationError": ("paddle.metric.classification", "CalibrationError"),
    "CohenKappa": ("paddle.metric.classification", "CohenKappa"),
    "EER": ("paddle.metric.classification", "EER"),
    "ExactMatch": ("paddle.metric.classification", "ExactMatch"),
    "HammingDistance": ("paddle.metric.classification", "HammingDistance"),
    "HingeLoss": ("paddle.metric.classification", "HingeLoss"),
    "JaccardIndex": ("paddle.metric.classification", "JaccardIndex"),
    "LogAUC": ("paddle.metric.classification", "LogAUC"),
    "MatthewsCorrCoef": ("paddle.metric.classification", "MatthewsCorrCoef"),
    "NegativePredictiveValue": (
        "paddle.metric.classification",
        "NegativePredictiveValue",
    ),
    "PrecisionAtFixedRecall": (
        "paddle.metric.classification",
        "PrecisionAtFixedRecall",
    ),
    "PrecisionRecallCurve": (
        "paddle.metric.classification",
        "PrecisionRecallCurve",
    ),
    "RecallAtFixedPrecision": (
        "paddle.metric.classification",
        "RecallAtFixedPrecision",
    ),
    "SensitivityAtSpecificity": (
        "paddle.metric.classification",
        "SensitivityAtSpecificity",
    ),
    "Specificity": ("paddle.metric.classification", "Specificity"),
    "SpecificityAtSensitivity": (
        "paddle.metric.classification",
        "SpecificityAtSensitivity",
    ),
    "StatScores": ("paddle.metric.classification", "StatScores"),
    # Regression
    "MeanSquaredError": ("paddle.metric.regression", "MeanSquaredError"),
    "MeanAbsoluteError": ("paddle.metric.regression", "MeanAbsoluteError"),
    "R2Score": ("paddle.metric.regression", "R2Score"),
    "PearsonCorrCoef": ("paddle.metric.regression", "PearsonCorrCoef"),
    "SpearmanCorrCoef": ("paddle.metric.regression", "SpearmanCorrCoef"),
    "CosineSimilarity": ("paddle.metric.regression", "CosineSimilarity"),
    "KLDivergence": ("paddle.metric.regression", "KLDivergence"),
    "LogCoshError": ("paddle.metric.regression", "LogCoshError"),
    "MeanAbsolutePercentageError": (
        "paddle.metric.regression",
        "MeanAbsolutePercentageError",
    ),
    "MeanSquaredLogError": ("paddle.metric.regression", "MeanSquaredLogError"),
    "MinkowskiDistance": ("paddle.metric.regression", "MinkowskiDistance"),
    "ConcordanceCorrCoef": ("paddle.metric.regression", "ConcordanceCorrCoef"),
    "ExplainedVariance": ("paddle.metric.regression", "ExplainedVariance"),
    "KendallRankCorrCoef": ("paddle.metric.regression", "KendallRankCorrCoef"),
    "SymmetricMeanAbsolutePercentageError": (
        "paddle.metric.regression",
        "SymmetricMeanAbsolutePercentageError",
    ),
    "TweedieDevianceScore": (
        "paddle.metric.regression",
        "TweedieDevianceScore",
    ),
    "WeightedMeanAbsolutePercentageError": (
        "paddle.metric.regression",
        "WeightedMeanAbsolutePercentageError",
    ),
    "JensenShannonDivergence": (
        "paddle.metric.regression",
        "JensenShannonDivergence",
    ),
    "NormalizedRootMeanSquaredError": (
        "paddle.metric.regression",
        "NormalizedRootMeanSquaredError",
    ),
    "RelativeSquaredError": (
        "paddle.metric.regression",
        "RelativeSquaredError",
    ),
    "ContinuousRankedProbabilityScore": (
        "paddle.metric.regression",
        "ContinuousRankedProbabilityScore",
    ),
    "CriticalSuccessIndex": (
        "paddle.metric.regression",
        "CriticalSuccessIndex",
    ),
    # Wrappers
    "BootStrapper": ("paddle.metric.wrappers", "BootStrapper"),
    "ClasswiseWrapper": ("paddle.metric.wrappers", "ClasswiseWrapper"),
    "MinMaxMetric": ("paddle.metric.wrappers", "MinMaxMetric"),
    "MetricTracker": ("paddle.metric.wrappers", "MetricTracker"),
    "MultitaskWrapper": ("paddle.metric.wrappers", "MultitaskWrapper"),
    "MultioutputWrapper": ("paddle.metric.wrappers", "MultioutputWrapper"),
    "Running": ("paddle.metric.wrappers", "Running"),
    "FeatureShareMetric": ("paddle.metric.wrappers", "FeatureShare"),
}


def __getattr__(name: str) -> object:
    """Lazy imports for domain-specific metrics."""
    if name in _LAZY_IMPORTS:
        module_path, attr_name = _LAZY_IMPORTS[name]
        import importlib

        mod = importlib.import_module(module_path)
        return getattr(mod, attr_name)
    raise AttributeError(f"module 'paddle.metric' has no attribute {name!r}")


def __dir__() -> list[str]:
    return __all__ + list(_LAZY_IMPORTS.keys())
