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

from paddle.metric.regression.concordance import ConcordanceCorrCoef
from paddle.metric.regression.cosine_similarity import CosineSimilarity
from paddle.metric.regression.crps import ContinuousRankedProbabilityScore
from paddle.metric.regression.csi import CriticalSuccessIndex
from paddle.metric.regression.explained_variance import ExplainedVariance
from paddle.metric.regression.js_divergence import JensenShannonDivergence
from paddle.metric.regression.kendall import KendallRankCorrCoef
from paddle.metric.regression.kl_divergence import KLDivergence
from paddle.metric.regression.log_cosh import LogCoshError
from paddle.metric.regression.log_mse import MeanSquaredLogError
from paddle.metric.regression.mae import MeanAbsoluteError
from paddle.metric.regression.mape import MeanAbsolutePercentageError
from paddle.metric.regression.minkowski import MinkowskiDistance
from paddle.metric.regression.mse import MeanSquaredError
from paddle.metric.regression.nrmse import NormalizedRootMeanSquaredError
from paddle.metric.regression.pearson import PearsonCorrCoef
from paddle.metric.regression.r2 import R2Score
from paddle.metric.regression.rse import RelativeSquaredError
from paddle.metric.regression.spearman import SpearmanCorrCoef
from paddle.metric.regression.symmetric_mape import (
    SymmetricMeanAbsolutePercentageError,
)
from paddle.metric.regression.tweedie_deviance import TweedieDevianceScore
from paddle.metric.regression.wmape import WeightedMeanAbsolutePercentageError

__all__ = [
    "ConcordanceCorrCoef",
    "ContinuousRankedProbabilityScore",
    "CosineSimilarity",
    "CriticalSuccessIndex",
    "ExplainedVariance",
    "JensenShannonDivergence",
    "KLDivergence",
    "KendallRankCorrCoef",
    "LogCoshError",
    "MeanAbsoluteError",
    "MeanAbsolutePercentageError",
    "MeanSquaredError",
    "MeanSquaredLogError",
    "MinkowskiDistance",
    "NormalizedRootMeanSquaredError",
    "PearsonCorrCoef",
    "R2Score",
    "RelativeSquaredError",
    "SpearmanCorrCoef",
    "SymmetricMeanAbsolutePercentageError",
    "TweedieDevianceScore",
    "WeightedMeanAbsolutePercentageError",
]
