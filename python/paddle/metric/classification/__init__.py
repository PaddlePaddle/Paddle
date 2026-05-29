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

from paddle.metric.classification.accuracy import (
    Accuracy,
    BinaryAccuracy,
    MulticlassAccuracy,
    MultilabelAccuracy,
)
from paddle.metric.classification.auroc import (
    AUROC,
    BinaryAUROC,
    MulticlassAUROC,
    MultilabelAUROC,
)
from paddle.metric.classification.average_precision import (
    AveragePrecision,
    BinaryAveragePrecision,
    MulticlassAveragePrecision,
    MultilabelAveragePrecision,
)
from paddle.metric.classification.calibration_error import (
    BinaryCalibrationError,
    CalibrationError,
    MulticlassCalibrationError,
)
from paddle.metric.classification.cohen_kappa import (
    BinaryCohenKappa,
    CohenKappa,
    MulticlassCohenKappa,
)
from paddle.metric.classification.confusion_matrix import (
    BinaryConfusionMatrix,
    ConfusionMatrix,
    MulticlassConfusionMatrix,
    MultilabelConfusionMatrix,
)
from paddle.metric.classification.eer import (
    EER,
    BinaryEER,
    MulticlassEER,
    MultilabelEER,
)
from paddle.metric.classification.exact_match import (
    ExactMatch,
    MulticlassExactMatch,
    MultilabelExactMatch,
)
from paddle.metric.classification.f_beta import (
    BinaryF1Score,
    BinaryFBetaScore,
    F1Score,
    FBetaScore,
    MulticlassF1Score,
    MulticlassFBetaScore,
    MultilabelF1Score,
    MultilabelFBetaScore,
)
from paddle.metric.classification.group_fairness import (
    BinaryFairness,
    BinaryGroupStatRates,
)
from paddle.metric.classification.hamming import (
    BinaryHammingDistance,
    HammingDistance,
    MulticlassHammingDistance,
    MultilabelHammingDistance,
)
from paddle.metric.classification.hinge import (
    BinaryHingeLoss,
    HingeLoss,
    MulticlassHingeLoss,
)
from paddle.metric.classification.jaccard import (
    BinaryJaccardIndex,
    JaccardIndex,
    MulticlassJaccardIndex,
    MultilabelJaccardIndex,
)
from paddle.metric.classification.logauc import (
    BinaryLogAUC,
    LogAUC,
    MulticlassLogAUC,
    MultilabelLogAUC,
)
from paddle.metric.classification.matthews_corrcoef import (
    BinaryMatthewsCorrCoef,
    MatthewsCorrCoef,
    MulticlassMatthewsCorrCoef,
    MultilabelMatthewsCorrCoef,
)
from paddle.metric.classification.negative_predictive_value import (
    BinaryNegativePredictiveValue,
    MulticlassNegativePredictiveValue,
    MultilabelNegativePredictiveValue,
    NegativePredictiveValue,
)
from paddle.metric.classification.precision_fixed_recall import (
    BinaryPrecisionAtFixedRecall,
    MulticlassPrecisionAtFixedRecall,
    MultilabelPrecisionAtFixedRecall,
    PrecisionAtFixedRecall,
)
from paddle.metric.classification.precision_recall import (
    BinaryPrecision,
    BinaryRecall,
    MulticlassPrecision,
    MulticlassRecall,
    MultilabelPrecision,
    MultilabelRecall,
    Precision,
    Recall,
)
from paddle.metric.classification.precision_recall_curve import (
    BinaryPrecisionRecallCurve,
    MulticlassPrecisionRecallCurve,
    MultilabelPrecisionRecallCurve,
    PrecisionRecallCurve,
)
from paddle.metric.classification.ranking import (
    MultilabelCoverageError,
    MultilabelRankingAveragePrecision,
    MultilabelRankingLoss,
)
from paddle.metric.classification.recall_fixed_precision import (
    BinaryRecallAtFixedPrecision,
    MulticlassRecallAtFixedPrecision,
    MultilabelRecallAtFixedPrecision,
    RecallAtFixedPrecision,
)
from paddle.metric.classification.roc import (
    ROC,
    BinaryROC,
    MulticlassROC,
    MultilabelROC,
)
from paddle.metric.classification.sensitivity_specificity import (
    BinarySensitivityAtSpecificity,
    MulticlassSensitivityAtSpecificity,
    MultilabelSensitivityAtSpecificity,
    SensitivityAtSpecificity,
)
from paddle.metric.classification.specificity import (
    BinarySpecificity,
    MulticlassSpecificity,
    MultilabelSpecificity,
    Specificity,
)
from paddle.metric.classification.specificity_sensitivity import (
    BinarySpecificityAtSensitivity,
    MulticlassSpecificityAtSensitivity,
    MultilabelSpecificityAtSensitivity,
    SpecificityAtSensitivity,
)
from paddle.metric.classification.stat_scores import (
    BinaryStatScores,
    MulticlassStatScores,
    MultilabelStatScores,
    StatScores,
)

__all__ = [
    "AUROC",
    "EER",
    "ROC",
    "Accuracy",
    "AveragePrecision",
    "BinaryAUROC",
    "BinaryAccuracy",
    "BinaryAveragePrecision",
    "BinaryCalibrationError",
    "BinaryCohenKappa",
    "BinaryConfusionMatrix",
    "BinaryEER",
    "BinaryF1Score",
    "BinaryFBetaScore",
    "BinaryFairness",
    "BinaryGroupStatRates",
    "BinaryHammingDistance",
    "BinaryHingeLoss",
    "BinaryJaccardIndex",
    "BinaryLogAUC",
    "BinaryMatthewsCorrCoef",
    "BinaryNegativePredictiveValue",
    "BinaryPrecision",
    "BinaryPrecisionAtFixedRecall",
    "BinaryPrecisionRecallCurve",
    "BinaryROC",
    "BinaryRecall",
    "BinaryRecallAtFixedPrecision",
    "BinarySensitivityAtSpecificity",
    "BinarySpecificity",
    "BinarySpecificityAtSensitivity",
    "BinaryStatScores",
    "CalibrationError",
    "CohenKappa",
    "ConfusionMatrix",
    "ExactMatch",
    "F1Score",
    "FBetaScore",
    "HammingDistance",
    "HingeLoss",
    "JaccardIndex",
    "LogAUC",
    "MatthewsCorrCoef",
    "MulticlassAUROC",
    "MulticlassAccuracy",
    "MulticlassAveragePrecision",
    "MulticlassCalibrationError",
    "MulticlassCohenKappa",
    "MulticlassConfusionMatrix",
    "MulticlassEER",
    "MulticlassExactMatch",
    "MulticlassF1Score",
    "MulticlassFBetaScore",
    "MulticlassHammingDistance",
    "MulticlassHingeLoss",
    "MulticlassJaccardIndex",
    "MulticlassLogAUC",
    "MulticlassMatthewsCorrCoef",
    "MulticlassNegativePredictiveValue",
    "MulticlassPrecision",
    "MulticlassPrecisionAtFixedRecall",
    "MulticlassPrecisionRecallCurve",
    "MulticlassROC",
    "MulticlassRecall",
    "MulticlassRecallAtFixedPrecision",
    "MulticlassSensitivityAtSpecificity",
    "MulticlassSpecificity",
    "MulticlassSpecificityAtSensitivity",
    "MulticlassStatScores",
    "MultilabelAUROC",
    "MultilabelAccuracy",
    "MultilabelAveragePrecision",
    "MultilabelConfusionMatrix",
    "MultilabelCoverageError",
    "MultilabelEER",
    "MultilabelExactMatch",
    "MultilabelF1Score",
    "MultilabelFBetaScore",
    "MultilabelHammingDistance",
    "MultilabelJaccardIndex",
    "MultilabelLogAUC",
    "MultilabelMatthewsCorrCoef",
    "MultilabelNegativePredictiveValue",
    "MultilabelPrecision",
    "MultilabelPrecisionAtFixedRecall",
    "MultilabelPrecisionRecallCurve",
    "MultilabelROC",
    "MultilabelRankingAveragePrecision",
    "MultilabelRankingLoss",
    "MultilabelRecall",
    "MultilabelRecallAtFixedPrecision",
    "MultilabelSensitivityAtSpecificity",
    "MultilabelSpecificity",
    "MultilabelSpecificityAtSensitivity",
    "MultilabelStatScores",
    "NegativePredictiveValue",
    "Precision",
    "PrecisionAtFixedRecall",
    "PrecisionRecallCurve",
    "Recall",
    "RecallAtFixedPrecision",
    "SensitivityAtSpecificity",
    "Specificity",
    "SpecificityAtSensitivity",
    "StatScores",
]
