from __future__ import annotations

from paddle.metric.functional.classification.accuracy import (
    accuracy,
    binary_accuracy,
    multiclass_accuracy,
    multilabel_accuracy,
)
from paddle.metric.functional.classification.auroc import (
    auroc,
    binary_auroc,
    multiclass_auroc,
    multilabel_auroc,
)
from paddle.metric.functional.classification.average_precision import (
    average_precision,
    binary_average_precision,
    multiclass_average_precision,
    multilabel_average_precision,
)
from paddle.metric.functional.classification.calibration_error import (
    binary_calibration_error,
    calibration_error,
    multiclass_calibration_error,
)
from paddle.metric.functional.classification.cohen_kappa import (
    binary_cohen_kappa,
    cohen_kappa,
    multiclass_cohen_kappa,
)
from paddle.metric.functional.classification.confusion_matrix import (
    binary_confusion_matrix,
    confusion_matrix,
    multiclass_confusion_matrix,
    multilabel_confusion_matrix,
)
from paddle.metric.functional.classification.eer import (
    binary_eer,
    eer,
    multiclass_eer,
    multilabel_eer,
)
from paddle.metric.functional.classification.exact_match import (
    exact_match,
    multiclass_exact_match,
    multilabel_exact_match,
)
from paddle.metric.functional.classification.f_beta import (
    binary_f1_score,
    binary_fbeta_score,
    f1_score,
    fbeta_score,
    multiclass_f1_score,
    multiclass_fbeta_score,
    multilabel_f1_score,
    multilabel_fbeta_score,
)
from paddle.metric.functional.classification.group_fairness import (
    binary_fairness,
    binary_groups_stat_rates,
    demographic_parity,
    equal_opportunity,
)
from paddle.metric.functional.classification.hamming import (
    binary_hamming_distance,
    hamming_distance,
    multiclass_hamming_distance,
    multilabel_hamming_distance,
)
from paddle.metric.functional.classification.hinge import (
    binary_hinge_loss,
    hinge_loss,
    multiclass_hinge_loss,
)
from paddle.metric.functional.classification.jaccard import (
    binary_jaccard_index,
    jaccard_index,
    multiclass_jaccard_index,
    multilabel_jaccard_index,
)
from paddle.metric.functional.classification.logauc import (
    binary_logauc,
    logauc,
    multiclass_logauc,
    multilabel_logauc,
)
from paddle.metric.functional.classification.matthews_corrcoef import (
    binary_matthews_corrcoef,
    matthews_corrcoef,
    multiclass_matthews_corrcoef,
    multilabel_matthews_corrcoef,
)
from paddle.metric.functional.classification.negative_predictive_value import (
    binary_negative_predictive_value,
    multiclass_negative_predictive_value,
    multilabel_negative_predictive_value,
    negative_predictive_value,
)
from paddle.metric.functional.classification.precision_fixed_recall import (
    binary_precision_at_fixed_recall,
    multiclass_precision_at_fixed_recall,
    multilabel_precision_at_fixed_recall,
    precision_at_fixed_recall,
)
from paddle.metric.functional.classification.precision_recall import (
    binary_precision,
    binary_recall,
    multiclass_precision,
    multiclass_recall,
    multilabel_precision,
    multilabel_recall,
    precision,
    recall,
)
from paddle.metric.functional.classification.precision_recall_curve import (
    binary_precision_recall_curve,
    multiclass_precision_recall_curve,
    multilabel_precision_recall_curve,
    precision_recall_curve,
)
from paddle.metric.functional.classification.ranking import (
    multilabel_coverage_error,
    multilabel_ranking_average_precision,
    multilabel_ranking_loss,
)
from paddle.metric.functional.classification.recall_fixed_precision import (
    binary_recall_at_fixed_precision,
    multiclass_recall_at_fixed_precision,
    multilabel_recall_at_fixed_precision,
    recall_at_fixed_precision,
)
from paddle.metric.functional.classification.roc import (
    binary_roc,
    multiclass_roc,
    multilabel_roc,
    roc,
)
from paddle.metric.functional.classification.sensitivity_specificity import (
    binary_sensitivity_at_specificity,
    multiclass_sensitivity_at_specificity,
    multilabel_sensitivity_at_specificity,
    sensitivity_at_specificity,
)
from paddle.metric.functional.classification.specificity import (
    binary_specificity,
    multiclass_specificity,
    multilabel_specificity,
    specificity,
)
from paddle.metric.functional.classification.specificity_sensitivity import (
    binary_specificity_at_sensitivity,
    multiclass_specificity_at_sensitivity,
    multilabel_specificity_at_sensitivity,
    specificity_at_sensitivity,
)
from paddle.metric.functional.classification.stat_scores import (
    binary_stat_scores,
    multiclass_stat_scores,
    multilabel_stat_scores,
    stat_scores,
)

__all__ = [
    "accuracy",
    "auroc",
    "average_precision",
    "binary_accuracy",
    "binary_auroc",
    "binary_average_precision",
    "binary_calibration_error",
    "binary_cohen_kappa",
    "binary_confusion_matrix",
    "binary_eer",
    "binary_f1_score",
    "binary_fairness",
    "binary_fbeta_score",
    "binary_groups_stat_rates",
    "binary_hamming_distance",
    "binary_hinge_loss",
    "binary_jaccard_index",
    "binary_logauc",
    "binary_matthews_corrcoef",
    "binary_negative_predictive_value",
    "binary_precision",
    "binary_precision_at_fixed_recall",
    "binary_precision_recall_curve",
    "binary_recall",
    "binary_recall_at_fixed_precision",
    "binary_roc",
    "binary_sensitivity_at_specificity",
    "binary_specificity",
    "binary_specificity_at_sensitivity",
    "binary_stat_scores",
    "calibration_error",
    "cohen_kappa",
    "confusion_matrix",
    "demographic_parity",
    "eer",
    "equal_opportunity",
    "exact_match",
    "f1_score",
    "fbeta_score",
    "hamming_distance",
    "hinge_loss",
    "jaccard_index",
    "logauc",
    "matthews_corrcoef",
    "multiclass_accuracy",
    "multiclass_auroc",
    "multiclass_average_precision",
    "multiclass_calibration_error",
    "multiclass_cohen_kappa",
    "multiclass_confusion_matrix",
    "multiclass_eer",
    "multiclass_exact_match",
    "multiclass_f1_score",
    "multiclass_fbeta_score",
    "multiclass_hamming_distance",
    "multiclass_hinge_loss",
    "multiclass_jaccard_index",
    "multiclass_logauc",
    "multiclass_matthews_corrcoef",
    "multiclass_negative_predictive_value",
    "multiclass_precision",
    "multiclass_precision_at_fixed_recall",
    "multiclass_precision_recall_curve",
    "multiclass_recall",
    "multiclass_recall_at_fixed_precision",
    "multiclass_roc",
    "multiclass_sensitivity_at_specificity",
    "multiclass_specificity",
    "multiclass_specificity_at_sensitivity",
    "multiclass_stat_scores",
    "multilabel_accuracy",
    "multilabel_auroc",
    "multilabel_average_precision",
    "multilabel_confusion_matrix",
    "multilabel_coverage_error",
    "multilabel_eer",
    "multilabel_exact_match",
    "multilabel_f1_score",
    "multilabel_fbeta_score",
    "multilabel_hamming_distance",
    "multilabel_jaccard_index",
    "multilabel_logauc",
    "multilabel_matthews_corrcoef",
    "multilabel_negative_predictive_value",
    "multilabel_precision",
    "multilabel_precision_at_fixed_recall",
    "multilabel_precision_recall_curve",
    "multilabel_ranking_average_precision",
    "multilabel_ranking_loss",
    "multilabel_recall",
    "multilabel_recall_at_fixed_precision",
    "multilabel_roc",
    "multilabel_sensitivity_at_specificity",
    "multilabel_specificity",
    "multilabel_specificity_at_sensitivity",
    "multilabel_stat_scores",
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
