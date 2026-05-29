from __future__ import annotations

from paddle.metric.functional.regression.concordance import concordance_corrcoef
from paddle.metric.functional.regression.cosine_similarity import (
    cosine_similarity,
)
from paddle.metric.functional.regression.crps import (
    continuous_ranked_probability_score,
)
from paddle.metric.functional.regression.csi import critical_success_index
from paddle.metric.functional.regression.explained_variance import (
    explained_variance,
)
from paddle.metric.functional.regression.js_divergence import (
    jensen_shannon_divergence,
)
from paddle.metric.functional.regression.kendall import kendall_rank_corrcoef
from paddle.metric.functional.regression.kl_divergence import kl_divergence
from paddle.metric.functional.regression.log_cosh import log_cosh_error
from paddle.metric.functional.regression.log_mse import mean_squared_log_error
from paddle.metric.functional.regression.mae import mean_absolute_error
from paddle.metric.functional.regression.mape import (
    mean_absolute_percentage_error,
)
from paddle.metric.functional.regression.minkowski import minkowski_distance
from paddle.metric.functional.regression.mse import mean_squared_error
from paddle.metric.functional.regression.nrmse import (
    normalized_root_mean_squared_error,
)
from paddle.metric.functional.regression.pearson import pearson_corrcoef
from paddle.metric.functional.regression.r2 import r2_score
from paddle.metric.functional.regression.rse import relative_squared_error
from paddle.metric.functional.regression.spearman import spearman_corrcoef
from paddle.metric.functional.regression.symmetric_mape import (
    symmetric_mean_absolute_percentage_error,
)
from paddle.metric.functional.regression.tweedie_deviance import (
    tweedie_deviance_score,
)
from paddle.metric.functional.regression.wmape import (
    weighted_mean_absolute_percentage_error,
)

__all__ = [
    "concordance_corrcoef",
    "continuous_ranked_probability_score",
    "cosine_similarity",
    "critical_success_index",
    "explained_variance",
    "jensen_shannon_divergence",
    "kendall_rank_corrcoef",
    "kl_divergence",
    "log_cosh_error",
    "mean_absolute_error",
    "mean_absolute_percentage_error",
    "mean_absolute_percentage_error",
    "mean_squared_error",
    "mean_squared_log_error",
    "minkowski_distance",
    "normalized_root_mean_squared_error",
    "pearson_corrcoef",
    "r2_score",
    "relative_squared_error",
    "spearman_corrcoef",
    "symmetric_mean_absolute_percentage_error",
    "tweedie_deviance_score",
    "weighted_mean_absolute_percentage_error",
]
