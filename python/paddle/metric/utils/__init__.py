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

"""Paddle metric utilities."""

from paddle.metric.utils.checks import check_forward_full_state_property
from paddle.metric.utils.data import (
    apply_to_collection,
    dim_zero_cat,
    dim_zero_max,
    dim_zero_mean,
    dim_zero_min,
    dim_zero_sum,
)
from paddle.metric.utils.distributed import (
    class_reduce,
    gather_all_tensors,
    reduce,
)
from paddle.metric.utils.prints import (
    rank_zero_debug,
    rank_zero_info,
    rank_zero_warn,
)

__all__ = [
    "apply_to_collection",
    "check_forward_full_state_property",
    "class_reduce",
    "dim_zero_cat",
    "dim_zero_max",
    "dim_zero_mean",
    "dim_zero_min",
    "dim_zero_sum",
    "gather_all_tensors",
    "rank_zero_debug",
    "rank_zero_info",
    "rank_zero_warn",
    "reduce",
]
