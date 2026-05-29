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

from typing import Any

from paddle.metric.metric import Metric


class _ClassificationTaskWrapper(Metric):
    """Base class for wrapper metrics for classification tasks."""

    def update(self, *args: Any, **kwargs: Any) -> None:
        """Update metric state."""
        raise NotImplementedError(
            f"{self.__class__.__name__} metric does not have a global `update` method. Use the task specific metric."
        )

    def compute(self) -> None:
        """Compute metric."""
        raise NotImplementedError(
            f"{self.__class__.__name__} metric does not have a global `compute` method. Use the task specific metric."
        )
