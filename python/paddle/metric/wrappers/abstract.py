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

from typing import Any, Callable

from paddle.metric.metric import Metric


class WrapperMetric(Metric):
    """Abstract base class for wrapper metrics.

    Wrapper metrics are characterized by them wrapping another metric, and forwarding all calls to the wrapped metric.
    This means that all logic regarding synchronization etc. is handled by the wrapped metric, and the wrapper metric
    should not do anything in this regard.

    This class therefore overwrites all methods that are related to synchronization, and does nothing in them.

    Additionally, the forward method is not implemented by default as custom logic is required for each wrapper metric.

    """

    def _wrap_update(self, update: Callable) -> Callable:
        """Overwrite to do nothing, because the default wrapped functionality is handled by the wrapped metric."""
        return update

    def _wrap_compute(self, compute: Callable) -> Callable:
        """Overwrite to do nothing, because the default wrapped functionality is handled by the wrapped metric."""
        return compute

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Overwrite to do nothing, because the default wrapped functionality is handled by the wrapped metric."""
        raise NotImplementedError
