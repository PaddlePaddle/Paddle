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

import logging
import os
import warnings
from functools import partial, wraps
from typing import Any, Callable

# Set up logger for paddle.metric
log = logging.getLogger("paddle.metric")


def rank_zero_only(fn: Callable) -> Callable:
    """Call a function only on rank 0 in distributed settings."""

    @wraps(fn)
    def wrapped_fn(*args: Any, **kwargs: Any) -> Any:
        if rank_zero_only.rank == 0:
            return fn(*args, **kwargs)
        return None

    return wrapped_fn


rank_zero_only.rank = getattr(
    rank_zero_only, "rank", int(os.environ.get("LOCAL_RANK", 0))
)


def _warn(*args: Any, **kwargs: Any) -> None:
    warnings.warn(*args, **kwargs)


def _info(*args: Any, **kwargs: Any) -> None:
    log.info(*args, **kwargs)


def _debug(*args: Any, **kwargs: Any) -> None:
    log.debug(*args, **kwargs)


rank_zero_debug = rank_zero_only(_debug)
rank_zero_info = rank_zero_only(_info)
rank_zero_warn = rank_zero_only(_warn)
_future_warning = partial(warnings.warn, category=FutureWarning)


def _deprecated_root_import_class(name: str, domain: str) -> None:
    """Warn user that importing class from deprecated location."""
    _future_warning(
        f"Importing `{name}` from `paddle.metric` was deprecated. Import `{name}` from `paddle.metric.{domain}` instead."
    )


def _deprecated_root_import_func(name: str, domain: str) -> None:
    """Warn user that importing function from deprecated location."""
    _future_warning(
        f"Importing `{name}` from `paddle.metric.functional` was deprecated. Import `{name}` from `paddle.metric.{domain}` instead."
    )
