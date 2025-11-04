# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

import functools

_SENTINEL = object()


def compute_legacy_reduction(reduce_val, size_average_val):
    if reduce_val is False:
        return 'none'
    if reduce_val is True:
        return 'sum' if size_average_val is False else 'mean'
    return 'sum' if size_average_val is False else 'mean'


def check_deprecated_params(cls):
    """
    Class decorator: intercept deprecated 'reduce' and 'size_average' kwargs.
    - If provided by user, raise ValueError immediately with the suggested
      'reduction' value.
    - Pop these keys so that future policies (e.g., warn-and-continue) won't
      pass unknown kwargs into the original __init__.
    """
    original_init = cls.__init__

    @functools.wraps(original_init)
    def new_init(self, *args, **kwargs):
        reduce_raw = kwargs.pop('reduce', _SENTINEL)
        size_avg_raw = kwargs.pop('size_average', _SENTINEL)

        has_reduce = reduce_raw is not _SENTINEL
        has_size_avg = size_avg_raw is not _SENTINEL

        if has_reduce or has_size_avg:
            reduce_val = None if reduce_raw is _SENTINEL else reduce_raw
            size_avg_val = None if size_avg_raw is _SENTINEL else size_avg_raw
            suggested = compute_legacy_reduction(reduce_val, size_avg_val)
            raise ValueError(
                f"{cls.__name__} no longer supports 'reduce' and 'size_average'."
                f"\nDetected: reduce={reduce_val}, size_average={size_avg_val}"
                f"\nPlease use: reduction='{suggested}' instead."
            )

        return original_init(self, *args, **kwargs)

    cls.__init__ = new_init
    return cls
